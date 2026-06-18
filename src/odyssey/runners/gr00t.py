"""GR00T training runner.

Fine-tunes an NVIDIA Isaac GR00T checkpoint (N1.7 family) by invoking
the upstream ``gr00t.experiment.launch_finetune`` entry point as a
subprocess — the Isaac-GR00T package must be installed in the
environment (``pip install -e`` of https://github.com/NVIDIA/Isaac-GR00T).

Mission ``config:`` keys map 1:1 onto the upstream CLI flags with
underscores translated to dashes (``embodiment_tag`` →
``--embodiment-tag``), the same passthrough pattern the OpenVLA runner
uses for draccus flags.

Routing: GR00T registers as a wildcard training runner *behind* OpenVLA,
so it is selected explicitly via the task-level ``config: {runner: gr00t}``
override (see ``examples/quickstart-gr00t/mission.yaml``). The eventual
model-family-aware dispatch is a Lovell-platform concern; the explicit
override is the OSS-local mechanism.

Licensing note: GR00T checkpoints are distributed under NVIDIA's terms.
This runner contains no NVIDIA code — it shells out to the user's own
Isaac-GR00T checkout — but users must accept the upstream license to
download the weights.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from odyssey.runners.base import (
    WILDCARD_TYPE,
    Runner,
    TaskContext,
)

# Shared with the OpenVLA runner; extract to a common module when a
# third runner needs them.
from odyssey.runners.openvla import _flatten_config, _resolve_and_fetch_hf_model
from odyssey.runners.subprocess import (
    TrainingProcessSpec,
    output_path,
    run_training_subprocess,
)
from odyssey.spec.refs import DatasetSource, HFModelRef
from odyssey.spec.tasks import TaskKind, TrainingTask, TrainingType

logger = logging.getLogger(__name__)


_ENTRY_MODULE = "gr00t.experiment.launch_finetune"
_DEFAULT_REPO_PATH = "/srv/isaac-gr00t"

# GR00T's finetune drives a HuggingFace Trainer, so stdout carries the
# Trainer's dict-style log lines and tqdm progress bars:
#   "{'loss': 0.693, 'grad_norm': 1.2, 'learning_rate': 1e-05, 'epoch': 0.25}"
#   " 10%|█         | 100/1000 [00:42<06:18,  2.38it/s]"
#   "Saving model checkpoint to ./checkpoint-500"
_GR00T_LOSS_RE = re.compile(r"'loss':\s*([\d.eE+-]+)")
_GR00T_EPOCH_RE = re.compile(r"'epoch':\s*([\d.]+)")
_GR00T_TQDM_RE = re.compile(r"\b(\d+)/(\d+)\s*\[")
_GR00T_SAVE_RE = re.compile(
    r"(?i)(saving model checkpoint|model weights saved|saving.*(checkpoint|model))"
)
_GR00T_DATASET_RE = re.compile(
    r"(?i)\b(loading|generating|building)\b.*\b(dataset|episodes|lerobot)\b"
)
_GR00T_LOAD_RE = re.compile(
    r"(?i)\bloading\b.*\b(checkpoint|shards|weights|model|processor|tokenizer)\b"
)


def parse_gr00t_line(line: str) -> dict[str, Any] | None:
    """Extract a progress-event dict from a GR00T finetune stdout line.

    Public for tests and for users embedding GR00T stdout parsing in a
    custom runner.
    """
    loss_match = _GR00T_LOSS_RE.search(line)
    if loss_match:
        payload: dict[str, Any] = {
            "stage": "executing",
            "step": "training_step",
            "step_label": f"loss={loss_match.group(1)}",
        }
        epoch_match = _GR00T_EPOCH_RE.search(line)
        if epoch_match:
            payload["step_label"] += f" epoch={epoch_match.group(1)}"
        return payload
    tqdm_match = _GR00T_TQDM_RE.search(line)
    if tqdm_match:
        return {
            "stage": "executing",
            "step": "training_step",
            "step_index": int(tqdm_match.group(1)),
            "step_total": int(tqdm_match.group(2)),
        }
    if _GR00T_SAVE_RE.search(line):
        return {"stage": "checkpoint_saving"}
    if _GR00T_DATASET_RE.search(line):
        return {"stage": "dataset_loading"}
    if _GR00T_LOAD_RE.search(line):
        return {"stage": "model_loading", "step": "load_artifacts"}
    return None


def _path_env_for_hf_id(hf_id: str) -> str | None:
    """Convention: ``nvidia/GR00T-N1.7-3B`` → ``NVIDIA_GR00T_N1_7_3B_PATH``.

    Same scheme as the OpenVLA runner, extended with ``.`` → ``_`` so
    dotted GR00T version ids form valid env-var names.
    """
    slug = hf_id.replace("/", "_").replace("-", "_").replace(".", "_").upper()
    return os.getenv(f"{slug}_PATH")


def _resolve_dataset_path(task: TrainingTask) -> str | None:
    """Resolve the task's dataset ref to a path/id for ``--dataset-path``.

    Relative ``local`` refs resolve against ``$ISAAC_GR00T_REPO_PATH`` —
    the quickstart points at the demo data shipped inside the
    Isaac-GR00T checkout. Everything else (absolute paths, HF ids)
    passes through for the upstream loader to interpret.
    """
    if task.dataset is None:
        return None
    ref = task.dataset.ref
    if task.dataset.source == DatasetSource.LOCAL and not os.path.isabs(ref):
        repo_path = os.getenv("ISAAC_GR00T_REPO_PATH", _DEFAULT_REPO_PATH)
        return os.path.join(repo_path, ref)
    return ref


def build_gr00t_argv(
    *,
    task: TrainingTask,
    agent_model_base: str | None,
    output_dir: Path,
) -> list[str]:
    """Build the GR00T ``launch_finetune`` CLI argv.

    Resolution order for ``--base-model-path``:
      1. ``task.config["base_model_path"]`` (operator override, also
         where the runner injects a starting checkpoint or fetched HF dir)
      2. ``<HF_ID>_PATH`` env var derived from the agent's HF base id
      3. ``agent_model_base`` itself (an HF hub id; first call triggers
         a download by the upstream loader)
    """
    config = task.config or {}
    base_model_path = (
        config.get("base_model_path")
        or (_path_env_for_hf_id(agent_model_base) if agent_model_base else None)
        or agent_model_base
    )
    if not base_model_path:
        raise RuntimeError(
            "GR00T runner: cannot resolve base_model_path. The agent's "
            "model must be a HuggingFace ref, or the runner must pre-fill "
            "task.config['base_model_path'] from a starting checkpoint."
        )

    dataset_path = config.get("dataset_path") or _resolve_dataset_path(task)

    argv: list[str] = ["--base-model-path", str(base_model_path)]
    if dataset_path:
        argv += ["--dataset-path", str(dataset_path)]
    argv += ["--output-dir", str(output_dir)]

    # Pass through remaining config keys as kebab-case flags. ``runner``
    # is the registry-override key, not an upstream flag.
    handled = {"base_model_path", "dataset_path", "runner"}
    repo_path = os.getenv("ISAAC_GR00T_REPO_PATH", _DEFAULT_REPO_PATH)
    for key, value in _flatten_config(config):
        if key in handled:
            continue
        # A relative ``modality_config_path`` resolves against the Isaac-GR00T
        # checkout (same convention as the dataset ref). The NEW_EMBODIMENT tag
        # *requires* a modality config file, so this lets a mission point at a
        # repo-relative config (e.g. examples/SO100/so100_config.py) portably.
        if key == "modality_config_path" and isinstance(value, str) and not os.path.isabs(value):
            value = os.path.join(repo_path, value)
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return argv


def _resolve_output_checkpoint(output_dir: Path) -> Path | None:
    """Locate the trained checkpoint under ``--output-dir``.

    The Trainer saves the final model at the output root (``config.json``
    plus weights); intermediate saves land in ``checkpoint-<step>/``
    subdirs. Prefer the root; fall back to the newest checkpoint dir.
    The fetched HF base model lives in ``base_model/`` and is never a
    candidate.
    """
    if not output_dir.is_dir():
        return None
    marker_files = ("config.json", "model.safetensors")
    if any((output_dir / fname).is_file() for fname in marker_files):
        return output_dir
    candidates = [
        entry
        for entry in output_dir.iterdir()
        if entry.is_dir()
        and entry.name.startswith("checkpoint-")
        and any((entry / fname).is_file() for fname in marker_files)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


class GR00TRunner(Runner):
    """Fine-tune a GR00T model. Subprocess-based — actual training
    happens in the upstream ``launch_finetune`` entry point."""

    @property
    def name(self) -> str:
        return "gr00t"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.TRAINING}

    @property
    def supported_types(self) -> set[str]:
        # Registered behind OpenVLA's wildcard; reached via the
        # task-level ``config: {runner: gr00t}`` override.
        return {WILDCARD_TYPE}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        spec = context.task.spec
        if not isinstance(spec, TrainingTask):
            raise TypeError(
                f"GR00TRunner expects TrainingTask, got {type(spec).__name__}"
            )
        if context.agent is None:
            raise RuntimeError(
                "GR00TRunner: TaskContext.agent is None — training tasks "
                "must be invoked through the engine, which resolves the "
                "agent from spec.robot.agents[task.agent_id]."
            )

        output_dir = output_path(context)

        # Decide what base_model_path the subprocess starts from — same
        # ladder as the OpenVLA runner: prior checkpoint, then provider
        # fetch of the agent's HF ref, then env-var / HF-id fallback
        # inside build_gr00t_argv.
        config = dict(spec.config)
        agent_model_base: str | None = None
        if context.starting_checkpoint is not None:
            config.setdefault("base_model_path", context.starting_checkpoint)
        elif (
            context.providers is not None
            and isinstance(context.agent.model, HFModelRef)
        ):
            resolved_path = await _resolve_and_fetch_hf_model(
                context, context.agent.model, output_dir / "base_model"
            )
            config.setdefault("base_model_path", str(resolved_path))

        if isinstance(context.agent.model, HFModelRef):
            agent_model_base = context.agent.model.base

        process_spec = TrainingProcessSpec(
            entry_module=_ENTRY_MODULE,
            argv_extra=build_gr00t_argv(
                task=spec.model_copy(update={"config": config}),
                agent_model_base=agent_model_base,
                output_dir=output_dir,
            ),
            line_parser=parse_gr00t_line,
        )

        rc = await run_training_subprocess(context, process_spec)
        if context.cancelled():
            logger.info("GR00T task %s cancelled by user", context.task.id)
            return {"cancelled": True}
        if rc != 0:
            raise RuntimeError(f"gr00t launch_finetune exited with code {rc}")

        checkpoint = _resolve_output_checkpoint(output_dir)
        if checkpoint is None:
            raise RuntimeError(
                f"gr00t launch_finetune finished but no checkpoint found "
                f"under {output_dir!r}"
            )
        return {
            "checkpoint_path": str(checkpoint),
            "agent_id": context.agent.id,
            "training_config": spec.config,
            "training_type": (
                spec.training_type.value
                if isinstance(spec.training_type, TrainingType)
                else spec.training_type
            ),
        }
