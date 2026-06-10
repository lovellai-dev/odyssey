"""OpenVLA training runner and inference policy.

Training
--------
Adapted from ``lai-inference/.../jobs/training/openvla_runner.py``.
OpenVLA uses draccus for config parsing — flags are flat
(``--vla_path``, ``--data_root_dir``, ``--batch_size``, etc.). LoRA is
the default fine-tuning mode (``--use_lora True``), producing a
HuggingFace-format adapter directory under ``run_root_dir``.

What's different from the lai-inference version:

  * Drops ``_stage_for_lai_trainer`` — OSS leaves the checkpoint in the
    task's output_dir for the user to consume. Lovell's hosted runner
    can re-add the staging step.
  * Uses ``odyssey.runners.subprocess`` instead of
    ``..jobs.training.subprocess_runner``.
  * Reads identifiers off ``ctx.task.spec`` instead of ``ctx``: the
    OSS TaskContext is spec-centric.

Scope B (v0.1.0-alpha) caveat: the runner expects the OpenVLA repo to be
on disk and either pointed at via ``$OPENVLA_REPO_PATH`` or located at
``/srv/openvla``. The first-class HF resolve+fetch flow lands in Batch 3
once the HF provider exists; for now ``vla_path`` is read from
``task.config["vla_path"]`` (or the env var pattern below) without
re-resolving against the model spec.

Inference policy
----------------
``make_openvla_policy()`` loads a LoRA-finetuned checkpoint via
HuggingFace transformers + peft and returns a ``Policy`` callable that
maps robosuite observation dicts to 7-DoF actions using the model's
built-in ``predict_action()`` method.

All heavy inference imports (transformers, peft, torch, PIL) are deferred
so the module can be imported in environments without GPU dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from odyssey.providers.base import ResolvedModel
from odyssey.runners.base import (
    WILDCARD_TYPE,
    Runner,
    TaskContext,
)
from odyssey.runners.subprocess import (
    TrainingProcessSpec,
    output_path,
    run_training_subprocess,
)
from odyssey.spec.refs import HFModelRef
from odyssey.spec.tasks import TaskKind, TrainingTask, TrainingType

logger = logging.getLogger(__name__)


# OpenVLA logs step progress like:
#   "step: 1234 | loss: 0.234 | grad_norm: 1.23 | lr: 5e-5"
_OPENVLA_STEP_RE = re.compile(
    r"\bstep:\s*(\d+)\s*\|\s*loss:\s*([\d.eE+-]+)", re.IGNORECASE
)
_OPENVLA_LOAD_RE = re.compile(
    r"(loading|Loading).*?(VLA|processor|tokenizer|checkpoint)"
)
_OPENVLA_DATASET_RE = re.compile(
    r"(building|Building|loading).*?(RLDS|dataset)", re.IGNORECASE
)
_OPENVLA_SAVE_RE = re.compile(
    r"(saving|Saved).*?(adapter|checkpoint|model)", re.IGNORECASE
)


_DEFAULT_REPO_PATH = "/srv/openvla"
_FINETUNE_SCRIPT_REL = "vla-scripts/finetune.py"


def parse_openvla_line(line: str) -> dict[str, Any] | None:
    """Extract a progress-event dict from an OpenVLA finetune stdout line.

    Public for tests and for users who want to embed OpenVLA stdout
    parsing in a custom runner.
    """
    m = _OPENVLA_STEP_RE.search(line)
    if m:
        return {
            "stage": "executing",
            "step": "training_step",
            "step_index": int(m.group(1)),
            "step_label": f"loss={m.group(2)}",
        }
    if _OPENVLA_LOAD_RE.search(line):
        return {"stage": "model_loading", "step": "load_artifacts"}
    if _OPENVLA_DATASET_RE.search(line):
        return {"stage": "dataset_loading"}
    if _OPENVLA_SAVE_RE.search(line):
        return {"stage": "checkpoint_saving"}
    return None


def _flatten_config(d: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for k, v in d.items():
        flat_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_flatten_config(v, flat_key))
        else:
            out.append((flat_key, v))
    return out


def _path_env_for_hf_id(hf_id: str) -> str | None:
    """Convention: ``openvla/openvla-7b`` → ``OPENVLA_OPENVLA_7B_PATH``."""
    slug = hf_id.replace("/", "_").replace("-", "_").upper()
    return os.getenv(f"{slug}_PATH")


def build_openvla_argv(
    *,
    task: TrainingTask,
    agent_model_base: str | None,
    output_dir: Path,
    run_id: str,
) -> list[str]:
    """Build the OpenVLA draccus CLI argv.

    Resolution order for ``vla_path``:
      1. ``task.config["vla_path"]`` (operator override, also where the
         runner injects a starting-checkpoint path or a fetched HF dir)
      2. ``<HF_ID>_PATH`` env var derived from the agent's HF base id
      3. ``agent_model_base`` itself (treated as an HF hub id; first
         call triggers a download)

    ``agent_model_base`` is the HF base id pulled off the agent's model
    ref (``AgentSpec.model.base`` for an HFModelRef). ``None`` when the
    agent's model isn't an HF ref — in which case ``vla_path`` must
    have been pre-filled by the runner via the config override.
    """
    config = task.config or {}
    vla_path = (
        config.get("vla_path")
        or (_path_env_for_hf_id(agent_model_base) if agent_model_base else None)
        or agent_model_base
    )
    if not vla_path:
        raise RuntimeError(
            "OpenVLA runner: cannot resolve vla_path. The agent's model "
            "must be a HuggingFace ref, or the runner must pre-fill "
            "task.config['vla_path'] from a starting checkpoint."
        )

    dataset_id = config.get("data_root_dir")
    if dataset_id is None and task.dataset is not None:
        dataset_id = task.dataset.ref

    # OXE dataset name: prefer dataset.ref (the OXE registry key) over
    # config fallback, so users declare the key in the dataset spec
    # rather than burying it in hyperparameter config.
    dataset_name = (
        task.dataset.ref if task.dataset else config.get("dataset_name", task.name)
    )

    argv: list[str] = ["--vla_path", str(vla_path)]
    if dataset_id:
        argv += ["--data_root_dir", str(dataset_id)]
    argv += ["--dataset_name", str(dataset_name)]
    argv += ["--run_root_dir", str(output_dir)]
    argv += ["--adapter_tmp_dir", str(output_dir / "adapter_tmp")]
    argv += ["--run_id", run_id]
    if "use_lora" not in config:
        argv += ["--use_lora", "True"]

    # Pass through any remaining flat overrides (skip the keys we
    # already injected). `use_lora` is intentionally NOT in `handled` —
    # the dedicated branch above only fires when it's missing from
    # config, so when the operator sets it the override loop here is
    # what propagates their value.
    # Keys consumed by Odyssey's mission spec but not accepted by the
    # upstream finetune.py draccus config.
    handled = {
        "vla_path",
        "data_root_dir",
        "dataset_name",
        "method",
        "lora_alpha",
        "epochs",
    }
    for key, value in _flatten_config(config):
        if key in handled:
            continue
        argv += [f"--{key}", str(value)]
    return argv


async def _resolve_and_fetch_hf_model(
    context: TaskContext, ref: HFModelRef, dest: Path
) -> Path:
    """Resolve + fetch an HF model via the engine's ProviderRegistry."""
    assert context.providers is not None  # checked by caller
    provider = context.providers.for_model_ref(ref)
    await context.emit_progress(
        "model_loading",
        step="resolve_hf",
        step_label=f"{ref.base}@{ref.revision or 'HEAD'}",
    )
    resolved: ResolvedModel = await provider.resolve(ref)
    await context.emit_progress(
        "model_loading",
        step="fetch_hf",
        step_label=f"{resolved.identifier}@{resolved.revision[:8]}",
    )
    return await provider.fetch(resolved, dest)


def _resolve_finetune_script() -> str:
    repo_path = os.getenv("OPENVLA_REPO_PATH", _DEFAULT_REPO_PATH)
    script_path = os.path.join(repo_path, _FINETUNE_SCRIPT_REL)
    if not os.path.isfile(script_path):
        raise RuntimeError(
            f"openvla finetune script not found at {script_path!r}; "
            "clone https://github.com/openvla/openvla and set OPENVLA_REPO_PATH."
        )
    return script_path


def _resolve_run_subdir(output_dir: Path, run_id: str) -> Path | None:
    """OpenVLA writes to ``{run_root_dir}/<vla_name>+<run_id>+<suffix>/``.

    Pick whichever subdir contains an adapter (LoRA) or full-finetune
    config and includes ``run_id`` in its name.  Then look inside for
    the latest ``checkpoint-NNN/`` subdir with an ``adapter_config.json``
    — that's the actual LoRA checkpoint the eval needs.
    """
    if not output_dir.is_dir():
        return None
    candidates: list[Path] = []
    for entry in output_dir.iterdir():
        if not entry.is_dir() or entry.name == "adapter_tmp":
            continue
        has_config = any(
            (entry / fname).is_file()
            for fname in ("adapter_config.json", "config.json")
        )
        if has_config and run_id in entry.name:
            candidates.append(entry)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    run_dir = candidates[0]

    # Look for the latest checkpoint-NNN/ subdir with adapter_config.json
    checkpoint = _resolve_latest_checkpoint(run_dir)
    return checkpoint if checkpoint is not None else run_dir


def _resolve_latest_checkpoint(run_dir: Path) -> Path | None:
    """Find the latest ``checkpoint-NNN/`` subdir containing a LoRA adapter."""
    checkpoints: list[tuple[int, Path]] = []
    for entry in run_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("checkpoint-"):
            continue
        if (entry / "adapter_config.json").is_file():
            # Extract step number for sorting
            try:
                step_num = int(entry.name.split("-", 1)[1])
            except (ValueError, IndexError):
                step_num = 0
            checkpoints.append((step_num, entry))
    if not checkpoints:
        return None
    checkpoints.sort(reverse=True)
    return checkpoints[0][1]


class OpenVLARunner(Runner):
    """Fine-tune an OpenVLA model via LoRA. Subprocess-based — actual
    training happens in the upstream ``finetune.py`` script."""

    @property
    def name(self) -> str:
        return "openvla"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.TRAINING}

    @property
    def supported_types(self) -> set[str]:
        # Any training_type pairs with OpenVLA — the framework decides
        # based on dataset shape, not the type enum.
        return {WILDCARD_TYPE}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        spec = context.task.spec
        if not isinstance(spec, TrainingTask):
            raise TypeError(
                f"OpenVLARunner expects TrainingTask, got {type(spec).__name__}"
            )
        if context.agent is None:
            raise RuntimeError(
                "OpenVLARunner: TaskContext.agent is None — training tasks "
                "must be invoked through the engine, which resolves the "
                "agent from spec.robot.agents[task.agent_id]."
            )

        output_dir = output_path(context)
        script_path = _resolve_finetune_script()

        # Decide what vla_path the subprocess should start from:
        #   1. A prior training task on this agent already produced a
        #      checkpoint → start from that local path.
        #   2. The agent's base model is an HF ref and we have a
        #      provider registry → fetch it locally first.
        #   3. Otherwise fall through to build_openvla_argv's env-var
        #      / HF-id resolution against the agent's base model.
        config = dict(spec.config)
        agent_model_base: str | None = None
        if context.starting_checkpoint is not None:
            config.setdefault("vla_path", context.starting_checkpoint)
        elif (
            context.providers is not None
            and isinstance(context.agent.model, HFModelRef)
        ):
            resolved_path = await _resolve_and_fetch_hf_model(
                context, context.agent.model, output_dir / "model"
            )
            config.setdefault("vla_path", str(resolved_path))

        if isinstance(context.agent.model, HFModelRef):
            agent_model_base = context.agent.model.base

        process_spec = TrainingProcessSpec(
            script_path=script_path,
            use_torchrun=True,
            argv_extra=build_openvla_argv(
                task=spec.model_copy(update={"config": config}),
                agent_model_base=agent_model_base,
                output_dir=output_dir,
                run_id=context.task.id,
            ),
            line_parser=parse_openvla_line,
        )

        rc = await run_training_subprocess(context, process_spec)
        if context.cancelled():
            logger.info("OpenVLA task %s cancelled by user", context.task.id)
            return {"cancelled": True}
        if rc != 0:
            raise RuntimeError(f"openvla finetune exited with code {rc}")

        run_subdir = _resolve_run_subdir(output_dir, context.task.id)
        if run_subdir is None:
            raise RuntimeError(
                f"openvla finetune finished but no run subdir found under "
                f"{output_dir!r}"
            )
        return {
            "checkpoint_path": str(run_subdir),
            "agent_id": context.agent.id,
            "training_config": spec.config,
            "training_type": (
                spec.training_type.value
                if isinstance(spec.training_type, TrainingType)
                else spec.training_type
            ),
        }


# ---------------------------------------------------------------------------
# Inference policy
# ---------------------------------------------------------------------------

# Default natural-language instructions per Robosuite benchmark.
_DEFAULT_INSTRUCTIONS: dict[str, str] = {
    "Lift": "pick up the red cube",
    "Stack": "stack the red cube on top of the green cube",
    "NutAssembly": "pick up the nut and place it on the peg",
    "NutAssemblySquare": "pick up the square nut and place it on the square peg",
    "NutAssemblyRound": "pick up the round nut and place it on the round peg",
    "PickPlace": "pick up the object and place it in the bin",
    "Door": "open the door",
    "Wipe": "wipe the table",
    "ToolHang": "hang the tool on the rack",
    "TwoArmLift": "lift the pot together",
}


def _resolve_base_model(checkpoint_path: Path) -> str:
    """Read ``adapter_config.json`` to find the base model name."""
    config_file = checkpoint_path / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in {checkpoint_path}. "
            "Expected a peft LoRA checkpoint directory."
        )
    with open(config_file) as f:
        config = json.load(f)
    base = config.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            f"adapter_config.json in {checkpoint_path} is missing "
            "'base_model_name_or_path' key."
        )
    return str(base)


def _find_image_key(obs: dict[str, Any], preferred: str) -> str:
    """Find a camera image key in the observation dict."""
    if preferred in obs:
        return preferred
    # Fall back to any key ending with "_image"
    for key in obs:
        if key.endswith("_image"):
            return key
    raise KeyError(
        f"No image key found in observation dict. "
        f"Looked for {preferred!r} and any key ending with '_image'. "
        f"Available keys: {sorted(obs.keys())}"
    )


def _is_lora_checkpoint(checkpoint_path: Path) -> bool:
    """Check whether the checkpoint is a LoRA adapter or a full model."""
    return (checkpoint_path / "adapter_config.json").is_file()


def make_openvla_policy(
    checkpoint_path: Path,
    *,
    config: dict[str, Any] | None = None,
    benchmark_name: str = "Lift",
) -> Any:
    """Build an OpenVLA inference policy from a checkpoint.

    Supports both LoRA adapter checkpoints (with ``adapter_config.json``)
    and full merged model checkpoints (with ``config.json`` and safetensors).

    Returns a callable ``policy(obs_dict) -> action_array`` suitable for
    use as a ``Policy`` in ``RobosuiteRunner``.
    """
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore[attr-defined]
    except ImportError as e:
        raise NotImplementedError(
            "OpenVLA inference policy requires the 'openvla' extra. "
            "Install with: pip install 'lovell-odyssey[openvla]'"
        ) from e

    cfg = config or {}
    unnorm_key = cfg.get("unnorm_key", "bridge_orig")
    task_instruction = cfg.get("task_instruction") or _DEFAULT_INSTRUCTIONS.get(
        benchmark_name, "complete the task"
    )
    image_key = cfg.get("image_key", "agentview_image")
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    is_lora = _is_lora_checkpoint(checkpoint_path)

    if is_lora:
        from peft import PeftModel

        base_model_name = _resolve_base_model(checkpoint_path)
        logger.info("Loading OpenVLA base model: %s", base_model_name)
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            base_model_name, trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info("Applying LoRA adapter from: %s", checkpoint_path)
        model = PeftModel.from_pretrained(model, str(checkpoint_path))

        # merge_and_unload() for faster inference — but only if the merged
        # model retains the custom predict_action method.
        if hasattr(model, "merge_and_unload"):
            merged = model.merge_and_unload()
            if hasattr(merged, "predict_action"):
                model = merged
                logger.info("LoRA merged and unloaded for faster inference")
            else:
                logger.info(
                    "Skipping merge_and_unload — predict_action not on merged model"
                )
    else:
        logger.info("Loading full merged model from: %s", checkpoint_path)
        processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            str(checkpoint_path), trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            str(checkpoint_path),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    model = model.to(device)

    logger.info(
        "OpenVLA policy ready — instruction=%r, unnorm_key=%r, image_key=%r",
        task_instruction,
        unnorm_key,
        image_key,
    )

    def policy(obs: dict[str, Any]) -> Any:
        import numpy as np

        key = _find_image_key(obs, image_key)
        img_array = obs[key]

        # Convert numpy HWC uint8 → PIL Image
        if not isinstance(img_array, Image.Image):
            img_array = Image.fromarray(img_array.astype("uint8"), "RGB")

        action = model.predict_action(
            processor,
            img_array,
            task_instruction,
            unnorm_key=unnorm_key,
            do_sample=False,
        )
        return np.array(action, dtype=np.float64)

    return policy
