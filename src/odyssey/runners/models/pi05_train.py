"""π0.5 (openpi) training runner.

Fine-tunes a Physical Intelligence π0.5 checkpoint by shelling out to the
upstream **openpi** training entry points — the openpi package must be
installed and its repo checked out on disk (``pip install -e`` of
https://github.com/Physical-Intelligence/openpi), pointed at via
``$OPENPI_REPO_PATH`` (default ``/srv/openpi``).

Why this looks different from the GR00T / OpenVLA runners
--------------------------------------------------------
GR00T and OpenVLA take a flat bag of ``--flag value`` overrides. openpi is
**config-name-driven**: training is selected by a registered ``TrainConfig``
name (e.g. ``pi05_libero``) that bundles the data pipeline, weight loader and
optimizer, and CLI ``--overrides`` (parsed by *tyro*) tweak individual fields.
So the mission's ``config:`` here carries:

  * ``config_name`` — REQUIRED, the registered openpi ``TrainConfig`` (positional
    arg to both scripts). Fine-tuning a *new* embodiment/dataset means adding a
    config entry to openpi's ``src/openpi/training/config.py`` (the π0.5 analogue
    of GR00T's ``modality_config_path`` file) whose ``data.repo_id`` points at the
    LeRobot dataset — see ``examples/quickstart-pi05-train/README.md``.
  * everything else — passed through as tyro overrides (``num_train_steps`` →
    ``--num-train-steps``, ``data.repo_id`` → ``--data.repo-id``). Booleans become
    tyro flags (``--overwrite`` / ``--no-overwrite``), NOT ``--flag True``.

Two subprocesses, in order (openpi requires norm stats before training):

  1. ``scripts/compute_norm_stats.py <config_name>`` — computes dataset
     normalization statistics into ``assets/`` (skippable via
     ``config: {compute_norm_stats: false}`` when they already exist).
  2. ``scripts/train.py <config_name> --exp-name <name> [--overwrite] [overrides]``.

Both run with ``cwd = <task output_dir>`` so openpi's cwd-relative ``./assets``
and ``./checkpoints`` land under the Odyssey task dir (no guessing an upstream
``--checkpoint-base-dir`` flag name). openpi is import-resolved from the installed
package, so the cwd only steers where artifacts are written.

Routing: registered behind OpenVLA's wildcard training slot, reached via the
task-level ``config: {runner: pi05}`` override — same mechanism as ``gr00t``.

Licensing note: π0.5 weights are distributed under Physical Intelligence's terms.
This runner contains no openpi code — it shells out to the user's own checkout.
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

# Shared flatten helper (dotted keys for nested dicts).
from odyssey.runners.models.openvla import _flatten_config
from odyssey.runners.subprocess import (
    TrainingProcessSpec,
    output_path,
    run_training_subprocess,
)
from odyssey.spec.refs import DatasetSource
from odyssey.spec.tasks import TaskKind, TrainingTask, TrainingType

logger = logging.getLogger(__name__)


_DEFAULT_REPO_PATH = "/srv/openpi"
_TRAIN_SCRIPT_REL = "scripts/train.py"
_NORM_STATS_SCRIPT_REL = "scripts/compute_norm_stats.py"

# Config keys the runner consumes directly — never forwarded as tyro overrides.
_CONTROL_KEYS = frozenset(
    {"config_name", "runner", "exp_name", "overwrite", "resume", "compute_norm_stats"}
)

# openpi's train loop logs metrics dicts and a tqdm bar, e.g.:
#   "step=1000 loss=0.234 grad_norm=1.2"
#   "Step 1000: {'loss': 0.234, 'learning_rate': 1e-05}"
#   " 10%|█  | 100/1000 [00:42<06:18,  2.38it/s]"
#   "Saving checkpoint to ./checkpoints/pi05_libero/exp/1000"
_PI05_LOSS_RE = re.compile(r"(?:'loss':|\bloss=)\s*([\d.eE+-]+)")
_PI05_STEP_RE = re.compile(r"(?:'step':|\bstep=|\bStep\s+)\s*(\d+)", re.IGNORECASE)
_PI05_TQDM_RE = re.compile(r"\b(\d+)/(\d+)\s*\[")
_PI05_SAVE_RE = re.compile(r"(?i)\b(saving|saved|wrote)\b.*\bcheckpoint\b")
_PI05_NORM_RE = re.compile(r"(?i)\b(computing|writing)\b.*\bnorm(alization)?\s*stat")
_PI05_DATASET_RE = re.compile(r"(?i)\b(loading|building)\b.*\b(dataset|lerobot)\b")


def parse_pi05_train_line(line: str) -> dict[str, Any] | None:
    """Extract a progress-event dict from an openpi train/norm-stats stdout line.

    Public for tests and for users embedding openpi stdout parsing in a custom
    runner. Precedence mirrors the GR00T parser: a metric line (loss / step)
    wins over the coarse phase markers.
    """
    loss_match = _PI05_LOSS_RE.search(line)
    step_match = _PI05_STEP_RE.search(line)
    if loss_match or step_match:
        payload: dict[str, Any] = {"stage": "executing", "step": "training_step"}
        if step_match:
            payload["step_index"] = int(step_match.group(1))
        if loss_match:
            payload["step_label"] = f"loss={loss_match.group(1)}"
        return payload
    tqdm_match = _PI05_TQDM_RE.search(line)
    if tqdm_match:
        return {
            "stage": "executing",
            "step": "training_step",
            "step_index": int(tqdm_match.group(1)),
            "step_total": int(tqdm_match.group(2)),
        }
    if _PI05_SAVE_RE.search(line):
        return {"stage": "checkpoint_saving"}
    if _PI05_NORM_RE.search(line):
        return {"stage": "dataset_loading", "step": "compute_norm_stats"}
    if _PI05_DATASET_RE.search(line):
        return {"stage": "dataset_loading"}
    return None


def _resolve_openpi_script(rel: str) -> str:
    """Absolute path to an openpi script under ``$OPENPI_REPO_PATH``.

    Raises a runner-actionable error (not a bare FileNotFoundError) when the
    checkout is missing, matching the OpenVLA runner's contract.
    """
    repo_path = os.getenv("OPENPI_REPO_PATH", _DEFAULT_REPO_PATH)
    script_path = os.path.join(repo_path, rel)
    if not os.path.isfile(script_path):
        raise RuntimeError(
            f"openpi script not found at {script_path!r}; clone "
            "https://github.com/Physical-Intelligence/openpi and set "
            "OPENPI_REPO_PATH (or place it at /srv/openpi)."
        )
    return script_path


def _tyro_overrides(config: dict[str, Any]) -> list[str]:
    """Flatten ``config`` into tyro CLI overrides, skipping control keys.

    Nested dicts become dotted flags (``data.repo_id`` → ``--data.repo-id``);
    booleans become tyro toggle flags (``--overwrite`` / ``--no-overwrite``)
    rather than ``--flag True`` (which tyro rejects for bool fields).
    """
    argv: list[str] = []
    for key, value in _flatten_config(config):
        # _flatten_config yields the dotted path; only the LEAF may be a control
        # key (nested overrides like data.repo_id are always forwarded).
        if "." not in key and key in _CONTROL_KEYS:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            argv.append(flag if value else f"--no-{key.replace('_', '-')}")
        else:
            argv += [flag, str(value)]
    return argv


def build_pi05_train_argv(*, task: TrainingTask, exp_name: str) -> list[str]:
    """Build the openpi ``scripts/train.py`` argv.

    Shape: ``<config_name> --exp-name <name> [--overwrite|--resume] [overrides…]``.
    ``config_name`` is REQUIRED — it selects the registered openpi ``TrainConfig``.
    ``--overwrite`` is emitted by default (re-running a mission clobbers the prior
    exp dir); set ``config: {overwrite: false}`` to keep it, or ``resume: true`` to
    continue from the latest checkpoint instead.
    """
    config = task.config or {}
    config_name = config.get("config_name")
    if not config_name:
        raise RuntimeError(
            "π0.5 runner: config['config_name'] is required — it names the "
            "registered openpi TrainConfig (e.g. 'pi05_libero'). Add a config "
            "entry to openpi's src/openpi/training/config.py for a custom dataset."
        )

    argv: list[str] = [str(config_name), "--exp-name", exp_name]
    if config.get("resume"):
        argv.append("--resume")
    elif config.get("overwrite", True):
        argv.append("--overwrite")
    argv += _tyro_overrides(config)
    return argv


def build_pi05_norm_stats_argv(*, task: TrainingTask) -> list[str]:
    """Build the openpi ``scripts/compute_norm_stats.py`` argv.

    Only the positional ``config_name`` is required; the dataset it reads is fixed
    by the config's ``data`` factory.
    """
    config = task.config or {}
    config_name = config.get("config_name")
    if not config_name:
        raise RuntimeError(
            "π0.5 runner: config['config_name'] is required for norm-stats."
        )
    return [str(config_name)]


def _lerobot_env_for_dataset(task: TrainingTask) -> dict[str, str]:
    """Env overlay so openpi's LeRobot loader finds a LOCAL dataset.

    openpi resolves ``data.repo_id`` against ``HF_LEROBOT_HOME`` (default
    ``~/.cache/huggingface/lerobot``). For an absolute local dataset dir we point
    that at the PARENT so ``repo_id`` == the dataset folder name resolves. HF/hub
    refs and relative paths are left to the config to interpret.
    """
    if task.dataset is None:
        return {}
    ref = task.dataset.ref
    if task.dataset.source == DatasetSource.LOCAL and os.path.isabs(ref):
        parent = os.path.dirname(os.path.normpath(ref))
        # Only HF_LEROBOT_HOME — do NOT also set the legacy LEROBOT_HOME. Recent
        # lerobot HARD-FAILS at import (raises ValueError) if the deprecated
        # LEROBOT_HOME var is present, which killed compute_norm_stats.
        return {"HF_LEROBOT_HOME": parent}
    return {}


def _resolve_output_checkpoint(output_dir: Path) -> Path | None:
    """Locate the trained checkpoint under ``<output_dir>/checkpoints``.

    openpi writes ``checkpoints/<config_name>/<exp_name>/<step>/`` (step is an
    integer dir holding ``params/``). Return the highest-numbered step dir; None
    if training wrote nothing.
    """
    checkpoints_root = output_dir / "checkpoints"
    if not checkpoints_root.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for entry in checkpoints_root.rglob("*"):
        if entry.is_dir() and entry.name.isdigit():
            step = int(entry.name)
            if best is None or step > best[0]:
                best = (step, entry)
    return best[1] if best is not None else None


class Pi05Runner(Runner):
    """Fine-tune a π0.5 model. Subprocess-based — actual training happens in
    openpi's ``scripts/compute_norm_stats.py`` then ``scripts/train.py``."""

    @property
    def name(self) -> str:
        return "pi05"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.TRAINING}

    @property
    def supported_types(self) -> set[str]:
        # Registered behind OpenVLA's wildcard; reached via the task-level
        # ``config: {runner: pi05}`` override (same pattern as GR00T).
        return {WILDCARD_TYPE}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        spec = context.task.spec
        if not isinstance(spec, TrainingTask):
            raise TypeError(
                f"Pi05Runner expects TrainingTask, got {type(spec).__name__}"
            )
        if context.agent is None:
            raise RuntimeError(
                "Pi05Runner: TaskContext.agent is None — training tasks must be "
                "invoked through the engine, which resolves the agent from "
                "spec.robot.agents[task.agent_id]."
            )

        output_dir = output_path(context)
        output_dir.mkdir(parents=True, exist_ok=True)
        config = dict(spec.config)
        exp_name = str(config.get("exp_name") or spec.name)
        timeout = getattr(spec, "timeout_seconds", None)
        child_env = _lerobot_env_for_dataset(spec)

        # Step 1: normalization statistics (openpi requires them before training).
        if config.get("compute_norm_stats", True):
            await context.emit_progress(
                "dataset_loading", step="compute_norm_stats", step_label=exp_name
            )
            norm_spec = TrainingProcessSpec(
                timeout_seconds=timeout,
                script_path=_resolve_openpi_script(_NORM_STATS_SCRIPT_REL),
                argv_extra=build_pi05_norm_stats_argv(task=spec),
                env=child_env,
                cwd=str(output_dir),
                line_parser=parse_pi05_train_line,
            )
            rc = await run_training_subprocess(context, norm_spec)
            if context.cancelled():
                logger.info("π0.5 task %s cancelled during norm-stats", context.task.id)
                return {"cancelled": True}
            if rc != 0:
                raise RuntimeError(
                    f"openpi compute_norm_stats exited with code {rc}"
                )

        # Step 2: fine-tune.
        train_spec = TrainingProcessSpec(
            timeout_seconds=timeout,
            script_path=_resolve_openpi_script(_TRAIN_SCRIPT_REL),
            argv_extra=build_pi05_train_argv(task=spec, exp_name=exp_name),
            env=child_env,
            cwd=str(output_dir),
            line_parser=parse_pi05_train_line,
        )
        rc = await run_training_subprocess(context, train_spec)
        if context.cancelled():
            logger.info("π0.5 task %s cancelled by user", context.task.id)
            return {"cancelled": True}
        if rc != 0:
            raise RuntimeError(f"openpi train.py exited with code {rc}")

        checkpoint = _resolve_output_checkpoint(output_dir)
        if checkpoint is None:
            raise RuntimeError(
                f"openpi train.py finished but no checkpoint found under "
                f"{output_dir / 'checkpoints'!r}"
            )
        return {
            "checkpoint_path": str(checkpoint),
            "agent_id": context.agent.id,
            "exp_name": exp_name,
            "config_name": config.get("config_name"),
            "training_config": spec.config,
            "training_type": (
                spec.training_type.value
                if isinstance(spec.training_type, TrainingType)
                else spec.training_type
            ),
        }
