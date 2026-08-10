"""RoboLab evaluation runner — first-class ``evaluation_type: robolab``.

`RoboLab <https://github.com/NVlabs/RoboLab>`_ is NVlabs' Isaac Lab-based
simulation benchmark of the **DROID rig** (Franka + dual mounted cameras,
tabletop task-generalist manipulation) — the benchmark NVIDIA scores the
Cosmos 3 ``Policy-DROID`` checkpoints on. Architecture is client/server:
RoboLab's ``policies/<backend>/run.py`` drives Isaac Sim and streams
observations to a pre-started policy server (cosmos-framework's WebSocket
``action_policy_server_robolab``), which answers action chunks.

This runner promotes what began as a ``evaluation_type: custom`` bridge into a
sibling of ``IsaacLabRunner``/``LiberoRunner``. Division of labour:

  * Odyssey owns the launch, cancellation, scoring, and artifact copy;
  * RoboLab's ``run.py`` owns everything sim-specific (Isaac app boot, env
    registration, per-episode logging). It does NOT speak the ``ODYSSEY_*``
    stdout protocol — instead it writes ``episode_results.jsonl`` (one JSON
    object per episode, ``success: bool`` + score/reason/events) under
    ``<robolab_root>/output/<folder>/``, and ``--output-folder-name`` lets the
    runner pick that folder deterministically (pinned from
    ``robolab/eval/runner.py`` + ``robolab/core/logging/results.py``).

Launch contract (all upstream flags accept snake_case aliases)::

    <eval_python> policies/<backend>/run.py \
        --task <benchmark_name> --num-runs R --num-envs N --headless \
        --remote-host H --remote-port P --output-folder-name <folder> \
        [--<config-key> <value> ...]

``spec.num_episodes`` maps to ``--num-runs ceil(num_episodes / num_envs)``
(RoboLab runs ``num_runs`` rounds of ``num_envs`` parallel envs). The policy
server is EXTERNALLY-SERVED (π0.5/cosmos3 posture) — the runner never loads
weights; ``config.checkpoint`` is recorded in the summary only.

Family-wide: ``config.entry_script`` picks the RoboLab policy backend
(default ``policies/cosmos3/run.py``), so a future backend needs a mission
edit, not a new runner. RoboLab needs Isaac Sim — point ``config.eval_python``
at the interpreter inside RoboLab's docker/venv.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import sys
from pathlib import Path
from typing import Any

from odyssey.runners.base import Runner, TaskContext
from odyssey.runners.evals._common import (
    build_eval_summary,
    resolve_eval_checkpoint,
)
from odyssey.runners.subprocess import (
    TrainingProcessSpec,
    run_training_subprocess,
)
from odyssey.spec.tasks import EvaluationTask, EvaluationType, TaskKind

logger = logging.getLogger(__name__)

DEFAULT_ENTRY_SCRIPT = "policies/cosmos3/run.py"
RESULTS_FILENAME = "episode_results.jsonl"

# Config keys the runner consumes itself — never forwarded as passthrough flags.
_HANDLED_CONFIG_KEYS = {
    "robolab_root", "eval_python", "entry_script", "runner", "checkpoint",
    "num_envs", "output_folder_name",
}


def resolve_robolab_root(config: dict[str, Any]) -> Path:
    """``config.robolab_root`` — the NVlabs/RoboLab checkout (required)."""
    root = config.get("robolab_root")
    if not root:
        raise RuntimeError(
            "RoboLab eval requires config.robolab_root — the path to a "
            "NVlabs/RoboLab checkout (its run.py drives Isaac Sim; see "
            "examples/cosmos3-robolab/README.md for the docker setup)."
        )
    return Path(str(root)).expanduser()


def resolve_interpreter(config: dict[str, Any]) -> list[str]:
    """``config.eval_python`` (RoboLab's docker/venv interpreter) or ours."""
    return [str(config.get("eval_python") or sys.executable)]


def output_folder_name(context: TaskContext) -> str:
    """Deterministic per-task results folder under ``<robolab_root>/output``."""
    return f"odyssey_{context.task.id}"


def build_robolab_argv(
    *,
    spec: EvaluationTask,
    folder: str,
) -> list[str]:
    """Build ``run.py`` argv: contract flags + config passthrough.

    ``num_episodes`` maps to ``--num-runs`` x ``num_envs``; every other config
    key passes through as ``--key value`` with underscores dashed
    (``remote_port`` → ``--remote-port``): RoboLab's own flags are
    kebab-case-first (the backend-specific ones exclusively so), minus the
    keys the runner consumes itself.
    """
    cfg = spec.config or {}
    num_envs = int(cfg.get("num_envs", 1))
    num_runs = max(1, math.ceil(spec.num_episodes / max(num_envs, 1)))
    argv: list[str] = [
        "--task", spec.benchmark_name,
        "--num-runs", str(num_runs),
        "--num-envs", str(num_envs),
        "--output-folder-name", folder,
        "--headless",
    ]
    for key, value in cfg.items():
        if key in _HANDLED_CONFIG_KEYS:
            continue
        flag = f"--{key.replace('_', '-')}"
        # Booleans map to bare store_true/store_false flags (RoboLab's
        # --disable-subtask etc. take no value); False simply omits the flag.
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        argv += [flag, str(value)]
    return argv


def read_episode_results(results_file: Path) -> list[dict[str, Any]]:
    """Parse RoboLab's ``episode_results.jsonl`` (one JSON object per episode)."""
    if not results_file.is_file():
        raise RuntimeError(
            f"RoboLab run exited 0 but wrote no results at {results_file}. "
            "Expected episode_results.jsonl (one JSON per episode) under the "
            "--output-folder-name directory — check the run.py log."
        )
    episodes: list[dict[str, Any]] = []
    for line in results_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Unparseable line in {results_file}: {line[:120]!r}"
            ) from e
        if isinstance(record, dict):
            episodes.append(record)
    if not episodes:
        raise RuntimeError(f"RoboLab results file {results_file} holds no episodes.")
    return episodes


def summarize(
    *,
    episodes: list[dict[str, Any]],
    spec: EvaluationTask,
    checkpoint: Path,
) -> dict[str, Any]:
    """Fold per-episode records into the shared eval summary shape."""
    successes = sum(1 for ep in episodes if bool(ep.get("success", False)))
    failure_reasons: dict[str, int] = {}
    for ep in episodes:
        if not ep.get("success", False):
            reason = str(ep.get("reason") or ep.get("error") or "unknown")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    return build_eval_summary(
        num_episodes=len(episodes),
        successes=successes,
        episode_returns=[],
        benchmark_name=spec.benchmark_name,
        checkpoint_path=checkpoint,
        metrics={
            "per_task": _per_task_rates(episodes),
            "failure_reasons": failure_reasons,
        },
    )


def _per_task_rates(episodes: list[dict[str, Any]]) -> dict[str, str]:
    """``{task: "successes/episodes"}`` for multi-task sweeps."""
    counts: dict[str, list[int]] = {}
    for ep in episodes:
        task = str(ep.get("task") or ep.get("env_name") or "task")
        bucket = counts.setdefault(task, [0, 0])
        bucket[0] += int(bool(ep.get("success", False)))
        bucket[1] += 1
    return {task: f"{won}/{total}" for task, (won, total) in counts.items()}


class RobolabRunner(Runner):
    """Evaluation runner for ``evaluation_type: robolab`` tasks."""

    @property
    def name(self) -> str:
        return "robolab"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.EVALUATION}

    @property
    def supported_types(self) -> set[str]:
        return {EvaluationType.ROBOLAB.value}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        spec = context.task.spec
        if not isinstance(spec, EvaluationTask):
            raise TypeError(
                f"RobolabRunner expects EvaluationTask, got {type(spec).__name__}"
            )

        config = spec.config or {}
        checkpoint = resolve_eval_checkpoint(context)
        robolab_root = resolve_robolab_root(config)
        entry = robolab_root / str(config.get("entry_script", DEFAULT_ENTRY_SCRIPT))
        folder = str(config.get("output_folder_name") or output_folder_name(context))
        results_file = robolab_root / "output" / folder / RESULTS_FILENAME

        await context.emit_progress(
            "executing",
            step="launch",
            step_label=f"task={spec.benchmark_name} backend={entry.parent.name}",
        )

        process_spec = TrainingProcessSpec(
            timeout_seconds=getattr(spec, "timeout_seconds", None),
            script_path=str(entry),
            launcher=resolve_interpreter(config),
            argv_extra=build_robolab_argv(spec=spec, folder=folder),
            cwd=str(robolab_root),
        )

        rc = await run_training_subprocess(context, process_spec)
        if context.cancelled():
            logger.info("RoboLab task %s cancelled by user", context.task.id)
            return {"cancelled": True}
        if rc != 0:
            raise RuntimeError(f"RoboLab run.py exited with code {rc}")

        episodes = read_episode_results(results_file)
        if context.output_dir is not None:  # keep the raw record as an artifact
            context.output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(results_file, context.output_dir / RESULTS_FILENAME)

        return summarize(episodes=episodes, spec=spec, checkpoint=checkpoint)
