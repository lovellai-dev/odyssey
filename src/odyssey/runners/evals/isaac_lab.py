"""Isaac Lab evaluation runner — subprocess contract.

Unlike the in-process Robosuite runner, Isaac Lab evaluation must run
as a subprocess: Isaac Lab scripts execute under Isaac Sim's bundled
Python (``isaaclab.sh -p``) and boot the Omniverse kit app, neither of
which can live inside Odyssey's process or dependency set.

The runner therefore launches an *eval script* inside the user's Isaac
Lab installation and consumes a small JSON-line protocol on stdout.
The script itself is the pluggable piece — the slot where a blessed
GR00T/VLA evaluation recipe lands (see the GR00T integration issue) —
Odyssey owns the launch, the protocol, cancellation, and scoring.

Launch contract::

    ${ISAACLAB_PATH}/isaaclab.sh -p <eval_script> \\
        --task <benchmark_name> --num_episodes <N> \\
        --checkpoint <path> --headless [--<config-key> <value> ...]

``eval_script`` comes from ``task.config["eval_script"]`` or
``$ISAACLAB_EVAL_SCRIPT``. When ``$ISAACLAB_PATH`` is unset the script
runs under plain ``python`` (an env where ``isaaclab`` is importable).

Stdout protocol (other lines are ignored; Isaac Sim boot noise is
expected)::

    ODYSSEY_EPISODE {"index": 1, "total": 10, "success": true, "return": 1.5}
    ODYSSEY_RESULT {"success_rate": 0.4, "performance_score": 0.7, "metrics": {}}
    ODYSSEY_REASONING {"episode": 1, "instruction": "...", "reasoning": "..."}

``ODYSSEY_RESULT`` is optional — when absent, the summary is computed
from the episode lines. ``ODYSSEY_REASONING`` is also optional — a
per-episode intent trace from the Cosmos-Reason sidecar
(``cosmos_reason.py``); when present, the traces are carried into
``metrics["reasoning"]`` so Command Center / Episode Review can show
reasoning beside the grade.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from odyssey.runners.base import Runner, TaskContext

# Shared eval helpers; extract to a common module when a third
# evaluation runner needs them.
from odyssey.runners.evals.robosuite import _grade, _resolve_eval_checkpoint
from odyssey.runners.subprocess import (
    TrainingProcessSpec,
    run_training_subprocess,
)
from odyssey.spec.tasks import EvaluationTask, EvaluationType, TaskKind

logger = logging.getLogger(__name__)


_EPISODE_PREFIX = "ODYSSEY_EPISODE "
_RESULT_PREFIX = "ODYSSEY_RESULT "
# Optional per-episode intent trace from the Cosmos-Reason sidecar. Purely
# additive: an eval that never emits it behaves exactly as before.
_REASONING_PREFIX = "ODYSSEY_REASONING "

# Config keys consumed by the runner itself, never forwarded as flags.
_HANDLED_CONFIG_KEYS = {"eval_script", "runner", "headless"}


class EvalProtocolCollector:
    """Parses the ODYSSEY_* stdout protocol while collecting results.

    Doubles as the subprocess ``line_parser``: each parsed episode is
    recorded *and* turned into a progress event. Malformed protocol
    lines are logged and skipped — a single bad line never fails the
    eval.
    """

    def __init__(self) -> None:
        self.episodes: list[dict[str, Any]] = []
        self.result: dict[str, Any] | None = None
        self.reasoning: list[dict[str, Any]] = []

    def parse(self, line: str) -> dict[str, Any] | None:
        stripped = line.strip()
        if stripped.startswith(_EPISODE_PREFIX):
            payload = self._load_json(stripped[len(_EPISODE_PREFIX):])
            if payload is None:
                return None
            self.episodes.append(payload)
            success = bool(payload.get("success"))
            event: dict[str, Any] = {
                "stage": "executing",
                "step": "episode_complete",
                "step_label": f"episode: {'PASS' if success else 'FAIL'}",
            }
            index = payload.get("index")
            total = payload.get("total")
            if isinstance(index, int):
                event["step_index"] = index
            if isinstance(total, int):
                event["step_total"] = total
            return event
        if stripped.startswith(_RESULT_PREFIX):
            payload = self._load_json(stripped[len(_RESULT_PREFIX):])
            if payload is None:
                return None
            self.result = payload
            return {"stage": "executing", "step": "result_received"}
        if stripped.startswith(_REASONING_PREFIX):
            payload = self._load_json(stripped[len(_REASONING_PREFIX):])
            if payload is None:
                return None
            self.reasoning.append(payload)
            reasoning_event: dict[str, Any] = {
                "stage": "executing", "step": "reasoning_received",
                "step_label": "intent trace",
            }
            episode = payload.get("episode")
            if isinstance(episode, int):
                reasoning_event["step_index"] = episode
            return reasoning_event
        return None

    @staticmethod
    def _load_json(raw: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Malformed ODYSSEY_* protocol line skipped: %r", raw)
            return None
        if not isinstance(payload, dict):
            logger.warning("ODYSSEY_* payload is not an object: %r", raw)
            return None
        return payload


def resolve_eval_script(config: dict[str, Any]) -> str:
    """``task.config["eval_script"]`` or ``$ISAACLAB_EVAL_SCRIPT``."""
    script = config.get("eval_script") or os.getenv("ISAACLAB_EVAL_SCRIPT")
    if not script:
        raise RuntimeError(
            "Isaac Lab eval requires an eval script: set "
            "task.config['eval_script'] (or $ISAACLAB_EVAL_SCRIPT) to a "
            "script that runs the benchmark and prints the ODYSSEY_EPISODE "
            "/ ODYSSEY_RESULT stdout protocol. See "
            "src/odyssey/runners/isaac_lab.py for the contract."
        )
    return str(script)


def resolve_launcher() -> list[str] | None:
    """``[$ISAACLAB_PATH/isaaclab.sh, -p]`` when the env var is set.

    None means plain ``python`` — fine for environments where
    ``isaaclab`` is importable directly (and for tests).
    """
    isaaclab_path = os.getenv("ISAACLAB_PATH")
    if not isaaclab_path:
        return None
    return [os.path.join(isaaclab_path, "isaaclab.sh"), "-p"]


def build_isaac_lab_argv(
    *,
    task: EvaluationTask,
    checkpoint: Path,
) -> list[str]:
    """Build the eval script's argv per the launch contract.

    Isaac Lab scripts use argparse with snake_case flags, so config
    keys pass through verbatim (``num_envs`` → ``--num_envs``), unlike
    the kebab-case training runners.
    """
    config = task.config or {}
    argv: list[str] = [
        "--task", task.benchmark_name,
        "--num_episodes", str(task.num_episodes),
        "--checkpoint", str(checkpoint),
    ]
    if config.get("headless", True):
        argv.append("--headless")
    for key, value in config.items():
        if key in _HANDLED_CONFIG_KEYS:
            continue
        argv += [f"--{key}", str(value)]
    return argv


def summarize(
    *,
    collector: EvalProtocolCollector,
    spec: EvaluationTask,
    checkpoint: Path,
    eval_script: str,
) -> dict[str, Any]:
    """Build the result_summary from collected protocol output.

    Same shape as the Robosuite runner (what lai-trainer's
    ``update_evaluation_results`` expects). An explicit ODYSSEY_RESULT
    wins; otherwise the summary is computed from the episode lines.
    """
    episodes = collector.episodes
    successes = sum(1 for e in episodes if e.get("success"))
    episode_returns = [float(e.get("return", 0.0)) for e in episodes]
    attempted = len(episodes)

    if collector.result is not None:
        result = collector.result
        success_rate = float(
            result.get(
                "success_rate", (successes / attempted) if attempted else 0.0
            )
        )
        performance_score = float(result.get("performance_score", success_rate))
        metrics = dict(result.get("metrics") or {})
        num_episodes = int(result.get("num_episodes", attempted or spec.num_episodes))
    elif attempted:
        success_rate = successes / attempted
        performance_score = sum(episode_returns) / attempted
        metrics = {}
        num_episodes = attempted
    else:
        raise RuntimeError(
            "Isaac Lab eval script completed without emitting any "
            "ODYSSEY_EPISODE or ODYSSEY_RESULT lines — it does not speak "
            "the protocol. See src/odyssey/runners/isaac_lab.py."
        )

    metrics.setdefault("successes", successes)
    metrics.setdefault(
        "episode_returns", [round(r, 4) for r in episode_returns]
    )
    metrics.setdefault("benchmark", spec.benchmark_name)
    metrics.setdefault("checkpoint_path", str(checkpoint))
    metrics.setdefault("eval_script", eval_script)
    # Carry the per-episode intent traces (if the eval emitted any) into the
    # summary so Command Center / Episode Review can show reasoning beside the
    # grade. Absent when the eval doesn't speak ODYSSEY_REASONING.
    if collector.reasoning:
        metrics.setdefault("reasoning", collector.reasoning)
    return {
        "num_episodes": num_episodes,
        "success_rate": round(success_rate, 4),
        "performance_score": round(performance_score, 4),
        "letter_grade": _grade(success_rate),
        "passed": success_rate >= 0.5,
        "metrics": metrics,
    }


class IsaacLabRunner(Runner):
    """Evaluation runner for ``evaluation_type: isaac_lab`` tasks."""

    @property
    def name(self) -> str:
        return "isaac_lab"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.EVALUATION}

    @property
    def supported_types(self) -> set[str]:
        return {EvaluationType.ISAAC_LAB.value}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        spec = context.task.spec
        if not isinstance(spec, EvaluationTask):
            raise TypeError(
                f"IsaacLabRunner expects EvaluationTask, got {type(spec).__name__}"
            )

        checkpoint = _resolve_eval_checkpoint(context)
        eval_script = resolve_eval_script(spec.config or {})
        launcher = resolve_launcher()

        await context.emit_progress(
            "executing",
            step="env_construct",
            step_label=(
                f"benchmark={spec.benchmark_name} "
                f"launcher={'isaaclab.sh' if launcher else 'python'}"
            ),
        )

        collector = EvalProtocolCollector()
        process_spec = TrainingProcessSpec(
            script_path=eval_script,
            launcher=launcher,
            argv_extra=build_isaac_lab_argv(task=spec, checkpoint=checkpoint),
            line_parser=collector.parse,
        )

        rc = await run_training_subprocess(context, process_spec)
        if context.cancelled():
            logger.info("Isaac Lab task %s cancelled by user", context.task.id)
            return {"cancelled": True}
        if rc != 0:
            raise RuntimeError(f"Isaac Lab eval script exited with code {rc}")

        return summarize(
            collector=collector,
            spec=spec,
            checkpoint=checkpoint,
            eval_script=eval_script,
        )
