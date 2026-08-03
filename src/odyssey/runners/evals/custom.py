"""Custom evaluation runner — run an arbitrary eval script as a subprocess.

The runner for ``evaluation_type: custom``. It is deliberately decoupled from
Isaac Lab / Robosuite / MuJoCo: no sim-specific argv (``--task``/``--headless``),
no per-episode stdout protocol, and no policy-serving dependency. Odyssey owns
the launch, the checkpoint path, cancellation, and reading a machine-readable
metrics file; the *script* owns everything domain-specific — including how it
loads or serves the checkpoint.

Without this runner, ``evaluation_type: custom`` falls through to the wildcard
``CPUMockRunner`` (a fake eval), so the only ways to run a real project-specific
evaluation were to abuse the ``isaac_lab`` runner's contract or to mock it out.

Launch contract::

    <eval_python> <eval_script> \\
        --checkpoint <path> --out-json <path> [--<config-key> <value> ...]

* ``eval_script`` — ``task.config["eval_script"]`` or ``$ODYSSEY_EVAL_SCRIPT``
  (required). The script that runs the evaluation.
* ``eval_python`` — ``task.config["eval_python"]``, default ``sys.executable``.
  Lets the script run under a separate venv (e.g. a GR00T / openpi env) without
  Odyssey having to import those deps.
* ``--checkpoint`` — the trained PILOT checkpoint (via ``resolve_eval_checkpoint``):
  a local path or an HF repo id. The runner does **not** serve it — the script
  loads/serves it however it needs (contrast the ``isaac_lab`` runner). This is
  what keeps ``CustomEvalRunner`` free of any policy-serving dependency.
* ``--out-json`` — a path the runner allocates under the task's output dir; the
  script MUST write its metrics there as a JSON object (see below). This is the
  metrics channel — there is no stdout protocol.
* Every other ``config.*`` key passes through as ``--key value`` verbatim
  (snake_case preserved), the same passthrough style as ``isaac_lab``.

out-json contract — the script writes one JSON object; every key is optional::

    {
      "success_rate": 0.42,          # float — when present, the summary carries a
      "performance_score": 0.7,      #   trainer letter grade + pass/fail
      "num_episodes": 50,
      "metrics": { ... }             # arbitrary extra metrics (e.g. per-joint MAE)
    }

Unknown top-level keys are folded into ``metrics``. When ``success_rate`` is
absent the eval is treated as **metric-only** (e.g. the open-loop GT eval, whose
honest metric is per-joint MAE + gripper agreement, not a success rate): the
summary surfaces the metrics *without* fabricating a 0.0 / grade-F pass/fail.

Serving is intentionally the script's job — there is no ``config.serve_checkpoint``.
The motivating open-loop GT eval loads the checkpoint in-process, and delegating
serving keeps this runner sim- and policy-server-agnostic. A closed-loop eval that
needs a served policy starts (or connects to) the server from inside the script.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
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

# Config keys the runner consumes itself, never forwarded as passthrough flags.
# ``checkpoint`` is resolved by ``resolve_eval_checkpoint`` (config.checkpoint is
# how eval-only missions name one) so it must not also appear as a flag.
_HANDLED_CONFIG_KEYS = {"eval_script", "eval_python", "runner", "checkpoint", "out_json"}

# Top-level out-json keys the runner interprets; anything else is folded into
# ``metrics`` so a script can emit flat payloads too.
_RESERVED_METRIC_KEYS = {"success_rate", "performance_score", "num_episodes", "metrics"}

_METRICS_FILENAME = "custom_eval_metrics.json"


def resolve_eval_script(config: dict[str, Any]) -> str:
    """``task.config["eval_script"]`` or ``$ODYSSEY_EVAL_SCRIPT``."""
    script = config.get("eval_script") or os.getenv("ODYSSEY_EVAL_SCRIPT")
    if not script:
        raise RuntimeError(
            "Custom eval requires an eval script: set "
            "task.config['eval_script'] (or $ODYSSEY_EVAL_SCRIPT) to a script "
            "that accepts --checkpoint / --out-json and writes its metrics as a "
            "JSON object. See src/odyssey/runners/evals/custom.py for the contract."
        )
    return str(script)


def resolve_interpreter(config: dict[str, Any]) -> list[str]:
    """Launcher for the eval script — ``[config['eval_python']]`` or the
    interpreter Odyssey itself runs under (``sys.executable``)."""
    return [str(config.get("eval_python") or sys.executable)]


def build_custom_argv(
    *,
    checkpoint: Path,
    out_json: Path,
    config: dict[str, Any],
) -> list[str]:
    """Build the eval script's argv per the launch contract.

    The runner owns ``--checkpoint`` and ``--out-json``; every other config key
    passes through verbatim as ``--key value`` (snake_case preserved).
    """
    argv: list[str] = [
        "--checkpoint", str(checkpoint),
        "--out-json", str(out_json),
    ]
    for key, value in config.items():
        if key in _HANDLED_CONFIG_KEYS:
            continue
        argv += [f"--{key}", str(value)]
    return argv


def read_metrics(out_json: Path) -> dict[str, Any]:
    """Load the JSON object the script wrote to ``out_json``.

    Raises when the file is missing (the script exited 0 but never wrote it),
    isn't valid JSON, or isn't a JSON object — all script-contract violations.
    """
    if not out_json.is_file():
        raise RuntimeError(
            f"Custom eval script exited 0 but wrote no metrics file at {out_json}. "
            "The script must write its results there as a JSON object (the "
            "--out-json path). See src/odyssey/runners/evals/custom.py."
        )
    try:
        payload = json.loads(out_json.read_text())
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Custom eval metrics file {out_json} is not valid JSON: {e}"
        ) from e
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Custom eval metrics file {out_json} must contain a JSON object, "
            f"got {type(payload).__name__}."
        )
    return payload


def summarize(
    *,
    payload: dict[str, Any],
    spec: EvaluationTask,
    checkpoint: Path,
    eval_script: str,
) -> dict[str, Any]:
    """Turn the script's out-json payload into a ``result_summary``.

    With ``success_rate`` present, routes through ``build_eval_summary`` so the
    summary matches every other eval runner (letter grade + pass/fail). Without
    it, returns a metric-only summary — the metrics stand on their own rather
    than being forced into a pass/fail the eval never measured.
    """
    metrics = dict(payload.get("metrics") or {})
    for key, value in payload.items():
        if key not in _RESERVED_METRIC_KEYS:
            metrics.setdefault(key, value)
    metrics.setdefault("eval_script", eval_script)

    num_episodes = int(payload.get("num_episodes", spec.num_episodes))

    success_rate = payload.get("success_rate")
    if success_rate is not None:
        success_rate = float(success_rate)
        performance_score = float(payload.get("performance_score", success_rate))
        return build_eval_summary(
            num_episodes=num_episodes,
            successes=round(success_rate * num_episodes),
            episode_returns=[],
            benchmark_name=spec.benchmark_name,
            checkpoint_path=checkpoint,
            metrics=metrics,
            success_rate=success_rate,
            performance_score=performance_score,
        )

    # Metric-only eval (e.g. open-loop GT MAE): no pass/fail grade to report.
    metrics.setdefault("benchmark", spec.benchmark_name)
    metrics.setdefault("checkpoint_path", str(checkpoint))
    return {
        "num_episodes": num_episodes,
        "metrics": metrics,
    }


class CustomEvalRunner(Runner):
    """Evaluation runner for ``evaluation_type: custom`` tasks."""

    @property
    def name(self) -> str:
        return "custom"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.EVALUATION}

    @property
    def supported_types(self) -> set[str]:
        return {EvaluationType.CUSTOM.value}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        spec = context.task.spec
        if not isinstance(spec, EvaluationTask):
            raise TypeError(
                f"CustomEvalRunner expects EvaluationTask, got {type(spec).__name__}"
            )

        config = spec.config or {}
        checkpoint = resolve_eval_checkpoint(context)
        eval_script = resolve_eval_script(config)
        interpreter = resolve_interpreter(config)
        out_json = self._metrics_path(context)

        await context.emit_progress(
            "executing",
            step="launch",
            step_label=(
                f"script={Path(eval_script).name} "
                f"python={Path(interpreter[0]).name}"
            ),
        )

        process_spec = TrainingProcessSpec(
            timeout_seconds=getattr(spec, "timeout_seconds", None),
            script_path=eval_script,
            launcher=interpreter,
            argv_extra=build_custom_argv(
                checkpoint=checkpoint, out_json=out_json, config=config
            ),
        )

        rc = await run_training_subprocess(context, process_spec)
        if context.cancelled():
            logger.info("Custom eval task %s cancelled by user", context.task.id)
            return {"cancelled": True}
        if rc != 0:
            raise RuntimeError(f"Custom eval script exited with code {rc}")

        payload = read_metrics(out_json)
        return summarize(
            payload=payload,
            spec=spec,
            checkpoint=checkpoint,
            eval_script=eval_script,
        )

    @staticmethod
    def _metrics_path(context: TaskContext) -> Path:
        """Where the script writes its metrics — under the task output dir when
        the engine provided one, else a temp dir (context-free test setups)."""
        base = context.output_dir or Path(tempfile.gettempdir())
        base.mkdir(parents=True, exist_ok=True)
        return base / _METRICS_FILENAME
