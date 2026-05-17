"""Robosuite evaluation runner — Scope B skeleton.

What this DOES today:
  * Resolves the task's model ref into a local checkpoint path,
    including ``FromTaskModelRef`` → earlier task's
    ``result_summary["checkpoint_path"]``.
  * Constructs a robosuite environment via a pluggable
    ``env_factory`` (defaults to real ``robosuite.make``).
  * Runs ``num_episodes`` rollouts, querying a pluggable ``policy``
    for actions per step, tallying successes.
  * Returns the result_summary shape lai-trainer's
    ``update_evaluation_results`` expects (``success_rate``,
    ``performance_score``, ``letter_grade``, ``passed``, ``metrics``).

What this DOES NOT do today (the integration gap):
  * Ship a built-in policy that translates an OpenVLA-loaded checkpoint
    into robosuite-shaped actions. That requires:
      - Loading the OpenVLA adapter (HF + peft)
      - Converting robosuite observations (dict of arrays) into the
        prompt + image inputs OpenVLA expects
      - Decoding OpenVLA outputs into 7-DoF (or task-specific) robosuite
        actions
    Each of these is a non-trivial integration that ships in v0.2.x.
    Until then, callers must supply a ``policy`` callable to the
    constructor; the default raises ``NotImplementedError`` with a
    pointer to docs.

What this lets us claim today: the eval-runner *plumbing* works
end-to-end. Pair it with a custom policy and you get real numbers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from odyssey.runners.base import Runner, TaskContext
from odyssey.spec.refs import FromTaskModelRef, HFModelRef, LovellModelRef
from odyssey.spec.tasks import EvaluationTask, EvaluationType, TaskKind

logger = logging.getLogger(__name__)


Policy = Callable[[dict[str, Any]], Any]
"""Maps one robosuite observation dict to one action.

Action shape is robosuite-task-specific (typically a 7-element
ndarray for end-effector control).
"""

PolicyFactory = Callable[[Path], Policy]
"""Builds a Policy from a local checkpoint path. Called once per task
before the episode loop."""

EnvFactory = Callable[[str], Any]
"""Builds a robosuite environment from a benchmark name. Default uses
``robosuite.make``."""


def _default_env_factory(benchmark_name: str) -> Any:
    try:
        import robosuite
    except ImportError as e:
        raise RuntimeError(
            "Robosuite eval requires the 'robosuite' extra. "
            "Install with: pip install 'lovell-odyssey[robosuite]'"
        ) from e
    return robosuite.make(
        env_name=benchmark_name,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )


def _default_policy_factory(checkpoint_path: Path) -> Policy:
    raise NotImplementedError(
        "RobosuiteRunner has no built-in policy for v0.1.0-alpha. "
        "Pass policy_factory=... to RobosuiteRunner(...) with a function "
        f"that loads {checkpoint_path!r} and returns a callable mapping "
        "robosuite observations to actions. See "
        "https://github.com/lovell/odyssey/blob/main/docs/runners/robosuite.md"
    )


class RobosuiteRunner(Runner):
    """Evaluation runner for ``evaluation_type: robosuite`` tasks."""

    def __init__(
        self,
        *,
        env_factory: EnvFactory | None = None,
        policy_factory: PolicyFactory | None = None,
        max_steps_per_episode: int = 500,
    ):
        self._env_factory = env_factory or _default_env_factory
        self._policy_factory = policy_factory or _default_policy_factory
        self._max_steps = max_steps_per_episode

    @property
    def name(self) -> str:
        return "robosuite"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.EVALUATION}

    @property
    def supported_types(self) -> set[str]:
        return {EvaluationType.ROBOSUITE.value}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        spec = context.task.spec
        if not isinstance(spec, EvaluationTask):
            raise TypeError(
                f"RobosuiteRunner expects EvaluationTask, got {type(spec).__name__}"
            )

        checkpoint = _resolve_checkpoint_path(context, spec)
        await context.emit_progress(
            "model_loading",
            step="load_policy",
            step_label=str(checkpoint),
        )
        policy = self._policy_factory(checkpoint)

        await context.emit_progress(
            "executing",
            step="env_construct",
            step_label=f"benchmark={spec.benchmark_name}",
        )
        env = self._env_factory(spec.benchmark_name)

        num_episodes = spec.num_episodes
        successes = 0
        episode_returns: list[float] = []

        for ep in range(1, num_episodes + 1):
            if context.cancelled():
                logger.info(
                    "Robosuite task %s cancelled at episode %d/%d",
                    context.task.id,
                    ep,
                    num_episodes,
                )
                break

            obs = env.reset()
            episode_return = 0.0
            success = False
            for _step in range(self._max_steps):
                if context.cancelled():
                    break
                action = policy(obs if isinstance(obs, dict) else {"observation": obs})
                step_result = env.step(action)
                # robosuite returns (obs, reward, done, info)
                obs, reward, done, info = step_result
                episode_return += float(reward)
                if done:
                    success = bool(
                        info.get("success", reward > 0)
                        if isinstance(info, dict)
                        else reward > 0
                    )
                    break

            if success:
                successes += 1
            episode_returns.append(episode_return)
            await context.emit_progress(
                "executing",
                step="episode_complete",
                step_index=ep,
                step_total=num_episodes,
                step_label=f"episode {ep}: {'PASS' if success else 'FAIL'} return={episode_return:.3f}",
            )

        attempted = len(episode_returns)
        success_rate = (successes / attempted) if attempted else 0.0
        performance_score = (
            sum(episode_returns) / max(attempted, 1)
            if episode_returns
            else 0.0
        )
        return {
            "num_episodes": attempted,
            "success_rate": round(success_rate, 4),
            "performance_score": round(performance_score, 4),
            "letter_grade": _grade(success_rate),
            "passed": success_rate >= 0.5,
            "metrics": {
                "successes": successes,
                "episode_returns": [round(r, 4) for r in episode_returns],
                "benchmark": spec.benchmark_name,
                "checkpoint_path": str(checkpoint),
            },
        }


def _grade(success_rate: float) -> str:
    if success_rate >= 0.9:
        return "A"
    if success_rate >= 0.8:
        return "B"
    if success_rate >= 0.7:
        return "C"
    if success_rate >= 0.6:
        return "D"
    return "F"


def _resolve_checkpoint_path(context: TaskContext, spec: EvaluationTask) -> Path:
    """Resolve the model ref to a local checkpoint path on disk.

    Today only ``FromTaskModelRef`` is handled — the engine's other model
    refs (HF, Lovell) for *evaluation* are an unusual case (you'd be
    eval'ing an unmodified base model) and aren't wired here yet.
    """
    ref = spec.model
    if isinstance(ref, FromTaskModelRef):
        target = next(
            (t for t in context.mission.tasks if t.spec.name == ref.from_task),
            None,
        )
        if target is None:
            raise ValueError(
                f"from_task reference {ref.from_task!r} not found in mission"
            )
        checkpoint = target.result_summary.get("checkpoint_path")
        if not checkpoint:
            raise ValueError(
                f"earlier task {ref.from_task!r} has no checkpoint_path in its "
                "result_summary — did training fail or get skipped?"
            )
        return Path(checkpoint)

    if isinstance(ref, HFModelRef):
        raise NotImplementedError(
            "Evaluating an HF base model directly isn't supported yet — "
            "use a from_task ref pointing at a training task."
        )
    if isinstance(ref, LovellModelRef):
        raise NotImplementedError(
            "LovellModelRef evaluation is a hosted-mode feature."
        )
    raise TypeError(f"Unrecognized model ref type: {type(ref).__name__}")
