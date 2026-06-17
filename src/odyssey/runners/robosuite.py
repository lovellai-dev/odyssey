"""Robosuite evaluation runner — Scope B skeleton.

What this DOES today:
  * Walks the robot's loadout (today exactly one agent) and pulls the
    latest completed training task's checkpoint for it via
    ``mission.latest_checkpoint_for(agent.id)``.
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
from odyssey.spec.mission import RobotSpec
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

EnvFactory = Callable[[str, str | None], Any]
"""Builds a robosuite environment from ``(benchmark_name, robosuite_robot)``.

``robosuite_robot`` is the PascalCase name Robosuite expects
(``Panda``, ``Sawyer``, ...). None means "let Robosuite pick its
per-env default" — used when the mission's robot is a URDF or a
hosted Lovell embodiment whose name isn't in
``ROBOSUITE_ROBOT_NAMES``.
"""


# Odyssey embodiment → Robosuite robot model. The map deliberately
# mirrors the trimmed LocalRobotProvider.KNOWN_EMBODIMENTS so the same
# names that pass spec validation also run end-to-end.
#
# Anything missing here either has no Robosuite equivalent (Unitree
# quadrupeds, Tiago, Stretch3) or hasn't been wired yet. For those,
# operators can use ``urdf:`` and we'll fall through to Robosuite's
# default robot for the benchmark.
ROBOSUITE_ROBOT_NAMES: dict[str, str] = {
    "franka_panda": "Panda",
    "panda": "Panda",
    "sawyer": "Sawyer",
    "iiwa": "IIWA",
    "jaco": "Jaco",
    "kinova_gen3": "Kinova3",
    "ur5e": "UR5e",
    "baxter": "Baxter",
}


def _default_env_factory(benchmark_name: str, robosuite_robot: str | None) -> Any:
    try:
        import robosuite
    except ImportError as e:
        raise RuntimeError(
            "Robosuite eval requires the 'robosuite' extra. "
            "Install with: pip install 'lovell-odyssey[robosuite]'"
        ) from e
    kwargs: dict[str, Any] = {
        "env_name": benchmark_name,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
    }
    if robosuite_robot is not None:
        # Robosuite accepts either a string ("Panda") or a list
        # (["Panda"], ["Panda", "Sawyer"] for dual-arm envs). We pass a
        # string so single-arm envs work; dual-arm envs (TwoArmLift,
        # etc.) will need an explicit list form when we wire them.
        kwargs["robots"] = robosuite_robot
    return robosuite.make(**kwargs)


def _make_eval_env(
    benchmark_name: str,
    robosuite_robot: str | None,
    config: dict[str, Any] | None = None,
) -> Any:
    """Create a camera-enabled robosuite env for policy evaluation."""
    try:
        import robosuite
    except ImportError as e:
        raise RuntimeError(
            "Robosuite eval requires the 'robosuite' extra. "
            "Install with: pip install 'lovell-odyssey[robosuite]'"
        ) from e
    cfg = config or {}
    camera_names = cfg.get("camera_names", "agentview")
    camera_height = cfg.get("camera_height", 256)
    camera_width = cfg.get("camera_width", 256)
    kwargs: dict[str, Any] = {
        "env_name": benchmark_name,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "camera_names": camera_names,
        "camera_heights": camera_height,
        "camera_widths": camera_width,
    }
    if robosuite_robot is not None:
        kwargs["robots"] = robosuite_robot
    return robosuite.make(**kwargs)


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

        checkpoint = _resolve_eval_checkpoint(context)
        await context.emit_progress(
            "model_loading",
            step="load_policy",
            step_label=str(checkpoint),
        )

        # Policy: use OpenVLA when no custom factory was injected
        if self._policy_factory is _default_policy_factory:
            from odyssey.runners.openvla import make_openvla_policy

            policy = make_openvla_policy(
                checkpoint,
                config=spec.config,
                benchmark_name=spec.benchmark_name,
            )
        else:
            policy = self._policy_factory(checkpoint)

        robosuite_robot = _resolve_robosuite_robot(context.mission.spec.robot)
        await context.emit_progress(
            "executing",
            step="env_construct",
            step_label=(
                f"benchmark={spec.benchmark_name} "
                f"robot={robosuite_robot or 'robosuite-default'}"
            ),
        )

        # Env: use camera-enabled env when no custom factory was injected
        if self._env_factory is _default_env_factory:
            env = _make_eval_env(spec.benchmark_name, robosuite_robot, spec.config)
        else:
            env = self._env_factory(spec.benchmark_name, robosuite_robot)

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


def _resolve_robosuite_robot(robot: RobotSpec) -> str | None:
    """Translate the mission's robot spec into a Robosuite robot name.

    Returns the PascalCase string Robosuite expects, or None when no
    override should be passed (URDF specs, hosted Lovell ids, or no
    embodiment at all — in which case Robosuite picks its per-env
    default robot).

    Raises ValueError when ``embodiment`` is set but has no Robosuite
    equivalent. That's preferable to silently swapping in Robosuite's
    default and producing eval numbers under a robot the operator
    never asked for.
    """
    if robot.embodiment is None:
        return None
    name = ROBOSUITE_ROBOT_NAMES.get(robot.embodiment)
    if name is None:
        raise ValueError(
            f"Robosuite has no built-in robot for embodiment "
            f"{robot.embodiment!r}. Supported: "
            f"{sorted(ROBOSUITE_ROBOT_NAMES)}. Either pick one of these, "
            "or supply ``urdf:`` and a custom env_factory."
        )
    return name


def _resolve_eval_checkpoint(context: TaskContext) -> Path:
    """Find the checkpoint the eval should run.

    Today: walks the robot's loadout (exactly one agent), takes the
    latest completed training task's checkpoint for that agent, fails
    cleanly if none exists.

    When multi-agent evaluation arrives, this becomes "compose the
    loadout" — assemble each agent's current checkpoint and hand the
    runtime a brain rather than a single policy.
    """
    agents = context.mission.spec.robot.agents
    if len(agents) != 1:
        raise NotImplementedError(
            f"RobosuiteRunner expects exactly one agent on the robot "
            f"(got {len(agents)}). Multi-agent eval arrives with the "
            "multi-agent runtime."
        )
    agent = agents[0]
    checkpoint = context.mission.latest_checkpoint_for(agent.id)
    if not checkpoint:
        raise ValueError(
            f"No completed training task produced a checkpoint for agent "
            f"{agent.id!r} on this mission — cannot evaluate."
        )
    return Path(checkpoint)
