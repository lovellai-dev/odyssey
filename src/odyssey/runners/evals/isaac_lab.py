"""Isaac Lab evaluation runner.

Evaluates a trained policy inside NVIDIA Isaac Lab simulation environments.
Isaac Lab is a GPU-accelerated robotics simulation framework built on top of
NVIDIA Isaac Sim / Omniverse, using PhysX for physics.

Structural mirror of ``RobosuiteRunner`` with three key differences:

  1. **Gymnasium 5-tuple API** — Isaac Lab envs return
     ``(obs, reward, terminated, truncated, info)`` on ``step()``, and
     ``(obs, info)`` on ``reset()``, matching the modern Gymnasium contract.
  2. **Observation conversion** — Isaac Lab may return torch tensors on GPU.
     The runner converts them to numpy/dicts so VLA policies (which expect
     PIL-convertible image arrays) work unchanged.
  3. **App initialization** — Isaac Lab requires a one-time ``AppLauncher``
     boot in headless mode before any env can be created.

All heavy Isaac Lab imports are deferred so this module is importable in
environments without NVIDIA dependencies (CI, macOS, tests).
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


# ---------------------------------------------------------------------------
# Type aliases (same shape as robosuite runner for consistency)
# ---------------------------------------------------------------------------

Policy = Callable[[dict[str, Any]], Any]
"""Maps one observation dict to one action array."""

PolicyFactory = Callable[[Path], Policy]
"""Builds a Policy from a local checkpoint path."""

EnvFactory = Callable[[str, str | None], Any]
"""Builds an Isaac Lab env from ``(env_id, isaac_robot_name)``.

``isaac_robot_name`` is None when the mission's embodiment has no known
Isaac Lab mapping — the factory should use the env's default robot.
"""


# ---------------------------------------------------------------------------
# Embodiment mapping
# ---------------------------------------------------------------------------

ISAAC_LAB_ROBOT_NAMES: dict[str, str] = {
    "franka_panda": "Franka",
    "panda": "Franka",
    "ur5e": "UR5e",
    "ur10": "UR10",
    "kinova_gen3": "KinovaGen3",
}
"""Odyssey embodiment name -> Isaac Lab robot name.

Isaac Lab uses PascalCase robot names embedded in environment IDs
(e.g. ``Isaac-Lift-Cube-Franka-v0``). This map covers the robots
that exist in both Odyssey's spec and Isaac Lab's env catalogue.
"""


# ---------------------------------------------------------------------------
# Benchmark -> env ID mapping
# ---------------------------------------------------------------------------

ISAAC_LAB_BENCHMARKS: dict[str, str] = {
    "Lift": "Isaac-Lift-Cube-{robot}-v0",
    "Reach": "Isaac-Reach-{robot}-v0",
}
"""Odyssey benchmark name -> Isaac Lab env ID template.

The ``{robot}`` placeholder is replaced by the resolved Isaac Lab robot
name. Users can bypass this mapping entirely by passing a full Isaac Lab
env ID as ``benchmark_name`` (e.g. ``Isaac-Lift-Cube-Franka-v0``).
"""


# ---------------------------------------------------------------------------
# Env factory helpers
# ---------------------------------------------------------------------------

_isaac_app_initialized = False


def _ensure_isaac_app() -> None:
    """Lazily boot Isaac Lab's simulation app in headless mode.

    Must be called before any ``gymnasium.make("Isaac-*")`` call.
    Safe to call multiple times — only the first call does work.
    """
    global _isaac_app_initialized
    if _isaac_app_initialized:
        return
    try:
        from isaaclab.app import AppLauncher
    except ImportError as e:
        raise RuntimeError(
            "Isaac Lab eval requires NVIDIA Isaac Lab. "
            "Install following https://isaac-sim.github.io/IsaacLab/main/source/setup/installation.html "
            "or: pip install 'lovell-odyssey[isaac_lab]'"
        ) from e
    launcher = AppLauncher(headless=True)
    _ = launcher.app
    _isaac_app_initialized = True


def _default_env_factory(env_id: str, _isaac_robot: str | None) -> Any:
    """Create an Isaac Lab gymnasium env for policy evaluation."""
    try:
        _ensure_isaac_app()
        import gymnasium as gym
    except ImportError as e:
        raise RuntimeError(
            "Isaac Lab eval requires NVIDIA Isaac Lab. "
            "Install following https://isaac-sim.github.io/IsaacLab/main/source/setup/installation.html "
            "or: pip install 'lovell-odyssey[isaac_lab]'"
        ) from e
    return gym.make(env_id)


def _default_policy_factory(checkpoint_path: Path) -> Policy:
    raise NotImplementedError(
        "IsaacLabRunner has no built-in policy for v0.1.0-alpha. "
        "Pass policy_factory=... to IsaacLabRunner(...) with a function "
        f"that loads {checkpoint_path!r} and returns a callable mapping "
        "observations to actions."
    )


# ---------------------------------------------------------------------------
# Observation / action helpers
# ---------------------------------------------------------------------------

def _to_numpy(value: Any) -> Any:
    """Convert a torch tensor to a numpy array; pass through otherwise."""
    if hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return value


def _convert_obs(
    obs_raw: Any,
    *,
    isaac_camera_key: str = "rgb",
    policy_image_key: str = "agentview_image",
) -> dict[str, Any]:
    """Normalise Isaac Lab observations for VLA policies.

    Isaac Lab may return torch tensors on GPU. VLA policies expect a dict
    with numpy image arrays keyed by ``*_image`` (``_find_image_key`` in
    ``openvla.py`` searches for this suffix).

    When running a single env (``num_envs=1``), Isaac Lab still returns
    batch-first tensors — we squeeze the leading dimension so the policy
    sees plain ``(H, W, C)`` images.
    """
    import numpy as np

    if not isinstance(obs_raw, dict):
        arr = _to_numpy(obs_raw)
        if isinstance(arr, np.ndarray) and arr.ndim >= 2 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        return {"observation": arr}

    result: dict[str, Any] = {}
    for key, value in obs_raw.items():
        arr = _to_numpy(value)
        # Squeeze batch dim for single-env runs
        if isinstance(arr, np.ndarray) and arr.ndim >= 2 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        out_key = policy_image_key if key == isaac_camera_key else key
        result[out_key] = arr
    return result


def _scalar(value: Any) -> float:
    """Extract a Python float from a (possibly batched) reward."""
    if hasattr(value, "item"):
        return float(value.item())
    if hasattr(value, "__getitem__") and hasattr(value, "__len__") and len(value) == 1:
        return float(value[0])
    return float(value)


def _is_done(value: Any) -> bool:
    """Extract a Python bool from a (possibly batched) done flag."""
    if hasattr(value, "item"):
        return bool(value.item())
    if hasattr(value, "__getitem__") and hasattr(value, "__len__") and len(value) == 1:
        return bool(value[0])
    return bool(value)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def _resolve_env_id(benchmark_name: str, isaac_robot: str | None) -> str:
    """Map ``benchmark_name`` + robot to an Isaac Lab gymnasium env ID.

    If ``benchmark_name`` already looks like a full Isaac Lab env ID
    (starts with ``Isaac-``), it is returned as-is. Otherwise, the name is
    looked up in ``ISAAC_LAB_BENCHMARKS`` and the ``{robot}`` placeholder
    is filled in.
    """
    if benchmark_name.startswith("Isaac-"):
        return benchmark_name

    template = ISAAC_LAB_BENCHMARKS.get(benchmark_name)
    if template is None:
        raise ValueError(
            f"Unknown Isaac Lab benchmark {benchmark_name!r}. "
            f"Supported shorthand names: {sorted(ISAAC_LAB_BENCHMARKS)}. "
            "Alternatively, pass a full Isaac Lab env ID "
            "(e.g. 'Isaac-Lift-Cube-Franka-v0') as benchmark_name."
        )
    robot = isaac_robot or "Franka"
    return template.format(robot=robot)


def _resolve_isaac_lab_robot(robot: RobotSpec) -> str | None:
    """Translate the mission's robot spec into an Isaac Lab robot name.

    Returns None for URDF specs or embodiments without a known Isaac Lab
    equivalent — the env factory should use the env's built-in default.
    """
    if robot.embodiment is None:
        return None
    name = ISAAC_LAB_ROBOT_NAMES.get(robot.embodiment)
    if name is None:
        raise ValueError(
            f"Isaac Lab has no built-in robot for embodiment "
            f"{robot.embodiment!r}. Supported: "
            f"{sorted(ISAAC_LAB_ROBOT_NAMES)}. Either pick one of these, "
            "or supply ``urdf:`` and a custom env_factory."
        )
    return name


def _resolve_eval_checkpoint(context: TaskContext) -> Path:
    """Find the PILOT checkpoint the eval should run against.

    Uses ``context.agent_checkpoints`` (populated by the engine) to find
    the first PILOT agent with a trained checkpoint. Multi-agent loadouts
    (PILOT + SPECIALIST) are supported.
    """
    from odyssey.spec.agents import AgentRole

    for agent in context.agents or context.mission.spec.robot.agents:
        if agent.role != AgentRole.PILOT:
            continue
        checkpoint = (context.agent_checkpoints or {}).get(agent.id)
        if not checkpoint:
            checkpoint = context.mission.latest_checkpoint_for(agent.id)
        if checkpoint:
            return Path(checkpoint)

    raise ValueError(
        "No completed training task produced a checkpoint for any PILOT "
        "agent on this mission — cannot evaluate."
    )


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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class IsaacLabRunner(Runner):
    """Evaluation runner for ``evaluation_type: isaac_lab`` tasks.

    Uses NVIDIA Isaac Lab (Gymnasium API) for simulation. Structural
    analogue of ``RobosuiteRunner`` adapted for the 5-tuple Gymnasium
    step contract and Isaac Lab's observation format.
    """

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
        await context.emit_progress(
            "model_loading",
            step="load_policy",
            step_label=str(checkpoint),
        )

        # Policy: use OpenVLA when no custom factory was injected
        if self._policy_factory is _default_policy_factory:
            from odyssey.runners.models.openvla import make_openvla_policy

            policy = make_openvla_policy(
                checkpoint,
                config=spec.config,
                benchmark_name=spec.benchmark_name,
            )
        else:
            policy = self._policy_factory(checkpoint)

        # Resolve env ID from benchmark_name + robot
        isaac_robot = _resolve_isaac_lab_robot(context.mission.spec.robot)
        env_id = _resolve_env_id(spec.benchmark_name, isaac_robot)
        await context.emit_progress(
            "executing",
            step="env_construct",
            step_label=f"env_id={env_id} robot={isaac_robot or 'default'}",
        )

        env = self._env_factory(env_id, isaac_robot)

        cfg = spec.config or {}
        isaac_camera_key = cfg.get("isaac_camera_key", "rgb")
        policy_image_key = cfg.get("image_key", "agentview_image")

        num_episodes = spec.num_episodes
        successes = 0
        episode_returns: list[float] = []

        for ep in range(1, num_episodes + 1):
            if context.cancelled():
                logger.info(
                    "Isaac Lab task %s cancelled at episode %d/%d",
                    context.task.id,
                    ep,
                    num_episodes,
                )
                break

            # Gymnasium reset: (obs, info)
            reset_result = env.reset()
            obs_raw = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            obs = _convert_obs(
                obs_raw,
                isaac_camera_key=isaac_camera_key,
                policy_image_key=policy_image_key,
            )
            episode_return = 0.0
            success = False

            for _step in range(self._max_steps):
                if context.cancelled():
                    break

                action = policy(obs)

                # Gymnasium step: 5-tuple (obs, reward, terminated, truncated, info)
                step_result = env.step(action)
                obs_raw, reward, terminated, truncated, info = step_result

                obs = _convert_obs(
                    obs_raw,
                    isaac_camera_key=isaac_camera_key,
                    policy_image_key=policy_image_key,
                )
                reward_val = _scalar(reward)
                episode_return += reward_val

                done = _is_done(terminated) or _is_done(truncated)
                if done:
                    success = bool(
                        info.get("success", reward_val > 0)
                        if isinstance(info, dict)
                        else reward_val > 0
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
                step_label=(
                    f"episode {ep}: {'PASS' if success else 'FAIL'} "
                    f"return={episode_return:.3f}"
                ),
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
                "env_id": env_id,
                "checkpoint_path": str(checkpoint),
            },
        }
