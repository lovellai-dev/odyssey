"""Tests for the Isaac Lab eval runner.

All tests inject ``env_factory`` and ``policy_factory`` so they run without
NVIDIA Isaac Lab / Isaac Sim installed. Covers the episode loop (5-tuple
Gymnasium API), success tallying, result_summary shape, checkpoint resolution,
embodiment mapping, env-ID resolution, observation conversion, and
cancellation.

Mirrors ``test_robosuite_runner.py`` structurally — same fixture helpers,
same coverage targets, adapted for Isaac Lab's Gymnasium contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from odyssey.engine.lifecycle import TaskStatus
from odyssey.engine.records import MissionRun, TaskRun
from odyssey.runners.base import TaskContext
from odyssey.runners.evals.isaac_lab import (
    ISAAC_LAB_ROBOT_NAMES,
    IsaacLabRunner,
    _convert_obs,
    _grade,
    _is_done,
    _resolve_env_id,
    _scalar,
)
from odyssey.spec import (
    AgentRole,
    AgentSpec,
    EvaluationTask,
    EvaluationType,
    HFModelRef,
    Mission,
    MissionMetadata,
    RobotSpec,
    TaskKind,
    TrainingTask,
    TrainingType,
)
from odyssey.telemetry import EventPublisher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NullPublisher(EventPublisher):
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        return


class _FakeIsaacEnv:
    """Isaac Lab-shaped env using the Gymnasium 5-tuple API.

    ``reset()`` returns ``(obs_dict, info)``.
    ``step(action)`` returns ``(obs, reward, terminated, truncated, info)``.
    """

    def __init__(
        self,
        *,
        steps_per_episode: int = 3,
        success: bool = True,
        image_shape: tuple[int, ...] = (256, 256, 3),
    ):
        self.steps_per_episode = steps_per_episode
        self.success = success
        self._image_shape = image_shape
        self._step = 0

    def _obs(self) -> dict[str, Any]:
        return {
            "rgb": np.random.randint(0, 255, self._image_shape, dtype=np.uint8),
            "joint_pos": np.zeros(7, dtype=np.float32),
        }

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self._step = 0
        return self._obs(), {}

    def step(
        self, action: Any
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step += 1
        done = self._step >= self.steps_per_episode
        reward = 1.0 if done and self.success else 0.0
        terminated = done
        truncated = False
        info: dict[str, Any] = {"success": self.success} if done else {}
        return self._obs(), reward, terminated, truncated, info


def _mission(
    *,
    train_checkpoint: str | None,
    embodiment: str = "franka_panda",
) -> MissionRun:
    spec = Mission(
        metadata=MissionMetadata(name="msn-isaac-eval"),
        objective="o",
        acceptance_criteria="a",
        robot=RobotSpec(
            embodiment=embodiment,
            agents=[
                AgentSpec(
                    id="pilot",
                    role=AgentRole.PILOT,
                    model=HFModelRef(base="openvla/openvla-7b"),
                ),
            ],
        ),
        tasks=[
            TrainingTask(
                name="train",
                training_type=TrainingType.DEMONSTRATION,
                agent_id="pilot",
            ),
            EvaluationTask(
                name="eval-isaac",
                evaluation_type=EvaluationType.ISAAC_LAB,
                benchmark_name="Lift",
                num_episodes=2,
            ),
        ],
    )
    mission = MissionRun.from_spec(spec)
    if train_checkpoint is not None:
        mission.tasks[0].status = TaskStatus.COMPLETED
        mission.tasks[0].result_summary = {
            "checkpoint_path": train_checkpoint,
            "agent_id": "pilot",
        }
    return mission


def _ctx(mission: MissionRun, eval_task: TaskRun, tmp_path: Path) -> TaskContext:
    return TaskContext(
        task=eval_task,
        mission=mission,
        publisher=_NullPublisher(),
        output_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# Runner surface
# ---------------------------------------------------------------------------

def test_supports_evaluation_isaac_lab_only() -> None:
    runner = IsaacLabRunner(
        env_factory=lambda _env_id, _robot: _FakeIsaacEnv(),
        policy_factory=lambda _: lambda obs: np.zeros(7),
    )
    assert runner.name == "isaac_lab"
    assert runner.supported_kinds == {TaskKind.EVALUATION}
    assert runner.supported_types == {EvaluationType.ISAAC_LAB.value}


# ---------------------------------------------------------------------------
# Happy path — episode loop with 5-tuple API
# ---------------------------------------------------------------------------

async def test_runs_all_episodes_and_tallies_successes(tmp_path: Path) -> None:
    mission = _mission(train_checkpoint="/tmp/checkpoint")
    eval_task = mission.tasks[1]
    runner = IsaacLabRunner(
        env_factory=lambda _e, _r: _FakeIsaacEnv(steps_per_episode=2, success=True),
        policy_factory=lambda _: lambda obs: np.zeros(7),
    )
    result = await runner.run(_ctx(mission, eval_task, tmp_path))

    assert result["num_episodes"] == 2
    assert result["success_rate"] == 1.0
    assert result["passed"] is True
    assert result["letter_grade"] == "A"
    assert result["metrics"]["checkpoint_path"] == "/tmp/checkpoint"
    assert "env_id" in result["metrics"]


async def test_failed_episodes_drop_success_rate(tmp_path: Path) -> None:
    mission = _mission(train_checkpoint="/tmp/checkpoint")
    eval_task = mission.tasks[1]
    runner = IsaacLabRunner(
        env_factory=lambda _e, _r: _FakeIsaacEnv(steps_per_episode=2, success=False),
        policy_factory=lambda _: lambda obs: np.zeros(7),
    )
    result = await runner.run(_ctx(mission, eval_task, tmp_path))
    assert result["success_rate"] == 0.0
    assert result["passed"] is False
    assert result["letter_grade"] == "F"


async def test_mixed_episodes(tmp_path: Path) -> None:
    """First episode succeeds, second fails -> 50% success rate."""

    class _AlternatingEnv:
        def __init__(self) -> None:
            self._episode = 0
            self._step = 0

        def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
            self._step = 0
            return {"rgb": np.zeros((64, 64, 3), dtype=np.uint8)}, {}

        def step(
            self, action: Any
        ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
            self._step += 1
            done = self._step >= 2
            if done:
                success = self._episode % 2 == 0
                self._episode += 1
                reward = 1.0 if success else 0.0
                return (
                    {"rgb": np.zeros((64, 64, 3), dtype=np.uint8)},
                    reward,
                    True,
                    False,
                    {"success": success},
                )
            return (
                {"rgb": np.zeros((64, 64, 3), dtype=np.uint8)},
                0.0,
                False,
                False,
                {},
            )

    mission = _mission(train_checkpoint="/tmp/cp")
    eval_task = mission.tasks[1]
    runner = IsaacLabRunner(
        env_factory=lambda _e, _r: _AlternatingEnv(),
        policy_factory=lambda _: lambda obs: np.zeros(7),
    )
    result = await runner.run(_ctx(mission, eval_task, tmp_path))
    assert result["success_rate"] == 0.5
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

async def test_cancellation_stops_episodes(tmp_path: Path) -> None:
    mission = _mission(train_checkpoint="/tmp/checkpoint")
    # Override to more episodes
    mission.spec.tasks[1] = EvaluationTask(
        name="eval-isaac",
        evaluation_type=EvaluationType.ISAAC_LAB,
        benchmark_name="Lift",
        num_episodes=100,
    )
    mission_new = MissionRun.from_spec(mission.spec)
    mission_new.tasks[0].status = TaskStatus.COMPLETED
    mission_new.tasks[0].result_summary = {
        "checkpoint_path": "/tmp/checkpoint",
        "agent_id": "pilot",
    }
    eval_task = mission_new.tasks[1]
    ctx = _ctx(mission_new, eval_task, tmp_path)

    # Cancel after the first episode completes
    steps_seen = 0
    original_env = _FakeIsaacEnv(steps_per_episode=1, success=True)
    original_reset = original_env.reset

    def _counting_reset() -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal steps_seen
        steps_seen += 1
        if steps_seen >= 2:
            ctx.request_cancel()
        return original_reset()

    original_env.reset = _counting_reset  # type: ignore[assignment]

    runner = IsaacLabRunner(
        env_factory=lambda _e, _r: original_env,
        policy_factory=lambda _: lambda obs: np.zeros(7),
    )
    result = await runner.run(ctx)
    # Should have run far fewer than 100 episodes
    assert result["num_episodes"] < 100


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------

async def test_raises_when_no_training_has_completed(tmp_path: Path) -> None:
    mission = _mission(train_checkpoint=None)
    eval_task = mission.tasks[1]
    runner = IsaacLabRunner(
        env_factory=lambda _e, _r: _FakeIsaacEnv(),
        policy_factory=lambda _: lambda obs: np.zeros(7),
    )
    with pytest.raises(ValueError, match="No completed training task"):
        await runner.run(_ctx(mission, eval_task, tmp_path))


async def test_picks_latest_checkpoint(tmp_path: Path) -> None:
    mission = _mission(train_checkpoint="/tmp/older")
    later_train = TaskRun(
        spec=TrainingTask(
            name="train-2",
            training_type=TrainingType.DEMONSTRATION,
            agent_id="pilot",
        ),
        status=TaskStatus.COMPLETED,
        result_summary={"checkpoint_path": "/tmp/newer", "agent_id": "pilot"},
    )
    eval_task = mission.tasks[-1]
    mission.tasks = [mission.tasks[0], later_train, eval_task]

    captured: dict[str, Any] = {}

    def _policy_factory(cp: Path) -> Any:
        captured["cp"] = str(cp)
        return lambda obs: np.zeros(7)

    runner = IsaacLabRunner(
        env_factory=lambda _e, _r: _FakeIsaacEnv(),
        policy_factory=_policy_factory,
    )
    await runner.run(_ctx(mission, eval_task, tmp_path))
    assert captured["cp"] == "/tmp/newer"


# ---------------------------------------------------------------------------
# Default policy raises with guidance
# ---------------------------------------------------------------------------

async def test_default_policy_factory_raises_with_guidance(tmp_path: Path) -> None:
    mission = _mission(train_checkpoint="/tmp/checkpoint")
    eval_task = mission.tasks[1]
    runner = IsaacLabRunner(env_factory=lambda _e, _r: _FakeIsaacEnv())
    with pytest.raises(NotImplementedError, match="policy"):
        await runner.run(_ctx(mission, eval_task, tmp_path))


# ---------------------------------------------------------------------------
# Embodiment -> Isaac Lab robot mapping
# ---------------------------------------------------------------------------

async def test_env_factory_receives_translated_robot(tmp_path: Path) -> None:
    mission = _mission(train_checkpoint="/tmp/checkpoint")
    eval_task = mission.tasks[1]
    captured: dict[str, Any] = {}

    def _capturing_factory(env_id: str, robot: str | None) -> _FakeIsaacEnv:
        captured["env_id"] = env_id
        captured["robot"] = robot
        return _FakeIsaacEnv()

    runner = IsaacLabRunner(
        env_factory=_capturing_factory,
        policy_factory=lambda _: lambda obs: np.zeros(7),
    )
    await runner.run(_ctx(mission, eval_task, tmp_path))
    assert captured["robot"] == "Franka"
    assert captured["env_id"] == "Isaac-Lift-Cube-Franka-v0"


async def test_env_factory_gets_none_for_urdf_robot(tmp_path: Path) -> None:
    urdf = tmp_path / "arm.urdf"
    urdf.write_text("<robot/>")
    mission = _mission(train_checkpoint="/tmp/checkpoint")
    mission.spec.robot = RobotSpec(
        urdf=str(urdf),
        agents=mission.spec.robot.agents,
    )
    eval_task = mission.tasks[1]
    captured: dict[str, Any] = {}

    def _capturing_factory(env_id: str, robot: str | None) -> _FakeIsaacEnv:
        captured["robot"] = robot
        return _FakeIsaacEnv()

    runner = IsaacLabRunner(
        env_factory=_capturing_factory,
        policy_factory=lambda _: lambda obs: np.zeros(7),
    )
    await runner.run(_ctx(mission, eval_task, tmp_path))
    assert captured["robot"] is None


async def test_unsupported_embodiment_fails_with_listing(tmp_path: Path) -> None:
    mission = _mission(train_checkpoint="/tmp/checkpoint")
    mission.spec.robot = RobotSpec.model_construct(
        embodiment="unitree_go2",
        agents=mission.spec.robot.agents,
    )
    eval_task = mission.tasks[1]
    runner = IsaacLabRunner(
        env_factory=lambda _e, _r: _FakeIsaacEnv(),
        policy_factory=lambda _: lambda obs: np.zeros(7),
    )
    with pytest.raises(ValueError, match="Isaac Lab has no built-in robot"):
        await runner.run(_ctx(mission, eval_task, tmp_path))


# ---------------------------------------------------------------------------
# Env ID resolution
# ---------------------------------------------------------------------------

def test_resolve_env_id_shorthand() -> None:
    assert _resolve_env_id("Lift", "Franka") == "Isaac-Lift-Cube-Franka-v0"
    assert _resolve_env_id("Reach", "UR5e") == "Isaac-Reach-UR5e-v0"


def test_resolve_env_id_passthrough() -> None:
    full_id = "Isaac-Custom-Task-Robot-v2"
    assert _resolve_env_id(full_id, "Franka") == full_id


def test_resolve_env_id_default_robot() -> None:
    assert _resolve_env_id("Lift", None) == "Isaac-Lift-Cube-Franka-v0"


def test_resolve_env_id_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown Isaac Lab benchmark"):
        _resolve_env_id("UnknownTask", "Franka")


# ---------------------------------------------------------------------------
# Observation conversion
# ---------------------------------------------------------------------------

def test_convert_obs_dict_with_camera_key() -> None:
    obs_raw = {
        "rgb": np.zeros((256, 256, 3), dtype=np.uint8),
        "joint_pos": np.array([1.0, 2.0]),
    }
    result = _convert_obs(obs_raw, isaac_camera_key="rgb", policy_image_key="agentview_image")
    assert "agentview_image" in result
    assert "joint_pos" in result
    assert "rgb" not in result


def test_convert_obs_squeezes_batch_dim() -> None:
    obs_raw = {
        "rgb": np.zeros((1, 64, 64, 3), dtype=np.uint8),
    }
    result = _convert_obs(obs_raw, isaac_camera_key="rgb", policy_image_key="agentview_image")
    assert result["agentview_image"].shape == (64, 64, 3)


def test_convert_obs_non_dict() -> None:
    obs_raw = np.zeros((1, 10), dtype=np.float32)
    result = _convert_obs(obs_raw, isaac_camera_key="rgb", policy_image_key="agentview_image")
    assert "observation" in result
    assert result["observation"].shape == (10,)


# ---------------------------------------------------------------------------
# Scalar / done helpers
# ---------------------------------------------------------------------------

def test_scalar_from_float() -> None:
    assert _scalar(1.5) == 1.5


def test_scalar_from_array() -> None:
    assert _scalar(np.array([2.5])) == 2.5


def test_is_done_from_bool() -> None:
    assert _is_done(True) is True
    assert _is_done(False) is False


def test_is_done_from_array() -> None:
    assert _is_done(np.array([True])) is True
    assert _is_done(np.array([False])) is False


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def test_grade_boundaries() -> None:
    assert _grade(1.0) == "A"
    assert _grade(0.9) == "A"
    assert _grade(0.85) == "B"
    assert _grade(0.75) == "C"
    assert _grade(0.65) == "D"
    assert _grade(0.5) == "F"
    assert _grade(0.0) == "F"


# ---------------------------------------------------------------------------
# Robot name mapping coverage
# ---------------------------------------------------------------------------

def test_known_embodiments_map() -> None:
    assert ISAAC_LAB_ROBOT_NAMES["franka_panda"] == "Franka"
    assert ISAAC_LAB_ROBOT_NAMES["panda"] == "Franka"
    assert ISAAC_LAB_ROBOT_NAMES["ur5e"] == "UR5e"
