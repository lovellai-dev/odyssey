"""Tests for the Robosuite eval runner skeleton.

We inject ``env_factory`` and ``policy_factory`` so the tests run without
robosuite / mujoco installed. Covers the lifecycle plumbing (episode
loop, success tally, result_summary shape), the loadout-walk that
resolves the checkpoint to evaluate, and the embodiment → Robosuite
robot translation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from odyssey.engine.lifecycle import TaskStatus
from odyssey.engine.records import MissionRun, TaskRun
from odyssey.runners.base import TaskContext
from odyssey.runners.evals.robosuite import RobosuiteRunner
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


class _NullPublisher(EventPublisher):
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        return


class _FakeEnv:
    """Robosuite-shaped env that succeeds in N steps then returns done.

    ``reset()`` returns a dict observation. ``step(action)`` returns the
    ``(obs, reward, done, info)`` tuple robosuite gives.
    """

    def __init__(self, *, steps_per_episode: int = 3, success: bool = True):
        self.steps_per_episode = steps_per_episode
        self.success = success
        self._step = 0

    def reset(self) -> dict[str, Any]:
        self._step = 0
        return {"observation": [0.0]}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._step += 1
        done = self._step >= self.steps_per_episode
        reward = 1.0 if done and self.success else 0.0
        info = {"success": self.success} if done else {}
        return {"observation": [0.0]}, reward, done, info


def _mission(
    *,
    train_checkpoint: str | None,
    embodiment: str = "franka_panda",
) -> MissionRun:
    spec = Mission(
        metadata=MissionMetadata(name="msn-eval"),
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
                name="eval",
                evaluation_type=EvaluationType.ROBOSUITE,
                benchmark_name="Lift",
                num_episodes=2,
            ),
        ],
    )
    mission = MissionRun.from_spec(spec)
    if train_checkpoint is not None:
        # Simulate a completed training task.
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
# Surface
# ---------------------------------------------------------------------------

def test_supports_evaluation_robosuite_only() -> None:
    runner = RobosuiteRunner(
        env_factory=lambda _bench, _robot: _FakeEnv(),
        policy_factory=lambda _: lambda obs: [0.0],
    )
    assert runner.name == "robosuite"
    assert runner.supported_kinds == {TaskKind.EVALUATION}
    assert runner.supported_types == {EvaluationType.ROBOSUITE.value}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_runs_all_episodes_and_tallies_successes(tmp_path: Path) -> None:
    mission = _mission(train_checkpoint="/tmp/checkpoint")
    eval_task = mission.tasks[1]
    runner = RobosuiteRunner(
        env_factory=lambda _b, _r: _FakeEnv(steps_per_episode=2, success=True),
        policy_factory=lambda _: lambda obs: [0.0],
    )
    result = await runner.run(_ctx(mission, eval_task, tmp_path))

    assert result["num_episodes"] == 2
    assert result["success_rate"] == 1.0
    assert result["passed"] is True
    assert result["letter_grade"] == "A"
    assert result["metrics"]["checkpoint_path"] == "/tmp/checkpoint"


async def test_failed_episodes_drop_success_rate(tmp_path: Path) -> None:
    mission = _mission(train_checkpoint="/tmp/checkpoint")
    eval_task = mission.tasks[1]
    runner = RobosuiteRunner(
        env_factory=lambda _b, _r: _FakeEnv(steps_per_episode=2, success=False),
        policy_factory=lambda _: lambda obs: [0.0],
    )
    result = await runner.run(_ctx(mission, eval_task, tmp_path))
    assert result["success_rate"] == 0.0
    assert result["passed"] is False
    assert result["letter_grade"] == "F"


# ---------------------------------------------------------------------------
# Loadout-walk checkpoint resolution
# ---------------------------------------------------------------------------

async def test_raises_when_no_training_has_completed(tmp_path: Path) -> None:
    """The eval walks the loadout for the agent's latest training
    output; if no training has completed, it must fail cleanly."""
    mission = _mission(train_checkpoint=None)
    eval_task = mission.tasks[1]
    runner = RobosuiteRunner(
        env_factory=lambda _bench, _robot: _FakeEnv(),
        policy_factory=lambda _: lambda obs: [0.0],
    )
    with pytest.raises(ValueError, match="No completed training task"):
        await runner.run(_ctx(mission, eval_task, tmp_path))


async def test_picks_latest_checkpoint_when_multiple_trainings(
    tmp_path: Path,
) -> None:
    """When several training tasks for the same agent have completed,
    the eval should run against the latest checkpoint (walk-in-reverse
    semantic on the per-agent chain)."""
    mission = _mission(train_checkpoint="/tmp/older")
    # Insert a second completed training task on the same agent with a
    # newer checkpoint. Append before the eval so the spec order stays
    # train, train, eval — matches the per-agent chain semantics.
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
        return lambda obs: [0.0]

    runner = RobosuiteRunner(
        env_factory=lambda _b, _r: _FakeEnv(),
        policy_factory=_policy_factory,
    )
    await runner.run(_ctx(mission, eval_task, tmp_path))
    assert captured["cp"] == "/tmp/newer"


# ---------------------------------------------------------------------------
# Default policy raises with a clear pointer
# ---------------------------------------------------------------------------

async def test_default_policy_factory_raises_with_guidance(tmp_path: Path) -> None:
    mission = _mission(train_checkpoint="/tmp/checkpoint")
    eval_task = mission.tasks[1]
    runner = RobosuiteRunner(env_factory=lambda _b, _r: _FakeEnv())
    with pytest.raises(NotImplementedError, match="policy"):
        await runner.run(_ctx(mission, eval_task, tmp_path))


# ---------------------------------------------------------------------------
# Embodiment → Robosuite robot plumbing
# ---------------------------------------------------------------------------

async def test_env_factory_receives_translated_robot(tmp_path: Path) -> None:
    """Mission says franka_panda → env_factory must see ``Panda``."""
    mission = _mission(train_checkpoint="/tmp/checkpoint")
    eval_task = mission.tasks[1]
    captured: dict[str, Any] = {}

    def _capturing_factory(bench: str, robot: str | None) -> _FakeEnv:
        captured["bench"] = bench
        captured["robot"] = robot
        return _FakeEnv()

    runner = RobosuiteRunner(
        env_factory=_capturing_factory,
        policy_factory=lambda _: lambda obs: [0.0],
    )
    await runner.run(_ctx(mission, eval_task, tmp_path))
    assert captured["bench"] == "Lift"
    assert captured["robot"] == "Panda"


async def test_env_factory_gets_none_for_urdf_robot(tmp_path: Path) -> None:
    """URDF specs have no Robosuite equivalent — fall back to env default."""
    urdf = tmp_path / "arm.urdf"
    urdf.write_text("<robot/>")
    mission = _mission(train_checkpoint="/tmp/checkpoint")
    mission.spec.robot = RobotSpec(
        urdf=str(urdf),
        agents=mission.spec.robot.agents,
    )
    eval_task = mission.tasks[1]
    captured: dict[str, Any] = {}

    def _capturing_factory(bench: str, robot: str | None) -> _FakeEnv:
        captured["robot"] = robot
        return _FakeEnv()

    runner = RobosuiteRunner(
        env_factory=_capturing_factory,
        policy_factory=lambda _: lambda obs: [0.0],
    )
    await runner.run(_ctx(mission, eval_task, tmp_path))
    assert captured["robot"] is None


async def test_unsupported_embodiment_fails_with_listing(tmp_path: Path) -> None:
    """Embodiment that the spec accepts but Robosuite can't simulate
    should fail loudly rather than silently default to Panda."""
    mission = _mission(train_checkpoint="/tmp/checkpoint")
    # Bypass the RobotSpec validator (which also gates on
    # KNOWN_EMBODIMENTS via the local provider in normal flow) so we
    # can exercise the runner-level guard directly.
    mission.spec.robot = RobotSpec.model_construct(
        embodiment="unitree_go2",
        agents=mission.spec.robot.agents,
    )
    eval_task = mission.tasks[1]
    runner = RobosuiteRunner(
        env_factory=lambda _b, _r: _FakeEnv(),
        policy_factory=lambda _: lambda obs: [0.0],
    )
    with pytest.raises(ValueError, match="Robosuite has no built-in robot"):
        await runner.run(_ctx(mission, eval_task, tmp_path))
