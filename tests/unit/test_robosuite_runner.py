"""Tests for the Robosuite eval runner skeleton.

We inject ``env_factory`` and ``policy_factory`` so the tests run without
robosuite / mujoco installed. Covers the lifecycle plumbing (episode
loop, success tally, result_summary shape, cancellation) and the
``FromTaskModelRef`` checkpoint resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from odyssey.engine.lifecycle import TaskStatus
from odyssey.engine.records import MissionRun, TaskRun
from odyssey.runners.base import TaskContext
from odyssey.runners.robosuite import RobosuiteRunner
from odyssey.spec import (
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
from odyssey.spec.refs import FromTaskModelRef
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


def _mission_with_from_task(train_checkpoint: str | None) -> MissionRun:
    spec = Mission(
        metadata=MissionMetadata(name="msn-eval"),
        objective="o",
        acceptance_criteria="a",
        robot=RobotSpec(embodiment="franka_panda"),
        tasks=[
            TrainingTask(
                name="train",
                training_type=TrainingType.DEMONSTRATION,
                model=HFModelRef(base="openvla/openvla-7b"),
                target_agent_id="pilot",
            ),
            EvaluationTask(
                name="eval",
                evaluation_type=EvaluationType.ROBOSUITE,
                benchmark_name="Lift",
                model=FromTaskModelRef(from_task="train"),
                target_agent_id="pilot",
                num_episodes=2,
            ),
        ],
    )
    mission = MissionRun.from_spec(spec)
    if train_checkpoint is not None:
        # Simulate a completed training task.
        mission.tasks[0].status = TaskStatus.COMPLETED
        mission.tasks[0].result_summary = {"checkpoint_path": train_checkpoint}
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
        env_factory=lambda _: _FakeEnv(),
        policy_factory=lambda _: lambda obs: [0.0],
    )
    assert runner.name == "robosuite"
    assert runner.supported_kinds == {TaskKind.EVALUATION}
    assert runner.supported_types == {EvaluationType.ROBOSUITE.value}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_runs_all_episodes_and_tallies_successes(tmp_path: Path) -> None:
    mission = _mission_with_from_task("/tmp/checkpoint")
    eval_task = mission.tasks[1]
    runner = RobosuiteRunner(
        env_factory=lambda _: _FakeEnv(steps_per_episode=2, success=True),
        policy_factory=lambda _: lambda obs: [0.0],
    )
    result = await runner.run(_ctx(mission, eval_task, tmp_path))

    assert result["num_episodes"] == 2
    assert result["success_rate"] == 1.0
    assert result["passed"] is True
    assert result["letter_grade"] == "A"
    assert result["metrics"]["checkpoint_path"] == "/tmp/checkpoint"


async def test_failed_episodes_drop_success_rate(tmp_path: Path) -> None:
    mission = _mission_with_from_task("/tmp/checkpoint")
    eval_task = mission.tasks[1]
    runner = RobosuiteRunner(
        env_factory=lambda _: _FakeEnv(steps_per_episode=2, success=False),
        policy_factory=lambda _: lambda obs: [0.0],
    )
    result = await runner.run(_ctx(mission, eval_task, tmp_path))
    assert result["success_rate"] == 0.0
    assert result["passed"] is False
    assert result["letter_grade"] == "F"


# ---------------------------------------------------------------------------
# from_task checkpoint resolution
# ---------------------------------------------------------------------------

async def test_raises_when_earlier_task_has_no_checkpoint(tmp_path: Path) -> None:
    mission = _mission_with_from_task(None)  # training task has no checkpoint_path
    eval_task = mission.tasks[1]
    runner = RobosuiteRunner(
        env_factory=lambda _: _FakeEnv(),
        policy_factory=lambda _: lambda obs: [0.0],
    )
    with pytest.raises(ValueError, match="no checkpoint_path"):
        await runner.run(_ctx(mission, eval_task, tmp_path))


async def test_raises_for_unknown_from_task(tmp_path: Path) -> None:
    mission = _mission_with_from_task("/tmp/checkpoint")
    # Mutate the eval task's model ref to point at a non-existent task.
    mission.tasks[1].spec.model = FromTaskModelRef(from_task="ghost-task")
    runner = RobosuiteRunner(
        env_factory=lambda _: _FakeEnv(),
        policy_factory=lambda _: lambda obs: [0.0],
    )
    with pytest.raises(ValueError, match="not found in mission"):
        await runner.run(_ctx(mission, mission.tasks[1], tmp_path))


# ---------------------------------------------------------------------------
# Default policy raises with a clear pointer
# ---------------------------------------------------------------------------

async def test_default_policy_factory_raises_with_guidance(tmp_path: Path) -> None:
    mission = _mission_with_from_task("/tmp/checkpoint")
    eval_task = mission.tasks[1]
    runner = RobosuiteRunner(env_factory=lambda _: _FakeEnv())
    with pytest.raises(NotImplementedError, match="policy"):
        await runner.run(_ctx(mission, eval_task, tmp_path))
