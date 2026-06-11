"""End-to-end engine tests against InMemoryPersistence + CPUMockRunner.

These are the lifecycle tests called out in the publication plan's Week 2
deliverable: cover the state-machine paths from mission-service-guide.md,
plus the no-runner / runner-raises / cancel-from-active failure modes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from odyssey.engine import (
    InvalidStateTransitionError,
    MissionEngine,
    MissionNotFoundError,
    MissionStatus,
    TaskStatus,
)
from odyssey.persistence import InMemoryPersistence
from odyssey.runners import (
    WILDCARD_TYPE,
    CPUMockRunner,
    Runner,
    RunnerRegistry,
    TaskContext,
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
    load_mission,
)
from odyssey.telemetry import EventPublisher

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MISSION = REPO_ROOT / "examples" / "quickstart-openvla" / "mission.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CapturingPublisher(EventPublisher):
    """Records every published event in order for test assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))

    @property
    def event_types(self) -> list[str]:
        return [e[0] for e in self.events]


class _RaisingRunner(Runner):
    @property
    def name(self) -> str:
        return "raising"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.TRAINING, TaskKind.EVALUATION}

    @property
    def supported_types(self) -> set[str]:
        return {WILDCARD_TYPE}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        raise RuntimeError("simulated runner failure")


class _SlowRunner(Runner):
    """Blocks on its cancel_event until cancelled. Lets tests exercise the
    cancel-from-ACTIVE path without timing-sensitive sleeps."""

    @property
    def name(self) -> str:
        return "slow"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.TRAINING, TaskKind.EVALUATION}

    @property
    def supported_types(self) -> set[str]:
        return {WILDCARD_TYPE}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        await context.emit_progress("waiting")
        try:
            await asyncio.wait_for(context.cancel_event.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            return {"_finished_naturally": True}
        return {"_cancelled": True}


def _spec() -> Mission:
    return Mission(
        metadata=MissionMetadata(name="msn-test"),
        objective="objective",
        acceptance_criteria="acceptance",
        robot=RobotSpec(
            embodiment="franka_panda",
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


async def _make_engine(
    runner: Runner | None = None,
) -> tuple[MissionEngine, CapturingPublisher]:
    persistence = InMemoryPersistence()
    runners = RunnerRegistry()
    runners.register(runner if runner is not None else CPUMockRunner())
    publisher = CapturingPublisher()
    engine = MissionEngine(
        persistence=persistence, runners=runners, event_publisher=publisher
    )
    await engine.initialize()
    return engine, publisher


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_create_mission_persists_in_draft() -> None:
    engine, pub = await _make_engine()
    run = await engine.create_mission(_spec())

    assert run.status == MissionStatus.DRAFT
    assert all(t.status == TaskStatus.PENDING for t in run.tasks)
    fetched = await engine.get_mission(run.id)
    assert fetched.id == run.id
    assert pub.event_types == ["mission.created"]


async def test_start_mission_drives_to_completed() -> None:
    engine, pub = await _make_engine()
    run = await engine.create_mission(_spec())
    final = await engine.start_mission(run.id)

    assert final.status == MissionStatus.COMPLETED
    assert all(t.status == TaskStatus.COMPLETED for t in final.tasks)
    assert final.started_at is not None
    assert final.completed_at is not None
    assert final.overall_grade is not None  # CPU mock returns a score for eval

    # Lifecycle event order: created → queued → started → task events → completed
    assert pub.event_types[0] == "mission.created"
    assert pub.event_types[1] == "mission.queued"
    assert pub.event_types[2] == "mission.started"
    assert pub.event_types[-1] == "mission.completed"
    assert "task.completed" in pub.event_types
    assert "task.progress" in pub.event_types


async def test_overall_grade_averages_eval_scores() -> None:
    engine, _ = await _make_engine()
    run = await engine.create_mission(_spec())
    final = await engine.start_mission(run.id)

    eval_task = next(t for t in final.tasks if t.spec.kind == "evaluation")
    expected = eval_task.result_summary["performance_score"]
    assert final.overall_grade == pytest.approx(expected)


async def test_shipped_example_runs_end_to_end() -> None:
    """The OpenVLA quickstart YAML should drive cleanly through the engine
    when paired with the CPU mock runner (no GPU required)."""
    engine, _ = await _make_engine()
    spec = load_mission(EXAMPLE_MISSION)
    run = await engine.create_mission(spec)
    final = await engine.start_mission(run.id)
    assert final.status == MissionStatus.COMPLETED


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------

async def test_runner_exception_marks_mission_failed() -> None:
    engine, pub = await _make_engine(runner=_RaisingRunner())
    run = await engine.create_mission(_spec())
    final = await engine.start_mission(run.id)

    assert final.status == MissionStatus.FAILED
    assert final.tasks[0].status == TaskStatus.FAILED
    assert final.tasks[0].error_code == "runner_exception"
    assert "simulated runner failure" in (final.tasks[0].error_message or "")
    # Second task never ran because the first failed (sequential dispatch
    # short-circuits, then _finalize_mission sees a FAILED task).
    assert final.tasks[1].status == TaskStatus.PENDING
    assert "mission.failed" in pub.event_types
    assert "task.failed" in pub.event_types


async def test_no_runner_registered_fails_task() -> None:
    persistence = InMemoryPersistence()
    runners = RunnerRegistry()  # nothing registered
    publisher = CapturingPublisher()
    engine = MissionEngine(
        persistence=persistence, runners=runners, event_publisher=publisher
    )
    await engine.initialize()

    run = await engine.create_mission(_spec())
    final = await engine.start_mission(run.id)

    assert final.status == MissionStatus.FAILED
    assert final.tasks[0].status == TaskStatus.FAILED
    assert final.tasks[0].error_code == "no_runner_registered"


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

async def test_cancel_from_draft() -> None:
    engine, pub = await _make_engine()
    run = await engine.create_mission(_spec())
    cancelled = await engine.cancel_mission(run.id)

    assert cancelled.status == MissionStatus.CANCELLED
    # No tasks ever ran, so they should all be CANCELLED in the record.
    assert all(t.status == TaskStatus.CANCELLED for t in cancelled.tasks)
    assert "mission.cancelled" in pub.event_types


async def test_cancel_from_active_stops_runner() -> None:
    engine, _ = await _make_engine(runner=_SlowRunner())
    run = await engine.create_mission(_spec())

    start_task = asyncio.create_task(engine.start_mission(run.id))

    # Spin until the mission reaches ACTIVE and the slow runner has started.
    for _ in range(100):
        await asyncio.sleep(0.01)
        snap = await engine.get_mission(run.id)
        if snap.status == MissionStatus.ACTIVE and snap.tasks[0].status == TaskStatus.IN_PROGRESS:
            break
    else:
        start_task.cancel()
        pytest.fail("mission never reached ACTIVE/IN_PROGRESS")

    cancelled = await engine.cancel_mission(run.id)
    assert cancelled.status == MissionStatus.CANCELLED

    final_run = await asyncio.wait_for(start_task, timeout=2.0)
    assert final_run.status == MissionStatus.CANCELLED
    assert final_run.tasks[0].status == TaskStatus.CANCELLED
    assert final_run.tasks[1].status == TaskStatus.CANCELLED


async def test_cancel_terminal_mission_is_noop() -> None:
    engine, _ = await _make_engine()
    run = await engine.create_mission(_spec())
    completed = await engine.start_mission(run.id)
    assert completed.status == MissionStatus.COMPLETED

    again = await engine.cancel_mission(run.id)
    assert again.status == MissionStatus.COMPLETED  # unchanged


# ---------------------------------------------------------------------------
# Error surfaces
# ---------------------------------------------------------------------------

async def test_get_unknown_mission_raises() -> None:
    engine, _ = await _make_engine()
    with pytest.raises(MissionNotFoundError):
        await engine.get_mission("does-not-exist")


async def test_start_already_started_raises() -> None:
    engine, _ = await _make_engine()
    run = await engine.create_mission(_spec())
    await engine.start_mission(run.id)  # now COMPLETED
    with pytest.raises(InvalidStateTransitionError):
        await engine.start_mission(run.id)


# ---------------------------------------------------------------------------
# Multi-agent: engine resolves agents + checkpoints for eval tasks
# ---------------------------------------------------------------------------

class _CapturingRunner(Runner):
    """Captures the TaskContext it receives so tests can inspect it."""

    def __init__(self) -> None:
        self.contexts: list[TaskContext] = []

    @property
    def name(self) -> str:
        return "capturing"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.TRAINING, TaskKind.EVALUATION}

    @property
    def supported_types(self) -> set[str]:
        return {WILDCARD_TYPE}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        self.contexts.append(context)
        await context.emit_progress("done")
        if context.task.spec.kind == "training":
            return {
                "checkpoint_path": "/tmp/trained-checkpoint",
                "agent_id": context.task.spec.agent_id,  # type: ignore[union-attr]
            }
        return {"success_rate": 0.5, "performance_score": 0.5}


async def test_eval_context_receives_all_agents_and_checkpoints() -> None:
    """When the engine runs an eval task on a multi-agent robot, the
    TaskContext must carry all agents and their checkpoint mappings."""
    spec = Mission(
        metadata=MissionMetadata(name="msn-multi"),
        objective="objective",
        acceptance_criteria="acceptance",
        robot=RobotSpec(
            embodiment="franka_panda",
            agents=[
                AgentSpec(
                    id="pilot",
                    role=AgentRole.PILOT,
                    model=HFModelRef(base="openvla/openvla-7b"),
                ),
                AgentSpec(
                    id="task-planner",
                    role=AgentRole.SPECIALIST,
                    model=HFModelRef(
                        base="google/gemma-3-4b-it", quantization="int4"
                    ),
                ),
            ],
        ),
        tasks=[
            TrainingTask(
                name="train-pilot",
                training_type=TrainingType.DEMONSTRATION,
                agent_id="pilot",
            ),
            EvaluationTask(
                name="eval-lift",
                evaluation_type=EvaluationType.ROBOSUITE,
                benchmark_name="Lift",
                num_episodes=2,
            ),
        ],
    )

    capturing = _CapturingRunner()
    persistence = InMemoryPersistence()
    runners = RunnerRegistry()
    runners.register(capturing)
    publisher = CapturingPublisher()
    engine = MissionEngine(
        persistence=persistence, runners=runners, event_publisher=publisher
    )
    await engine.initialize()

    run = await engine.create_mission(spec)
    final = await engine.start_mission(run.id)
    assert final.status == MissionStatus.COMPLETED

    # The capturing runner saw two tasks: training + eval
    assert len(capturing.contexts) == 2

    # Training context: agent set, agents/agent_checkpoints empty
    train_ctx = capturing.contexts[0]
    assert train_ctx.agent is not None
    assert train_ctx.agent.id == "pilot"

    # Eval context: agents populated with both agents, checkpoints resolved
    eval_ctx = capturing.contexts[1]
    assert len(eval_ctx.agents) == 2
    agent_ids = [a.id for a in eval_ctx.agents]
    assert "pilot" in agent_ids
    assert "task-planner" in agent_ids

    # Pilot was trained -> has a checkpoint
    assert eval_ctx.agent_checkpoints["pilot"] == "/tmp/trained-checkpoint"
    # Specialist was never trained -> None (uses base model)
    assert eval_ctx.agent_checkpoints["task-planner"] is None
