"""Tests for SqlitePersistence.

Same surface as the InMemory tests so anyone swapping implementations
sees identical behavior. Each test gets its own tmp db so they don't
share state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from odyssey.engine.errors import MissionNotFoundError, TaskNotFoundError
from odyssey.engine.lifecycle import MissionStatus, TaskStatus
from odyssey.engine.records import MissionRun
from odyssey.persistence import SqlitePersistence
from odyssey.spec import (
    EvaluationTask,
    EvaluationType,
    HFModelRef,
    Mission,
    MissionMetadata,
    RobotSpec,
    TrainingTask,
    TrainingType,
)


def _spec(name: str = "msn-sql") -> Mission:
    return Mission(
        metadata=MissionMetadata(name=name),
        objective="objective",
        acceptance_criteria="acceptance",
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
                model=HFModelRef(base="openvla/openvla-7b"),
                target_agent_id="pilot",
            ),
        ],
    )


async def _make(tmp_path: Path) -> SqlitePersistence:
    p = SqlitePersistence(str(tmp_path / "missions.db"))
    await p.initialize()
    return p


# ---------------------------------------------------------------------------
# Initialize + save/get roundtrip
# ---------------------------------------------------------------------------

async def test_initialize_creates_db_file(tmp_path: Path) -> None:
    p = SqlitePersistence(str(tmp_path / "nested" / "missions.db"))
    await p.initialize()
    assert (tmp_path / "nested" / "missions.db").exists()


async def test_save_and_get_roundtrip(tmp_path: Path) -> None:
    p = await _make(tmp_path)
    run = MissionRun.from_spec(_spec())
    await p.save_mission(run)

    fetched = await p.get_mission(run.id)
    assert fetched is not None
    assert fetched.id == run.id
    assert fetched.spec.metadata.name == "msn-sql"
    assert fetched.status == MissionStatus.DRAFT
    assert len(fetched.tasks) == 2


async def test_get_unknown_returns_none(tmp_path: Path) -> None:
    p = await _make(tmp_path)
    assert await p.get_mission("does-not-exist") is None


async def test_save_is_upsert(tmp_path: Path) -> None:
    p = await _make(tmp_path)
    run = MissionRun.from_spec(_spec())
    await p.save_mission(run)
    run.status = MissionStatus.ACTIVE
    await p.save_mission(run)

    fetched = await p.get_mission(run.id)
    assert fetched is not None
    assert fetched.status == MissionStatus.ACTIVE


# ---------------------------------------------------------------------------
# list_missions
# ---------------------------------------------------------------------------

async def test_list_missions_returns_all_by_default(tmp_path: Path) -> None:
    p = await _make(tmp_path)
    a = MissionRun.from_spec(_spec("aaa"))
    b = MissionRun.from_spec(_spec("bbb"))
    await p.save_mission(a)
    await p.save_mission(b)

    rows = await p.list_missions()
    assert {r.id for r in rows} == {a.id, b.id}


async def test_list_missions_filters_by_status(tmp_path: Path) -> None:
    p = await _make(tmp_path)
    draft = MissionRun.from_spec(_spec("draft-one"))
    completed = MissionRun.from_spec(_spec("done-one"))
    completed.status = MissionStatus.COMPLETED
    await p.save_mission(draft)
    await p.save_mission(completed)

    rows = await p.list_missions(status="COMPLETED")
    assert [r.id for r in rows] == [completed.id]


async def test_list_missions_honors_limit_and_offset(tmp_path: Path) -> None:
    p = await _make(tmp_path)
    runs = [MissionRun.from_spec(_spec(f"msn-{i}")) for i in range(5)]
    for r in runs:
        await p.save_mission(r)

    page1 = await p.list_missions(limit=2, offset=0)
    page2 = await p.list_missions(limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    assert {r.id for r in page1}.isdisjoint({r.id for r in page2})


# ---------------------------------------------------------------------------
# delete_mission
# ---------------------------------------------------------------------------

async def test_delete_existing_returns_true(tmp_path: Path) -> None:
    p = await _make(tmp_path)
    run = MissionRun.from_spec(_spec())
    await p.save_mission(run)
    assert await p.delete_mission(run.id) is True
    assert await p.get_mission(run.id) is None


async def test_delete_unknown_returns_false(tmp_path: Path) -> None:
    p = await _make(tmp_path)
    assert await p.delete_mission("never-existed") is False


# ---------------------------------------------------------------------------
# update_task
# ---------------------------------------------------------------------------

async def test_update_task_persists_field_changes(tmp_path: Path) -> None:
    p = await _make(tmp_path)
    run = MissionRun.from_spec(_spec())
    await p.save_mission(run)

    train_task_id = run.tasks[0].id
    updated = await p.update_task(
        run.id,
        train_task_id,
        status=TaskStatus.COMPLETED,
        error_code=None,
        result_summary={"performance_score": 0.91},
    )
    assert updated.status == TaskStatus.COMPLETED
    assert updated.result_summary == {"performance_score": 0.91}

    fetched = await p.get_mission(run.id)
    assert fetched is not None
    assert fetched.tasks[0].status == TaskStatus.COMPLETED
    assert fetched.tasks[0].result_summary == {"performance_score": 0.91}


async def test_update_task_on_unknown_mission_raises(tmp_path: Path) -> None:
    p = await _make(tmp_path)
    with pytest.raises(MissionNotFoundError):
        await p.update_task("nope", "also-nope", status=TaskStatus.COMPLETED)


async def test_update_task_on_unknown_task_raises(tmp_path: Path) -> None:
    p = await _make(tmp_path)
    run = MissionRun.from_spec(_spec())
    await p.save_mission(run)
    with pytest.raises(TaskNotFoundError):
        await p.update_task(run.id, "no-such-task", status=TaskStatus.COMPLETED)


# ---------------------------------------------------------------------------
# Cross-implementation parity: SqlitePersistence drives the engine end-to-end
# ---------------------------------------------------------------------------

async def test_engine_runs_mission_against_sqlite(tmp_path: Path) -> None:
    """End-to-end: MissionEngine with SqlitePersistence + CPUMockRunner
    completes a mission and the terminal state is durable across reads."""
    from odyssey.engine import MissionEngine, MissionStatus
    from odyssey.runners import CPUMockRunner, RunnerRegistry
    from odyssey.telemetry import EventPublisher

    class _NullPublisher(EventPublisher):
        async def publish(self, event_type: str, payload: dict) -> None:
            return

    p = await _make(tmp_path)
    runners = RunnerRegistry()
    runners.register(CPUMockRunner())
    engine = MissionEngine(
        persistence=p, runners=runners, event_publisher=_NullPublisher()
    )
    await engine.initialize()

    run = await engine.create_mission(_spec("end-to-end"))
    final = await engine.start_mission(run.id)
    assert final.status == MissionStatus.COMPLETED

    # Persisted record reflects the terminal state without going through
    # the engine's in-memory copy.
    fresh = SqlitePersistence(p._db_path)  # type: ignore[attr-defined]
    await fresh.initialize()
    durable = await fresh.get_mission(run.id)
    assert durable is not None
    assert durable.status == MissionStatus.COMPLETED
