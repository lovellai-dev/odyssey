"""Integration tests for `odyssey list` and `odyssey status`."""

from __future__ import annotations

import asyncio
from pathlib import Path

from click.testing import CliRunner

from odyssey.cli.main import cli
from odyssey.engine.lifecycle import MissionStatus
from odyssey.engine.records import MissionRun
from odyssey.persistence import SqlitePersistence
from odyssey.spec import (
    AgentRole,
    AgentSpec,
    EvaluationTask,
    EvaluationType,
    HFModelRef,
    Mission,
    MissionMetadata,
    RobotSpec,
    TrainingTask,
    TrainingType,
)


def _spec(name: str) -> Mission:
    return Mission(
        metadata=MissionMetadata(name=name),
        objective="o",
        acceptance_criteria="a",
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
            ),
        ],
    )


def _seed_db(db_path: Path, *names: str) -> list[str]:
    """Sync helper — set up a tmp DB with N missions, return their ids."""
    p = SqlitePersistence(str(db_path))

    async def _do() -> list[str]:
        await p.initialize()
        ids: list[str] = []
        for name in names:
            run = MissionRun.from_spec(_spec(name))
            await p.save_mission(run)
            ids.append(run.id)
        return ids

    return asyncio.run(_do())


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def test_list_empty_db_shows_no_missions(tmp_path: Path) -> None:
    db = tmp_path / "missions.db"
    _seed_db(db)  # initialize but no rows
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--db", str(db)])
    assert result.exit_code == 0
    assert "(no missions)" in result.output


def test_list_shows_all_missions(tmp_path: Path) -> None:
    db = tmp_path / "missions.db"
    _seed_db(db, "alpha", "beta", "gamma")
    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--db", str(db)])
    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "beta" in result.output
    assert "gamma" in result.output
    assert "DRAFT" in result.output


def test_list_filters_by_status(tmp_path: Path) -> None:
    db = tmp_path / "missions.db"
    ids = _seed_db(db, "draft-one", "to-complete")

    async def _complete_second() -> None:
        p = SqlitePersistence(str(db))
        await p.initialize()
        run = await p.get_mission(ids[1])
        assert run is not None
        run.status = MissionStatus.COMPLETED
        await p.save_mission(run)
    asyncio.run(_complete_second())

    runner = CliRunner()
    result = runner.invoke(cli, ["list", "--db", str(db), "--status", "COMPLETED"])
    assert result.exit_code == 0
    assert "to-complete" in result.output
    assert "draft-one" not in result.output


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def test_status_shows_mission_detail(tmp_path: Path) -> None:
    db = tmp_path / "missions.db"
    ids = _seed_db(db, "the-one")
    runner = CliRunner()
    result = runner.invoke(cli, ["status", ids[0], "--db", str(db)])
    assert result.exit_code == 0
    assert "the-one" in result.output
    assert "DRAFT" in result.output
    assert "train" in result.output
    assert "eval" in result.output


def test_status_accepts_id_prefix(tmp_path: Path) -> None:
    db = tmp_path / "missions.db"
    ids = _seed_db(db, "the-one")
    runner = CliRunner()
    result = runner.invoke(cli, ["status", ids[0][:8], "--db", str(db)])
    assert result.exit_code == 0
    assert "the-one" in result.output


def test_status_unknown_id_exits_nonzero(tmp_path: Path) -> None:
    db = tmp_path / "missions.db"
    _seed_db(db)
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "no-such-id", "--db", str(db)])
    assert result.exit_code == 1
    assert "NOT FOUND" in result.output
