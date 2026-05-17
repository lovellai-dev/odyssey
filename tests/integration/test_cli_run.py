"""Integration test for `odyssey run`.

End-to-end: invoke the CLI against the shipped example with
``--use-mock-runner``, point persistence + working_dir at a tmp_path, and
verify the mission lands in COMPLETED with the right shape in SQLite.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from click.testing import CliRunner

from odyssey.cli.main import cli
from odyssey.engine.lifecycle import MissionStatus
from odyssey.persistence import SqlitePersistence

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MISSION = REPO_ROOT / "examples" / "quickstart-openvla" / "mission.yaml"


def test_run_example_with_mock_runner_succeeds(tmp_path: Path) -> None:
    db_path = tmp_path / "missions.db"
    work_dir = tmp_path / "runs"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            str(EXAMPLE_MISSION),
            "--use-mock-runner",
            "--db",
            str(db_path),
            "--working-dir",
            str(work_dir),
        ],
    )
    # Print on failure for debuggability — pytest captures unless -s.
    assert result.exit_code == 0, f"CLI failed: {result.output}\n{result.exception}"
    assert "COMPLETED" in result.output
    assert db_path.exists()

    # The DB should hold exactly one mission in COMPLETED.
    persistence = SqlitePersistence(str(db_path))
    asyncio.run(persistence.initialize())
    missions = asyncio.run(persistence.list_missions())
    assert len(missions) == 1
    assert missions[0].status == MissionStatus.COMPLETED


def test_run_invalid_yaml_exits_nonzero(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: a [valid mission\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["run", str(bad), "--use-mock-runner"])
    assert result.exit_code == 1
    assert "INVALID" in result.output
