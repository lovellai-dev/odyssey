"""Integration tests for `odyssey init`."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from odyssey.cli.main import cli
from odyssey.spec.loader import load_mission


def test_init_openvla_writes_valid_mission(tmp_path: Path) -> None:
    target = tmp_path / "my-mission"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", str(target), "--template", "openvla", "--name", "my-mission", "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert "Created" in result.output

    mission_path = target / "mission.yaml"
    assert mission_path.is_file()

    spec = load_mission(mission_path)
    assert spec.metadata.name == "my-mission"
    assert any(t.kind == "training" for t in spec.tasks)
    assert any(t.kind == "evaluation" for t in spec.tasks)


def test_init_cpu_mock_writes_valid_mission(tmp_path: Path) -> None:
    target = tmp_path / "smoke"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", str(target), "--template", "cpu_mock", "--name", "smoke-test", "--yes"],
    )
    assert result.exit_code == 0, result.output

    spec = load_mission(target / "mission.yaml")
    assert spec.metadata.name == "smoke-test"
    # cpu_mock template uses `from_task` for the eval — sanity-check it
    # parses through the cross-task ref validator.
    eval_tasks = [t for t in spec.tasks if t.kind == "evaluation"]
    assert len(eval_tasks) == 1


def test_init_defaults_template_under_yes(tmp_path: Path) -> None:
    target = tmp_path / "no-template-flag"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", str(target), "--name", "default-template", "--yes"],
    )
    assert result.exit_code == 0, result.output
    assert "template     : openvla" in result.output


def test_init_derives_name_from_directory_basename(tmp_path: Path) -> None:
    target = tmp_path / "derived-name"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", str(target), "--template", "cpu_mock", "--yes"],
    )
    assert result.exit_code == 0, result.output
    spec = load_mission(target / "mission.yaml")
    assert spec.metadata.name == "derived-name"


def test_init_slugifies_directory_basename(tmp_path: Path) -> None:
    target = tmp_path / "Has Spaces and CAPS"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", str(target), "--template", "cpu_mock", "--yes"],
    )
    assert result.exit_code == 0, result.output
    spec = load_mission(target / "mission.yaml")
    assert spec.metadata.name == "has-spaces-and-caps"


def test_init_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    target = tmp_path / "exists"
    target.mkdir()
    (target / "mission.yaml").write_text("pre-existing", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", str(target), "--template", "cpu_mock", "--name", "ok", "--yes"],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output
    # Untouched.
    assert (target / "mission.yaml").read_text(encoding="utf-8") == "pre-existing"


def test_init_overwrites_with_force(tmp_path: Path) -> None:
    target = tmp_path / "exists"
    target.mkdir()
    (target / "mission.yaml").write_text("pre-existing", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "init",
            str(target),
            "--template",
            "cpu_mock",
            "--name",
            "fresh",
            "--yes",
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    spec = load_mission(target / "mission.yaml")
    assert spec.metadata.name == "fresh"


def test_init_rejects_invalid_explicit_name_under_yes(tmp_path: Path) -> None:
    target = tmp_path / "bad-name"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "init",
            str(target),
            "--template",
            "cpu_mock",
            "--name",
            "Bad_Name_With_Underscores",
            "--yes",
        ],
    )
    # The CLI itself accepts the string; validation kicks in when the
    # rendered YAML is loaded, which triggers the rollback branch.
    assert result.exit_code != 0
    assert not (target / "mission.yaml").exists()


def test_init_fails_under_yes_when_directory_has_no_valid_slug(tmp_path: Path) -> None:
    target = tmp_path / "_"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", str(target), "--template", "cpu_mock", "--yes"],
    )
    assert result.exit_code != 0
    assert "valid mission name" in result.output
