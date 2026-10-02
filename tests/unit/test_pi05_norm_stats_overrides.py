"""Norm stats must be computed for the dataset training actually reads.

openpi's ``train.py`` applies the mission's tyro overrides (``--data.repo-id``…)
but ``compute_norm_stats.py`` only takes ``--config-name``. Run directly, the
norm-stats step computes statistics for the registered config's DEFAULT
dataset. These tests reproduce that with a fake ``openpi`` package (same
config-CLI shape, same output layout as upstream) and show that the bootstrap
makes both steps see the overridden dataset. No real openpi or GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from odyssey.runners.models import openpi_bootstrap
from odyssey.runners.models.pi05_train import (
    _OPENPI_BOOTSTRAP,
    build_pi05_norm_stats_argv,
    build_pi05_norm_stats_launch,
    build_pi05_train_argv,
)
from odyssey.spec import TrainingTask, TrainingType

_CFG = "my_cfg"

# Mirrors upstream openpi.training.config: a frozen TrainConfig in
# _CONFIGS_DICT, get_config(name), and cli() = overridable_config_cli
# (positional config name, then --dotted.overrides; --exp-name is required).
_FAKE_CONFIG = textwrap.dedent(
    """
    import dataclasses, sys

    @dataclasses.dataclass(frozen=True)
    class Data:
        repo_id: str

    @dataclasses.dataclass(frozen=True)
    class TrainConfig:
        name: str
        data: Data
        exp_name: str = ""

    _CONFIGS_DICT = {"my_cfg": TrainConfig(name="my_cfg", data=Data("default_dataset"))}

    def get_config(name):
        return _CONFIGS_DICT[name]

    def cli():
        name, *rest = sys.argv[1:]
        cfg = _CONFIGS_DICT[name]
        it = iter(rest)
        for flag in it:
            value = next(it)
            if flag == "--data.repo-id":
                cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=value))
            elif flag == "--exp-name":
                cfg = dataclasses.replace(cfg, exp_name=value)
        if not cfg.exp_name:
            raise SystemExit("--exp-name is required")
        return cfg
    """
)

# Mirrors upstream scripts/compute_norm_stats.py: tyro.cli(main) with only
# --config-name; writes assets/<config.name>/<repo_id>/norm_stats.json.
_FAKE_NORM_STATS = textwrap.dedent(
    """
    import argparse, json, pathlib
    from openpi.training import config as _config

    p = argparse.ArgumentParser()
    p.add_argument("--config-name", required=True)
    args = p.parse_args()
    cfg = _config.get_config(args.config_name)
    out = pathlib.Path("assets") / cfg.name / cfg.data.repo_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "norm_stats.json").write_text(json.dumps({"repo_id": cfg.data.repo_id}))
    """
)


@pytest.fixture
def fake_openpi(tmp_path: Path) -> dict[str, Path]:
    pkg = tmp_path / "site" / "openpi"
    (pkg / "training").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "training" / "__init__.py").write_text("")
    (pkg / "training" / "config.py").write_text(_FAKE_CONFIG)
    script = tmp_path / "openpi_repo" / "scripts" / "compute_norm_stats.py"
    script.parent.mkdir(parents=True)
    script.write_text(_FAKE_NORM_STATS)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return {"site": tmp_path / "site", "script": script, "run": run_dir}


def _task(config: dict[str, Any]) -> TrainingTask:
    return TrainingTask(
        name="finetune",
        training_type=TrainingType.DEMONSTRATION,
        agent_id="pilot",
        config={"runner": "pi05", "config_name": _CFG, **config},
    )


def _run(script: str, argv: list[str], env_site: Path, cwd: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(env_site)}
    subprocess.run([sys.executable, script, *argv], cwd=cwd, env=env, check=True,
                   capture_output=True, text=True)


def _stats_written(run_dir: Path) -> list[str]:
    return sorted(
        json.loads(p.read_text())["repo_id"]
        for p in (run_dir / "assets").rglob("norm_stats.json")
    )


# ---------------------------------------------------------------------------
# The defect, reproduced
# ---------------------------------------------------------------------------

def test_direct_norm_stats_ignores_repo_id_override(fake_openpi: dict[str, Path]) -> None:
    """What develop does: stats land under the DEFAULT dataset, not the mission's."""
    task = _task({"data": {"repo_id": "my_dataset"}})
    _run(str(fake_openpi["script"]), build_pi05_norm_stats_argv(task=task),
         fake_openpi["site"], fake_openpi["run"])
    assert _stats_written(fake_openpi["run"]) == ["default_dataset"]
    # train.py would look for assets/my_cfg/my_dataset/norm_stats.json: missing.
    assert not (fake_openpi["run"] / "assets" / _CFG / "my_dataset").exists()


# ---------------------------------------------------------------------------
# The fix
# ---------------------------------------------------------------------------

def test_bootstrap_computes_stats_for_overridden_dataset(
    fake_openpi: dict[str, Path],
) -> None:
    task = _task({"data": {"repo_id": "my_dataset"}})
    script, argv = build_pi05_norm_stats_launch(
        task=task, exp_name="exp", norm_stats_script=str(fake_openpi["script"])
    )
    assert script == _OPENPI_BOOTSTRAP
    _run(script, argv, fake_openpi["site"], fake_openpi["run"])
    assert _stats_written(fake_openpi["run"]) == ["my_dataset"]
    # Exactly where train.py (config name unchanged + same repo_id) reads them.
    assert (fake_openpi["run"] / "assets" / _CFG / "my_dataset" / "norm_stats.json").is_file()


def test_without_overrides_runs_norm_stats_directly(fake_openpi: dict[str, Path]) -> None:
    """No overrides: unchanged behaviour, no bootstrap in the loop."""
    task = _task({})
    script, argv = build_pi05_norm_stats_launch(
        task=task, exp_name="exp", norm_stats_script=str(fake_openpi["script"])
    )
    assert script == str(fake_openpi["script"])
    assert argv == ["--config-name", _CFG]
    _run(script, argv, fake_openpi["site"], fake_openpi["run"])
    assert _stats_written(fake_openpi["run"]) == ["default_dataset"]


def test_norm_stats_and_train_receive_the_same_overrides() -> None:
    """The bootstrap gets exactly train.py's overrides and exp-name."""
    task = _task({"data": {"repo_id": "my_dataset"}, "num_train_steps": 100})
    train_argv = build_pi05_train_argv(task=task, exp_name="exp")
    _, norm_argv = build_pi05_norm_stats_launch(
        task=task, exp_name="exp", norm_stats_script="/x/compute_norm_stats.py"
    )
    norm_overrides = norm_argv[norm_argv.index("--") + 1 :]
    # train argv: <config_name> --exp-name exp --overwrite <overrides…>
    train_overrides = train_argv[4:]
    assert norm_overrides[: len(train_overrides)] == train_overrides
    assert norm_overrides[len(train_overrides):] == ["--exp-name", "exp"]


def test_launch_requires_config_name() -> None:
    task = TrainingTask(
        name="finetune",
        training_type=TrainingType.DEMONSTRATION,
        agent_id="pilot",
        config={"data": {"repo_id": "my_dataset"}},
    )
    with pytest.raises(RuntimeError, match="config_name"):
        build_pi05_norm_stats_launch(task=task, exp_name="e", norm_stats_script="/x.py")


# ---------------------------------------------------------------------------
# Bootstrap argv contract
# ---------------------------------------------------------------------------

def test_bootstrap_parse_argv() -> None:
    assert openpi_bootstrap.parse_argv(["t.py", "cfg", "--", "--a", "1"]) == (
        "t.py",
        "cfg",
        ["--a", "1"],
    )


@pytest.mark.parametrize("argv", [["t.py", "cfg"], ["t.py", "--", "x"], ["--", "x"]])
def test_bootstrap_rejects_malformed_argv(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        openpi_bootstrap.parse_argv(argv)


def test_bootstrap_key_is_private() -> None:
    assert openpi_bootstrap.registry_key(_CFG) != _CFG
