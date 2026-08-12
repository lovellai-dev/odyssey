"""Tests for the Cosmos 3 (cosmos-framework) SFT training runner's argv builders,
recipe resolution, env overlays, checkpoint resolution and stdout parser.

We don't invoke the real ``cosmos_framework.scripts.*`` entry points — those need
the cosmos-framework package, torch with CUDA, multi-GPU and a registered SFT
recipe TOML. The testable pieces are the pure functions that shape the three
subprocess argvs from a ``TrainingTask`` spec, the dataset/path env overlays, the
run-dir + checkpoint locators, and the parser that turns cosmos stdout into
progress events.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from odyssey.runners import Cosmos3Runner, RunnerRegistry
from odyssey.runners.models.cosmos3_train import (
    _dataset_env,
    _path_env,
    _resolve_base_model,
    _resolve_cosmos_recipe,
    _resolve_run_dir,
    _resolve_trained_checkpoint,
    build_cosmos3_dcp_argv,
    build_cosmos3_export_argv,
    build_cosmos3_train_argv,
    parse_cosmos3_train_line,
)
from odyssey.spec import DatasetRef, DatasetSource, TrainingTask, TrainingType


def _task(**overrides: Any) -> TrainingTask:
    fields: dict[str, Any] = {
        "name": "finetune-cosmos3",
        "training_type": TrainingType.DEMONSTRATION,
        "agent_id": "pilot",
    }
    fields.update(overrides)
    return TrainingTask(**fields)


def _repo_with_recipe(tmp_path: Path, name: str = "action_policy_libero_nano") -> Path:
    """Materialize a fake cosmos-framework checkout holding one recipe TOML."""
    recipe_dir = tmp_path / "examples" / "toml" / "sft_config"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / f"{name}.toml").write_text("# recipe\n")
    return tmp_path


# ---------------------------------------------------------------------------
# recipe resolution
# ---------------------------------------------------------------------------

def test_resolve_recipe_against_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo_with_recipe(tmp_path)
    monkeypatch.setenv("COSMOS_FRAMEWORK_REPO_PATH", str(repo))
    resolved = _resolve_cosmos_recipe("action_policy_libero_nano")
    assert resolved == str(
        repo / "examples" / "toml" / "sft_config" / "action_policy_libero_nano.toml"
    )


def test_resolve_recipe_accepts_toml_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo_with_recipe(tmp_path)
    monkeypatch.setenv("COSMOS_FRAMEWORK_REPO_PATH", str(repo))
    # Passing the name already suffixed must not double the extension.
    resolved = _resolve_cosmos_recipe("action_policy_libero_nano.toml")
    assert resolved.endswith("action_policy_libero_nano.toml")
    assert not resolved.endswith(".toml.toml")


def test_resolve_recipe_accepts_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = tmp_path / "custom.toml"
    recipe.write_text("# recipe\n")
    # An absolute path bypasses the repo-relative lookup entirely.
    monkeypatch.setenv("COSMOS_FRAMEWORK_REPO_PATH", "/nonexistent")
    assert _resolve_cosmos_recipe(str(recipe)) == str(recipe)


def test_resolve_recipe_missing_raises_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COSMOS_FRAMEWORK_REPO_PATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="COSMOS_FRAMEWORK_REPO_PATH"):
        _resolve_cosmos_recipe("action_policy_libero_nano")


# ---------------------------------------------------------------------------
# train argv builder
# ---------------------------------------------------------------------------

def test_train_argv_requires_config_name() -> None:
    with pytest.raises(RuntimeError, match="config_name"):
        build_cosmos3_train_argv(task=_task())


def test_train_argv_sft_toml_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo_with_recipe(tmp_path)
    monkeypatch.setenv("COSMOS_FRAMEWORK_REPO_PATH", str(repo))
    task = _task(config={"config_name": "action_policy_libero_nano"})
    argv = build_cosmos3_train_argv(task=task)
    assert argv[0].startswith("--sft-toml=")
    assert argv[0].endswith("action_policy_libero_nano.toml")


def test_train_argv_passthrough_is_kebab_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo_with_recipe(tmp_path)
    monkeypatch.setenv("COSMOS_FRAMEWORK_REPO_PATH", str(repo))
    task = _task(
        config={
            "config_name": "action_policy_libero_nano",
            "trainer": {"max_iter": 10},
        }
    )
    argv = build_cosmos3_train_argv(task=task)
    idx = argv.index("--trainer.max-iter")
    assert argv[idx + 1] == "10"


def test_train_argv_bool_true_is_bare_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo_with_recipe(tmp_path)
    monkeypatch.setenv("COSMOS_FRAMEWORK_REPO_PATH", str(repo))
    task = _task(config={"config_name": "action_policy_libero_nano", "wandb": True})
    argv = build_cosmos3_train_argv(task=task)
    assert "--wandb" in argv
    assert "True" not in argv


def test_train_argv_bool_false_is_no_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo_with_recipe(tmp_path)
    monkeypatch.setenv("COSMOS_FRAMEWORK_REPO_PATH", str(repo))
    task = _task(config={"config_name": "action_policy_libero_nano", "wandb": False})
    argv = build_cosmos3_train_argv(task=task)
    assert "--no-wandb" in argv


def test_train_argv_excludes_control_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo_with_recipe(tmp_path)
    monkeypatch.setenv("COSMOS_FRAMEWORK_REPO_PATH", str(repo))
    task = _task(
        config={
            "config_name": "action_policy_libero_nano",
            "runner": "cosmos3",
            "base_model": "Cosmos3-Nano",
            "convert_dcp": False,
            "export": False,
            "nproc_per_node": 8,
            "wan_vae_path": "/x/vae.pth",
            "filter_dir": "/x/filters",
            "dataset_path": "/x/data",
        }
    )
    argv = build_cosmos3_train_argv(task=task)
    for control in (
        "--runner",
        "--base-model",
        "--convert-dcp",
        "--no-convert-dcp",
        "--export",
        "--no-export",
        "--nproc-per-node",
        "--wan-vae-path",
        "--filter-dir",
        "--dataset-path",
        "--config-name",
    ):
        assert control not in argv


# ---------------------------------------------------------------------------
# dcp + export argv builders
# ---------------------------------------------------------------------------

def test_dcp_argv_shape() -> None:
    argv = build_cosmos3_dcp_argv(
        base_model="Cosmos3-Nano", base_checkpoint_path="/out/dcp"
    )
    assert argv == ["--checkpoint-path", "Cosmos3-Nano", "-o", "/out/dcp"]


def test_export_argv_shape() -> None:
    argv = build_cosmos3_export_argv(
        checkpoint_path="/run/checkpoints/iter_1000",
        config_file="/run/config.yaml",
        out_dir="/run/model",
    )
    assert argv == [
        "--checkpoint-path",
        "/run/checkpoints/iter_1000",
        "--config-file",
        "/run/config.yaml",
        "-o",
        "/run/model",
    ]


# ---------------------------------------------------------------------------
# base model resolution
# ---------------------------------------------------------------------------

def test_base_model_config_wins() -> None:
    task = _task(config={"config_name": "c", "base_model": "Cosmos3-Edge"})
    assert _resolve_base_model(task, "Cosmos3-Nano") == "Cosmos3-Edge"


def test_base_model_falls_back_to_agent_base() -> None:
    task = _task(config={"config_name": "c"})
    assert _resolve_base_model(task, "nvidia/Cosmos3-Nano") == "nvidia/Cosmos3-Nano"


def test_base_model_defaults_when_unknown() -> None:
    assert _resolve_base_model(_task(config={"config_name": "c"}), None) == "Cosmos3-Nano"


# ---------------------------------------------------------------------------
# dataset env overlay (the LeRobot footgun)
# ---------------------------------------------------------------------------

def test_dataset_env_maps_absolute_local_verbatim() -> None:
    # DATASET_PATH points at the PARENT of success/ — i.e. the named dataset dir
    # itself, NOT success/. The loader infers schema from this dir NAME, so it is
    # passed through unchanged (contrast pi05, which strips to the parent).
    task = _task(
        dataset=DatasetRef(
            source=DatasetSource.LOCAL,
            ref="/data/droid_plus_lerobot_640x360_20260412",
        )
    )
    env = _dataset_env(task)
    assert env["DATASET_PATH"] == "/data/droid_plus_lerobot_640x360_20260412"


def test_dataset_env_name_is_recipe_specific() -> None:
    # LIBERO recipe reads LIBERO_ROOT, not DATASET_PATH (verified on
    # cosmos-framework). dataset_env overrides the env VAR NAME; value unchanged.
    task = _task(
        config={"config_name": "action_policy_libero_10_nano", "dataset_env": "LIBERO_ROOT"},
        dataset=DatasetRef(
            source=DatasetSource.LOCAL, ref="/data/LIBERO_LeRobot_v3/libero_10"
        ),
    )
    env = _dataset_env(task)
    assert env == {"LIBERO_ROOT": "/data/LIBERO_LeRobot_v3/libero_10"}
    assert "DATASET_PATH" not in env


def test_dataset_env_defaults_to_dataset_path() -> None:
    # No dataset_env -> DROID default DATASET_PATH.
    task = _task(
        dataset=DatasetRef(source=DatasetSource.LOCAL, ref="/data/droid_lerobot")
    )
    assert _dataset_env(task) == {"DATASET_PATH": "/data/droid_lerobot"}


def test_dataset_env_explicit_config_wins() -> None:
    task = _task(
        config={"config_name": "c", "dataset_path": "/override/data"},
        dataset=DatasetRef(source=DatasetSource.LOCAL, ref="/data/ignored"),
    )
    assert _dataset_env(task)["DATASET_PATH"] == "/override/data"


def test_dataset_env_empty_for_hf_ref() -> None:
    task = _task(dataset=DatasetRef(source=DatasetSource.HUGGINGFACE, ref="nvidia/Cosmos3-DROID"))
    assert _dataset_env(task) == {}


def test_dataset_env_empty_when_no_dataset() -> None:
    assert _dataset_env(_task()) == {}


# ---------------------------------------------------------------------------
# path env bridging
# ---------------------------------------------------------------------------

def test_path_env_sets_required_and_output_root(tmp_path: Path) -> None:
    env = _path_env(_task(config={"config_name": "c"}), tmp_path, "/out/dcp")
    assert env["BASE_CHECKPOINT_PATH"] == "/out/dcp"
    assert env["IMAGINAIRE_OUTPUT_ROOT"] == str(tmp_path)
    assert env["NPROC_PER_NODE"] == "8"  # default


def test_path_env_optional_paths_only_when_configured(tmp_path: Path) -> None:
    # WAN_VAE_PATH / FILTER_DIR are the operator's to provide via shell unless the
    # mission pins them — never emitted empty.
    env = _path_env(_task(config={"config_name": "c"}), tmp_path, "/out/dcp")
    assert "WAN_VAE_PATH" not in env
    assert "FILTER_DIR" not in env

    task = _task(
        config={
            "config_name": "c",
            "nproc_per_node": 16,
            "wan_vae_path": "/x/Wan2.2_VAE.pth",
            "filter_dir": "/x/droid_filters",
        }
    )
    env2 = _path_env(task, tmp_path, "/out/dcp")
    assert env2["NPROC_PER_NODE"] == "16"
    assert env2["WAN_VAE_PATH"] == "/x/Wan2.2_VAE.pth"
    assert env2["FILTER_DIR"] == "/x/droid_filters"


# ---------------------------------------------------------------------------
# run-dir + trained-checkpoint resolution
# ---------------------------------------------------------------------------

def test_resolve_run_dir_picks_newest_config_yaml(tmp_path: Path) -> None:
    old = tmp_path / "run_old"
    new = tmp_path / "run_new"
    old.mkdir()
    new.mkdir()
    (old / "config.yaml").write_text("a: 1\n")
    (new / "config.yaml").write_text("a: 2\n")
    # Make `new` unambiguously newer regardless of write ordering.
    os.utime(old / "config.yaml", (1_000, 1_000))
    os.utime(new / "config.yaml", (2_000, 2_000))
    assert _resolve_run_dir(tmp_path) == new


def test_resolve_run_dir_none_when_empty(tmp_path: Path) -> None:
    assert _resolve_run_dir(tmp_path) is None


def test_resolve_trained_checkpoint_picks_highest_iter(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints"
    for step in ("iter_000001000", "iter_000005000", "iter_000002000"):
        (ckpt / step).mkdir(parents=True)
    resolved = _resolve_trained_checkpoint(tmp_path)
    assert resolved is not None
    assert resolved.name == "iter_000005000"


def test_resolve_trained_checkpoint_falls_back_to_root(tmp_path: Path) -> None:
    # A checkpoints dir with no numbered subdir still resolves to the root, not None.
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    assert _resolve_trained_checkpoint(tmp_path) == ckpt


def test_resolve_trained_checkpoint_none_when_missing(tmp_path: Path) -> None:
    assert _resolve_trained_checkpoint(tmp_path) is None


# ---------------------------------------------------------------------------
# registry dispatch
# ---------------------------------------------------------------------------

def test_runner_selected_by_config_override() -> None:
    registry = RunnerRegistry()
    registry.register(Cosmos3Runner())
    task = _task(config={"runner": "cosmos3", "config_name": "action_policy_libero_nano"})
    # The engine forwards config['runner'] as the select override.
    assert isinstance(registry.select(task, override="cosmos3"), Cosmos3Runner)


def test_runner_name_and_kind() -> None:
    runner = Cosmos3Runner()
    assert runner.name == "cosmos3"
    from odyssey.spec.tasks import TaskKind

    assert runner.supported_kinds == {TaskKind.TRAINING}


# ---------------------------------------------------------------------------
# stdout parser
# ---------------------------------------------------------------------------

def test_parse_loss_and_iter_line() -> None:
    event = parse_cosmos3_train_line("iter=1000 loss=0.234 grad_norm=1.2")
    assert event is not None
    assert event["stage"] == "executing"
    assert event["step"] == "training_step"
    assert event["step_index"] == 1000
    assert "loss=0.234" in event["step_label"]


def test_parse_tqdm_progress_line() -> None:
    event = parse_cosmos3_train_line(" 10%|█  | 100/1000 [00:42<06:18,  2.38it/s]")
    assert event is not None
    assert event["step_index"] == 100
    assert event["step_total"] == 1000


def test_parse_checkpoint_save_line() -> None:
    event = parse_cosmos3_train_line(
        "Saving checkpoint to outputs/train/exp/checkpoints/iter_000001000"
    )
    assert event is not None
    assert event["stage"] == "checkpoint_saving"


def test_parse_dcp_line() -> None:
    event = parse_cosmos3_train_line("Converting model to DCP format ...")
    assert event is not None
    assert event["stage"] == "dataset_loading"
    assert event["step"] == "convert_dcp"


def test_parse_dataset_loading_line() -> None:
    event = parse_cosmos3_train_line("Loading LeRobot dataset droid_plus_lerobot_640x360")
    assert event is not None
    assert event["stage"] == "dataset_loading"


def test_parse_unrelated_line_returns_none() -> None:
    assert parse_cosmos3_train_line("nothing interesting here") is None
