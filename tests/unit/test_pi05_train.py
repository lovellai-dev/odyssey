"""Tests for the π0.5 (openpi) training runner's argv builders, env overlay,
checkpoint resolution and stdout parser.

We don't invoke the real openpi ``scripts/train.py`` — that needs the openpi
package, JAX/torch with CUDA, a GPU and a registered ``TrainConfig``. The
testable pieces are the pure functions that shape the two subprocess argvs from
a ``TrainingTask`` spec, the LeRobot env overlay, the checkpoint locator, and the
parser that turns openpi stdout into progress events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from odyssey.runners.models.pi05_train import (
    _lerobot_env_for_dataset,
    _resolve_output_checkpoint,
    build_pi05_norm_stats_argv,
    build_pi05_train_argv,
    parse_pi05_train_line,
)
from odyssey.spec import DatasetRef, DatasetSource, TrainingTask, TrainingType


def _task(**overrides: Any) -> TrainingTask:
    fields: dict[str, Any] = {
        "name": "finetune-pi05",
        "training_type": TrainingType.DEMONSTRATION,
        "agent_id": "pilot",
    }
    fields.update(overrides)
    return TrainingTask(**fields)


# ---------------------------------------------------------------------------
# train argv builder
# ---------------------------------------------------------------------------

def test_train_argv_requires_config_name() -> None:
    with pytest.raises(RuntimeError, match="config_name"):
        build_pi05_train_argv(task=_task(), exp_name="exp")


def test_train_argv_positional_config_name_first() -> None:
    task = _task(config={"config_name": "pi05_libero"})
    argv = build_pi05_train_argv(task=task, exp_name="exp")
    assert argv[0] == "pi05_libero"


def test_train_argv_includes_exp_name() -> None:
    task = _task(config={"config_name": "pi05_libero"})
    argv = build_pi05_train_argv(task=task, exp_name="my-exp")
    idx = argv.index("--exp-name")
    assert argv[idx + 1] == "my-exp"


def test_train_argv_defaults_to_overwrite() -> None:
    task = _task(config={"config_name": "pi05_libero"})
    argv = build_pi05_train_argv(task=task, exp_name="exp")
    assert "--overwrite" in argv


def test_train_argv_overwrite_false_omits_flag() -> None:
    task = _task(config={"config_name": "pi05_libero", "overwrite": False})
    argv = build_pi05_train_argv(task=task, exp_name="exp")
    assert "--overwrite" not in argv


def test_train_argv_resume_replaces_overwrite() -> None:
    task = _task(config={"config_name": "pi05_libero", "resume": True})
    argv = build_pi05_train_argv(task=task, exp_name="exp")
    assert "--resume" in argv
    assert "--overwrite" not in argv


def test_train_argv_passthrough_is_kebab_case() -> None:
    task = _task(config={"config_name": "pi05_libero", "num_train_steps": 30000})
    argv = build_pi05_train_argv(task=task, exp_name="exp")
    idx = argv.index("--num-train-steps")
    assert argv[idx + 1] == "30000"


def test_train_argv_nested_config_becomes_dotted_flag() -> None:
    # tyro overrides address nested dataclass fields via dots; underscores in a
    # leaf still become dashes (data.repo_id -> --data.repo-id).
    task = _task(
        config={"config_name": "pi05_ur10e", "data": {"repo_id": "ur10e_partial_cond_aug"}}
    )
    argv = build_pi05_train_argv(task=task, exp_name="exp")
    idx = argv.index("--data.repo-id")
    assert argv[idx + 1] == "ur10e_partial_cond_aug"


def test_train_argv_bool_true_is_bare_flag() -> None:
    task = _task(config={"config_name": "c", "wandb_enabled": True})
    argv = build_pi05_train_argv(task=task, exp_name="exp")
    assert "--wandb-enabled" in argv
    # A bare flag carries no value token after it.
    assert "True" not in argv


def test_train_argv_bool_false_is_no_flag() -> None:
    task = _task(config={"config_name": "c", "wandb_enabled": False})
    argv = build_pi05_train_argv(task=task, exp_name="exp")
    assert "--no-wandb-enabled" in argv


def test_train_argv_excludes_control_keys() -> None:
    task = _task(
        config={
            "config_name": "c",
            "runner": "pi05",
            "exp_name": "ignored",
            "compute_norm_stats": False,
        }
    )
    argv = build_pi05_train_argv(task=task, exp_name="exp")
    assert "--runner" not in argv
    assert "--config-name" not in argv
    assert "--compute-norm-stats" not in argv
    # exp_name is supplied via the dedicated flag, not passed through twice.
    assert argv.count("--exp-name") == 1


# ---------------------------------------------------------------------------
# norm-stats argv builder
# ---------------------------------------------------------------------------

def test_norm_stats_argv_is_config_name_only() -> None:
    task = _task(config={"config_name": "pi05_libero", "num_train_steps": 10})
    assert build_pi05_norm_stats_argv(task=task) == ["pi05_libero"]


def test_norm_stats_argv_requires_config_name() -> None:
    with pytest.raises(RuntimeError, match="config_name"):
        build_pi05_norm_stats_argv(task=_task())


# ---------------------------------------------------------------------------
# LeRobot env overlay
# ---------------------------------------------------------------------------

def test_lerobot_env_points_at_parent_for_absolute_local() -> None:
    task = _task(
        dataset=DatasetRef(
            source=DatasetSource.LOCAL, ref="/data/ur10e_drugsort_v0/ur10e_partial_cond_aug"
        ),
    )
    env = _lerobot_env_for_dataset(task)
    assert env["HF_LEROBOT_HOME"] == "/data/ur10e_drugsort_v0"
    # The deprecated LEROBOT_HOME must NOT be set — recent lerobot hard-fails on it.
    assert "LEROBOT_HOME" not in env


def test_lerobot_env_empty_for_hf_dataset() -> None:
    task = _task(
        dataset=DatasetRef(source=DatasetSource.HUGGINGFACE, ref="org/some-lerobot"),
    )
    assert _lerobot_env_for_dataset(task) == {}


def test_lerobot_env_empty_when_no_dataset() -> None:
    assert _lerobot_env_for_dataset(_task()) == {}


# ---------------------------------------------------------------------------
# checkpoint resolution
# ---------------------------------------------------------------------------

def test_resolve_output_checkpoint_picks_highest_step(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints" / "pi05_ur10e" / "exp"
    for step in (1000, 5000, 2000):
        (root / str(step) / "params").mkdir(parents=True)
    resolved = _resolve_output_checkpoint(tmp_path)
    assert resolved is not None
    assert resolved.name == "5000"


def test_resolve_output_checkpoint_none_when_empty(tmp_path: Path) -> None:
    assert _resolve_output_checkpoint(tmp_path) is None


# ---------------------------------------------------------------------------
# stdout parser
# ---------------------------------------------------------------------------

def test_parse_loss_and_step_line() -> None:
    event = parse_pi05_train_line("step=1000 loss=0.234 grad_norm=1.2")
    assert event is not None
    assert event["stage"] == "executing"
    assert event["step"] == "training_step"
    assert event["step_index"] == 1000
    assert "loss=0.234" in event["step_label"]


def test_parse_dict_style_metric_line() -> None:
    event = parse_pi05_train_line("Step 1000: {'loss': 0.234, 'learning_rate': 1e-05}")
    assert event is not None
    assert event["step_index"] == 1000
    assert "loss=0.234" in event["step_label"]


def test_parse_tqdm_progress_line() -> None:
    event = parse_pi05_train_line(" 10%|█  | 100/1000 [00:42<06:18,  2.38it/s]")
    assert event is not None
    assert event["step_index"] == 100
    assert event["step_total"] == 1000


def test_parse_checkpoint_save_line() -> None:
    event = parse_pi05_train_line("Saving checkpoint to ./checkpoints/pi05_libero/exp/1000")
    assert event is not None
    assert event["stage"] == "checkpoint_saving"


def test_parse_norm_stats_line() -> None:
    event = parse_pi05_train_line("Computing normalization statistics for pi05_ur10e")
    assert event is not None
    assert event["stage"] == "dataset_loading"
    assert event["step"] == "compute_norm_stats"


def test_parse_dataset_loading_line() -> None:
    event = parse_pi05_train_line("Loading LeRobot dataset ur10e_partial_cond_aug")
    assert event is not None
    assert event["stage"] == "dataset_loading"


def test_parse_unrelated_line_returns_none() -> None:
    assert parse_pi05_train_line("nothing interesting here") is None
