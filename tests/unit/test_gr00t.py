"""Tests for the GR00T runner's argv builder + stdout parser.

We don't invoke the real ``launch_finetune`` — that requires the
Isaac-GR00T package, torch with CUDA, and a GPU. The testable pieces are
the construction of the CLI argv from a TrainingTask spec (with the
agent's model base passed alongside) and the parser that turns the
HF-Trainer-style stdout into progress events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from odyssey.runners.models.gr00t import build_gr00t_argv, parse_gr00t_line
from odyssey.spec import DatasetRef, DatasetSource, TrainingTask, TrainingType


def _task(**overrides: Any) -> TrainingTask:
    fields: dict[str, Any] = {
        "name": "finetune",
        "training_type": TrainingType.DEMONSTRATION,
        "agent_id": "pilot",
    }
    fields.update(overrides)
    return TrainingTask(**fields)


HF_BASE = "nvidia/GR00T-N1.7-3B"


# ---------------------------------------------------------------------------
# argv builder
# ---------------------------------------------------------------------------

def test_argv_uses_agent_model_base_when_no_override(tmp_path: Path) -> None:
    argv = build_gr00t_argv(
        task=_task(), agent_model_base=HF_BASE, output_dir=tmp_path
    )
    idx = argv.index("--base-model-path")
    assert argv[idx + 1] == HF_BASE


def test_argv_prefers_config_base_model_path(tmp_path: Path) -> None:
    task = _task(config={"base_model_path": "/local/gr00t-n1.7-3b"})
    argv = build_gr00t_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path
    )
    idx = argv.index("--base-model-path")
    assert argv[idx + 1] == "/local/gr00t-n1.7-3b"


def test_argv_uses_env_var_when_no_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Dots in the version id map to underscores in the env-var name.
    monkeypatch.setenv("NVIDIA_GR00T_N1_7_3B_PATH", "/env/gr00t-n1.7-3b")
    argv = build_gr00t_argv(
        task=_task(), agent_model_base=HF_BASE, output_dir=tmp_path
    )
    idx = argv.index("--base-model-path")
    assert argv[idx + 1] == "/env/gr00t-n1.7-3b"


def test_argv_errors_when_nothing_resolvable(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="base_model_path"):
        build_gr00t_argv(task=_task(), agent_model_base=None, output_dir=tmp_path)


def test_argv_includes_output_dir(tmp_path: Path) -> None:
    argv = build_gr00t_argv(
        task=_task(), agent_model_base=HF_BASE, output_dir=tmp_path
    )
    idx = argv.index("--output-dir")
    assert argv[idx + 1] == str(tmp_path)


def test_argv_resolves_relative_local_dataset_against_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ISAAC_GR00T_REPO_PATH", "/opt/isaac-gr00t")
    task = _task(
        dataset=DatasetRef(
            source=DatasetSource.LOCAL, ref="demo_data/cube_to_bowl_5"
        ),
    )
    argv = build_gr00t_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path
    )
    idx = argv.index("--dataset-path")
    assert argv[idx + 1] == "/opt/isaac-gr00t/demo_data/cube_to_bowl_5"


def test_argv_resolves_relative_modality_config_against_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # NEW_EMBODIMENT finetunes require a modality config file; a relative ref
    # resolves against the Isaac-GR00T checkout, exactly like the dataset ref.
    monkeypatch.setenv("ISAAC_GR00T_REPO_PATH", "/opt/isaac-gr00t")
    task = _task(config={"modality_config_path": "examples/SO100/so100_config.py"})
    argv = build_gr00t_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path
    )
    idx = argv.index("--modality-config-path")
    assert argv[idx + 1] == "/opt/isaac-gr00t/examples/SO100/so100_config.py"


def test_argv_passes_absolute_modality_config_unchanged(tmp_path: Path) -> None:
    task = _task(config={"modality_config_path": "/abs/cfg.py"})
    argv = build_gr00t_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path
    )
    idx = argv.index("--modality-config-path")
    assert argv[idx + 1] == "/abs/cfg.py"


def test_argv_passes_absolute_local_dataset_unchanged(tmp_path: Path) -> None:
    task = _task(
        dataset=DatasetRef(source=DatasetSource.LOCAL, ref="/data/my_episodes"),
    )
    argv = build_gr00t_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path
    )
    idx = argv.index("--dataset-path")
    assert argv[idx + 1] == "/data/my_episodes"


def test_argv_passes_hf_dataset_ref_through(tmp_path: Path) -> None:
    task = _task(
        dataset=DatasetRef(
            source=DatasetSource.HUGGINGFACE, ref="nvidia/some-lerobot-set"
        ),
    )
    argv = build_gr00t_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path
    )
    idx = argv.index("--dataset-path")
    assert argv[idx + 1] == "nvidia/some-lerobot-set"


def test_argv_passthrough_is_kebab_case(tmp_path: Path) -> None:
    task = _task(config={"embodiment_tag": "new_embodiment", "max_steps": 1000})
    argv = build_gr00t_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path
    )
    idx = argv.index("--embodiment-tag")
    assert argv[idx + 1] == "new_embodiment"
    idx = argv.index("--max-steps")
    assert argv[idx + 1] == "1000"


def test_argv_excludes_runner_routing_key(tmp_path: Path) -> None:
    task = _task(config={"runner": "gr00t", "max_steps": 10})
    argv = build_gr00t_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path
    )
    assert "--runner" not in argv


# ---------------------------------------------------------------------------
# stdout parser
# ---------------------------------------------------------------------------

def test_parse_trainer_loss_line() -> None:
    line = "{'loss': 0.6931, 'grad_norm': 1.21, 'learning_rate': 1e-05, 'epoch': 0.25}"
    event = parse_gr00t_line(line)
    assert event is not None
    assert event["stage"] == "executing"
    assert event["step"] == "training_step"
    assert "loss=0.6931" in event["step_label"]
    assert "epoch=0.25" in event["step_label"]


def test_parse_tqdm_progress_line() -> None:
    event = parse_gr00t_line(" 10%|█         | 100/1000 [00:42<06:18,  2.38it/s]")
    assert event is not None
    assert event["step_index"] == 100
    assert event["step_total"] == 1000


def test_parse_checkpoint_save_line() -> None:
    event = parse_gr00t_line("Saving model checkpoint to ./out/checkpoint-500")
    assert event is not None
    assert event["stage"] == "checkpoint_saving"


def test_parse_dataset_loading_line() -> None:
    event = parse_gr00t_line("Loading LeRobot dataset from demo_data/cube_to_bowl_5")
    assert event is not None
    assert event["stage"] == "dataset_loading"


def test_parse_model_loading_line() -> None:
    event = parse_gr00t_line("Loading checkpoint shards: 100%")
    assert event is not None
    assert event["stage"] == "model_loading"


def test_parse_unrelated_line_returns_none() -> None:
    assert parse_gr00t_line("nothing interesting here") is None
