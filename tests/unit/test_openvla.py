"""Tests for the OpenVLA runner's argv builder + stdout parser.

We don't invoke the real ``finetune.py`` — that requires the openvla repo,
torch with CUDA, and a GPU. The two testable pieces here are the
construction of the CLI argv from a TrainingTask spec (with the agent's
model base passed alongside), and the regex parser that turns OpenVLA's
stdout into progress events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from odyssey.runners.openvla import build_openvla_argv, parse_openvla_line
from odyssey.spec import TrainingTask, TrainingType


def _task(**overrides: Any) -> TrainingTask:
    fields: dict[str, Any] = {
        "name": "finetune",
        "training_type": TrainingType.DEMONSTRATION,
        "agent_id": "pilot",
    }
    fields.update(overrides)
    return TrainingTask(**fields)


HF_BASE = "openvla/openvla-7b"


# ---------------------------------------------------------------------------
# argv builder
# ---------------------------------------------------------------------------

def test_argv_uses_agent_model_base_when_no_override(tmp_path: Path) -> None:
    task = _task()
    argv = build_openvla_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path, run_id="run-1"
    )
    assert "--vla_path" in argv
    idx = argv.index("--vla_path")
    assert argv[idx + 1] == HF_BASE


def test_argv_prefers_config_vla_path(tmp_path: Path) -> None:
    task = _task(config={"vla_path": "/local/openvla-7b"})
    argv = build_openvla_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path, run_id="run-1"
    )
    idx = argv.index("--vla_path")
    assert argv[idx + 1] == "/local/openvla-7b"


def test_argv_uses_env_var_when_no_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENVLA_OPENVLA_7B_PATH", "/env/openvla-7b")
    task = _task()
    argv = build_openvla_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path, run_id="run-1"
    )
    idx = argv.index("--vla_path")
    assert argv[idx + 1] == "/env/openvla-7b"


def test_argv_includes_run_root_and_run_id(tmp_path: Path) -> None:
    task = _task()
    argv = build_openvla_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path, run_id="my-run"
    )
    rr_idx = argv.index("--run_root_dir")
    assert argv[rr_idx + 1] == str(tmp_path)
    rid_idx = argv.index("--run_id")
    assert argv[rid_idx + 1] == "my-run"


def test_argv_includes_adapter_tmp_dir(tmp_path: Path) -> None:
    task = _task()
    argv = build_openvla_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path, run_id="r"
    )
    idx = argv.index("--adapter_tmp_dir")
    assert argv[idx + 1] == str(tmp_path / "adapter_tmp")


def test_argv_defaults_use_lora_true(tmp_path: Path) -> None:
    task = _task()
    argv = build_openvla_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path, run_id="r"
    )
    assert "--use_lora" in argv
    idx = argv.index("--use_lora")
    assert argv[idx + 1] == "True"


def test_argv_does_not_override_explicit_use_lora(tmp_path: Path) -> None:
    task = _task(config={"use_lora": "False"})
    argv = build_openvla_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path, run_id="r"
    )
    # Should appear exactly once with the explicit value.
    idx = argv.index("--use_lora")
    assert argv[idx + 1] == "False"
    assert argv.count("--use_lora") == 1


def test_argv_passes_dataset_via_data_root_dir(tmp_path: Path) -> None:
    from odyssey.spec import DatasetRef, DatasetSource
    task = _task(
        dataset=DatasetRef(
            source=DatasetSource.HUGGINGFACE, ref="lerobot/bridge_v2"
        ),
    )
    argv = build_openvla_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path, run_id="r"
    )
    idx = argv.index("--data_root_dir")
    assert argv[idx + 1] == "lerobot/bridge_v2"


def test_argv_includes_flat_overrides(tmp_path: Path) -> None:
    task = _task(
        config={
            "batch_size": 4,
            "learning_rate": 5e-5,
        }
    )
    argv = build_openvla_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path, run_id="r"
    )
    assert "--batch_size" in argv and argv[argv.index("--batch_size") + 1] == "4"
    assert "--learning_rate" in argv


def test_argv_skips_handled_keys_in_overrides(tmp_path: Path) -> None:
    task = _task(config={"vla_path": "/x", "batch_size": 2})
    argv = build_openvla_argv(
        task=task, agent_model_base=HF_BASE, output_dir=tmp_path, run_id="r"
    )
    # vla_path should appear exactly once (from the dedicated handler),
    # not also as a flat override.
    assert argv.count("--vla_path") == 1


def test_argv_raises_when_no_vla_path_resolvable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No config override, no env var, no agent_model_base — nothing to
    # use as the starting checkpoint.
    monkeypatch.delenv("OPENVLA_OPENVLA_7B_PATH", raising=False)
    task = _task()
    with pytest.raises(RuntimeError, match="cannot resolve vla_path"):
        build_openvla_argv(
            task=task, agent_model_base=None, output_dir=tmp_path, run_id="r"
        )


# ---------------------------------------------------------------------------
# stdout parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "step: 1234 | loss: 0.234 | grad_norm: 1.23 | lr: 5e-5",
            {
                "stage": "executing",
                "step": "training_step",
                "step_index": 1234,
                "step_label": "loss=0.234",
            },
        ),
        (
            "Loading VLA checkpoint from disk",
            {"stage": "model_loading", "step": "load_artifacts"},
        ),
        (
            "Building RLDS dataset...",
            {"stage": "dataset_loading"},
        ),
        (
            "Saved adapter to /tmp/x",
            {"stage": "checkpoint_saving"},
        ),
    ],
)
def test_parser_recognizes_known_lines(line: str, expected: dict[str, Any]) -> None:
    assert parse_openvla_line(line) == expected


def test_parser_returns_none_for_unrecognized() -> None:
    assert parse_openvla_line("some random log line that doesn't match anything") is None
    assert parse_openvla_line("") is None
