"""The UR5e drug-sorting example mission parses and the GR00T runner assembles
the expected ``launch_finetune`` argv from it (no GPU / Isaac-GR00T needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from odyssey.runners.models.gr00t import build_gr00t_argv
from odyssey.spec.loader import load_mission
from odyssey.spec.tasks import TrainingTask

MISSION = Path("examples/ur5e-drugsort/mission.yaml")


def test_example_mission_parses_and_has_gr00t_training_task() -> None:
    mission = load_mission(MISSION)
    assert mission.metadata.name == "ur5e-drugsort-finetune"
    # exactly one training task (routed to gr00t) + one evaluation task, last.
    kinds = [t.kind for t in mission.tasks]
    assert kinds == ["training", "evaluation"]
    train = mission.tasks[0]
    assert isinstance(train, TrainingTask)
    assert train.config["runner"] == "gr00t"
    assert train.config["embodiment_tag"] == "new_embodiment"


def test_runner_assembles_expected_launch_finetune_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ISAAC_GR00T_REPO_PATH", "/opt/isaac-gr00t")
    train = load_mission(MISSION).tasks[0]
    assert isinstance(train, TrainingTask)

    argv = build_gr00t_argv(
        task=train, agent_model_base="nvidia/GR00T-N1.7-3B", output_dir=tmp_path
    )

    def val(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert val("--base-model-path") == "nvidia/GR00T-N1.7-3B"
    assert val("--dataset-path") == "/data/ur5e_drugsort"  # absolute -> passthrough
    assert val("--embodiment-tag") == "new_embodiment"
    # relative modality config resolves against $ISAAC_GR00T_REPO_PATH
    assert val("--modality-config-path") == (
        "/opt/isaac-gr00t/examples/UR5e_DrugSort/ur5e_config.py"
    )
    assert val("--max-steps") == "2000"
    assert val("--global-batch-size") == "16"
    # routing key must NOT leak onto the CLI
    assert "--runner" not in argv
