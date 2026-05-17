"""Unit tests for the mission spec module.

Covers the validators that matter most: cardinality, name uniqueness,
RobotSpec exactly-one, from_task ordering. Plus a smoke test that the
shipped example loads cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from odyssey.spec import (
    EvaluationTask,
    EvaluationType,
    HFModelRef,
    Mission,
    MissionMetadata,
    RobotSpec,
    TrainingTask,
    TrainingType,
    load_mission,
)
from odyssey.spec.loader import LoadError
from odyssey.spec.refs import FromTaskModelRef

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MISSION = REPO_ROOT / "examples" / "quickstart-openvla" / "mission.yaml"


# ---------------------------------------------------------------------------
# Builders to keep test bodies short
# ---------------------------------------------------------------------------

def _training_task(name: str = "train-one", **overrides: object) -> TrainingTask:
    fields: dict[str, object] = {
        "name": name,
        "training_type": TrainingType.DEMONSTRATION,
        "model": HFModelRef(base="openvla/openvla-7b"),
        "target_agent_id": "pilot",
    }
    fields.update(overrides)
    return TrainingTask(**fields)


def _eval_task(name: str = "eval-one", **overrides: object) -> EvaluationTask:
    fields: dict[str, object] = {
        "name": name,
        "evaluation_type": EvaluationType.ROBOSUITE,
        "benchmark_name": "Lift",
        "model": HFModelRef(base="openvla/openvla-7b"),
        "target_agent_id": "pilot",
    }
    fields.update(overrides)
    return EvaluationTask(**fields)


def _mission(tasks: list) -> Mission:
    return Mission(
        metadata=MissionMetadata(name="msn"),
        objective="objective",
        acceptance_criteria="acceptance",
        robot=RobotSpec(embodiment="franka_panda"),
        tasks=tasks,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_minimal_mission_parses() -> None:
    m = _mission([_training_task(), _eval_task()])
    assert m.odysseyVersion.value == "0.1"
    assert len(m.tasks) == 2


def test_shipped_example_loads() -> None:
    """The OpenVLA quickstart YAML must always be a valid spec."""
    mission = load_mission(EXAMPLE_MISSION)
    assert mission.metadata.name == "openvla-bridge-lift"
    assert sum(1 for t in mission.tasks if t.kind == "training") == 1
    assert sum(1 for t in mission.tasks if t.kind == "evaluation") == 1


# ---------------------------------------------------------------------------
# Mission cardinality
# ---------------------------------------------------------------------------

def test_zero_eval_tasks_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly one evaluation task"):
        _mission([_training_task("t1"), _training_task("t2")])


def test_two_eval_tasks_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly one evaluation task"):
        _mission([_training_task(), _eval_task("e1"), _eval_task("e2")])


def test_zero_training_tasks_rejected() -> None:
    # Need >=2 tasks to satisfy min_length; two eval tasks fails BOTH
    # cardinality rules — accept either error string.
    with pytest.raises(ValidationError):
        _mission([_eval_task("e1"), _eval_task("e2")])


# ---------------------------------------------------------------------------
# Task name uniqueness
# ---------------------------------------------------------------------------

def test_duplicate_task_names_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        _mission([_training_task("dup"), _eval_task("dup")])


# ---------------------------------------------------------------------------
# RobotSpec exactly-one
# ---------------------------------------------------------------------------

def test_robotspec_requires_exactly_one() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        RobotSpec()  # nothing set
    with pytest.raises(ValidationError, match="exactly one"):
        RobotSpec(embodiment="franka_panda", urdf="/tmp/x.urdf")  # two set


def test_robotspec_accepts_each_variant() -> None:
    assert RobotSpec(embodiment="franka_panda").embodiment == "franka_panda"
    assert RobotSpec(urdf="/tmp/x.urdf").urdf == "/tmp/x.urdf"
    assert RobotSpec(id="robot-123").id == "robot-123"


# ---------------------------------------------------------------------------
# from_task ordering
# ---------------------------------------------------------------------------

def test_from_task_ref_to_later_task_rejected() -> None:
    # eval task references a training task that comes AFTER it.
    eval_first = _eval_task(
        "eval-first",
        model=FromTaskModelRef(from_task="train-later"),
    )
    train_later = _training_task("train-later")
    with pytest.raises(ValidationError, match="later task"):
        _mission([eval_first, train_later])


def test_from_task_ref_to_unknown_task_rejected() -> None:
    eval_task = _eval_task(
        "eval-bad",
        model=FromTaskModelRef(from_task="does-not-exist"),
    )
    with pytest.raises(ValidationError, match="unknown task"):
        _mission([_training_task("train-one"), eval_task])


def test_from_task_ref_to_earlier_task_accepted() -> None:
    train = _training_task("train-first")
    eval_after = _eval_task(
        "eval-after",
        model=FromTaskModelRef(from_task="train-first"),
    )
    m = _mission([train, eval_after])
    assert isinstance(m.tasks[1].model, FromTaskModelRef)


# ---------------------------------------------------------------------------
# Loader error wrapping
# ---------------------------------------------------------------------------

def test_loader_wraps_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(LoadError, match="cannot read file"):
        load_mission(missing)


def test_loader_wraps_bad_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [unclosed\n", encoding="utf-8")
    with pytest.raises(LoadError, match="YAML syntax error"):
        load_mission(bad)


def test_loader_wraps_validation_error(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "metadata:\n  name: x\nobjective: o\nacceptance_criteria: a\n"
        "robot:\n  embodiment: franka_panda\ntasks: []\n",
        encoding="utf-8",
    )
    with pytest.raises(LoadError, match="spec validation failed"):
        load_mission(invalid)
