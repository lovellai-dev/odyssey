"""Unit tests for the mission spec module.

Covers the validators that matter most: cardinality, name uniqueness,
RobotSpec exactly-one-embodiment + agent loadout, training agent_id
resolution, eval-is-last. Plus a smoke test that the shipped example
loads cleanly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

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
    load_mission,
)
from odyssey.spec.loader import LoadError

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MISSION = REPO_ROOT / "examples" / "quickstart-openvla" / "mission.yaml"


# ---------------------------------------------------------------------------
# Builders to keep test bodies short
# ---------------------------------------------------------------------------

def _agent(id_: str = "pilot") -> AgentSpec:
    return AgentSpec(
        id=id_,
        role=AgentRole.PILOT,
        model=HFModelRef(base="openvla/openvla-7b"),
    )


def _robot(**overrides: object) -> RobotSpec:
    fields: dict[str, object] = {
        "embodiment": "franka_panda",
        "agents": [_agent()],
    }
    fields.update(overrides)
    return RobotSpec(**fields)


def _training_task(name: str = "train-one", **overrides: object) -> TrainingTask:
    fields: dict[str, object] = {
        "name": name,
        "training_type": TrainingType.DEMONSTRATION,
        "agent_id": "pilot",
    }
    fields.update(overrides)
    return TrainingTask(**fields)


def _eval_task(name: str = "eval-one", **overrides: object) -> EvaluationTask:
    fields: dict[str, object] = {
        "name": name,
        "evaluation_type": EvaluationType.ROBOSUITE,
        "benchmark_name": "Lift",
    }
    fields.update(overrides)
    return EvaluationTask(**fields)


def _mission(tasks: list, *, robot: RobotSpec | None = None) -> Mission:
    return Mission(
        metadata=MissionMetadata(name="msn"),
        objective="objective",
        acceptance_criteria="acceptance",
        robot=robot if robot is not None else _robot(),
        tasks=tasks,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_minimal_mission_parses() -> None:
    m = _mission([_training_task(), _eval_task()])
    assert m.odysseyVersion.value == "0.1"
    assert len(m.tasks) == 2
    assert m.robot.agents[0].id == "pilot"


def test_shipped_example_loads() -> None:
    """The OpenVLA quickstart YAML must always be a valid spec."""
    mission = load_mission(EXAMPLE_MISSION)
    assert mission.metadata.name == "openvla-bridge-lift"
    assert sum(1 for t in mission.tasks if t.kind == "training") == 1
    assert sum(1 for t in mission.tasks if t.kind == "evaluation") == 1
    assert mission.robot.agents[0].id == "pilot"


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
# Eval must be last
# ---------------------------------------------------------------------------

def test_eval_first_rejected() -> None:
    with pytest.raises(ValidationError, match="last entry"):
        _mission([_eval_task("ev"), _training_task("tr")])


def test_eval_between_trainings_rejected() -> None:
    with pytest.raises(ValidationError, match="last entry"):
        _mission(
            [
                _training_task("t1"),
                _eval_task("ev"),
                _training_task("t2"),
            ]
        )


def test_eval_last_accepted_with_multiple_trainings() -> None:
    m = _mission(
        [
            _training_task("t1"),
            _training_task("t2"),
            _eval_task("ev"),
        ]
    )
    assert m.tasks[-1].kind == "evaluation"


# ---------------------------------------------------------------------------
# RobotSpec exactly-one-embodiment + loadout
# ---------------------------------------------------------------------------

def test_robotspec_requires_exactly_one_embodiment() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        RobotSpec(agents=[_agent()])  # nothing of embodiment/urdf/id set
    with pytest.raises(ValidationError, match="exactly one"):
        RobotSpec(
            embodiment="franka_panda",
            urdf="/tmp/x.urdf",
            agents=[_agent()],
        )


def test_robotspec_accepts_each_embodiment_variant() -> None:
    assert RobotSpec(embodiment="franka_panda", agents=[_agent()]).embodiment == "franka_panda"
    assert RobotSpec(urdf="/tmp/x.urdf", agents=[_agent()]).urdf == "/tmp/x.urdf"
    assert RobotSpec(id="robot-123", agents=[_agent()]).id == "robot-123"


def test_robotspec_requires_at_least_one_agent() -> None:
    with pytest.raises(ValidationError):
        RobotSpec(embodiment="franka_panda", agents=[])


def test_robotspec_accepts_pilot_plus_specialist() -> None:
    specialist = AgentSpec(
        id="task-planner",
        role=AgentRole.SPECIALIST,
        model=HFModelRef(base="google/gemma-3-4b-it", quantization="int4"),
    )
    robot = RobotSpec(
        embodiment="franka_panda",
        agents=[_agent("pilot"), specialist],
    )
    assert len(robot.agents) == 2
    assert robot.agents[0].role == AgentRole.PILOT
    assert robot.agents[1].role == AgentRole.SPECIALIST


def test_robotspec_rejects_loadout_without_pilot() -> None:
    specialist = AgentSpec(
        id="planner",
        role=AgentRole.SPECIALIST,
        model=HFModelRef(base="google/gemma-3-4b-it"),
    )
    with pytest.raises(ValidationError, match="at least one PILOT"):
        RobotSpec(embodiment="franka_panda", agents=[specialist])


def test_robotspec_caps_agents_at_four() -> None:
    with pytest.raises(ValidationError):
        RobotSpec(
            embodiment="franka_panda",
            agents=[_agent(f"a{i}") for i in range(5)],
        )


def test_robotspec_duplicate_agent_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        RobotSpec(
            embodiment="franka_panda",
            agents=[_agent("pilot"), _agent("pilot")],
        )


# ---------------------------------------------------------------------------
# Training agent_id must resolve + no SPECIALIST training
# ---------------------------------------------------------------------------

def test_training_agent_id_must_resolve_to_robot_agent() -> None:
    with pytest.raises(ValidationError, match="not in the robot's loadout"):
        _mission([_training_task(agent_id="nobody"), _eval_task()])


def test_training_agent_id_to_known_agent_accepted() -> None:
    m = _mission([_training_task(agent_id="pilot"), _eval_task()])
    assert m.tasks[0].agent_id == "pilot"


def test_training_specialist_rejected() -> None:
    specialist = AgentSpec(
        id="task-planner",
        role=AgentRole.SPECIALIST,
        model=HFModelRef(base="google/gemma-3-4b-it"),
    )
    robot = RobotSpec(
        embodiment="franka_panda",
        agents=[_agent("pilot"), specialist],
    )
    with pytest.raises(ValidationError, match=r"SPECIALIST.*not trained"):
        _mission(
            [_training_task(agent_id="task-planner"), _eval_task()],
            robot=robot,
        )


def test_training_pilot_with_specialist_present_accepted() -> None:
    """Training a PILOT is fine even when a SPECIALIST is in the loadout."""
    specialist = AgentSpec(
        id="task-planner",
        role=AgentRole.SPECIALIST,
        model=HFModelRef(base="google/gemma-3-4b-it"),
    )
    robot = RobotSpec(
        embodiment="franka_panda",
        agents=[_agent("pilot"), specialist],
    )
    m = _mission(
        [_training_task(agent_id="pilot"), _eval_task()],
        robot=robot,
    )
    assert m.tasks[0].agent_id == "pilot"


# ---------------------------------------------------------------------------
# HFModelRef quantization field
# ---------------------------------------------------------------------------

def test_hfmodelref_accepts_quantization() -> None:
    ref = HFModelRef(base="google/gemma-3-4b-it", quantization="int4")
    assert ref.quantization == "int4"


def test_hfmodelref_quantization_optional() -> None:
    ref = HFModelRef(base="openvla/openvla-7b")
    assert ref.quantization is None


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
        "robot:\n  embodiment: franka_panda\n  agents: []\n"
        "tasks: []\n",
        encoding="utf-8",
    )
    with pytest.raises(LoadError, match="spec validation failed"):
        load_mission(invalid)
