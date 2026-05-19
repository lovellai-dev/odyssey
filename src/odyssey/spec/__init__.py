"""Pydantic schemas for the mission spec (`mission.yaml`).

Public surface is re-exported here so callers do
``from odyssey.spec import Mission`` without knowing the submodule layout.
"""

from odyssey.spec.agents import AgentRole, AgentSpec
from odyssey.spec.execution import ExecutionSpec
from odyssey.spec.graph import GraphSpec
from odyssey.spec.leaderboard import LeaderboardSpec
from odyssey.spec.loader import LoadError, load_mission
from odyssey.spec.mission import Mission, MissionMetadata, OdysseyVersion, RobotSpec
from odyssey.spec.refs import (
    DatasetFormat,
    DatasetRef,
    DatasetSource,
    HFModelRef,
    LovellModelRef,
    ModelRef,
)
from odyssey.spec.tasks import (
    EvaluationTask,
    EvaluationType,
    TaskKind,
    TaskSpec,
    TrainingTask,
    TrainingType,
)

__all__ = [
    "AgentRole",
    "AgentSpec",
    "DatasetFormat",
    "DatasetRef",
    "DatasetSource",
    "EvaluationTask",
    "EvaluationType",
    "ExecutionSpec",
    "GraphSpec",
    "HFModelRef",
    "LeaderboardSpec",
    "LoadError",
    "LovellModelRef",
    "Mission",
    "MissionMetadata",
    "ModelRef",
    "OdysseyVersion",
    "RobotSpec",
    "TaskKind",
    "TaskSpec",
    "TrainingTask",
    "TrainingType",
    "load_mission",
]
