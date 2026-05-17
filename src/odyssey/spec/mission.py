"""The Mission spec — top-level Pydantic model for `mission.yaml`.

Per design §2 v0.3.2:

  * `odysseyVersion` is the spec version, evolving independently from the
    framework version.
  * `objective` and `acceptance_criteria` are required prose fields —
    matching the CC missions-table NOT NULL columns. These are the inputs
    the mission materializer extracts structured artifacts from.
  * Mission cardinality: at least one training task, exactly one evaluation
    task. Enforced at load time, not at execution time.
  * Task names are unique within a mission.
  * `model.from_task` refs must point to earlier tasks.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from odyssey.spec.execution import ExecutionSpec
from odyssey.spec.graph import GraphSpec
from odyssey.spec.leaderboard import LeaderboardSpec
from odyssey.spec.refs import FromTaskModelRef
from odyssey.spec.tasks import TaskSpec

_NAME_PATTERN = r"^[a-z0-9][a-z0-9-]*[a-z0-9]$"


class OdysseyVersion(str, Enum):
    """Mission spec versions this loader accepts."""

    V0_1 = "0.1"


class MissionMetadata(BaseModel):
    """Mission identity. Prose fields live on Mission, not here."""

    name: Annotated[str, Field(pattern=_NAME_PATTERN, max_length=64)]
    description: str | None = None
    tags: list[str] = Field(default_factory=list)


class RobotSpec(BaseModel):
    """Exactly one of embodiment / urdf / id must be set."""

    embodiment: str | None = None
    urdf: str | None = None
    id: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> RobotSpec:
        set_count = sum(x is not None for x in (self.embodiment, self.urdf, self.id))
        if set_count != 1:
            raise ValueError(
                "RobotSpec requires exactly one of: embodiment, urdf, id"
            )
        return self


class Mission(BaseModel):
    odysseyVersion: OdysseyVersion = OdysseyVersion.V0_1
    kind: Literal["Mission"] = "Mission"
    metadata: MissionMetadata

    objective: str = Field(..., min_length=1)
    acceptance_criteria: str = Field(..., min_length=1)

    materialized_profile: str | None = None

    robot: RobotSpec
    tasks: list[TaskSpec] = Field(min_length=2)

    leaderboard: LeaderboardSpec = Field(default_factory=LeaderboardSpec)
    graph: GraphSpec = Field(default_factory=GraphSpec)
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)

    @model_validator(mode="after")
    def _task_names_unique(self) -> Mission:
        names = [t.name for t in self.tasks]
        if len(names) != len(set(names)):
            raise ValueError("Task names must be unique within a mission")
        return self

    @model_validator(mode="after")
    def _task_cardinality(self) -> Mission:
        training = sum(1 for t in self.tasks if t.kind == "training")
        evaluation = sum(1 for t in self.tasks if t.kind == "evaluation")
        if training < 1:
            raise ValueError(
                f"Mission must have at least one training task, got {training}"
            )
        if evaluation != 1:
            raise ValueError(
                f"Mission must have exactly one evaluation task, got {evaluation}"
            )
        return self

    @model_validator(mode="after")
    def _from_task_refs_resolve(self) -> Mission:
        index_by_name = {t.name: i for i, t in enumerate(self.tasks)}
        for i, task in enumerate(self.tasks):
            ref = task.model
            if isinstance(ref, FromTaskModelRef):
                if ref.from_task not in index_by_name:
                    raise ValueError(
                        f"Task {task.name!r} references unknown task {ref.from_task!r}"
                    )
                if index_by_name[ref.from_task] >= i:
                    raise ValueError(
                        f"Task {task.name!r} references later task {ref.from_task!r} "
                        "(must be earlier)"
                    )
        return self
