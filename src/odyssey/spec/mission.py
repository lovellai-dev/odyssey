"""The Mission spec — top-level Pydantic model for `mission.yaml`.

Per design §2 v0.3.2 (with the agent-aware spec correction):

  * `odysseyVersion` is the spec version, evolving independently from the
    framework version.
  * `objective` and `acceptance_criteria` are required prose fields —
    matching the CC missions-table NOT NULL columns. These are the inputs
    the mission materializer extracts structured artifacts from.
  * Mission cardinality: at least one training task, exactly one evaluation
    task, and the evaluation task is the last entry in ``tasks[]``.
  * A robot carries a loadout of agents (today: exactly one). Every
    training task names the agent it updates via ``agent_id``, which
    must resolve to an entry in ``robot.agents``.
  * Task names are unique within a mission; agent ids are unique within
    the robot's loadout.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from odyssey.spec.agents import AgentSpec
from odyssey.spec.execution import ExecutionSpec
from odyssey.spec.graph import GraphSpec
from odyssey.spec.leaderboard import LeaderboardSpec
from odyssey.spec.tasks import TaskSpec, TrainingTask

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
    """A robot: an embodiment plus a loadout of agents.

    Exactly one of ``embodiment`` (catalog name), ``urdf`` (local file
    path), or ``id`` (a robot registered in a Lovell account) names the
    embodiment. ``agents`` is the loadout — today exactly one entry;
    when SPECIALISTs ship the upper bound lifts. The single agent
    today is the PILOT (running a Vision-Language-Action model with
    physical authority over the actuators).

    See the Lovell robot-brain paper for the fuller agent shape that
    v0.0.x's ``AgentSpec`` does not yet model (persona, goals, success
    criteria, materialized artifacts).
    """

    embodiment: str | None = None
    urdf: str | None = None
    id: str | None = None
    agents: list[AgentSpec] = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def _exactly_one_embodiment(self) -> RobotSpec:
        set_count = sum(x is not None for x in (self.embodiment, self.urdf, self.id))
        if set_count != 1:
            raise ValueError(
                "RobotSpec requires exactly one of: embodiment, urdf, id"
            )
        return self

    @model_validator(mode="after")
    def _agent_ids_unique(self) -> RobotSpec:
        # Trivial for ``max_length=1`` today; forward-compat for when
        # multi-agent loadouts ship and operators can declare several.
        ids = [a.id for a in self.agents]
        if len(ids) != len(set(ids)):
            raise ValueError("Agent ids must be unique within a robot")
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
    def _eval_is_last(self) -> Mission:
        # Evaluation runs the robot after every training task has
        # completed. Today execution follows spec order (sequential, no
        # depends_on graph walk), so eval-last is enforced positionally.
        if self.tasks and self.tasks[-1].kind != "evaluation":
            raise ValueError(
                "Evaluation task must be the last entry in tasks[]; "
                f"got {self.tasks[-1].kind!r} ({self.tasks[-1].name!r}) last."
            )
        return self

    @model_validator(mode="after")
    def _training_agent_ids_resolve(self) -> Mission:
        known_agent_ids = {a.id for a in self.robot.agents}
        for task in self.tasks:
            if isinstance(task, TrainingTask) and task.agent_id not in known_agent_ids:
                raise ValueError(
                    f"Training task {task.name!r} targets agent "
                    f"{task.agent_id!r}, which is not in the robot's loadout "
                    f"(known: {sorted(known_agent_ids)})"
                )
        return self
