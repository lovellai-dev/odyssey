"""Runtime records for mission + task execution.

The Pydantic models in `odyssey.spec` describe a mission *spec* — what the
user authored in YAML. The records here describe an *instance* of running
that spec: the persisted state, status transitions, timestamps, error
messages, and any per-task result_summary the runner returned.

Naming: a `MissionRun` is one execution of a Mission spec. A `TaskRun` is
one execution of a TaskSpec. The spec is reused; the records track what
happened to it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from odyssey.engine.lifecycle import MissionStatus, TaskStatus
from odyssey.providers.base import ResolvedRobot
from odyssey.spec.agents import AgentSpec
from odyssey.spec.mission import Mission
from odyssey.spec.tasks import TaskSpec, TrainingTask


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class TaskRun(BaseModel):
    id: str = Field(default_factory=_new_id)
    spec: TaskSpec
    status: TaskStatus = TaskStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    result_summary: dict[str, Any] = Field(default_factory=dict)


class MissionRun(BaseModel):
    id: str = Field(default_factory=_new_id)
    spec: Mission
    status: MissionStatus = MissionStatus.DRAFT
    tasks: list[TaskRun]
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    overall_grade: float | None = None
    error_message: str | None = None
    # Set by ``MissionEngine.create_mission`` when a ProviderRegistry is
    # wired. None when the engine was built without providers (the unit
    # / CPU-mock test path).
    resolved_robot: ResolvedRobot | None = None

    @classmethod
    def from_spec(cls, spec: Mission) -> MissionRun:
        """Materialize a fresh MissionRun from a validated Mission spec.

        Each TaskSpec becomes a PENDING TaskRun in spec order.
        """
        return cls(
            spec=spec,
            tasks=[TaskRun(spec=t) for t in spec.tasks],
        )

    # ------------------------------------------------------------------
    # Agent + checkpoint lookup helpers
    # ------------------------------------------------------------------
    # The engine and runners use these to honor the "agents have models,
    # training tasks update an agent" semantics: each training task
    # starts from the latest completed checkpoint for its agent (or
    # the agent's base model if it's the first round).

    def agent_by_id(self, agent_id: str) -> AgentSpec:
        for agent in self.spec.robot.agents:
            if agent.id == agent_id:
                return agent
        known = sorted(a.id for a in self.spec.robot.agents)
        raise ValueError(
            f"Agent {agent_id!r} not found on the robot (known: {known})"
        )

    def latest_checkpoint_for(self, agent_id: str) -> str | None:
        """The most-recent completed training task's checkpoint for this
        agent, or None if no training has produced one yet.

        Walks ``tasks`` in reverse so the latest match wins. Used by
        the engine to fill ``TaskContext.starting_checkpoint`` on
        subsequent training tasks against the same agent, and by the
        eval runner to find the checkpoint to evaluate.
        """
        for task in reversed(self.tasks):
            if not isinstance(task.spec, TrainingTask):
                continue
            if task.spec.agent_id != agent_id:
                continue
            if task.status != TaskStatus.COMPLETED:
                continue
            checkpoint = task.result_summary.get("checkpoint_path")
            if checkpoint:
                return str(checkpoint)
        return None
