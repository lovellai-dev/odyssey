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
from odyssey.spec.mission import Mission
from odyssey.spec.tasks import TaskSpec


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

    @classmethod
    def from_spec(cls, spec: Mission) -> MissionRun:
        """Materialize a fresh MissionRun from a validated Mission spec.

        Each TaskSpec becomes a PENDING TaskRun in spec order.
        """
        return cls(
            spec=spec,
            tasks=[TaskRun(spec=t) for t in spec.tasks],
        )
