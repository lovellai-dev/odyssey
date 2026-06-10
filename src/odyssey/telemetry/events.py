"""Event-type vocabulary published by the engine.

Mirrors lai-trainer's `MissionEventType` and `TaskEventType` so the OSS
engine and the Lovell-hosted Command Center can consume the same stream
shape.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class MissionEventType(str, Enum):
    CREATED = "mission.created"
    QUEUED = "mission.queued"
    STARTED = "mission.started"
    COMPLETED = "mission.completed"
    FAILED = "mission.failed"
    CANCELLED = "mission.cancelled"


class TaskEventType(str, Enum):
    QUEUED = "task.queued"
    STARTED = "task.started"
    PROGRESS = "task.progress"
    COMPLETED = "task.completed"
    FAILED = "task.failed"
    CANCELLED = "task.cancelled"


class ProgressEvent(BaseModel):
    """Structured progress event emitted by runners.

    Validated at emission time by ``TaskContext.emit_progress()``.
    """

    mission_id: str
    task_id: str
    task_name: str
    stage: str
    seq: int
    step: str | None = None
    step_index: int | None = None
    step_total: int | None = None
    step_label: str | None = None
    metadata: dict[str, Any] | None = None
