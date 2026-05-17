"""Event-type vocabulary published by the engine.

Mirrors lai-trainer's `MissionEventType` and `TaskEventType` so the OSS
engine and the Lovell-hosted Command Center can consume the same stream
shape.
"""

from __future__ import annotations

from enum import Enum


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
