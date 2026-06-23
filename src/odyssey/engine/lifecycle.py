"""Mission + task status enums and transition rules.

Mirrors the 6-state vocabulary used by lai-trainer's `MissionStatus` /
`TaskStatus` and by lai-inference's `JobState` exactly. The three layers
share this vocabulary on purpose — adding or renaming a state would break
the existing webhook + Pub/Sub contracts between them.

Spec: mission-service-guide.md §3.4 and §4.1.
"""

from __future__ import annotations

from enum import Enum


class MissionStatus(str, Enum):
    DRAFT = "DRAFT"
    QUEUED = "QUEUED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


_TERMINAL_MISSION: frozenset[MissionStatus] = frozenset(
    {MissionStatus.COMPLETED, MissionStatus.FAILED, MissionStatus.CANCELLED}
)

_TERMINAL_TASK: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)


_MISSION_TRANSITIONS: dict[MissionStatus, frozenset[MissionStatus]] = {
    MissionStatus.DRAFT: frozenset({MissionStatus.QUEUED, MissionStatus.CANCELLED}),
    MissionStatus.QUEUED: frozenset({MissionStatus.ACTIVE, MissionStatus.CANCELLED}),
    MissionStatus.ACTIVE: frozenset(
        {MissionStatus.COMPLETED, MissionStatus.FAILED, MissionStatus.CANCELLED}
    ),
    MissionStatus.COMPLETED: frozenset(),
    MissionStatus.FAILED: frozenset(),
    MissionStatus.CANCELLED: frozenset(),
}


_TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset(
        {TaskStatus.QUEUED, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.QUEUED: frozenset(
        {TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.IN_PROGRESS: frozenset(
        {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}


# Human-readable, state-derived descriptions. These are the single source
# of truth for the lifecycle prose carried in event payloads' ``message``
# field — a readable rendering of the typed status, not a substitute for it.
# The structured ``status`` value stays in the payload for machines; the
# message is for humans. Keep them generic (they describe the *state*, not a
# specific mission) so they stay consistent across every run.
_MISSION_MESSAGES: dict[MissionStatus, str] = {
    MissionStatus.DRAFT: "Mission created and awaiting submission.",
    MissionStatus.QUEUED: "Mission queued — waiting for resources to start.",
    MissionStatus.ACTIVE: "Mission running — executing tasks in order.",
    MissionStatus.COMPLETED: "Mission completed successfully.",
    MissionStatus.FAILED: "Mission failed — see task errors for details.",
    MissionStatus.CANCELLED: "Mission cancelled before completion.",
}

_TASK_MESSAGES: dict[TaskStatus, str] = {
    TaskStatus.PENDING: "Task pending — not yet scheduled.",
    TaskStatus.QUEUED: "Task queued — waiting to start.",
    TaskStatus.IN_PROGRESS: "Task running.",
    TaskStatus.COMPLETED: "Task completed successfully.",
    TaskStatus.FAILED: "Task failed.",
    TaskStatus.CANCELLED: "Task cancelled before completion.",
}


def mission_message(status: MissionStatus) -> str:
    """Return the human-readable description for a mission status."""
    return _MISSION_MESSAGES[status]


def task_message(status: TaskStatus) -> str:
    """Return the human-readable description for a task status."""
    return _TASK_MESSAGES[status]


def is_terminal_mission(status: MissionStatus) -> bool:
    return status in _TERMINAL_MISSION


def is_terminal_task(status: TaskStatus) -> bool:
    return status in _TERMINAL_TASK


def can_transition_mission(from_state: MissionStatus, to_state: MissionStatus) -> bool:
    return to_state in _MISSION_TRANSITIONS.get(from_state, frozenset())


def can_transition_task(from_state: TaskStatus, to_state: TaskStatus) -> bool:
    return to_state in _TASK_TRANSITIONS.get(from_state, frozenset())
