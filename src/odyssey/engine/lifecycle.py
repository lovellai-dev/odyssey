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


def is_terminal_mission(status: MissionStatus) -> bool:
    return status in _TERMINAL_MISSION


def is_terminal_task(status: TaskStatus) -> bool:
    return status in _TERMINAL_TASK


def can_transition_mission(from_state: MissionStatus, to_state: MissionStatus) -> bool:
    return to_state in _MISSION_TRANSITIONS.get(from_state, frozenset())


def can_transition_task(from_state: TaskStatus, to_state: TaskStatus) -> bool:
    return to_state in _TASK_TRANSITIONS.get(from_state, frozenset())
