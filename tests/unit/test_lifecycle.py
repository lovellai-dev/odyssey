"""Tests for the mission + task state machines.

The transition tables in `engine/lifecycle.py` are load-bearing — every
other piece of the engine relies on them rejecting illegal transitions.
"""

from __future__ import annotations

import pytest

from odyssey.engine.lifecycle import (
    MissionStatus,
    TaskStatus,
    can_transition_mission,
    can_transition_task,
    is_terminal_mission,
    is_terminal_task,
)

# ---------------------------------------------------------------------------
# Terminal-state recognition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "status",
    [MissionStatus.COMPLETED, MissionStatus.FAILED, MissionStatus.CANCELLED],
)
def test_mission_terminal_states(status: MissionStatus) -> None:
    assert is_terminal_mission(status)


@pytest.mark.parametrize(
    "status",
    [MissionStatus.DRAFT, MissionStatus.QUEUED, MissionStatus.ACTIVE],
)
def test_mission_non_terminal_states(status: MissionStatus) -> None:
    assert not is_terminal_mission(status)


@pytest.mark.parametrize(
    "status",
    [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED],
)
def test_task_terminal_states(status: TaskStatus) -> None:
    assert is_terminal_task(status)


@pytest.mark.parametrize(
    "status",
    [TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.IN_PROGRESS],
)
def test_task_non_terminal_states(status: TaskStatus) -> None:
    assert not is_terminal_task(status)


# ---------------------------------------------------------------------------
# Mission transitions
# ---------------------------------------------------------------------------

_VALID_MISSION = [
    (MissionStatus.DRAFT, MissionStatus.QUEUED),
    (MissionStatus.DRAFT, MissionStatus.CANCELLED),
    (MissionStatus.QUEUED, MissionStatus.ACTIVE),
    (MissionStatus.QUEUED, MissionStatus.CANCELLED),
    (MissionStatus.ACTIVE, MissionStatus.COMPLETED),
    (MissionStatus.ACTIVE, MissionStatus.FAILED),
    (MissionStatus.ACTIVE, MissionStatus.CANCELLED),
]

_INVALID_MISSION = [
    (MissionStatus.DRAFT, MissionStatus.ACTIVE),       # skip QUEUED
    (MissionStatus.DRAFT, MissionStatus.COMPLETED),
    (MissionStatus.QUEUED, MissionStatus.COMPLETED),   # skip ACTIVE
    (MissionStatus.COMPLETED, MissionStatus.ACTIVE),   # terminal
    (MissionStatus.FAILED, MissionStatus.QUEUED),
    (MissionStatus.CANCELLED, MissionStatus.ACTIVE),
    (MissionStatus.ACTIVE, MissionStatus.DRAFT),       # never reverse
]


@pytest.mark.parametrize(("src", "dst"), _VALID_MISSION)
def test_mission_valid_transitions(src: MissionStatus, dst: MissionStatus) -> None:
    assert can_transition_mission(src, dst)


@pytest.mark.parametrize(("src", "dst"), _INVALID_MISSION)
def test_mission_invalid_transitions(src: MissionStatus, dst: MissionStatus) -> None:
    assert not can_transition_mission(src, dst)


def test_mission_same_state_is_not_a_transition() -> None:
    for s in MissionStatus:
        assert not can_transition_mission(s, s)


# ---------------------------------------------------------------------------
# Task transitions
# ---------------------------------------------------------------------------

_VALID_TASK = [
    (TaskStatus.PENDING, TaskStatus.QUEUED),
    (TaskStatus.PENDING, TaskStatus.CANCELLED),
    (TaskStatus.PENDING, TaskStatus.FAILED),
    (TaskStatus.QUEUED, TaskStatus.IN_PROGRESS),
    (TaskStatus.QUEUED, TaskStatus.CANCELLED),
    (TaskStatus.QUEUED, TaskStatus.FAILED),
    (TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED),
    (TaskStatus.IN_PROGRESS, TaskStatus.FAILED),
    (TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED),
]

_INVALID_TASK = [
    (TaskStatus.PENDING, TaskStatus.IN_PROGRESS),  # skip QUEUED
    (TaskStatus.PENDING, TaskStatus.COMPLETED),
    (TaskStatus.QUEUED, TaskStatus.COMPLETED),     # skip IN_PROGRESS
    (TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS),
    (TaskStatus.FAILED, TaskStatus.QUEUED),
    (TaskStatus.CANCELLED, TaskStatus.PENDING),
]


@pytest.mark.parametrize(("src", "dst"), _VALID_TASK)
def test_task_valid_transitions(src: TaskStatus, dst: TaskStatus) -> None:
    assert can_transition_task(src, dst)


@pytest.mark.parametrize(("src", "dst"), _INVALID_TASK)
def test_task_invalid_transitions(src: TaskStatus, dst: TaskStatus) -> None:
    assert not can_transition_task(src, dst)


def test_task_same_state_is_not_a_transition() -> None:
    for s in TaskStatus:
        assert not can_transition_task(s, s)
