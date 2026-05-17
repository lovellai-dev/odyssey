"""Engine exception hierarchy."""

from __future__ import annotations

from odyssey.engine.lifecycle import MissionStatus, TaskStatus


class MissionEngineError(Exception):
    """Base for all engine-level errors."""


class MissionNotFoundError(MissionEngineError):
    def __init__(self, mission_id: str):
        super().__init__(f"Mission not found: {mission_id!r}")
        self.mission_id = mission_id


class TaskNotFoundError(MissionEngineError):
    def __init__(self, task_id: str):
        super().__init__(f"Task not found: {task_id!r}")
        self.task_id = task_id


class InvalidStateTransitionError(MissionEngineError):
    """Raised when something tries to transition a mission or task to a
    state that the lifecycle rules forbid."""

    def __init__(
        self,
        what: str,
        from_state: MissionStatus | TaskStatus,
        to_state: MissionStatus | TaskStatus,
    ):
        super().__init__(
            f"Illegal {what} transition: {from_state.value} -> {to_state.value}"
        )
        self.what = what
        self.from_state = from_state
        self.to_state = to_state


class NoRunnerForTaskError(MissionEngineError):
    """No runner is registered for the (kind, type) pair of a task."""

    def __init__(self, kind: str, type_value: str):
        super().__init__(
            f"No runner registered for kind={kind!r} type={type_value!r}"
        )
        self.kind = kind
        self.type_value = type_value
