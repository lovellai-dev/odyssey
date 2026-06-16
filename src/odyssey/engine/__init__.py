"""MissionEngine — the orchestrator and its runtime types."""

from typing import TYPE_CHECKING, Any

from odyssey.engine.errors import (
    InvalidStateTransitionError,
    MissionEngineError,
    MissionNotFoundError,
    NoRunnerForTaskError,
    TaskNotFoundError,
)
from odyssey.engine.lifecycle import (
    MissionStatus,
    TaskStatus,
    can_transition_mission,
    can_transition_task,
    is_terminal_mission,
    is_terminal_task,
)
from odyssey.engine.records import MissionRun, TaskRun

if TYPE_CHECKING:
    from odyssey.engine.mission_engine import MissionEngine


def __getattr__(name: str) -> Any:
    """Lazily expose ``MissionEngine`` (PEP 562).

    ``mission_engine`` imports from ``odyssey.runners`` (TaskContext,
    RunnerRegistry), and ``runners`` imports back from ``odyssey.engine``
    (errors/records). Importing it eagerly here makes *any* import of an
    ``odyssey.engine`` submodule drag in ``mission_engine`` and deadlock the
    runners<->engine cycle — fatal when the out-of-process planner_server is
    launched via ``python -m odyssey.runners.agents.planner_server``. Deferring
    it until first access breaks the cycle for every entry point.
    """
    if name == "MissionEngine":
        from odyssey.engine.mission_engine import MissionEngine

        return MissionEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "InvalidStateTransitionError",
    "MissionEngine",
    "MissionEngineError",
    "MissionNotFoundError",
    "MissionRun",
    "MissionStatus",
    "NoRunnerForTaskError",
    "TaskNotFoundError",
    "TaskRun",
    "TaskStatus",
    "can_transition_mission",
    "can_transition_task",
    "is_terminal_mission",
    "is_terminal_task",
]
