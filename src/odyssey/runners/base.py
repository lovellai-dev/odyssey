"""Runner ABC + TaskContext.

A runner executes one task end-to-end. It receives a `TaskContext` from
the engine — the only object it needs to emit progress events, check
cancellation, and read task metadata.

`run()` returns a `result_summary` dict that is persisted on the task and
echoed in the terminal `task.completed` event.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from odyssey.engine.records import MissionRun, TaskRun
from odyssey.spec.tasks import TaskKind
from odyssey.telemetry.events import TaskEventType
from odyssey.telemetry.publishers.base import EventPublisher

# Sentinel meaning "this runner accepts any training_type / evaluation_type
# value." Used by CPUMockRunner so a single registration covers every task
# spec the registry might dispatch.
WILDCARD_TYPE = "*"


@dataclass
class TaskContext:
    """Everything a runner sees when it executes one task."""

    task: TaskRun
    mission: MissionRun
    publisher: EventPublisher
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def request_cancel(self) -> None:
        self.cancel_event.set()

    async def emit_progress(
        self,
        stage: str,
        *,
        step: str | None = None,
        step_index: int | None = None,
        step_total: int | None = None,
        step_label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "mission_id": self.mission.id,
            "task_id": self.task.id,
            "task_name": self.task.spec.name,
            "stage": stage,
        }
        if step is not None:
            payload["step"] = step
        if step_index is not None:
            payload["step_index"] = step_index
        if step_total is not None:
            payload["step_total"] = step_total
        if step_label is not None:
            payload["step_label"] = step_label
        if metadata:
            payload["metadata"] = metadata
        await self.publisher.publish(TaskEventType.PROGRESS.value, payload)


class Runner(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def supported_kinds(self) -> set[TaskKind]: ...

    @property
    @abstractmethod
    def supported_types(self) -> set[str]:
        """Training-type or evaluation-type values this runner can execute.

        A runner that accepts every type within its supported kinds should
        return ``{WILDCARD_TYPE}``.
        """

    @abstractmethod
    async def run(self, context: TaskContext) -> dict[str, Any]:
        """Execute the task. Returns a result_summary dict.

        Should check ``context.cancelled()`` at safe boundaries and exit
        cleanly when set. Raises only on unrecoverable conditions; routine
        failures should be surfaced via the returned summary or by raising
        a descriptive exception the engine converts into FAILED status.
        """
