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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from odyssey.providers.registry import ProviderRegistry
from odyssey.spec.agents import AgentSpec
from odyssey.spec.tasks import TaskKind
from odyssey.telemetry.events import ProgressEvent, TaskEventType
from odyssey.telemetry.publishers.base import EventPublisher

if TYPE_CHECKING:
    # Annotation-only: importing engine.records at runtime initializes
    # the engine package, whose mission_engine imports this module right
    # back — a cycle that bites when a runner module is the entry
    # import. Guarded by tests/unit/test_imports.py.
    from odyssey.engine.records import MissionRun, TaskRun

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
    _progress_seq: int = field(default=0, init=False, repr=False)
    # Per-task working directory where checkpoints, logs, and other
    # artifacts should land. The engine creates this dir before invoking
    # the runner. None only in tests that don't go through the engine.
    output_dir: Path | None = None
    # Provider registry the engine was configured with. Runners that need
    # to resolve / fetch models or datasets should use this rather than
    # talking to providers directly. None when the engine was built
    # without providers (the CPU-mock-only test setup).
    providers: ProviderRegistry | None = None
    # The agent this training task updates. Set by the engine for
    # training tasks; None for evaluation tasks (which walk all agents
    # via ``mission.spec.robot.agents`` themselves).
    agent: AgentSpec | None = None
    # Local path to the checkpoint a training task should start from.
    # Set by the engine when a prior completed training task on the
    # same agent produced one. None for the first training round
    # against an agent — runners fall back to ``agent.model`` (the
    # agent's base checkpoint).
    starting_checkpoint: str | None = None

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
        self._progress_seq += 1
        event = ProgressEvent(
            mission_id=self.mission.id,
            task_id=self.task.id,
            task_name=self.task.spec.name,
            stage=stage,
            seq=self._progress_seq,
            step=step,
            step_index=step_index,
            step_total=step_total,
            step_label=step_label,
            metadata=metadata,
        )
        await self.publisher.publish(
            TaskEventType.PROGRESS.value,
            event.model_dump(exclude_none=True),
        )


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
