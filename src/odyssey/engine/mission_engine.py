"""MissionEngine — the orchestrator.

Drives a Mission spec through its lifecycle:

  DRAFT --create-> persisted record
        --start--> QUEUED --auto--> ACTIVE --execute tasks-->
                                              COMPLETED | FAILED | CANCELLED

Per design §5, the engine speaks to persistence, runners, and events only
through ABCs. The constructor takes those collaborators; the engine does
no I/O of its own.

Scope B caveats:
  * Tasks execute sequentially in spec order. Parallel execution comes
    when the design's ``execution.parallelism`` is honored.
  * No materialized profile loading yet — deadlines, quality gates, and
    evaluation predicates from the materializer are wired in Week 3+.
  * No watchdog timers; CC has them but they need the materialized profile.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from odyssey.engine.errors import (
    InvalidStateTransitionError,
    MissionNotFoundError,
    NoRunnerForTaskError,
)
from odyssey.engine.lifecycle import (
    MissionStatus,
    TaskStatus,
    can_transition_mission,
    can_transition_task,
    is_terminal_mission,
    is_terminal_task,
    mission_message,
    task_message,
)
from odyssey.engine.records import MissionRun, TaskRun
from odyssey.persistence.base import Persistence
from odyssey.providers.registry import ProviderRegistry
from odyssey.runners.base import TaskContext
from odyssey.runners.registry import RunnerRegistry
from odyssey.spec.agents import AgentSpec
from odyssey.spec.mission import Mission
from odyssey.spec.tasks import EvaluationTask, TaskSpec, TrainingTask
from odyssey.telemetry.events import MissionEventType, TaskEventType
from odyssey.telemetry.publishers.base import EventPublisher

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _mission_payload(run: MissionRun) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "mission_id": run.id,
        "name": run.spec.metadata.name,
        "status": run.status.value,
        # Human-readable rendering of ``status``; the typed value above is
        # what machines key off. See lifecycle._MISSION_MESSAGES.
        "message": mission_message(run.status),
    }
    if run.overall_grade is not None:
        payload["overall_grade"] = run.overall_grade
    if run.error_message:
        payload["error_message"] = run.error_message
    return payload


def _task_payload(mission: MissionRun, task: TaskRun) -> dict[str, Any]:
    return {
        "mission_id": mission.id,
        "task_id": task.id,
        "task_name": task.spec.name,
        "task_kind": task.spec.kind,
        "status": task.status.value,
        # Human-readable rendering of ``status``. See lifecycle._TASK_MESSAGES.
        "message": task_message(task.status),
    }


def _runner_override(spec: TaskSpec) -> str | None:
    """Task-level runner selection: ``config: {runner: <name>}``.

    Lets a mission disambiguate when several runners serve the same
    (kind, type) routing key — e.g. OpenVLA and GR00T are both wildcard
    training runners, and the model family isn't visible to the
    registry. Non-string or empty values are ignored.
    """
    value = spec.config.get("runner")
    if isinstance(value, str) and value:
        return value
    return None


class MissionEngine:
    def __init__(
        self,
        *,
        persistence: Persistence,
        runners: RunnerRegistry,
        event_publisher: EventPublisher,
        working_dir: Path | None = None,
        providers: ProviderRegistry | None = None,
        force_runner: str | None = None,
    ):
        self._persistence = persistence
        self._runners = runners
        self._publisher = event_publisher
        # When set, every task dispatches to this runner regardless of
        # task-level ``config: {runner: ...}`` overrides. The CLI sets
        # it to "cpu_mock" for --use-mock-runner.
        self._force_runner = force_runner
        # Per-mission working dirs live under here. Lazy-init to a tempdir
        # on first use if the caller didn't supply one — keeps tests cheap.
        self._working_dir = working_dir
        # Threaded into every TaskContext so runners can resolve+fetch
        # models / datasets without holding their own registry. Optional —
        # runners that don't need providers (CPU mock) ignore it.
        self._providers = providers
        # One cancel-event per active mission. Lets cancel_mission() reach
        # into a running start_mission() coroutine without coupling to it.
        self._cancel_events: dict[str, asyncio.Event] = {}

    async def initialize(self) -> None:
        await self._persistence.initialize()

    def _task_output_dir(self, mission_id: str, task_id: str) -> Path:
        if self._working_dir is None:
            self._working_dir = Path(tempfile.mkdtemp(prefix="odyssey-"))
        out = self._working_dir / mission_id / task_id
        out.mkdir(parents=True, exist_ok=True)
        return out

    # ------------------------------------------------------------------
    # Mission CRUD
    # ------------------------------------------------------------------

    async def create_mission(self, spec: Mission) -> MissionRun:
        run = MissionRun.from_spec(spec)

        # If a provider registry is wired, resolve the robot up front so
        # an unknown embodiment or missing URDF fails at create-time
        # rather than mid-execution. Engines built without providers
        # (test / CPU-mock paths) leave ``resolved_robot`` as None and
        # runners that need a resolved robot can fail later with a
        # clearer message than "no providers registered."
        if self._providers is not None:
            provider = self._providers.for_robot_spec(spec.robot)
            run.resolved_robot = await provider.resolve(spec.robot)

        await self._persistence.save_mission(run)
        await self._publisher.publish(
            MissionEventType.CREATED.value, _mission_payload(run)
        )
        return run

    async def get_mission(self, mission_id: str) -> MissionRun:
        run = await self._persistence.get_mission(mission_id)
        if run is None:
            raise MissionNotFoundError(mission_id)
        return run

    async def list_missions(
        self,
        *,
        status: MissionStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MissionRun]:
        return await self._persistence.list_missions(
            status=status.value if status else None,
            limit=limit,
            offset=offset,
        )

    async def delete_mission(self, mission_id: str) -> bool:
        return await self._persistence.delete_mission(mission_id)

    # ------------------------------------------------------------------
    # Mission lifecycle
    # ------------------------------------------------------------------

    async def start_mission(self, mission_id: str) -> MissionRun:
        """Drive the mission from DRAFT to a terminal state.

        Blocks until all tasks finish (or a failure/cancel cuts the run
        short). For background execution the caller wraps this in their
        own ``asyncio.create_task``.
        """
        run = await self.get_mission(mission_id)
        if run.status != MissionStatus.DRAFT:
            raise InvalidStateTransitionError(
                "mission", run.status, MissionStatus.QUEUED
            )

        cancel_event = asyncio.Event()
        self._cancel_events[mission_id] = cancel_event
        try:
            await self._transition_mission(
                run, MissionStatus.QUEUED, MissionEventType.QUEUED
            )
            run.started_at = _utcnow()
            await self._transition_mission(
                run, MissionStatus.ACTIVE, MissionEventType.STARTED
            )

            for task in run.tasks:
                if cancel_event.is_set():
                    break
                await self._execute_task(run, task, cancel_event)
                # Honor execution.on_task_failure — the spec default is
                # "stop", which short-circuits the remaining tasks. Anything
                # left PENDING is finalized in _finalize_mission.
                if (
                    task.status == TaskStatus.FAILED
                    and run.spec.execution.on_task_failure == "stop"
                ):
                    break

            await self._finalize_mission(run, cancel_event)
        finally:
            self._cancel_events.pop(mission_id, None)

        return run

    async def cancel_mission(self, mission_id: str) -> MissionRun:
        run = await self.get_mission(mission_id)
        if is_terminal_mission(run.status):
            return run

        ev = self._cancel_events.get(mission_id)
        if ev is not None:
            ev.set()

        # Mark all non-terminal tasks CANCELLED so the persisted state
        # reflects reality even if start_mission isn't currently running.
        for task in run.tasks:
            if not is_terminal_task(task.status):
                task.status = TaskStatus.CANCELLED
                task.completed_at = _utcnow()

        run.status = MissionStatus.CANCELLED
        run.completed_at = _utcnow()
        await self._persistence.save_mission(run)
        await self._publisher.publish(
            MissionEventType.CANCELLED.value, _mission_payload(run)
        )
        return run

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    async def _execute_task(
        self,
        mission: MissionRun,
        task: TaskRun,
        cancel_event: asyncio.Event,
    ) -> None:
        await self._transition_task(
            mission, task, TaskStatus.QUEUED, TaskEventType.QUEUED
        )
        task.started_at = _utcnow()
        await self._transition_task(
            mission, task, TaskStatus.IN_PROGRESS, TaskEventType.STARTED
        )
        await self._persistence.update_task(
            mission.id, task.id, started_at=task.started_at
        )

        try:
            runner = self._runners.select(
                task.spec,
                override=self._force_runner or _runner_override(task.spec),
            )
        except NoRunnerForTaskError as e:
            await self._finalize_task(
                mission,
                task,
                TaskStatus.FAILED,
                error_code="no_runner_registered",
                error_message=str(e),
            )
            return

        # Resolve agent context depending on task kind.
        agent = None
        starting_checkpoint = None
        agents: list[AgentSpec] = []
        agent_checkpoints: dict[str, str | None] = {}

        if isinstance(task.spec, TrainingTask):
            # Training: resolve the single target agent + its starting
            # checkpoint so the runner doesn't walk the per-agent chain.
            agent = mission.agent_by_id(task.spec.agent_id)
            starting_checkpoint = mission.latest_checkpoint_for(task.spec.agent_id)
        elif isinstance(task.spec, EvaluationTask):
            # Evaluation: resolve ALL agents + their checkpoints so the
            # runner can compose multi-agent runtimes (planner + pilot).
            agents = list(mission.spec.robot.agents)
            agent_checkpoints = {
                a.id: mission.latest_checkpoint_for(a.id)
                for a in mission.spec.robot.agents
            }

        ctx = TaskContext(
            task=task,
            mission=mission,
            publisher=self._publisher,
            cancel_event=cancel_event,
            output_dir=self._task_output_dir(mission.id, task.id),
            providers=self._providers,
            agent=agent,
            starting_checkpoint=starting_checkpoint,
            agents=agents,
            agent_checkpoints=agent_checkpoints,
        )

        try:
            result = await runner.run(ctx)
        except Exception as e:
            logger.exception(
                "Runner %r raised for task %s — marking FAILED",
                runner.name,
                task.id,
            )
            await self._finalize_task(
                mission,
                task,
                TaskStatus.FAILED,
                error_code="runner_exception",
                error_message=str(e)[:500],
            )
            return

        if cancel_event.is_set():
            await self._finalize_task(
                mission,
                task,
                TaskStatus.CANCELLED,
                error_code="cancelled",
                result_summary=result or {},
            )
            return

        await self._finalize_task(
            mission,
            task,
            TaskStatus.COMPLETED,
            result_summary=result or {},
        )

    async def _finalize_mission(
        self,
        run: MissionRun,
        cancel_event: asyncio.Event,
    ) -> None:
        if cancel_event.is_set():
            terminal = MissionStatus.CANCELLED
            event = MissionEventType.CANCELLED
            # Any task that never had a chance to run should be persisted as
            # CANCELLED. Without this, _finalize_mission's full-record save
            # below would overwrite cancel_mission's bookkeeping with the
            # stale PENDING values from this coroutine's local copy.
            for task in run.tasks:
                if not is_terminal_task(task.status):
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = _utcnow()
        elif any(t.status == TaskStatus.FAILED for t in run.tasks):
            terminal = MissionStatus.FAILED
            event = MissionEventType.FAILED
        else:
            terminal = MissionStatus.COMPLETED
            event = MissionEventType.COMPLETED
            scores = [
                t.result_summary["performance_score"]
                for t in run.tasks
                if isinstance(t.spec, EvaluationTask)
                and isinstance(
                    t.result_summary.get("performance_score"), (int, float)
                )
            ]
            if scores:
                run.overall_grade = sum(scores) / len(scores)

        run.status = terminal
        run.completed_at = _utcnow()
        await self._persistence.save_mission(run)
        await self._publisher.publish(event.value, _mission_payload(run))

    # ------------------------------------------------------------------
    # Transition helpers
    # ------------------------------------------------------------------

    async def _transition_mission(
        self,
        run: MissionRun,
        new_status: MissionStatus,
        event_type: MissionEventType,
    ) -> None:
        if not can_transition_mission(run.status, new_status):
            raise InvalidStateTransitionError("mission", run.status, new_status)
        run.status = new_status
        await self._persistence.save_mission(run)
        await self._publisher.publish(event_type.value, _mission_payload(run))

    async def _transition_task(
        self,
        mission: MissionRun,
        task: TaskRun,
        new_status: TaskStatus,
        event_type: TaskEventType,
    ) -> None:
        if not can_transition_task(task.status, new_status):
            raise InvalidStateTransitionError("task", task.status, new_status)
        task.status = new_status
        await self._persistence.update_task(mission.id, task.id, status=new_status)
        await self._publisher.publish(event_type.value, _task_payload(mission, task))

    async def _finalize_task(
        self,
        mission: MissionRun,
        task: TaskRun,
        terminal_status: TaskStatus,
        *,
        error_code: str | None = None,
        error_message: str | None = None,
        result_summary: dict[str, Any] | None = None,
    ) -> None:
        task.status = terminal_status
        task.completed_at = _utcnow()
        if error_code is not None:
            task.error_code = error_code
        if error_message is not None:
            task.error_message = error_message
        if result_summary is not None:
            task.result_summary = result_summary

        update_fields: dict[str, Any] = {
            "status": terminal_status,
            "completed_at": task.completed_at,
        }
        if error_code is not None:
            update_fields["error_code"] = error_code
        if error_message is not None:
            update_fields["error_message"] = error_message
        if result_summary is not None:
            update_fields["result_summary"] = result_summary
        await self._persistence.update_task(mission.id, task.id, **update_fields)

        event_type = {
            TaskStatus.COMPLETED: TaskEventType.COMPLETED,
            TaskStatus.FAILED: TaskEventType.FAILED,
            TaskStatus.CANCELLED: TaskEventType.CANCELLED,
        }[terminal_status]
        payload = _task_payload(mission, task)
        if result_summary:
            payload["result_summary"] = result_summary
        if error_message:
            payload["error_message"] = error_message
        await self._publisher.publish(event_type.value, payload)
