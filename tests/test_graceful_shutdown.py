"""Tests for F15 — graceful SIGTERM/SIGINT shutdown.

The CLI ran a mission with no signal handlers, so a Cloud Run revision swap or a
preemptible-VM stop (SIGTERM) or a Ctrl-C (SIGINT) just killed the process — the
in-flight GPU subprocess was orphaned and the mission was left ACTIVE. Fix:
`MissionEngine.request_cancel` (signal-safe) sets the mission's cancel_event, and
the CLI installs SIGTERM/SIGINT handlers that call it, so start_mission finalizes
the run CANCELLED and _watch_cancel SIGTERMs the child.
"""
import asyncio
import os
import signal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from odyssey.cli.commands.run import _run_mission
from odyssey.engine import MissionEngine, MissionStatus
from odyssey.persistence import InMemoryPersistence
from odyssey.runners import WILDCARD_TYPE, CPUMockRunner, Runner, RunnerRegistry, TaskContext
from odyssey.spec import TaskKind, load_mission
from odyssey.telemetry import EventPublisher, MissionEventType

_RUN_PY = (
    Path(__file__).resolve().parents[1]
    / "src" / "odyssey" / "cli" / "commands" / "run.py"
)
_EXAMPLE_MISSION = (
    Path(__file__).resolve().parents[1]
    / "examples" / "quickstart-openvla" / "mission.yaml"
)


def _engine() -> MissionEngine:
    return MissionEngine(
        persistence=MagicMock(), runners=MagicMock(), event_publisher=MagicMock()
    )


# --- Behavioral end-to-end fixtures (per SoyGema's review of #50) -----------

class _CapturingPublisher(EventPublisher):
    """Records published events so tests can assert the CANCELLED transition."""

    def __init__(self) -> None:
        self.events: list[str] = []

    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append(event_type)


class _BlockingRunner(Runner):
    """Blocks on ctx.cancel_event until cancelled — lets a mission sit ACTIVE so
    we can exercise the real OS-signal → handler → engine → CANCELLED seam."""

    @property
    def name(self) -> str:
        return "blocking"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.TRAINING, TaskKind.EVALUATION}

    @property
    def supported_types(self) -> set[str]:
        return {WILDCARD_TYPE}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        await context.emit_progress("blocking")
        await asyncio.wait_for(context.cancel_event.wait(), timeout=5.0)
        return {"_cancelled": True}


async def _build_engine(
    runner: Runner, *, force_runner: str | None = None
) -> tuple[MissionEngine, _CapturingPublisher]:
    pub = _CapturingPublisher()
    engine = MissionEngine(
        persistence=InMemoryPersistence(),
        runners=_registry(runner),
        event_publisher=pub,
        force_runner=force_runner,
    )
    await engine.initialize()
    return engine, pub


def _registry(runner: Runner) -> RunnerRegistry:
    reg = RunnerRegistry()
    reg.register(runner)
    return reg


def test_request_cancel_sets_event_for_running_mission():
    engine = _engine()
    ev = asyncio.Event()
    engine._cancel_events["m1"] = ev
    assert engine.request_cancel("m1") is True
    assert ev.is_set()


def test_request_cancel_unknown_mission_returns_false():
    engine = _engine()
    assert engine.request_cancel("nope") is False


def test_request_cancel_is_synchronous():
    # Must be safe to call from an OS signal handler: no coroutine, no await.
    assert not asyncio.iscoroutinefunction(MissionEngine.request_cancel)


def test_cli_installs_sigterm_sigint_handlers_calling_request_cancel():
    src = _RUN_PY.read_text()
    body = src[src.index("async def _run_mission("):]
    assert "add_signal_handler" in body
    assert "request_cancel" in body
    assert "SIGTERM" in body and "SIGINT" in body
    # Handlers are torn down after the run (no leaked global state).
    assert "remove_signal_handler" in body


# --- Behavioral: the real OS-signal → handler → engine → CANCELLED seam --------

async def test_request_cancel_during_run_finalizes_cancelled():
    """The core contract: cancelling a running mission finalizes it CANCELLED
    (persisted) and publishes a CANCELLED event — not just sets a flag."""
    engine, pub = await _build_engine(_BlockingRunner())
    run = await engine.create_mission(load_mission(_EXAMPLE_MISSION))
    task = asyncio.create_task(engine.start_mission(run.id))
    await asyncio.sleep(0.05)                       # reach ACTIVE + the blocking runner
    assert engine.request_cancel(run.id) is True
    result = await task
    assert result.status == MissionStatus.CANCELLED
    assert MissionEventType.CANCELLED.value in pub.events


async def test_installed_sigint_handler_cancels_the_run():
    """A REAL SIGINT delivered through the handler _run_mission installs cancels
    the mission — exercises add_signal_handler for real, not by grep."""
    engine, _ = await _build_engine(_BlockingRunner())
    run_task = asyncio.create_task(_run_mission(engine, load_mission(_EXAMPLE_MISSION)))
    await asyncio.sleep(0.1)                        # let it install handlers + block
    os.kill(os.getpid(), signal.SIGINT)            # delivered to the asyncio handler
    result = await run_task
    assert result.status == MissionStatus.CANCELLED


async def test_signal_handlers_removed_after_run():
    """The finally tears the handlers down — no leaked global signal disposition.
    remove_signal_handler returns False when nothing was registered."""
    engine, _ = await _build_engine(CPUMockRunner(), force_runner="cpu_mock")
    await _run_mission(engine, load_mission(_EXAMPLE_MISSION))
    loop = asyncio.get_running_loop()
    assert loop.remove_signal_handler(signal.SIGTERM) is False
    assert loop.remove_signal_handler(signal.SIGINT) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
