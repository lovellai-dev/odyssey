"""Tests for F15 — graceful SIGTERM/SIGINT shutdown.

The CLI ran a mission with no signal handlers, so a Cloud Run revision swap or a
preemptible-VM stop (SIGTERM) or a Ctrl-C (SIGINT) just killed the process — the
in-flight GPU subprocess was orphaned and the mission was left ACTIVE. Fix:
`MissionEngine.request_cancel` (signal-safe) sets the mission's cancel_event, and
the CLI installs SIGTERM/SIGINT handlers that call it, so start_mission finalizes
the run CANCELLED and _watch_cancel SIGTERMs the child.
"""
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

from odyssey.engine.mission_engine import MissionEngine

_RUN_PY = (
    Path(__file__).resolve().parents[1]
    / "src" / "odyssey" / "cli" / "commands" / "run.py"
)


def _engine() -> MissionEngine:
    return MissionEngine(
        persistence=MagicMock(), runners=MagicMock(), event_publisher=MagicMock()
    )


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


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
