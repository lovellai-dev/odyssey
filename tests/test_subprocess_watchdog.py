"""Tests for F14/F16/F58 — the subprocess runner's watchdog + reap.

`run_training_subprocess` awaited `proc.wait()` with no timeout, so a hung eval
(Isaac boot, a ZMQ handshake to a policy server that never replies, a wedged
CUDA context) blocked the whole mission forever. The task schema already carried
a dormant `timeout_seconds`; `_wait_with_deadline` now enforces it: SIGTERM the
child's process group on deadline, escalate to SIGKILL, then raise
`SubprocessTimeoutError`.
"""
import asyncio
import os
import time
from unittest.mock import MagicMock

import pytest

from odyssey.runners.subprocess import (
    SubprocessTimeoutError,
    TrainingProcessSpec,
    _wait_with_deadline,
)


def _ctx():
    ctx = MagicMock()
    ctx.task.id = "t1"
    return ctx


async def _spawn(*argv):
    return await asyncio.create_subprocess_exec(
        *argv,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


async def test_deadline_kills_hung_subprocess():
    """A child that outlives its deadline is killed and raises SubprocessTimeoutError."""
    proc = await _spawn("sleep", "30")
    spec = TrainingProcessSpec(
        entry_module="x", timeout_seconds=0.3, sigterm_grace_seconds=0.5
    )

    started = time.monotonic()
    with pytest.raises(SubprocessTimeoutError):
        await _wait_with_deadline(_ctx(), proc, spec)

    # Killed promptly (well under the child's 30s), and reaped.
    assert time.monotonic() - started < 5.0
    assert proc.returncode is not None


async def test_no_deadline_waits_to_completion():
    """With timeout_seconds=None the wait is unbounded (legacy behavior)."""
    proc = await _spawn("true")
    spec = TrainingProcessSpec(entry_module="x")  # no deadline
    rc = await _wait_with_deadline(_ctx(), proc, spec)
    assert rc == 0


async def test_completes_before_deadline_returns_rc():
    """A child that finishes within its deadline returns its exit code, no kill."""
    proc = await _spawn("true")
    spec = TrainingProcessSpec(entry_module="x", timeout_seconds=10)
    rc = await _wait_with_deadline(_ctx(), proc, spec)
    assert rc == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
