"""Subprocess training-runner infrastructure.

Adapted from ``lai-inference/.../jobs/training/subprocess_runner.py`` —
the same pattern both GR00T and openpi use upstream:

  * Tyro / draccus / argparse CLI invoked as ``python -m <entry> <args>``
    or ``python <script> <args>``.
  * Step-based training loop logging ``Step N: loss=...`` or similar.
  * Periodic checkpoint saves.

This module owns the subprocess lifecycle (launch, stdout-line streaming,
SIGTERM-on-cancel, exit-code interpretation) so per-family runners only
have to provide an argv builder and an optional stdout line parser.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from odyssey.runners.base import TaskContext

logger = logging.getLogger(__name__)


LineParser = Callable[[str], dict[str, Any] | None]
"""Family-specific stdout line parser.

Receives one stdout line at a time. Returns a kwargs dict that's passed
to ``ctx.emit_progress(**kwargs)``, or ``None`` to skip emitting.
"""


@dataclass
class TrainingProcessSpec:
    """How to launch a specific family's training subprocess.

    Also used by subprocess-shaped evaluation runners (Isaac Lab).

    Exactly one of ``entry_module`` or ``script_path`` must be set:
      * ``entry_module`` invokes ``python -m <module>`` (suitable for
        packages like ``gr00t.experiment.launch_train`` or
        ``openpi.scripts.train``).
      * ``script_path`` invokes ``python <path>`` (used by openvla, whose
        ``vla-scripts/finetune.py`` isn't importable as a module name due
        to the hyphen).
    """

    entry_module: str | None = None
    script_path: str | None = None
    argv_extra: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    line_parser: LineParser | None = None
    cwd: str | None = None
    sigterm_grace_seconds: float = 30.0
    use_torchrun: bool = False
    torchrun_nproc: int = 1
    # Optional command prefix replacing the default ``python`` for
    # ``script_path`` invocations — e.g. ["/path/isaaclab.sh", "-p"] so the
    # script runs under Isaac Sim's bundled interpreter. Not valid with
    # ``entry_module`` (the ``-m`` form assumes a python launcher), and
    # mutually exclusive with ``use_torchrun`` (both define the prefix).
    launcher: list[str] | None = None

    def __post_init__(self) -> None:
        if (self.entry_module is None) == (self.script_path is None):
            raise ValueError(
                "TrainingProcessSpec requires exactly one of "
                "entry_module or script_path"
            )
        if self.launcher is not None and self.entry_module is not None:
            raise ValueError(
                "TrainingProcessSpec.launcher only applies to script_path "
                "invocations"
            )
        if self.launcher is not None and self.use_torchrun:
            raise ValueError(
                "TrainingProcessSpec.launcher and use_torchrun are mutually "
                "exclusive — both define the command prefix"
            )


async def run_training_subprocess(
    ctx: TaskContext, spec: TrainingProcessSpec
) -> int:
    """Launch the configured subprocess, stream stdout into progress
    events, watch for cancellation, return the exit code.

    The caller (a family-specific runner like ``openvla.run``) is
    responsible for building ``argv_extra``, picking the entry, and
    post-processing the resulting output directory.
    """
    if spec.launcher is not None:
        launcher = spec.launcher
    elif spec.use_torchrun:
        launcher = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={spec.torchrun_nproc}",
        ]
    else:
        launcher = ["python"]

    if spec.entry_module is not None:
        cmd = [*launcher, "-m", spec.entry_module, *spec.argv_extra]
    else:
        # script_path is guaranteed by __post_init__
        assert spec.script_path is not None
        cmd = [*launcher, spec.script_path, *spec.argv_extra]
    env = {**os.environ, **spec.env}

    logger.info(
        "Launching subprocess for task %s: %s",
        ctx.task.id,
        " ".join(cmd),
    )

    await ctx.emit_progress("executing", step="subprocess_start")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,  # merge to one stream
        env=env,
        cwd=spec.cwd,
        # New process group so SIGTERM targets only this child tree.
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        # tqdm progress bars use \r without \n, which can accumulate
        # into a single "line" that exceeds asyncio's default 64 KB
        # buffer.  10 MB is enough for long training runs.
        limit=10 * 1024 * 1024,
    )

    stdout_task = asyncio.create_task(
        _stream_stdout(ctx, proc, spec.line_parser),
        name=f"stdout-{ctx.task.id}",
    )
    cancel_task = asyncio.create_task(
        _watch_cancel(ctx, proc, grace=spec.sigterm_grace_seconds),
        name=f"cancel-watch-{ctx.task.id}",
    )

    try:
        rc = await proc.wait()
    finally:
        cancel_task.cancel()
        with suppress(asyncio.CancelledError):
            await cancel_task
        await stdout_task  # drain any remaining lines

    logger.info("Subprocess for task %s exited rc=%d", ctx.task.id, rc)
    return rc


def output_path(ctx: TaskContext, *parts: str) -> Path:
    """Convenience: build a path under the task's output_dir.

    Raises ``RuntimeError`` if the engine didn't supply one — almost
    always a sign that the runner is being invoked outside the engine.
    """
    if ctx.output_dir is None:
        raise RuntimeError(
            "TaskContext.output_dir is None — runner needs the engine "
            "to provide a working directory."
        )
    return ctx.output_dir.joinpath(*parts)


# ---------------------------------------------------------------------------
# stdout streaming
# ---------------------------------------------------------------------------

# Generic step + checkpoint patterns. Per-family parsers override; this
# fallback at least emits per-step heartbeats from typical training logs.
_GENERIC_STEP_RE = re.compile(r"\bstep[\s:]+(\d+)", re.IGNORECASE)
_CHECKPOINT_RE = re.compile(r"(checkpoint|save_state).*?(\d+)", re.IGNORECASE)


async def _stream_stdout(
    ctx: TaskContext,
    proc: asyncio.subprocess.Process,
    line_parser: LineParser | None,
) -> None:
    """Read lines from the child, emit progress events, mirror to logs.

    Best-effort: a single bad line doesn't fail the run. Every line goes
    to the parent log prefixed with the task id so operators can debug.
    """
    if proc.stdout is None:
        return
    last_step_emitted = -1
    last_checkpoint_emitted = -1
    buf = ""
    while True:
        try:
            raw = await proc.stdout.read(8192)
        except Exception:
            logger.exception(
                "Error reading subprocess stdout for task %s", ctx.task.id
            )
            break
        if not raw:
            break
        # Write raw bytes directly to terminal for real-time tqdm output
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()
        # Also parse lines for structured events
        buf += raw.decode("utf-8", errors="replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                continue
            logger.info("[%s] %s", ctx.task.id, line)

            parsed: dict[str, Any] | None = None
            if line_parser is not None:
                try:
                    parsed = line_parser(line)
                except Exception:
                    logger.exception(
                        "line_parser raised for task %s — using fallback", ctx.task.id
                    )

            if parsed is None:
                m = _GENERIC_STEP_RE.search(line)
                if m:
                    step = int(m.group(1))
                    if step != last_step_emitted:
                        last_step_emitted = step
                        parsed = {
                            "stage": "executing",
                            "step": "training_step",
                            "step_index": step,
                        }
                elif _CHECKPOINT_RE.search(line):
                    m2 = _CHECKPOINT_RE.search(line)
                    assert m2 is not None  # we just matched
                    step = int(m2.group(2))
                    if step != last_checkpoint_emitted:
                        last_checkpoint_emitted = step
                        parsed = {
                            "stage": "checkpoint_saving",
                            "step": "checkpoint_save",
                            "step_index": step,
                        }

            if parsed is None:
                continue

            try:
                await ctx.emit_progress(**parsed)
            except Exception:
                logger.exception(
                    "emit_progress failed for task %s — continuing", ctx.task.id
                )


# ---------------------------------------------------------------------------
# Cancellation watcher
# ---------------------------------------------------------------------------

async def _watch_cancel(
    ctx: TaskContext, proc: asyncio.subprocess.Process, grace: float
) -> None:
    """Wait for ctx.cancel_event; on trip, SIGTERM the child group;
    escalate to SIGKILL after `grace` seconds. Returns when the child
    has exited.
    """
    await ctx.cancel_event.wait()
    if proc.returncode is not None:
        return
    logger.warning(
        "Cancellation requested for task %s — SIGTERM pid=%s",
        ctx.task.id,
        proc.pid,
    )
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace)
        return
    except asyncio.TimeoutError:
        logger.warning(
            "Subprocess for task %s did not exit within %.0fs of SIGTERM — SIGKILL",
            ctx.task.id,
            grace,
        )
    with suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
