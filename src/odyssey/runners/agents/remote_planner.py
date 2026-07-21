"""RemotePlanner — drives an out-of-process SPECIALIST planner.

Implements the ``PlannerRuntime`` protocol but, instead of loading Gemma in
this process, launches ``planner_server`` in a *separate* venv/process. That
frees the planner from OpenVLA's pinned ``transformers==4.40.1`` (which lives
in this, the main, venv) so it can host the multimodal Gemma 4 planner (which
needs a modern ``transformers`` + ``torchvision``).

Communicates over the JSON-lines stdin/stdout protocol documented in
``planner_server``. The subprocess (and the model) starts lazily on the first
``plan()`` call and is reused across episodes — the planner runs once per
episode, so this is off the per-step hot loop. Robust by design: non-JSON
stdout lines are skipped, and any failure falls back to ``[instruction]``
(the same single-step fallback ``LLMPlanner`` uses).
"""

from __future__ import annotations

import atexit
import base64
import contextlib
import io
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from typing import IO, Any

from odyssey.runners.agents.runtime import RouteDecision

logger = logging.getLogger(__name__)

_SERVER_MODULE = "odyssey.runners.agents.planner_server"
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


def _popen_creation_kwargs() -> dict[str, Any]:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_process(proc: subprocess.Popen[str], sig: int) -> None:
    killpg = getattr(os, "killpg", None)
    getpgid = getattr(os, "getpgid", None)
    if killpg is not None and getpgid is not None:
        killpg(getpgid(proc.pid), sig)
    elif sig == signal.SIGTERM:
        proc.terminate()
    else:
        proc.kill()


def _encode_image(image: Any) -> str:
    """Encode a PIL Image or HWC uint8 ndarray as a base64 PNG string.

    Runs once per episode (off the per-step hot loop), so PNG's cost is
    irrelevant and it keeps the JSON-lines channel ASCII-clean.
    """
    from PIL import Image

    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class RemotePlanner:
    """Out-of-process planner. Satisfies the ``PlannerRuntime`` protocol.

    Parameters
    ----------
    model_base:
        HF id of the SPECIALIST model (e.g. ``google/gemma-4-E2B-it``). The
        server hosts it via the multimodal ``GemmaVLMGenerator``; ``plan`` ships
        the scene image alongside the instruction when one is provided.
    quantization:
        Quantization string passed through to the server (e.g. ``int4``) or None.
    python_path:
        Path to the *specialist* venv's python interpreter (from
        ``ODYSSEY_SPECIALIST_PYTHON``).
    startup_timeout:
        Seconds to wait for ``{"ready": true}`` — the first run downloads the
        model, so this is generous.
    request_timeout:
        Seconds to wait for a single plan response.
    check_timeout:
        Seconds to wait for a single completion-check response. Short by design
        (the check runs on the per-step loop, every K steps): a hung check
        falls back to ``False`` rather than freezing the episode.
    ground_timeout:
        Seconds to wait for a single grounding response. Grounding runs only at
        phase boundaries (not every step), so it can afford a longer budget than
        the completion check; a hung grounding call falls back to echoing the
        query rather than freezing the episode.
    launch_args:
        Argv (after ``python_path``) that starts the server. Defaults to
        ``("-m", planner_server)``; overridable for tests.
    """

    def __init__(
        self,
        model_base: str,
        quantization: str | None = None,
        *,
        python_path: str,
        startup_timeout: float = 600.0,
        request_timeout: float = 120.0,
        check_timeout: float = 5.0,
        ground_timeout: float = 15.0,
        route_timeout: float = 15.0,
        launch_args: Sequence[str] = ("-m", _SERVER_MODULE),
    ) -> None:
        self._model_base = model_base
        self._quantization = quantization
        self._python_path = python_path
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._check_timeout = check_timeout
        self._ground_timeout = ground_timeout
        self._route_timeout = route_timeout
        self._launch_args = list(launch_args)
        self._proc: subprocess.Popen[str] | None = None
        # A background thread drains stdout into this queue. We avoid
        # select()+readline() because buffered readline can pull several lines
        # into Python's buffer while select() only sees the OS fd, stalling on
        # data that's already been read. The queue gives clean per-get timeouts.
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        atexit.register(self.close)

    def _ensure_started(self) -> None:
        if self._proc is not None:
            return
        argv = [self._python_path, *self._launch_args, "--model", self._model_base]
        if self._quantization:
            argv += ["--quantization", self._quantization]
        logger.info("RemotePlanner: launching specialist server: %s", " ".join(argv))
        # stderr inherited -> model-loading logs stay visible in the terminal.
        # POSIX gets its own process group; Windows falls back to direct cleanup.
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            **_popen_creation_kwargs(),
        )
        assert self._proc.stdout is not None
        self._reader = threading.Thread(
            target=self._drain_stdout, args=(self._proc.stdout,), daemon=True
        )
        self._reader.start()
        msg = self._read_message(self._startup_timeout)
        if not msg or not msg.get("ready"):
            err = (msg or {}).get("error", "no ready signal / server exited")
            self.close()
            raise RuntimeError(f"specialist planner failed to start: {err}")

    def _drain_stdout(self, stdout: IO[str]) -> None:
        """Reader thread: push each stdout line onto the queue; None on EOF."""
        try:
            for line in stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)

    def _read_message(self, timeout: float) -> dict[str, Any] | None:
        """Return the next protocol JSON object, skipping noise.

        Returns None on timeout or server EOF.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("RemotePlanner: timed out waiting for server response")
                return None
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                logger.warning("RemotePlanner: timed out waiting for server response")
                return None
            if line is None:
                logger.warning("RemotePlanner: server stdout closed (process exited?)")
                return None
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("RemotePlanner: skipping non-JSON stdout line: %s", line[:200])
                continue
            if isinstance(obj, dict):
                return obj

    def plan(self, task_instruction: str, image: Any | None = None) -> list[str]:
        """Decompose a task instruction via the out-of-process planner.

        When ``image`` is given (and the server is multimodal), it is sent as
        a base64 PNG alongside the instruction. Falls back to
        ``[task_instruction]`` on any error (matches LLMPlanner).
        """
        try:
            self._ensure_started()
            proc = self._proc
            assert proc is not None and proc.stdin is not None
            request: dict[str, Any] = {"instruction": task_instruction}
            if image is not None:
                request["image"] = _encode_image(image)
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            msg = self._read_message(self._request_timeout)
            plan = (msg or {}).get("plan")
            if isinstance(plan, list) and plan:
                return [str(s) for s in plan]
            logger.warning(
                "RemotePlanner: bad/empty response %r for %r — single-step fallback",
                msg,
                task_instruction,
            )
        except Exception as e:
            logger.warning(
                "RemotePlanner.plan failed (%s) for %r — single-step fallback",
                e,
                task_instruction,
            )
        return [task_instruction]

    def check_done(self, instruction: str, image: Any) -> bool:
        """Ask the out-of-process SPECIALIST if ``instruction`` is completed.

        Satisfies ``CompletionDetector``. Reuses the same loaded model as
        ``plan`` over the same channel; unlike ``plan`` this runs on the
        per-step loop (every K steps), so it uses the short ``check_timeout``.
        Fails safe: any error, timeout, missing frame, or a server that doesn't
        support checks (``{"error": ...}``) returns ``False`` — a phase never
        advances on a bad check (the runtime's step cap still guarantees
        progress).
        """
        if image is None:
            return False
        try:
            self._ensure_started()
            proc = self._proc
            assert proc is not None and proc.stdin is not None
            request = {
                "check": {"instruction": instruction, "image": _encode_image(image)}
            }
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            msg = self._read_message(self._check_timeout)
            return bool((msg or {}).get("done"))
        except Exception as e:
            logger.warning(
                "RemotePlanner.check_done failed (%s) for %r — treating as not done",
                e,
                instruction,
            )
            return False

    def ground(self, target_query: str, image: Any) -> str:
        """Ask the out-of-process SPECIALIST to ground ``target_query`` in ``image``.

        Satisfies ``GroundingProvider``. Reuses the same loaded model as ``plan``
        over the same channel. Runs only at phase boundaries (not every step),
        so it uses the moderate ``ground_timeout``. Fails safe: any error,
        timeout, missing frame, or a server that doesn't support grounding
        (``{"error": ...}``) returns ``target_query`` unchanged — the pilot
        always receives an actionable instruction.
        """
        if image is None:
            return target_query
        try:
            self._ensure_started()
            proc = self._proc
            assert proc is not None and proc.stdin is not None
            request = {"ground": {"query": target_query, "image": _encode_image(image)}}
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            msg = self._read_message(self._ground_timeout)
            target = (msg or {}).get("target")
            if isinstance(target, str) and target.strip():
                return target.strip()
            logger.warning(
                "RemotePlanner: bad/empty grounding %r for %r — echoing query",
                msg,
                target_query,
            )
        except Exception as e:
            logger.warning(
                "RemotePlanner.ground failed (%s) for %r — echoing query",
                e,
                target_query,
            )
        return target_query

    def route(
        self, task_instruction: str, image: Any, history: list[str]
    ) -> RouteDecision:
        """Ask the out-of-process ORCHESTRATOR for the next sub-instruction.

        Satisfies ``OrchestratorRuntime``. Reuses the same loaded model over the
        same channel; runs at phase boundaries (not every step), so it uses
        ``route_timeout``. Fails safe: any error, timeout, missing frame, or a
        server that doesn't support routing (``{"error": ...}``) returns
        ``RouteDecision(subtask=task_instruction, done=False)`` — the episode
        never terminates on a bad route and the pilot still gets an instruction.
        """
        if image is None:
            return RouteDecision(subtask=task_instruction, done=False)
        try:
            self._ensure_started()
            proc = self._proc
            assert proc is not None and proc.stdin is not None
            request = {
                "route": {
                    "task": task_instruction,
                    "history": list(history),
                    "image": _encode_image(image),
                }
            }
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            msg = self._read_message(self._route_timeout) or {}
            if bool(msg.get("done")):
                return RouteDecision(subtask="", done=True)
            subtask = msg.get("subtask")
            if isinstance(subtask, str) and subtask.strip():
                return RouteDecision(subtask=subtask.strip(), done=False)
            logger.warning(
                "RemotePlanner: bad/empty route %r for %r — using task",
                msg,
                task_instruction,
            )
        except Exception as e:
            logger.warning(
                "RemotePlanner.route failed (%s) for %r — using task",
                e,
                task_instruction,
            )
        return RouteDecision(subtask=task_instruction, done=False)

    def close(self) -> None:
        """Shut down the server process. Idempotent; also runs at exit."""
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        if proc.stdin is not None and not proc.stdin.closed:
            with contextlib.suppress(BrokenPipeError, ValueError, OSError):
                proc.stdin.write(json.dumps({"shutdown": True}) + "\n")
                proc.stdin.flush()
            with contextlib.suppress(BrokenPipeError, ValueError, OSError):
                proc.stdin.close()
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            if proc.poll() is None:
                _terminate_process(proc, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _terminate_process(proc, _SIGKILL)
