"""RemoteGenerator — drives an out-of-process ``TextGenerator``.

Implements the ``TextGenerator`` protocol but, instead of loading Gemma in this
process, launches ``generator_server`` in a *separate* venv/process. That frees
the specialist from OpenVLA's pinned ``transformers==4.40.1`` (which lives in
the main venv) so it can host the multimodal Gemma 4 generator (which needs a
modern ``transformers`` + ``torchvision``).

This is a generic, prompt-agnostic transport: it sends fully-composed
``messages`` (+ optional image) and returns generated ``text``. It carries no
brain logic — prompting/composition is the caller's (the brain's) job.

Communicates over the JSON-lines stdin/stdout protocol documented in
``generator_server``. The subprocess (and the model) starts lazily on the first
``generate`` call and is reused across calls. Robust by design: non-JSON stdout
lines are skipped, and any failure returns ``""`` so an advisory turn degrades
gracefully rather than crashing the rollout.
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

logger = logging.getLogger(__name__)

_SERVER_MODULE = "odyssey.runners.models.generator_server"
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
    """Encode a PIL Image or HWC uint8 ndarray as a base64 PNG string."""
    from PIL import Image

    pil = image if isinstance(image, Image.Image) else Image.fromarray(image)
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class RemoteGenerator:
    """Out-of-process ``TextGenerator``. Satisfies the ``TextGenerator`` protocol.

    Parameters
    ----------
    model_base:
        HF id of the SPECIALIST model (e.g. ``google/gemma-4-E2B-it``).
    quantization:
        Quantization string passed through to the server (e.g. ``int4``) or None.
    python_path:
        Path to the *specialist* venv's python interpreter (from
        ``ODYSSEY_SPECIALIST_PYTHON``).
    startup_timeout:
        Seconds to wait for ``{"ready": true}`` — the first run downloads the
        model, so this is generous.
    request_timeout:
        Seconds to wait for a single generate response.
    launch_args:
        Argv (after ``python_path``) that starts the server. Defaults to
        ``("-m", generator_server)``; overridable for tests.
    """

    def __init__(
        self,
        model_base: str,
        quantization: str | None = None,
        *,
        python_path: str,
        startup_timeout: float = 600.0,
        request_timeout: float = 120.0,
        launch_args: Sequence[str] = ("-m", _SERVER_MODULE),
    ) -> None:
        self._model_base = model_base
        self._quantization = quantization
        self._python_path = python_path
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._launch_args = list(launch_args)
        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None
        atexit.register(self.close)

    def _ensure_started(self) -> None:
        if self._proc is not None:
            return
        argv = [self._python_path, *self._launch_args, "--model", self._model_base]
        if self._quantization:
            argv += ["--quantization", self._quantization]
        logger.info("RemoteGenerator: launching specialist server: %s", " ".join(argv))
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
            raise RuntimeError(f"specialist generator failed to start: {err}")

    def _drain_stdout(self, stdout: IO[str]) -> None:
        """Reader thread: push each stdout line onto the queue; None on EOF."""
        try:
            for line in stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)

    def _read_message(self, timeout: float) -> dict[str, Any] | None:
        """Return the next protocol JSON object, skipping noise. None on timeout/EOF."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("RemoteGenerator: timed out waiting for server response")
                return None
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                logger.warning("RemoteGenerator: timed out waiting for server response")
                return None
            if line is None:
                logger.warning("RemoteGenerator: server stdout closed (process exited?)")
                return None
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("RemoteGenerator: skipping non-JSON stdout line: %s", line[:200])
                continue
            if isinstance(obj, dict):
                return obj

    def generate(self, messages: list[dict[str, Any]], image: Any | None = None) -> str:
        """Generate text from chat messages via the out-of-process server.

        When ``image`` is given it is sent as a base64 PNG alongside the
        messages. Returns ``""`` on any error (the caller degrades gracefully).
        """
        try:
            self._ensure_started()
            proc = self._proc
            assert proc is not None and proc.stdin is not None
            request: dict[str, Any] = {"messages": messages}
            if image is not None:
                request["image"] = _encode_image(image)
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            msg = self._read_message(self._request_timeout)
            text = (msg or {}).get("text")
            if isinstance(text, str):
                return text
            logger.warning("RemoteGenerator: bad/empty response %r — empty text", msg)
        except Exception as e:
            logger.warning("RemoteGenerator.generate failed (%s) — empty text", e)
        return ""

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
