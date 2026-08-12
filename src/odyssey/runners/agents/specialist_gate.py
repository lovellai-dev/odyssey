"""Async SPECIALIST stuck-gate — VLM verdicts without blocking the sim loop.

The recovery loop (``runners/agents/recovery.py``) wants a SPECIALIST VLM's
"is the robot stuck?" verdict at chunk boundaries, but a served reasoner takes
~1-2 s per call while a sim step takes milliseconds. This gate runs the judge
call on a single background worker thread and delivers verdicts through a
non-blocking mailbox:

  * ``submit`` never blocks and never queues more than one call: while a call
    is in flight, further boundaries are *skipped* (a judge slower than a
    chunk degrades to a lower poll rate, never a backlog);
  * ``poll`` is a non-blocking mailbox read; a verdict is consumed at the
    *next* boundary after it lands — acceptable one-boundary lag, because
    stuck is a persistent state, not an edge;
  * verdicts are tagged ``(episode, chunk_index)``; cross-episode verdicts are
    dropped here, rollback-horizon staleness is the policy's job.

The detector is injected (same ``DetectorLike`` duck-typing as
``ChunkCompletionGate``): pass an ``OpenAICompatCompletionJudge`` built with
:data:`STUCK_PROMPT_TEMPLATE` and a YES reply *is* the stuck verdict. Pass
``executor=lambda job: job()`` in tests for fully synchronous determinism —
the thread machinery is then never touched (``remote_planner.py`` precedent:
daemon worker + queue + ``None`` sentinel + ``atexit`` close).
"""

from __future__ import annotations

import atexit
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from odyssey.runners.agents.completion_gate import DetectorLike, _as_callable

logger = logging.getLogger(__name__)

# Two-frame frozen-arm compare — the Phase 0 finding (design doc §5): the
# single-frame open-ended retry question answered NO everywhere (twice
# reproduced on H100), while this concrete two-frame comparison discriminated
# cleanly (active windows 8/8 NO, idle/stuck windows 10/10 YES, 2026-08-11).
# The wording is the validated sweep prompt verbatim. Task context is
# deliberately ABSENT — open-ended task judgements bias the reasoner to NO —
# so the instruction slot is consumed without being rendered
# (``{instruction:.0s}``, the reasoner-probe control-template trick).
# Callers submit a (frame_earlier, frame_now) TUPLE; the judge emits one
# image part per tuple element (``OpenAICompatCompletionJudge.build_payload``).
STUCK_PROMPT_TEMPLATE = (
    "These two images are frames of the SAME robot manipulation episode, "
    "the first taken a few seconds BEFORE the second. Comparing them, is the "
    "robot arm essentially FROZEN in the same pose (no meaningful movement "
    "between the frames)? Answer with exactly one word: YES or NO."
    "{instruction:.0s}"
)


@dataclass
class SpecialistVerdict:
    """One judge call's outcome, tagged with when its frame was captured."""

    episode: int
    chunk_index: int
    stuck: bool
    latency_s: float
    error: str | None = None


class SpecialistGate:
    """Chunk-boundary async wrapper around a stuck detector.

    Parameters
    ----------
    detector:
        A ``(frame, instruction) -> bool`` callable or any object exposing
        ``is_complete`` (the judge surface). ``True`` means **stuck** when the
        detector carries :data:`STUCK_PROMPT_TEMPLATE`.
    poll_every_chunks:
        Submit only for chunk indices divisible by this (default 1 = every
        boundary the gate is offered).
    executor:
        How a judge job runs. ``None`` (default) = lazy single daemon worker
        thread + request queue. Tests inject ``lambda job: job()`` (sync) or a
        deferred list to pin the in-flight semantics.
    """

    def __init__(
        self,
        *,
        detector: DetectorLike,
        poll_every_chunks: int = 1,
        executor: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        if int(poll_every_chunks) < 1:
            raise ValueError(f"poll_every_chunks must be >= 1, got {poll_every_chunks}")
        self._detect = _as_callable(detector)
        self._poll_every = int(poll_every_chunks)
        self._executor = executor
        self._mailbox: queue.Queue[SpecialistVerdict] = queue.Queue()
        self._requests: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._in_flight = False
        self._episode = 0
        self._closed = False

    # -- lifecycle -----------------------------------------------------------

    def begin_episode(self, episode: int) -> None:
        """Tag subsequent calls with ``episode``; drop any buffered verdicts."""
        self._episode = int(episode)
        while True:
            try:
                self._mailbox.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        """Stop the worker thread. Idempotent; also registered at exit."""
        if self._closed:
            return
        self._closed = True
        worker = self._worker
        if worker is not None:
            self._requests.put(None)
            worker.join(timeout=5.0)
            if worker.is_alive():
                logger.warning("specialist gate worker did not stop within 5s")
            self._worker = None

    # -- submit / poll --------------------------------------------------------

    @property
    def in_flight(self) -> bool:
        return self._in_flight

    def submit(self, *, chunk_index: int, frame: Any, instruction: str) -> bool:
        """Launch a judge call for this boundary; ``False`` = skipped.

        Skips (never blocks, never queues a backlog) when a call is already in
        flight, the boundary is throttled out, or the gate is closed.
        """
        if self._closed or self._in_flight or chunk_index % self._poll_every != 0:
            return False
        episode = self._episode
        self._in_flight = True

        def job() -> None:
            start = time.monotonic()
            stuck = False
            error: str | None = None
            try:
                stuck = bool(self._detect(frame, instruction))
            except Exception as exc:  # a flaky judge must not kill the episode
                error = f"{type(exc).__name__}: {exc}"
                logger.warning("specialist stuck-check failed", exc_info=True)
            finally:
                self._mailbox.put(
                    SpecialistVerdict(
                        episode=episode,
                        chunk_index=chunk_index,
                        stuck=stuck,
                        latency_s=time.monotonic() - start,
                        error=error,
                    )
                )
                self._in_flight = False

        if self._executor is not None:
            self._executor(job)
        else:
            self._ensure_worker()
            self._requests.put(job)
        return True

    def poll(self) -> SpecialistVerdict | None:
        """Non-blocking mailbox read; cross-episode verdicts are dropped."""
        while True:
            try:
                verdict = self._mailbox.get_nowait()
            except queue.Empty:
                return None
            if verdict.episode == self._episode:
                return verdict
            logger.debug(
                "specialist gate: dropping verdict from episode %d (now %d)",
                verdict.episode,
                self._episode,
            )

    # -- worker thread ---------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker is not None:
            return
        self._worker = threading.Thread(
            target=self._drain_requests, name="specialist-gate", daemon=True
        )
        self._worker.start()
        atexit.register(self.close)

    def _drain_requests(self) -> None:
        while True:
            job = self._requests.get()
            if job is None:
                return
            job()
