"""Chunk-aware completion / hand-back gate — the shared closed-loop trigger.

A *chunk-emitting* pilot (GR00T, π0.5, any FAST/flow-matching VLA) answers one
policy query with a whole action **chunk** of ``n_action_steps`` future actions,
which the env replays open-loop before the next query (see
``runners/agents/chunk_pilot.py``). This module decides, at each **chunk
boundary**, whether the active sub-instruction is done and control should be
*handed back* to the orchestrator (advance the planner's phase, or end the
episode on the final phase).

Why gate *on chunks* rather than *per step*:

  * **cost** — the completion judge is a VLM (Gemma int4, GR00T's
    ``cosmos_reason``); running it on every one of ``n_action_steps`` replayed
    steps is wasteful when the pilot won't re-plan until the chunk drains anyway.
  * **staleness** — the world barely moves within one open-loop chunk, so a
    per-step judgement mostly repeats itself. One judgement per chunk lines the
    completion check up with the pilot's own decision cadence.

Historically each pilot inlined its own advancement rule (GR00T's fixed
``for k in range(n_action_steps)`` loop; ``PlannedEvalRuntime``'s fixed-step /
timeout strategies). This gate factors the *closed-loop* rule out ONCE, as a
pilot-agnostic component, so GR00T, π0.5 and the orchestrator share a single
implementation instead of three that drift (the same move ``ChunkPilotAdapter``
made for chunk replay).

Dependency-free by construction: the completion detector is injected — either a
plain ``detector(observation, instruction) -> bool`` callable or any
``CompletionDetector`` (an object with ``is_complete``) — so the gate carries no
torch/VLM import and is unit-testable with a trivial fake.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# A detector is either a bare callable or a CompletionDetector-shaped object.
DetectorLike = Callable[[Any, str], bool] | Any


def _as_callable(detector: DetectorLike) -> Callable[[Any, str], bool]:
    """Normalise a detector to a ``(observation, instruction) -> bool`` callable.

    Accepts a bare callable or any ``CompletionDetector`` (an object exposing
    ``is_complete``), so callers can pass a VLM-judge object or a one-line lambda
    interchangeably.
    """
    is_complete = getattr(detector, "is_complete", None)
    if callable(is_complete):
        return is_complete
    if callable(detector):
        return detector
    raise TypeError(
        "detector must be callable(observation, instruction) -> bool or expose "
        f"an is_complete method; got {type(detector).__name__}"
    )


class ChunkCompletionGate:
    """Poll a completion judge at chunk boundaries and report hand-back.

    Call :meth:`update` once per env step (after the action is applied). The gate
    counts steps; only when a full chunk of ``n_action_steps`` has been replayed
    does it consult the injected detector, and it returns ``True`` exactly when
    the detector judges the current sub-instruction complete — the signal for the
    caller to hand control back (advance the phase / end the episode).

    Parameters
    ----------
    detector:
        The completion judge — a ``detector(observation, instruction) -> bool``
        callable or any :class:`CompletionDetector` (has ``is_complete``).
    n_action_steps:
        Chunk size — how many steps the pilot replays open-loop per query. The
        gate polls once every ``n_action_steps`` steps. Match it to the pilot's
        replay horizon (e.g. a :class:`ChunkPilotAdapter`'s ``n_action_steps``)
        so the completion check lines up with the pilot's re-query points.
    min_steps:
        Warm-up: never poll (never hand back) before this many steps have elapsed
        in the current phase. Guards against a premature "done" on the very first
        chunk before the arm has moved. Default ``0`` (poll from the first
        boundary).
    poll_every_chunks:
        Throttle: consult the detector only every Nth chunk boundary. Default
        ``1`` (every boundary). Use a higher value to trade responsiveness for
        fewer (expensive) judge calls.
    """

    def __init__(
        self,
        *,
        detector: DetectorLike,
        n_action_steps: int,
        min_steps: int = 0,
        poll_every_chunks: int = 1,
    ) -> None:
        if int(n_action_steps) < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}")
        if int(poll_every_chunks) < 1:
            raise ValueError(
                f"poll_every_chunks must be >= 1, got {poll_every_chunks}"
            )
        if int(min_steps) < 0:
            raise ValueError(f"min_steps must be >= 0, got {min_steps}")
        self._detect = _as_callable(detector)
        self._n = int(n_action_steps)
        self._min_steps = int(min_steps)
        self._poll_every = int(poll_every_chunks)
        self._steps = 0
        self._chunks = 0

    # -- introspection (handy for tests + logging) -----------------------------

    @property
    def n_action_steps(self) -> int:
        return self._n

    @property
    def steps_in_phase(self) -> int:
        """Steps elapsed since the last :meth:`reset` (i.e. in this phase)."""
        return self._steps

    @property
    def chunks_completed(self) -> int:
        """Chunk boundaries crossed since the last :meth:`reset`."""
        return self._chunks

    def at_chunk_boundary(self) -> bool:
        """Whether the last :meth:`update` landed on a chunk boundary."""
        return self._steps > 0 and self._steps % self._n == 0

    def reset(self) -> None:
        """Drop the counters so the next chunk window starts fresh.

        Call on every hand-back (new sub-instruction) and at the start of each
        episode. A mid-chunk reset re-aligns the gate's boundary with the pilot's
        own re-query — a chunk pilot flushes on the instruction change, so both
        restart their chunk window together.
        """
        self._steps = 0
        self._chunks = 0

    # -- the closed-loop trigger ----------------------------------------------

    def update(self, observation: Any, instruction: str) -> bool:
        """Advance one step; return ``True`` iff control should be handed back.

        Increments the step counter and, only at a chunk boundary (and past the
        warm-up / throttle), consults the detector. Between boundaries this is a
        cheap counter bump that always returns ``False`` — the detector (and its
        VLM cost) is untouched.

        Does **not** self-reset on a hand-back: the caller resets when it acts on
        the signal (advances the phase), which keeps a hand-back the caller
        chooses to ignore — e.g. on the final phase — from silently re-arming.
        """
        self._steps += 1
        if self._steps % self._n != 0:
            return False  # mid-chunk: no judge call
        self._chunks += 1
        if self._steps < self._min_steps:
            return False  # still warming up
        if self._chunks % self._poll_every != 0:
            return False  # throttled this boundary
        return bool(self._detect(observation, instruction))
