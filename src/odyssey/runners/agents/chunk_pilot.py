"""Chunk-aware pilot adapter — the shared engine behind chunk-emitting VLAs.

A *chunk-emitting* pilot (GR00T, π0.5, any FAST/flow-matching VLA) answers one
policy query with a whole **action chunk** of ``n_action_steps`` future actions,
which the env then replays open-loop before the next query. The GR00T LIBERO
recipe hard-codes that replay as an inline ``for k in range(n_action_steps)``
loop (``gr00t_libero_eval.py``); this module factors that loop out ONCE, as a
pilot-agnostic adapter, so π0.5 reuses the exact same gating instead of shipping
a second copy that drifts.

The adapter presents the single-step ``PilotRuntime`` surface (``act`` →
one action per call) on top of a chunk-emitting policy, hiding the buffer:

  * **buffer + cursor** — a query fills a chunk; each ``act`` returns the next
    step and advances a cursor;
  * **re-query on drain** — when the cursor reaches ``n_action_steps`` (or no
    chunk has been fetched yet) the next ``act`` queries the policy again;
  * **flush-on-instruction-change** — if the instruction differs from the one
    the current chunk was computed for, the chunk is discarded and re-queried
    immediately. This is what lets a closed-loop planner advance a phase
    mid-chunk (the multi-agent completion-gating path) without replaying stale
    actions planned for the previous sub-instruction.

Everything pilot-specific is injected, so the adapter carries **no** heavy deps
(no numpy/torch/openpi at import time) and is unit-testable with plain fakes:

  * ``predict_chunk(wire_obs) -> chunk`` — the policy call (e.g. an openpi
    ``WebsocketClientPolicy.infer`` or a GR00T ``PolicyClient.get_action``). May
    return a bare chunk or a ``(chunk, extra)`` tuple; the leading element is used.
  * ``action_decoder(chunk, k) -> action`` — map step ``k`` of the chunk to the
    env's action vector (e.g. ``pi05_action_to_libero`` / ``gr00t_action_to_libero``).
  * ``observation_builder(raw_obs, instruction) -> wire_obs`` *(optional)* —
    turn the env observation into the policy's wire format; identity if omitted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _unwrap_chunk(result: Any) -> Any:
    """A policy may return a bare chunk or ``(chunk, extra)``; take the chunk."""
    return result[0] if isinstance(result, tuple) else result


class ChunkPilotAdapter:
    """Replay a chunk-emitting policy one action per ``act`` call.

    Satisfies the ``PilotRuntime`` protocol structurally: ``act(observation,
    instruction) -> action``. The first argument is named ``observation`` (not
    ``image``) because a proprio-conditioned chunk pilot needs the full env
    observation — the injected ``observation_builder`` decides what the policy
    actually consumes, so an image-only pilot still fits by passing the image
    straight through.
    """

    def __init__(
        self,
        *,
        predict_chunk: Callable[[Any], Any],
        action_decoder: Callable[[Any, int], Any],
        n_action_steps: int,
        observation_builder: Callable[[Any, str], Any] | None = None,
    ) -> None:
        if int(n_action_steps) < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {n_action_steps}")
        self._predict = predict_chunk
        self._decode = action_decoder
        self._build_obs = observation_builder
        self._n = int(n_action_steps)
        self._chunk: Any = None
        self._cursor = 0
        self._instruction: str | None = None

    # -- introspection (handy for tests + logging; not part of PilotRuntime) --

    @property
    def n_action_steps(self) -> int:
        return self._n

    @property
    def steps_remaining(self) -> int:
        """Actions left in the current buffer before the next re-query."""
        if self._chunk is None:
            return 0
        return max(0, self._n - self._cursor)

    def reset(self) -> None:
        """Drop the buffer so the next ``act`` re-queries. Call per episode."""
        self._chunk = None
        self._cursor = 0
        self._instruction = None

    # -- PilotRuntime surface --------------------------------------------------

    def _needs_requery(self, instruction: str) -> bool:
        return (
            self._chunk is None
            or self._cursor >= self._n
            or instruction != self._instruction  # flush-on-instruction-change
        )

    def _requery(self, observation: Any, instruction: str) -> None:
        wire = (
            self._build_obs(observation, instruction)
            if self._build_obs is not None
            else observation
        )
        self._chunk = _unwrap_chunk(self._predict(wire))
        self._cursor = 0
        self._instruction = instruction

    def act(self, observation: Any, instruction: str) -> Any:
        """Return one action, re-querying the policy when the chunk is spent.

        A fresh chunk is fetched when there is none buffered, the buffer has
        been fully replayed, or ``instruction`` changed since the last query.
        """
        if self._needs_requery(instruction):
            self._requery(observation, instruction)
        action = self._decode(self._chunk, self._cursor)
        self._cursor += 1
        return action
