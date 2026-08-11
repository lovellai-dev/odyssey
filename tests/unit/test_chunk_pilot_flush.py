"""Tests for ``ChunkPilotAdapter.flush()`` — event-triggered chunk truncation.

``flush()`` is the trigger-side counterpart of flush-on-instruction-change:
a recovery/drift monitor discards the rest of a stale chunk so the next
``act`` re-queries the policy from the current observation (VLA-Corrector's
truncation semantics). Exercised with the same dependency-free fakes as the
adapter's own tests (``test_pi05_pilot.py``).
"""

from __future__ import annotations

from typing import Any

from odyssey.runners.agents.chunk_pilot import ChunkPilotAdapter


class _RecordingPolicy:
    """Fake chunk-emitting policy: records every query, returns a tagged chunk."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def infer(self, wire_obs: Any) -> Any:
        self.calls.append(wire_obs)
        return {"query": len(self.calls)}


def _index_decoder(chunk: Any, k: int) -> tuple[Any, int]:
    return (chunk, k)


def _make_adapter(policy: _RecordingPolicy, *, n_action_steps: int = 4) -> ChunkPilotAdapter:
    return ChunkPilotAdapter(
        predict_chunk=policy.infer,
        action_decoder=_index_decoder,
        n_action_steps=n_action_steps,
    )


def test_flush_forces_requery_next_act() -> None:
    policy = _RecordingPolicy()
    adapter = _make_adapter(policy, n_action_steps=4)

    adapter.act("obs", "instr")
    assert len(policy.calls) == 1

    adapter.flush()
    _chunk, k = adapter.act("obs", "instr")

    assert len(policy.calls) == 2  # flush dropped the buffer -> fresh query
    assert k == 0  # fresh chunk replays from its first step


def test_flush_midchunk_discards_remaining_actions() -> None:
    policy = _RecordingPolicy()
    adapter = _make_adapter(policy, n_action_steps=4)

    adapter.act("obs", "instr")  # cursor 0 of chunk 1
    adapter.act("obs", "instr")  # cursor 1 of chunk 1
    adapter.flush()  # steps 2..3 of chunk 1 are never decoded

    outs = [adapter.act("obs", "instr") for _ in range(2)]

    assert [k for _c, k in outs] == [0, 1]
    assert all(chunk == {"query": 2} for chunk, _k in outs)


def test_flush_preserves_instruction_and_reads_as_boundary() -> None:
    policy = _RecordingPolicy()
    adapter = _make_adapter(policy, n_action_steps=4)

    adapter.act("obs", "same instruction")
    assert adapter.steps_remaining == 3

    adapter.flush()

    # Boundary predicate: recipes detect chunk starts via steps_remaining == 0.
    assert adapter.steps_remaining == 0
    # Same instruction after flush -> exactly one new query, no double flush.
    adapter.act("obs", "same instruction")
    assert len(policy.calls) == 2


def test_flush_before_first_act_is_noop() -> None:
    policy = _RecordingPolicy()
    adapter = _make_adapter(policy, n_action_steps=4)

    adapter.flush()  # nothing buffered yet: must not raise
    adapter.act("obs", "instr")

    assert len(policy.calls) == 1
    assert adapter.steps_remaining == 3
