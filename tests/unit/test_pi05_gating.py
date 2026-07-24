"""Tests for the chunk-aware completion / hand-back gate (issue #74).

The completion gate is the closed-loop counterpart to the fixed-step / timeout
phase strategies: a chunk-emitting pilot (GR00T, π0.5) replays ``n_action_steps``
actions open-loop per query, and this gate consults a completion judge once per
CHUNK BOUNDARY to decide when the current sub-instruction is done and control
should be handed back (the orchestrator advances the phase).

Like ``ChunkPilotAdapter``, the gate is pilot-agnostic and dependency-free — the
completion detector is injected — so the whole thing is exercisable WITHOUT a
GPU, a served checkpoint, or a real VLM judge:

  * ``ChunkCompletionGate`` (``runners/agents/completion_gate.py``) is tested with
    plain fake detectors (callables + ``is_complete`` objects);
  * the orchestrator wiring (``PlannedEvalRuntime`` + ``PhaseStrategy.COMPLETION``)
    is tested with a fake pilot + scripted detector;
  * the shared boundary with the chunk pilots is checked against a real
    ``ChunkPilotAdapter``'s ``n_action_steps``.

All tests are named ``test_pi05_gating_*`` (the ``-k pi05_gating`` gate).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import odyssey.engine  # noqa: F401  (avoid circular import, as in test_agent_runtimes)
from odyssey.runners.agents.chunk_pilot import ChunkPilotAdapter
from odyssey.runners.agents.completion_gate import ChunkCompletionGate
from odyssey.runners.agents.planned import (
    PhaseConfig,
    PhaseStrategy,
    PlannedEvalRuntime,
)
from odyssey.runners.agents.runtime import CompletionDetector

# ---------------------------------------------------------------------------
# Fakes — completion detectors, both a bare callable and a CompletionDetector.
# ---------------------------------------------------------------------------

class _RecordingDetector:
    """CompletionDetector that records every poll and replays a scripted verdict.

    ``verdicts`` is consumed one per poll; once exhausted it returns ``False``
    (never complete). Records the ``(observation, instruction)`` of each poll so
    tests can assert the gate polls at the right cadence with the right args.
    """

    def __init__(self, verdicts: list[bool] | None = None) -> None:
        self.calls: list[tuple[Any, str]] = []
        self._verdicts = list(verdicts or [])

    def is_complete(self, observation: Any, instruction: str) -> bool:
        idx = len(self.calls)
        self.calls.append((observation, instruction))
        return self._verdicts[idx] if idx < len(self._verdicts) else False


def _never(_obs: Any, _instr: str) -> bool:
    return False


def _always(_obs: Any, _instr: str) -> bool:
    return True


class _FakePilot:
    """Records every (image, instruction) call and returns a fixed action."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, str]] = []

    def act(self, image: Any, instruction: str) -> Any:
        self.calls.append((image, instruction))
        return np.zeros(7, dtype=np.float64)


# ---------------------------------------------------------------------------
# ChunkCompletionGate — polls at chunk boundaries only.
# ---------------------------------------------------------------------------

def test_pi05_gating_polls_only_at_chunk_boundary() -> None:
    det = _RecordingDetector()
    gate = ChunkCompletionGate(detector=det, n_action_steps=3)

    # Steps 1,2 are mid-chunk: no judge call, always False.
    assert gate.update("obs", "instr") is False
    assert gate.update("obs", "instr") is False
    assert det.calls == []  # detector untouched between boundaries

    # Step 3 lands on the boundary -> exactly one poll.
    gate.update("obs", "instr")
    assert len(det.calls) == 1


def test_pi05_gating_polls_once_per_chunk() -> None:
    det = _RecordingDetector()
    gate = ChunkCompletionGate(detector=det, n_action_steps=2)

    for _ in range(6):  # three full chunks of 2
        gate.update("obs", "instr")

    assert len(det.calls) == 3  # one poll per drained chunk, not per step


def test_pi05_gating_hands_back_when_detector_says_done() -> None:
    # Complete on the SECOND boundary (first chunk NO, second chunk YES).
    det = _RecordingDetector(verdicts=[False, True])
    gate = ChunkCompletionGate(detector=det, n_action_steps=2)

    verdicts = [gate.update("obs", "instr") for _ in range(4)]
    # Only the 4th step (2nd boundary) hands back.
    assert verdicts == [False, False, False, True]


def test_pi05_gating_no_handback_when_detector_never_done() -> None:
    gate = ChunkCompletionGate(detector=_never, n_action_steps=2)
    assert not any(gate.update("obs", "instr") for _ in range(10))


def test_pi05_gating_passes_observation_and_instruction_to_detector() -> None:
    det = _RecordingDetector()
    gate = ChunkCompletionGate(detector=det, n_action_steps=1)  # boundary every step

    gate.update({"frame": 7}, "grasp the bowl")

    assert det.calls == [({"frame": 7}, "grasp the bowl")]


# ---------------------------------------------------------------------------
# ChunkCompletionGate — warm-up and throttle.
# ---------------------------------------------------------------------------

def test_pi05_gating_min_steps_warmup_skips_early_boundary() -> None:
    det = _RecordingDetector(verdicts=[True, True])
    # n=2, warm up past the first boundary (step 2) — first eligible poll at step 4.
    gate = ChunkCompletionGate(detector=det, n_action_steps=2, min_steps=4)

    v = [gate.update("obs", "instr") for _ in range(4)]
    assert v == [False, False, False, True]  # step-2 boundary suppressed by warm-up
    assert len(det.calls) == 1  # detector only consulted once (at step 4)


def test_pi05_gating_poll_every_chunks_throttles_detector() -> None:
    det = _RecordingDetector()
    gate = ChunkCompletionGate(detector=det, n_action_steps=2, poll_every_chunks=2)

    for _ in range(8):  # four chunk boundaries
        gate.update("obs", "instr")

    # Polled only on the 2nd and 4th boundary.
    assert len(det.calls) == 2


# ---------------------------------------------------------------------------
# ChunkCompletionGate — introspection, reset, no self-reset.
# ---------------------------------------------------------------------------

def test_pi05_gating_introspection_tracks_steps_and_chunks() -> None:
    gate = ChunkCompletionGate(detector=_never, n_action_steps=3)
    assert gate.n_action_steps == 3
    assert gate.steps_in_phase == 0 and gate.chunks_completed == 0
    assert not gate.at_chunk_boundary()

    gate.update("o", "i")
    gate.update("o", "i")
    assert gate.steps_in_phase == 2 and not gate.at_chunk_boundary()

    gate.update("o", "i")  # boundary
    assert gate.steps_in_phase == 3 and gate.chunks_completed == 1
    assert gate.at_chunk_boundary()


def test_pi05_gating_reset_zeroes_counters() -> None:
    det = _RecordingDetector()
    gate = ChunkCompletionGate(detector=det, n_action_steps=2)
    gate.update("o", "i")
    gate.update("o", "i")  # boundary, 1 poll
    assert gate.steps_in_phase == 2 and gate.chunks_completed == 1

    gate.reset()
    assert gate.steps_in_phase == 0 and gate.chunks_completed == 0
    # A reset mid-count re-aligns the boundary: the next poll is a fresh chunk away.
    gate.update("o", "i")
    assert len(det.calls) == 1  # still just the pre-reset poll; no boundary yet
    gate.update("o", "i")
    assert len(det.calls) == 2  # new boundary after reset


def test_pi05_gating_does_not_self_reset_on_handback() -> None:
    # Detector says done every boundary; without a caller reset the gate keeps
    # counting so an ignored hand-back re-fires only on the NEXT full chunk.
    gate = ChunkCompletionGate(detector=_always, n_action_steps=2)
    v = [gate.update("o", "i") for _ in range(4)]
    assert v == [False, True, False, True]  # boundaries at steps 2 and 4
    assert gate.chunks_completed == 2  # counters kept advancing (no self-reset)


# ---------------------------------------------------------------------------
# ChunkCompletionGate — detector shapes + validation.
# ---------------------------------------------------------------------------

def test_pi05_gating_accepts_completion_detector_object() -> None:
    det = _RecordingDetector(verdicts=[True])
    gate = ChunkCompletionGate(detector=det, n_action_steps=1)
    assert gate.update("o", "i") is True  # is_complete adapted to a callable


def test_pi05_gating_accepts_bare_callable_detector() -> None:
    gate = ChunkCompletionGate(detector=_always, n_action_steps=1)
    assert gate.update("o", "i") is True


def test_pi05_gating_rejects_non_detector() -> None:
    with pytest.raises(TypeError, match="detector"):
        ChunkCompletionGate(detector=object(), n_action_steps=1)


def test_pi05_gating_rejects_non_positive_n_action_steps() -> None:
    with pytest.raises(ValueError, match="n_action_steps"):
        ChunkCompletionGate(detector=_never, n_action_steps=0)


def test_pi05_gating_rejects_bad_poll_every_chunks() -> None:
    with pytest.raises(ValueError, match="poll_every_chunks"):
        ChunkCompletionGate(detector=_never, n_action_steps=1, poll_every_chunks=0)


def test_pi05_gating_rejects_negative_min_steps() -> None:
    with pytest.raises(ValueError, match="min_steps"):
        ChunkCompletionGate(detector=_never, n_action_steps=1, min_steps=-1)


def test_pi05_gating_recording_detector_satisfies_protocol() -> None:
    # runtime_checkable Protocol: structural match on is_complete(obs, instr).
    assert isinstance(_RecordingDetector(), CompletionDetector)


# ---------------------------------------------------------------------------
# Shared boundary with the chunk pilots (GR00T / π0.5) — same n_action_steps.
# ---------------------------------------------------------------------------

def test_pi05_gating_boundary_aligns_with_chunk_pilot_requery() -> None:
    """A gate built from the pilot's n_action_steps polls exactly at re-query."""
    queries: list[int] = []

    def predict(_wire: Any) -> dict[str, int]:
        queries.append(1)
        return {"q": len(queries)}

    pilot = ChunkPilotAdapter(
        predict_chunk=predict,
        action_decoder=lambda chunk, k: (chunk, k),
        n_action_steps=4,
    )
    det = _RecordingDetector()
    gate = ChunkCompletionGate(detector=det, n_action_steps=pilot.n_action_steps)

    # Drive them in lockstep: pilot.act (may re-query) then gate.update.
    for _ in range(8):
        pilot.act("obs", "instr")
        gate.update("obs", "instr")

    # Pilot re-queried twice (two chunks of 4); the gate polled once per chunk,
    # i.e. exactly at the pilot's re-query points.
    assert len(queries) == 2
    assert len(det.calls) == 2


# ---------------------------------------------------------------------------
# Orchestrator integration — PlannedEvalRuntime + PhaseStrategy.COMPLETION.
# ---------------------------------------------------------------------------

def _completion_runtime(pilot: _FakePilot, det: Any, *, n_action_steps: int,
                        steps: list[str]) -> PlannedEvalRuntime:
    gate = ChunkCompletionGate(detector=det, n_action_steps=n_action_steps)
    cfg = PhaseConfig(strategy=PhaseStrategy.COMPLETION)

    class _Planner:
        def plan(self, task: str, image: Any = None) -> list[str]:
            return list(steps)

    rt = PlannedEvalRuntime(pilot, _Planner(), phase_config=cfg, completion_gate=gate)
    rt.begin_episode("task")
    return rt


def test_pi05_gating_runtime_advances_phase_on_handback() -> None:
    pilot = _FakePilot()
    # n=2: hand back on the first boundary of each phase (done at every poll).
    rt = _completion_runtime(
        pilot, _always, n_action_steps=2, steps=["phase-1", "phase-2", "phase-3"]
    )
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    assert rt.current_phase_index == 0
    rt.get_action(img)  # step 1, mid-chunk
    assert rt.current_phase_index == 0
    rt.get_action(img)  # step 2, boundary -> hand back
    assert rt.current_phase_index == 1
    assert pilot.calls[-1][1] == "phase-1"  # step 2 still ran under phase-1


def test_pi05_gating_runtime_does_not_advance_without_completion() -> None:
    pilot = _FakePilot()
    rt = _completion_runtime(
        pilot, _never, n_action_steps=2, steps=["phase-1", "phase-2"]
    )
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    for _ in range(20):
        rt.get_action(img)
    # Detector never completes -> stays on the first phase the whole time.
    assert rt.current_phase_index == 0
    assert all(instr == "phase-1" for _img, instr in pilot.calls)


def test_pi05_gating_runtime_resets_gate_on_phase_advance() -> None:
    pilot = _FakePilot()
    det = _RecordingDetector()  # never completes -> we drive boundaries manually
    gate = ChunkCompletionGate(detector=det, n_action_steps=3)
    cfg = PhaseConfig(strategy=PhaseStrategy.COMPLETION)

    class _Planner:
        def plan(self, task: str, image: Any = None) -> list[str]:
            return ["a", "b"]

    rt = PlannedEvalRuntime(pilot, _Planner(), phase_config=cfg, completion_gate=gate)
    rt.begin_episode("task")
    img = np.zeros((4, 4, 3), dtype=np.uint8)

    rt.get_action(img)
    rt.get_action(img)
    assert gate.steps_in_phase == 2  # mid-chunk within phase "a"

    rt.begin_episode("task2")  # a fresh episode must re-arm the gate
    assert gate.steps_in_phase == 0
    assert rt.current_phase_index == 0


def test_pi05_gating_runtime_completion_detector_sees_current_phase() -> None:
    pilot = _FakePilot()
    det = _RecordingDetector(verdicts=[True, True])  # done at each phase's boundary
    gate = ChunkCompletionGate(detector=det, n_action_steps=1)  # boundary every step
    cfg = PhaseConfig(strategy=PhaseStrategy.COMPLETION)

    class _Planner:
        def plan(self, task: str, image: Any = None) -> list[str]:
            return ["reach", "grasp"]

    rt = PlannedEvalRuntime(pilot, _Planner(), phase_config=cfg, completion_gate=gate)
    rt.begin_episode("task")
    img = np.zeros((4, 4, 3), dtype=np.uint8)

    rt.get_action(img)  # phase "reach" -> boundary -> hand back
    rt.get_action(img)  # phase "grasp" -> boundary -> hand back
    # The judge was polled with each phase's active sub-instruction in turn.
    assert [instr for _obs, instr in det.calls] == ["reach", "grasp"]


def test_pi05_gating_runtime_completion_without_gate_never_advances() -> None:
    pilot = _FakePilot()
    cfg = PhaseConfig(strategy=PhaseStrategy.COMPLETION)

    class _Planner:
        def plan(self, task: str, image: Any = None) -> list[str]:
            return ["only-phase", "next"]

    # No completion_gate supplied: the runtime warns and runs the phase open-ended.
    rt = PlannedEvalRuntime(pilot, _Planner(), phase_config=cfg)
    rt.begin_episode("task")
    img = np.zeros((4, 4, 3), dtype=np.uint8)

    for _ in range(10):
        rt.get_action(img)
    assert rt.current_phase_index == 0
