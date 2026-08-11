"""Tests for the async SPECIALIST stuck-gate (``runners/agents/specialist_gate.py``).

Determinism strategy: all semantics are pinned with an injected synchronous
executor (``lambda job: job()``) or a deferred-list executor; only the two
thread-mode tests touch the real worker, gated by ``threading.Event``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from odyssey.runners.agents.openai_judge import OpenAICompatCompletionJudge
from odyssey.runners.agents.specialist_gate import (
    STUCK_PROMPT_TEMPLATE,
    SpecialistGate,
    SpecialistVerdict,
)

_SYNC = lambda job: job()  # noqa: E731


class _RecordingDetector:
    """Detector fake: records calls, replays scripted verdicts."""

    def __init__(self, verdicts: list[bool] | None = None) -> None:
        self.calls: list[tuple[Any, str]] = []
        self._verdicts = list(verdicts or [])

    def __call__(self, frame: Any, instruction: str) -> bool:
        idx = len(self.calls)
        self.calls.append((frame, instruction))
        return self._verdicts[idx] if idx < len(self._verdicts) else False


def _image() -> Any:
    return [[[0, 0, 0]]]  # anything; fakes never decode it


# ---------------------------------------------------------------------------
# Synchronous-executor semantics.
# ---------------------------------------------------------------------------


def test_gate_submit_and_poll_roundtrip_sync_executor() -> None:
    detector = _RecordingDetector(verdicts=[True])
    gate = SpecialistGate(detector=detector, executor=_SYNC)
    gate.begin_episode(3)

    assert gate.submit(chunk_index=0, frame="f0", instruction="pick") is True
    verdict = gate.poll()

    assert isinstance(verdict, SpecialistVerdict)
    assert verdict.stuck is True
    assert verdict.episode == 3 and verdict.chunk_index == 0
    assert verdict.latency_s >= 0.0 and verdict.error is None
    assert detector.calls == [("f0", "pick")]
    assert gate.poll() is None  # mailbox drained


def test_gate_skips_submit_while_in_flight() -> None:
    deferred: list[Callable[[], None]] = []
    gate = SpecialistGate(detector=_RecordingDetector(), executor=deferred.append)
    gate.begin_episode(1)

    assert gate.submit(chunk_index=0, frame="f", instruction="i") is True
    assert gate.in_flight
    # Boundaries 1..3 arrive while the call is still running: all skipped.
    for chunk in (1, 2, 3):
        assert gate.submit(chunk_index=chunk, frame="f", instruction="i") is False
    assert len(deferred) == 1

    deferred.pop()()  # the call finally completes
    assert not gate.in_flight
    assert gate.poll() is not None
    assert gate.submit(chunk_index=4, frame="f", instruction="i") is True


def test_gate_throttles_by_poll_every_chunks() -> None:
    detector = _RecordingDetector()
    gate = SpecialistGate(detector=detector, poll_every_chunks=2, executor=_SYNC)
    gate.begin_episode(1)

    accepted = [
        gate.submit(chunk_index=chunk, frame="f", instruction="i") for chunk in range(4)
    ]

    assert accepted == [True, False, True, False]
    assert len(detector.calls) == 2


def test_gate_drops_cross_episode_verdicts() -> None:
    gate = SpecialistGate(detector=_RecordingDetector(verdicts=[True]), executor=_SYNC)
    gate.begin_episode(1)
    gate.submit(chunk_index=0, frame="f", instruction="i")

    gate.begin_episode(2)  # episode ended before the verdict was consumed
    assert gate.poll() is None  # begin_episode drained the mailbox

    # A verdict landing *after* the episode bump is dropped by poll().
    deferred: list[Callable[[], None]] = []
    gate2 = SpecialistGate(detector=_RecordingDetector(verdicts=[True]), executor=deferred.append)
    gate2.begin_episode(1)
    gate2.submit(chunk_index=0, frame="f", instruction="i")
    gate2.begin_episode(2)
    deferred.pop()()  # verdict tagged episode 1 lands now
    assert gate2.poll() is None


def test_gate_detector_exception_reports_error_not_stuck() -> None:
    def broken(frame: Any, instruction: str) -> bool:
        raise RuntimeError("endpoint down")

    gate = SpecialistGate(detector=broken, executor=_SYNC)
    gate.begin_episode(1)
    gate.submit(chunk_index=0, frame="f", instruction="i")
    verdict = gate.poll()

    assert verdict is not None
    assert verdict.stuck is False  # conservative: never intervene on a broken judge
    assert verdict.error is not None and "endpoint down" in verdict.error
    assert not gate.in_flight  # the failure released the in-flight slot


def test_gate_wraps_openai_judge_with_stuck_template() -> None:
    payloads: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        payloads.append(payload)
        return {"choices": [{"message": {"content": "YES, clearly wedged."}}]}

    judge = OpenAICompatCompletionJudge(
        base_url="http://judge:8002/v1",
        model="nvidia/Cosmos3-Nano",
        prompt_template=STUCK_PROMPT_TEMPLATE,
        max_tokens=256,
        transport=transport,
    )
    gate = SpecialistGate(detector=judge, executor=_SYNC)
    gate.begin_episode(1)

    import numpy as np

    gate.submit(
        chunk_index=0,
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        instruction="put the bowl on the plate",
    )
    verdict = gate.poll()

    assert verdict is not None and verdict.stuck is True  # YES == stuck
    prompt = payloads[0]["messages"][0]["content"][1]["text"]
    assert "FAILED or STUCK" in prompt
    assert "put the bowl on the plate" in prompt


# ---------------------------------------------------------------------------
# Real worker thread.
# ---------------------------------------------------------------------------


def test_gate_thread_mode_delivers_verdict_and_close_joins() -> None:
    release = threading.Event()

    def slow_detector(frame: Any, instruction: str) -> bool:
        assert release.wait(timeout=5.0)
        return True

    gate = SpecialistGate(detector=slow_detector)
    gate.begin_episode(1)
    assert gate.submit(chunk_index=0, frame="f", instruction="i") is True
    assert gate.poll() is None  # nothing yet: the sim loop is not blocked

    release.set()
    deadline = threading.Event()
    verdict = None
    for _ in range(100):
        verdict = gate.poll()
        if verdict is not None:
            break
        deadline.wait(0.02)
    assert verdict is not None and verdict.stuck is True

    gate.close()
    gate.close()  # idempotent
    assert gate.submit(chunk_index=1, frame="f", instruction="i") is False  # closed


def test_gate_never_blocks_when_judge_slower_than_chunks() -> None:
    release = threading.Event()
    calls: list[int] = []

    def glacial(frame: Any, instruction: str) -> bool:
        calls.append(1)
        assert release.wait(timeout=5.0)
        return False

    gate = SpecialistGate(detector=glacial)
    gate.begin_episode(1)
    results = [gate.submit(chunk_index=c, frame="f", instruction="i") for c in range(5)]
    assert results == [True, False, False, False, False]  # 1 accepted, 4 skipped

    release.set()
    gate.close()
    assert sum(calls) == 1  # the backlog never formed
