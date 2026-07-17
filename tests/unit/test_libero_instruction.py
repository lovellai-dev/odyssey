"""Unit tests for LIBERO instruction resolution (the Option-B alignment contract).

``_resolve_libero_instruction`` is a pure helper — it needs neither the ``libero``
package nor a GPU — so the alignment contract between the mission's declared
``config.task_instruction`` and the benchmark's authoritative ``task.language`` is
testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from odyssey.runners.evals.libero import _resolve_libero_instruction


@dataclass
class _FakeTask:
    """Stand-in for a LIBERO benchmark task (only ``language`` is read)."""

    language: str | None


def test_uses_benchmark_language_when_no_declaration() -> None:
    task = _FakeTask(language="pick up the ketchup and place it in the basket")
    assert (
        _resolve_libero_instruction(task, {})
        == "pick up the ketchup and place it in the basket"
    )


def test_benchmark_language_wins_over_matching_declaration() -> None:
    gt = "pick up the ketchup and place it in the basket"
    task = _FakeTask(language=gt)
    # Declared value matches (modulo surrounding whitespace) → no warning, gt returned.
    assert _resolve_libero_instruction(task, {"task_instruction": f"  {gt}  "}) == gt


def test_drift_warns_but_returns_benchmark_language(
    caplog: pytest.LogCaptureFixture,
) -> None:
    gt = "pick up the ketchup and place it in the basket"
    task = _FakeTask(language=gt)
    cfg = {"task_instruction": "pick up the WRONG object"}
    with caplog.at_level("WARNING"):
        result = _resolve_libero_instruction(task, cfg)
    # The benchmark instruction is authoritative and still drives the eval,
    assert result == gt
    # but the drift is surfaced, not swallowed (the old precedence dropped it).
    assert any("instruction mismatch" in r.message.lower() for r in caplog.records)


def test_strict_mode_raises_on_drift() -> None:
    task = _FakeTask(language="pick up the ketchup and place it in the basket")
    cfg = {"task_instruction": "pick up the WRONG object", "strict_instruction": True}
    with pytest.raises(ValueError, match="instruction mismatch"):
        _resolve_libero_instruction(task, cfg)


def test_falls_back_to_declaration_when_benchmark_has_none() -> None:
    task = _FakeTask(language=None)
    assert (
        _resolve_libero_instruction(task, {"task_instruction": "do the thing"})
        == "do the thing"
    )


def test_final_fallback_when_nothing_available() -> None:
    task = _FakeTask(language=None)
    assert _resolve_libero_instruction(task, {}) == "complete the task"
