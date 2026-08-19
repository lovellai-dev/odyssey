"""Unit tests for the monitor-only SafetyBoundary."""

from __future__ import annotations

import numpy as np

from odyssey.runners.agents.brain.safety import SafetyBoundary


def test_clean_action_passes() -> None:
    sb = SafetyBoundary(expected_dim=7)
    result = sb.evaluate(np.zeros(7))
    assert result.ok and result.findings == []


def test_non_finite_is_a_violation() -> None:
    sb = SafetyBoundary(expected_dim=7)
    result = sb.evaluate(np.array([np.nan, 0, 0, 0, 0, 0, 0]))
    assert not result.ok
    assert any(f.check == "finite" and f.severity == "violation" for f in result.findings)


def test_wrong_shape_is_a_warning() -> None:
    sb = SafetyBoundary(expected_dim=7)
    result = sb.evaluate(np.zeros(4))
    assert any(f.check == "shape" for f in result.findings)


def test_out_of_range_is_reported_and_clamped_copy_exposed() -> None:
    sb = SafetyBoundary(expected_dim=7, abs_limit=1.0)
    result = sb.evaluate(np.array([5.0, 0, 0, 0, 0, 0, 0]))
    assert any(f.check == "range" for f in result.findings)
    assert float(result.clamped[0]) == 1.0


def test_monitor_mode_returns_action_unchanged() -> None:
    sb = SafetyBoundary(expected_dim=7, abs_limit=1.0, mode="monitor")
    action = np.array([5.0, 0, 0, 0, 0, 0, 0])
    out = sb.check(action)
    assert float(out[0]) == 5.0  # NOT clamped in monitor mode


def test_enforce_mode_returns_clamped_action() -> None:
    sb = SafetyBoundary(expected_dim=7, abs_limit=1.0, mode="enforce")
    out = sb.check(np.array([5.0, 0, 0, 0, 0, 0, 0]))
    assert float(out[0]) == 1.0
