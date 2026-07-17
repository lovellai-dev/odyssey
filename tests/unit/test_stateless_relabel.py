"""TDD for the memoryless corrective (dagger_relabel_assemble.stateless_corrective).

The latching relabeler's target-noise floor (~0.09 dagger-source val MSE across
six steering configs) traced to targets not being functions of state. This
locks in: determinism, the stage boundaries, and the regrasp fallthrough.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "examples" / "ur5e-drugsort" / "browser_capture"))

from dagger_relabel_assemble import stateless_corrective  # noqa: E402

Q = {n: np.full(6, i, dtype=float) for i, n in
     enumerate(("approach", "descend", "lift", "transport", "lower"))}
GRASP = np.array([-0.45, 0.0, 0.24])
PLACE = np.array([0.39, -0.18, 0.21])
Z0 = 0.22


def _c(pinch, grip, vial=(-0.45, 0.0, 0.22)):
    return stateless_corrective(np.asarray(pinch, float), np.asarray(vial, float),
                                grip, Q, GRASP, PLACE, Z0)


def test_deterministic_same_state_same_target():
    a = _c([-0.30, 0.10, 0.35], 0.0)
    b = _c([-0.30, 0.10, 0.35], 0.0)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


def test_open_far_goes_approach():
    q, g = _c([-0.30, 0.10, 0.35], 0.0)
    assert np.array_equal(q, Q["approach"]) and g == 0.0


def test_open_above_column_descends():
    q, g = _c([-0.45, 0.01, 0.32], 0.0)
    assert np.array_equal(q, Q["descend"]) and g == 0.0


def test_open_at_depth_closes():
    q, g = _c(GRASP + [0.0, 0.0, 0.01], 0.0)
    assert np.array_equal(q, Q["descend"]) and g == 1.0


def test_closed_not_lifted_lifts():
    q, g = _c(GRASP, 1.0, vial=(-0.45, 0.0, Z0 + 0.01))
    assert np.array_equal(q, Q["lift"]) and g == 1.0


def test_closed_lifted_transports():
    q, g = _c([-0.40, 0.0, 0.34], 1.0, vial=(-0.40, 0.0, 0.30))
    assert np.array_equal(q, Q["transport"]) and g == 1.0


def test_over_pocket_lowers_then_releases():
    q, g = _c(PLACE + [0.0, 0.0, 0.06], 1.0, vial=(0.39, -0.18, 0.30))
    assert np.array_equal(q, Q["lower"]) and g == 1.0
    q2, g2 = _c(PLACE + [0.0, 0.0, 0.02], 1.0, vial=(0.39, -0.18, 0.24))
    assert np.array_equal(q2, Q["lower"]) and g2 == 0.0


def test_hover_band_state_gets_moving_corrective():
    # THE fix target: policy hovering 4cm above the vial, open — corrective
    # must command the descend target, deterministically.
    q, g = _c(GRASP + [0.005, 0.005, 0.04], 0.0)
    assert np.array_equal(q, Q["descend"]) and g == 0.0
