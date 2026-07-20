"""TDD for the L5 bounded final-centimeter visual servo (scripts/servo_ur5e.py).

Locks the servo's ship-blocking invariants with a STUB FK (no mujoco / no GPU):

  (1) zero correction outside the active radius / in the wrong phase (and it
      returns the INPUT array object -> byte-identical, backward compat);
  (2) inside the radius it moves the pinch TOWARD the target (post FK error < pre);
  (3) ||Delta_q|| respects the absolute bound AND stays a minority of GR00T's own
      commanded step (dominance ratio < 1);
  (4) a correction that would breach a CBF barrier is shrunk to a feasible chunk;
  (5) a gated-off correction is byte-identical to the GR00T chunk.

Run:
    env -u PYTHONPATH .venv-ur5e/bin/python -m pytest tests/unit/test_servo_ur5e.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import cbf_constraints_ur5e as cbf  # noqa: E402
import servo_ur5e as sv  # noqa: E402

# Linear stub FK: pinch = A q + b. Chosen so the joints for a workspace pinch
# stay INSIDE the UR5e limit barrier (shoulder_lift q1 in [-3.14, 0] -> py = -q1;
# a 0.5 m z offset keeps q2 in range). Wrist joints add a small full-rank
# coupling so a single DLS step reduces the Cartesian error.
_A = np.array([
    [1.0,  0.0, 0.0, 0.10, 0.00, 0.00],
    [0.0, -1.0, 0.0, 0.00, 0.10, 0.00],
    [0.0,  0.0, 1.0, 0.00, 0.00, 0.10],
])
_B = np.array([0.0, 0.0, 0.5])


def stub_fk(qs: np.ndarray) -> np.ndarray:
    qs = np.asarray(qs, dtype=np.float64).reshape(-1, 6)
    return (qs @ _A.T) + _B


def q_for_pinch(pinch: np.ndarray) -> np.ndarray:
    """Arm config whose stub-FK pinch equals ``pinch`` (wrist joints zero)."""
    p = np.asarray(pinch, float).reshape(3)
    return np.array([p[0], -p[1], p[2] - 0.5, 0.0, 0.0, 0.0])


def _chunk(start_pinch, end_pinch, T=16):
    """A GR00T-like chunk moving the pinch start->end in a straight joint line."""
    q0 = q_for_pinch(start_pinch)
    q1 = q_for_pinch(end_pinch)
    return np.stack([q0 + (q1 - q0) * (t / (T - 1)) for t in range(T)]).astype(np.float32)


# ---------------------------------------------------------------------------
# (1) gate: wrong phase and out-of-radius -> untouched INPUT object
# ---------------------------------------------------------------------------
def test_wrong_phase_is_zero_correction_and_byte_identical():
    chunk = _chunk([0.30, 0.30, 0.25], [0.31, 0.30, 0.24])
    target = [0.305, 0.305, 0.245]  # within radius, but phase = TRANSPORT
    out, rep = sv.servo_correction(chunk, stub_fk, target, sv.TRANSPORT, False)
    assert out is chunk                      # same object -> byte-identical
    assert rep["applied"] is False
    assert rep["reason"] == "phase_gate"
    assert rep["delta_q_norm"] == 0.0


def test_out_of_radius_is_zero_correction():
    chunk = _chunk([0.30, 0.30, 0.25], [0.31, 0.30, 0.24])
    target = [0.60, 0.30, 0.25]   # ~30 cm away, > 8 cm radius
    out, rep = sv.servo_correction(chunk, stub_fk, target, sv.REACH, False)
    assert out is chunk
    assert rep["reason"] == "out_of_radius"
    assert rep["pinch_err_before_cm"] > 8.0


def test_none_target_is_noop():
    chunk = _chunk([0.30, 0.30, 0.25], [0.31, 0.30, 0.24])
    out, rep = sv.servo_correction(chunk, stub_fk, None, sv.REACH, False)
    assert out is chunk
    assert rep["applied"] is False


# ---------------------------------------------------------------------------
# (2) inside the radius, REACH/GRASP -> pinch moves TOWARD the target
# ---------------------------------------------------------------------------
def test_correction_moves_pinch_toward_target():
    start = np.array([0.30, 0.30, 0.25])
    chunk = _chunk(start, [0.33, 0.31, 0.24])       # GR00T commands real motion
    target = np.array([0.34, 0.335, 0.25])          # ~5 cm from start pinch
    out, rep = sv.servo_correction(chunk, stub_fk, target, sv.GRASP, False)
    assert rep["applied"] is True
    assert out is not chunk
    err_before = np.linalg.norm(target - stub_fk(chunk[0][None])[0])
    err_after = np.linalg.norm(target - stub_fk(out[0][None])[0])
    assert err_after < err_before                   # first-step pinch centred more
    assert rep["pinch_err_after_cm"] < rep["pinch_err_before_cm"]
    # every step shifted by the SAME bounded offset; grip dims never touched here
    assert out.shape == chunk.shape and out.dtype == chunk.dtype


def test_repeated_ticks_converge():
    # closed-loop: re-run the servo linearising at the (moved) pose each tick.
    target = np.array([0.34, 0.335, 0.25])
    cur = np.array([0.30, 0.30, 0.25])
    errs = [float(np.linalg.norm(target - cur))]
    for _ in range(6):
        chunk = _chunk(cur, cur + np.array([0.02, 0.0, 0.0]))
        out, rep = sv.servo_correction(chunk, stub_fk, target, sv.REACH, False)
        cur = stub_fk(out[0][None])[0]
        errs.append(float(np.linalg.norm(target - cur)))
    assert errs[-1] < errs[0]                        # monotone-ish shrink overall


# ---------------------------------------------------------------------------
# (3) bound + dominance: correction is a strict minority of GR00T's own step
# ---------------------------------------------------------------------------
def test_bound_and_dominance_respected():
    start = np.array([0.30, 0.30, 0.25])
    chunk = _chunk(start, [0.36, 0.30, 0.25])        # sizeable GR00T step
    target = np.array([0.37, 0.30, 0.25])            # 7 cm -> big raw nudge
    out, rep = sv.servo_correction(chunk, stub_fk, target, sv.REACH, False,
                                   bound=0.05, max_dominance=0.5)
    dq = (out - chunk)[0]
    assert np.all(np.abs(dq) <= 0.05 + 1e-9)         # absolute per-joint bound
    assert rep["delta_q_norm"] <= rep["groot_step_norm"] * 0.5 + 1e-9  # dominance
    assert rep["dominance_ratio"] < 1.0
    # broadcast: identical offset on every step
    assert np.allclose(out - chunk, dq[None, :])


def test_frozen_pilot_gets_no_correction():
    # GR00T commands NO motion -> the servo may not become the primary actuator.
    q = q_for_pinch([0.30, 0.30, 0.25])
    chunk = np.tile(q, (16, 1)).astype(np.float32)
    target = np.array([0.315, 0.30, 0.25])
    out, rep = sv.servo_correction(chunk, stub_fk, target, sv.REACH, False)
    assert out is chunk
    assert rep["delta_q_norm"] == 0.0


# ---------------------------------------------------------------------------
# (4) CBF: a nudge that would breach a barrier is shrunk to feasible
# ---------------------------------------------------------------------------
def test_cbf_shrinks_correction_to_feasible():
    # GR00T's OWN chunk stays inside a workspace wall (it travels in y, giving
    # the servo plenty of dominance headroom); the servo's x-nudge toward a
    # target past the wall would breach it -> must shrink, not veto GR00T.
    lim = cbf.Limits()
    lim.workspace_hi = np.array([0.33, 0.63, 0.90])   # x wall at 0.33
    start = np.array([0.30, 0.30, 0.25])              # 3 cm from the wall in x
    chunk = _chunk(start, [0.30, 0.12, 0.25])         # GR00T moves in y only
    target = np.array([0.36, 0.30, 0.25])             # past the wall, within radius
    out, rep = sv.servo_correction(chunk, stub_fk, target, sv.REACH, False, lim=lim,
                                   bound=0.5, max_dominance=1.0)
    assert rep["cbf_scale"] < 1.0                      # full nudge was infeasible
    assert rep["cbf_feasible"] is True
    # the kept chunk really passes the SAME filter best-of-N uses
    H = 8
    pinch = stub_fk(out[:H])
    states = [{"phase": sv.REACH, "pinch": pinch[t], "q": out[t], "grasped": False,
               "vial": target, "grasp_target": target} for t in range(H)]
    ok, _ = cbf.chunk_feasible(states, lim=lim)
    assert ok
    assert np.max(pinch[:, 0]) <= lim.workspace_hi[0] + 1e-9


def test_cbf_scale_zero_returns_groot_chunk():
    # If even scale 0 is infeasible the GR00T chunk is returned unchanged.
    lim = cbf.Limits()
    lim.workspace_hi = np.array([0.20, 0.63, 0.90])   # the GR00T pinch is already out
    start = np.array([0.30, 0.30, 0.25])
    chunk = _chunk(start, [0.31, 0.30, 0.25])
    target = np.array([0.315, 0.305, 0.25])
    out, rep = sv.servo_correction(chunk, stub_fk, target, sv.REACH, False, lim=lim)
    assert out is chunk                                # never worse than no-servo
    assert rep["cbf_feasible"] is False


# ---------------------------------------------------------------------------
# (5) backward compat: a gated-off correction is byte-identical
# ---------------------------------------------------------------------------
def test_gated_off_is_byte_identical():
    chunk = _chunk([0.30, 0.30, 0.25], [0.31, 0.30, 0.24])
    for phase in (sv.TRANSPORT, sv.PLACE):
        out, _ = sv.servo_correction(chunk, stub_fk, [0.305, 0.30, 0.25], phase, True)
        assert out is chunk
        assert np.array_equal(out, chunk)
