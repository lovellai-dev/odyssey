"""TDD for best-of-N selection (scripts/bestofn_select.py): CBF filters, CLF
ranks, HOLD falls back. Pure numpy with a stub FK."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import bestofn_select as bon  # noqa: E402

TGT = np.array([-0.45, 0.0, 0.05])
VIAL = np.array([-0.45, 0.0, 0.03])


def _fk(q6):
    """Stub FK: pinch = (q3, q4, q5) (wrist joints carry the pinch; limits ±6.28)."""
    q = np.asarray(q6, float)
    return q[3:6].copy()


def _traj_from_pinch(points):
    """arm traj whose stub-FK pinch follows `points`; joints 0-2 held safe."""
    T = len(points)
    arm = np.zeros((T, 6))
    arm[:, 1] = -1.5           # shoulder_lift away from its 0.0 upper limit
    arm[:, 3:6] = np.asarray(points, float)
    return arm


def _line(start, end, T=16):
    return np.linspace(np.asarray(start, float), np.asarray(end, float), T)


def test_selects_descending_over_hovering():
    hover = _traj_from_pinch(_line(TGT + [0, 0, 0.30], TGT + [0, 0, 0.295]))
    descend = _traj_from_pinch(_line(TGT + [0, 0, 0.30], TGT + [0, 0, 0.16]))
    grips = np.zeros((2, 16))
    idx, rep = bon.select(np.stack([hover, descend]), grips, _fk,
                          vial=VIAL, grasp_target=TGT, pocket=None,
                          phase=0, grasped=False)
    assert idx == 1 and rep["n_feasible"] == 2


def test_unsafe_fast_candidate_filtered_even_if_most_progress():
    # teleporting candidate makes the most CLF progress but violates step_motion
    teleport = _traj_from_pinch([TGT + [0, 0, 0.40], TGT + [0, 0, 0.06]] * 8)
    gentle = _traj_from_pinch(_line(TGT + [0, 0, 0.40], TGT + [0, 0, 0.30]))
    grips = np.zeros((2, 16))
    idx, rep = bon.select(np.stack([teleport, gentle]), grips, _fk,
                          vial=VIAL, grasp_target=TGT, pocket=None,
                          phase=0, grasped=False)
    assert idx == 1
    assert rep["n_feasible"] == 1 and rep["cbf_rejections"]


def test_all_infeasible_returns_hold():
    t1 = _traj_from_pinch([TGT + [0, 0, 0.40], TGT + [0.3, 0, 0.40]] * 8)
    t2 = _traj_from_pinch([TGT + [0, 0, 0.40], TGT + [0, 0.3, 0.40]] * 8)
    grips = np.zeros((2, 16))
    idx, rep = bon.select(np.stack([t1, t2]), grips, _fk,
                          vial=VIAL, grasp_target=TGT, pocket=None,
                          phase=0, grasped=False)
    assert idx is None and rep["fallback_hold"] is True


def test_grasp_phase_prefers_closing_candidate():
    at = _traj_from_pinch(np.tile(TGT + [0, 0, 0.005], (16, 1)))
    grips_open = np.zeros(16)
    grips_close = np.linspace(0, 1, 16)
    idx, rep = bon.select(np.stack([at, at]),
                          np.stack([grips_open, grips_close]), _fk,
                          vial=VIAL, grasp_target=TGT, pocket=None,
                          phase=1, grasped=False)
    assert idx == 1


def test_hold_chunk_shape_and_values():
    q, g = bon.hold_chunk(np.arange(6, dtype=float), 0.7, horizon=16)
    assert q.shape == (16, 6) and g.shape == (16,)
    assert np.allclose(q[5], np.arange(6)) and np.allclose(g, 0.7)


def test_grip_hold_guard_blocks_mid_transport_release():
    at = _traj_from_pinch(np.tile(TGT + [0, 0, 0.05], (16, 1)))
    hold = np.ones(16)
    # release INSIDE the executed prefix (steps 0-7) — a release at steps 8+
    # never executes and correctly does not trigger the guard
    release = np.concatenate([np.ones(3), np.zeros(13)])
    idx, rep = bon.select(np.stack([at, at]), np.stack([release, hold]), _fk,
                          vial=VIAL, grasp_target=TGT, pocket=[0.39, -0.18, 0.21],
                          phase=2, grasped=True)
    assert idx == 1
    assert "grip_hold" in rep["cbf_rejections"]


def test_grip_release_allowed_in_place_phase():
    # in PLACE phase, releasing is legitimate (that's how the vial is seated)
    at = _traj_from_pinch(np.tile(TGT + [0, 0, 0.05], (16, 1)))
    release = np.concatenate([np.ones(3), np.zeros(13)])
    idx, rep = bon.select(np.stack([at]), np.stack([release]), _fk,
                          vial=VIAL, grasp_target=TGT, pocket=[0.39, -0.18, 0.21],
                          phase=3, grasped=True)
    assert idx == 0 and "grip_hold" not in (rep["cbf_rejections"] or {})


def test_make_cem_seeds_elite_preserved():
    elite = np.random.default_rng(1).standard_normal((40, 132)).astype(np.float32)
    seeds = bon.make_cem_seeds(elite, 8, 0.2, rng=np.random.default_rng(2))
    assert seeds.shape == (8, 40, 132)
    assert np.array_equal(seeds[0], elite)          # slot 0 = the elite, exactly
    assert not np.array_equal(seeds[1], elite)      # others are jittered
    z = bon.make_cem_seeds(elite, 4, 0.0)
    assert np.allclose(z, elite[None])              # sigma 0 -> pure elite
