"""TDD for the CBF safety constraints (scripts/cbf_constraints_ur5e.py).

Locks in the measured failure modes: the vial-batting barrier (steered rollouts
knocked the vial 1-3 m), the descend-cone exception (safety must never fight
the grasp itself), and perception-aware tightening. Pure numpy:

    env -u PYTHONPATH .venv-ur5e/bin/python -m pytest tests/unit/test_cbf_constraints.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import cbf_constraints_ur5e as cbf  # noqa: E402

# Realistic base-frame geometry (measured expert envelope: grasp z ~0.21-0.27)
TGT = np.array([-0.45, 0.0, 0.25])
VIAL = np.array([-0.45, 0.0, 0.23])


def _s(pinch, vial=VIAL, grasped=False, q=None):
    return {"pinch": np.asarray(pinch, float), "vial": np.asarray(vial, float),
            "grasp_target": TGT, "grasped": grasped, "q": q}


def _traj(points, **kw):
    return [_s(p, **kw) for p in points]


def test_nominal_in_workspace_trajectory_is_feasible():
    # descent that stays OUTSIDE the vial-protection radius (>=13cm above)
    pts = [[-0.45, 0.0, 0.45 - 0.01 * k] for k in range(8)]
    ok, rep = cbf.chunk_feasible(_traj(pts))
    assert ok, rep


def test_table_clearance_blocks_dive_far_from_target():
    pts = [[-0.20, 0.30, 0.30], [-0.20, 0.30, 0.10]]          # dives below table_z away from target
    ok, rep = cbf.chunk_feasible(_traj(pts))
    assert not ok and "table_clear" in rep["violated"]


def test_descend_cone_allows_slow_grasp_descent_at_target():
    # Inside the cone AND inside the vial-protection radius: descent is allowed
    # but must be SLOW (<= vial_near_step_max) — the barrier composition
    # enforces the expert's own gentle final approach.
    # DECELERATING descent into the cap zone — the graded barrier's contract:
    # full speed at the radius edge, expert-pace at contact range.
    zs = [0.330, 0.315, 0.302, 0.291, 0.282, 0.275]   # each step ~80% of the graded cap
    pts = [[-0.45, 0.0, z] for z in zs]
    ok, rep = cbf.chunk_feasible(_traj(pts))
    assert ok, rep


def test_step_motion_cap_blocks_teleports():
    pts = [[-0.45, 0.0, 0.40], [-0.45, 0.30, 0.40]]           # 30cm in one tick
    ok, rep = cbf.chunk_feasible(_traj(pts))
    assert not ok and "step_motion" in rep["violated"]


def _vial_h(pinch, prev, step_dist, grasped=False):
    """vial_protect barrier at `pinch` (d from VIAL) after a `step_dist` move."""
    lim = cbf.Limits()
    prev_s = _s(prev, vial=VIAL, grasped=grasped)
    # place prev at exactly step_dist from pinch along -x so the recorded step matches
    hv = cbf.barrier_values(_s(pinch, vial=VIAL, grasped=grasped), prev_s, lim)
    return hv


def test_vial_protection_tapers_fast_approach_to_gentle_contact():
    lim = cbf.Limits()
    # contact (d=1cm): cap = 0.02 + 0.14*0.1 = 0.034
    c = VIAL + [0.01, 0.0, 0.0]                      # 1cm from vial (in x)
    h_slam = cbf.barrier_values(_s(c, vial=VIAL), _s(c + [0.04, 0, 0], vial=VIAL), lim)
    assert h_slam["vial_protect"] < 0                # 4cm/tick > 0.034 -> reject
    h_gentle = cbf.barrier_values(_s(c, vial=VIAL), _s(c + [0.015, 0, 0], vial=VIAL), lim)
    assert h_gentle["vial_protect"] > 0              # 1.5cm/tick < 0.034 -> ok
    # at 6cm (cap = 0.02+0.14*0.6 = 0.104) a 6cm/tick step is ADMITTED
    mid = VIAL + [0.06, 0.0, 0.0]
    h_mid = cbf.barrier_values(_s(mid, vial=VIAL), _s(mid + [0.06, 0, 0], vial=VIAL), lim)
    assert h_mid["vial_protect"] > 0
    # once GRASPED the vial barrier does not apply at all
    h_grasp = cbf.barrier_values(_s(c, vial=VIAL, grasped=True),
                                 _s(c + [0.04, 0, 0], vial=VIAL, grasped=True), lim)
    assert "vial_protect" not in h_grasp


def test_joint_margin_barrier():
    q_ok = np.zeros(6); q_ok[1] = -1.5
    q_bad = np.zeros(6); q_bad[1] = -3.13     # against the shoulder_lift limit
    ok, _ = cbf.chunk_feasible([_s([-0.45, 0, 0.4], q=q_ok)] * 2)
    bad, rep = cbf.chunk_feasible([_s([-0.45, 0, 0.4], q=q_bad)] * 2)
    assert ok and not bad and "joint_margin" in rep["violated"]


def test_decay_condition_flags_barrier_erosion():
    # workspace margin eroding faster than gamma=0.4 allows per step
    pts = [[-0.45, 0.0, 0.60], [-0.45, 0.0, 0.86]]  # jumps toward the z ceiling 0.90
    ok, rep = cbf.chunk_feasible(_traj(pts), gamma=0.1)
    assert not ok
    assert any(v.endswith(":decay") or v == "step_motion" for v in rep["violated"])


def test_perception_aware_shrink_tightens_motion():
    lim = cbf.Limits()
    tight = lim.shrink(0.0)
    assert tight.step_motion_max < 0.2 * lim.step_motion_max + 1e-9
    assert lim.shrink(1.0).step_motion_max == lim.step_motion_max


def test_filter_candidates_mask_and_reports():
    good = _traj([[-0.45, 0.0, 0.40], [-0.45, 0.0, 0.385]])
    bad = _traj([[-0.45, 0.0, 0.40], [-0.45, 0.30, 0.40]])
    mask, reps = cbf.filter_candidates([good, bad])
    assert mask.tolist() == [True, False]
    assert reps[1]["violated"]


def test_taper_cap_scales_with_distance():
    lim = cbf.Limits()
    # cap(d) = 0.02 + 0.14*(d/0.10); a 5cm/tick step: rejected at 2cm (cap 0.048),
    # admitted at 5cm (cap 0.09).
    d2 = VIAL + [0.02, 0.0, 0.0]
    d5 = VIAL + [0.05, 0.0, 0.0]
    h2 = cbf.barrier_values(_s(d2, vial=VIAL), _s(d2 + [0.05, 0, 0], vial=VIAL), lim)
    h5 = cbf.barrier_values(_s(d5, vial=VIAL), _s(d5 + [0.05, 0, 0], vial=VIAL), lim)
    assert h2["vial_protect"] < 0 < h5["vial_protect"]
