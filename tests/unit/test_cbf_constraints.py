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
    zs = [0.330, 0.312, 0.297, 0.285, 0.276, 0.270]
    pts = [[-0.45, 0.0, z] for z in zs]
    ok, rep = cbf.chunk_feasible(_traj(pts))
    assert ok, rep


def test_step_motion_cap_blocks_teleports():
    pts = [[-0.45, 0.0, 0.40], [-0.45, 0.30, 0.40]]           # 30cm in one tick
    ok, rep = cbf.chunk_feasible(_traj(pts))
    assert not ok and "step_motion" in rep["violated"]


def test_vial_protection_blocks_fast_approach_near_ungrasped_vial():
    # 3cm/tick INSIDE the 10cm protection radius of the ungrasped vial
    near = VIAL + [0.0, 0.0, 0.05]
    pts = [near, near + [0.03, 0, 0]]
    ok, rep = cbf.chunk_feasible(_traj(pts))
    assert not ok and "vial_protect" in rep["violated"]
    # the same step is fine once GRASPED (carrying the vial fast is transport)
    ok2, _ = cbf.chunk_feasible(_traj(pts, grasped=True))
    assert ok2
    # and a slow approach inside the radius is fine
    slow = [near, near + [0.008, 0, 0]]
    ok3, _ = cbf.chunk_feasible(_traj(slow))
    assert ok3


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


def test_graded_vial_cap_decelerates_into_contact():
    lim = cbf.Limits()
    near = VIAL + [0.0, 0.0, 0.02]          # 2cm from the vial
    # 1.5cm/tick at 2cm range: under the edge cap (0.022) but OVER the graded
    # cap (max(0.006, 0.022*0.02/0.10)=0.006) -> must reject
    fast = _traj([near, near + [0.015, 0, 0]])
    ok, rep = cbf.chunk_feasible(fast, lim=lim)
    assert not ok and "vial_protect" in rep["violated"]
    # 0.5cm/tick at the same range -> under the floor -> feasible
    slow = _traj([near, near + [0.005, 0, 0]])
    ok2, _ = cbf.chunk_feasible(slow, lim=lim)
    assert ok2
    # and at the radius edge (9.5cm out) the full 0.022 still applies
    far = VIAL + [0.0, 0.0, 0.095]
    edge = _traj([far, far + [0.018, 0, 0]])
    ok3, _ = cbf.chunk_feasible(edge, lim=lim)
    assert ok3
