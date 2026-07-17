"""TDD for the trajectory-shape domain randomization (build_adaptive_phases dr=).

Tests the DR LOGIC with a STUB IK (solve returns the target as q) so no mujoco
model is needed. Locks in: dr=None reproduces the deterministic expert exactly
(eval unchanged); dr= jitters clearance/approach-offset/grasp-depth/velocity so
trajectory SHAPE varies while the DESCEND grasp column stays on the vial.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from odyssey.embodiments.ur5e_drugsort.ik import IKSolveInfo, build_adaptive_phases  # noqa: E402

VIAL = np.array([-0.45, 0.0, 0.23])
POCKET = np.array([-0.39, 0.174, 0.228])


class _StubIK:
    """solve(target, seed) -> IKSolveInfo whose q encodes the target xyz (so the
    test can read back which Cartesian anchor each phase targeted)."""
    def solve(self, target, seed):
        t = np.asarray(target, dtype=float).reshape(3)
        q = (float(t[0]), float(t[1]), float(t[2]), 0.0, 0.0, 0.0)
        return IKSolveInfo(q=q, pos_error=0.0, iters=1, converged=True)


def _phase(plan, name):
    return [w for w in plan.phases if w.name == name][0]


def test_dr_none_is_deterministic():
    a = build_adaptive_phases(_StubIK(), vial_xyz=VIAL, pocket_xyz=POCKET)
    b = build_adaptive_phases(_StubIK(), vial_xyz=VIAL, pocket_xyz=POCKET)
    for wa, wb in zip(a.phases, b.phases):
        assert np.allclose(wa.q, wb.q) and wa.move_steps == wb.move_steps
    # descend targets the vial column exactly, at the nominal grasp depth
    dq = _phase(a, "descend").q
    assert np.allclose(dq[:2], VIAL[:2]) and abs(dq[2] - (VIAL[2] - 0.010)) < 1e-9


def test_dr_varies_approach_clearance_and_offset():
    approach_z, approach_xy = [], []
    for seed in range(8):
        dr = {"rng": np.random.default_rng(seed), "clearance": 0.04,
              "approach_xy": 0.03, "grasp_z": 0.003, "carry_z": 0.03,
              "vel_scale": (0.7, 1.4)}
        p = build_adaptive_phases(_StubIK(), vial_xyz=VIAL, pocket_xyz=POCKET, dr=dr)
        aq = _phase(p, "approach").q
        approach_z.append(aq[2]); approach_xy.append(aq[:2])
    assert np.std(approach_z) > 1e-3                       # clearance varies
    assert np.std(np.array(approach_xy), axis=0).max() > 1e-3  # angled approach varies


def test_dr_velocity_scale_varies():
    ms = []
    for seed in range(8):
        dr = {"rng": np.random.default_rng(seed), "vel_scale": (0.7, 1.4)}
        p = build_adaptive_phases(_StubIK(), vial_xyz=VIAL, pocket_xyz=POCKET, dr=dr)
        ms.append(_phase(p, "approach").move_steps)
    assert len(set(ms)) > 1                                # per-episode speed varies


def test_dr_descend_column_stays_on_vial():
    # the grasp itself must be unaffected: DESCEND targets the true vial xy even
    # under DR (only the APPROACH hover is offset)
    dr = {"rng": np.random.default_rng(5), "approach_xy": 0.03, "clearance": 0.04}
    p = build_adaptive_phases(_StubIK(), vial_xyz=VIAL, pocket_xyz=POCKET, dr=dr)
    dq = _phase(p, "descend").q
    aq = _phase(p, "approach").q
    assert np.allclose(dq[:2], VIAL[:2])                  # descend on the vial column
    assert not np.allclose(aq[:2], VIAL[:2])              # approach offset from it
