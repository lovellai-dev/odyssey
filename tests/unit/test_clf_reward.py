"""TDD for the CLF (Control Lyapunov Function) reward (scripts/clf_reward_ur5e.py).

Locks in the two design rules this program's own data demanded: distance
hacking earns zero (the v0.2 ep007 vial-pushing exploit), and phase
transitions pay fixed bonuses with V resets (no cross-phase jump reward).
Pure numpy — runs locally:

    env -u PYTHONPATH .venv-ur5e/bin/python -m pytest tests/unit/test_clf_reward.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import clf_reward_ur5e as clf  # noqa: E402

TGT = np.array([-0.45, 0.0, 0.25])
POCKET = np.array([0.39, -0.18, 0.20])


def _s(phase, pinch, grip=0.0, grasped=False, vial=(0.45, 0.15, 0.22)):
    return {"phase": phase, "pinch": np.asarray(pinch, float), "grip": grip,
            "grasped": grasped, "grasp_target": TGT, "vial": np.asarray(vial, float),
            "pocket": POCKET}


# ---- V definitions ----------------------------------------------------------

def test_reach_V_decreases_toward_target_and_zero_at_goal():
    far = clf.clf_value(_s(clf.REACH, TGT + [0.1, 0.1, 0.1]))
    near = clf.clf_value(_s(clf.REACH, TGT + [0.01, 0, 0]))
    assert far > near > 0.0
    assert clf.clf_value(_s(clf.REACH, TGT)) == 0.0


def test_grasp_V_rewards_closing_at_target():
    open_ = clf.clf_value(_s(clf.GRASP, TGT, grip=0.0))
    closed = clf.clf_value(_s(clf.GRASP, TGT, grip=1.0))
    assert open_ > closed == 0.0


def test_transport_and_place_gated_on_grasped():
    # UNGRASPED vial near the pocket contributes nothing — the ep007 exploit.
    assert clf.clf_value(_s(clf.TRANSPORT, TGT, grasped=False)) is None
    assert clf.clf_value(_s(clf.PLACE, TGT, grasped=False)) is None
    assert clf.clf_value(_s(clf.TRANSPORT, TGT, grasped=True)) is not None


# ---- shaping reward ---------------------------------------------------------

def test_reward_positive_on_approach_negative_on_retreat():
    a = _s(clf.REACH, TGT + [0.10, 0, 0])
    b = _s(clf.REACH, TGT + [0.05, 0, 0])
    assert clf.clf_reward(a, b) > 0
    assert clf.clf_reward(b, a) < 0


def test_pushing_vial_toward_pocket_ungrasped_earns_zero():
    # vial moves 20cm toward the pocket but grasped=False, phase TRANSPORT
    a = _s(clf.TRANSPORT, TGT, grasped=False, vial=POCKET + [0.30, 0, 0])
    b = _s(clf.TRANSPORT, TGT, grasped=False, vial=POCKET + [0.10, 0, 0])
    assert clf.clf_reward(a, b) == 0.0


def test_phase_advance_pays_bonus_not_jump():
    # REACH (V small) -> GRASP (V large because gripper open): the raw V jump is
    # positive-large; the reward must be exactly the GRASP completion bonus.
    a = _s(clf.REACH, TGT + [0.001, 0, 0])
    b = _s(clf.GRASP, TGT, grip=0.0)
    assert clf.clf_reward(a, b) == clf.COMPLETION_BONUS[clf.GRASP]


def test_phase_regression_charges_bonus_back():
    a = _s(clf.GRASP, TGT, grip=0.9)
    b = _s(clf.REACH, TGT + [0.02, 0, 0])
    assert clf.clf_reward(a, b) == -clf.COMPLETION_BONUS[clf.GRASP]
    # drop-regrasp cycles cannot farm net bonus
    assert clf.clf_reward(a, b) + clf.clf_reward(b, a) == 0.0


# ---- chunk scoring ----------------------------------------------------------

def _reach_traj(dists):
    return [_s(clf.REACH, TGT + [d, 0, 0]) for d in dists]


def test_score_chunk_ranks_descent_over_hover():
    descend = clf.score_chunk(_reach_traj(np.linspace(0.10, 0.01, 8)))
    hover = clf.score_chunk(_reach_traj([0.06] * 8))
    retreat = clf.score_chunk(_reach_traj(np.linspace(0.02, 0.12, 8)))
    assert descend["total_reward"] > hover["total_reward"] > retreat["total_reward"]
    assert descend["monotone_frac"] == 1.0
    assert retreat["violation_frac"] == 1.0


def test_exp_margin_flags_stalls():
    ok = clf.exp_decrease_margin([0.01, 0.005, 0.002, 0.0008], alpha=0.5)
    stall = clf.exp_decrease_margin([0.01, 0.0099, 0.0098, 0.0098], alpha=0.5)
    assert ok <= 0.0 < stall


def test_reach_series_metrics_shapes():
    pinch = TGT[None, :] + np.linspace(0.1, 0.0, 6)[:, None] * np.array([1.0, 0, 0])
    m = clf.reach_series_metrics(pinch, TGT)
    assert m["total_reward"] > 0 and m["monotone_frac"] == 1.0


def test_score_chunk_handles_mixed_phases_without_cross_jump():
    traj = _reach_traj(np.linspace(0.08, 0.005, 4)) + [
        _s(clf.GRASP, TGT, grip=g) for g in (0.0, 0.5, 1.0)]
    m = clf.score_chunk(traj)
    reach_gain = 0.08 ** 2 - 0.005 ** 2
    grasp_gain = 1.0  # open (V=1) -> closed (V=0)
    expected = reach_gain + clf.COMPLETION_BONUS[clf.GRASP] + grasp_gain
    assert abs(m["total_reward"] - expected) < 1e-9


def test_closing_credit_is_graded_by_centering():
    # centered: closing reduces V to exactly 0
    assert clf.clf_value(_s(clf.GRASP, TGT, grip=1.0)) == 0.0
    assert clf.clf_value(_s(clf.GRASP, TGT, grip=0.0)) > 0.5
    # 3cm off-center: closing earns only a small fraction of the centered credit
    off = TGT + [0.03, 0, 0]
    credit_off = (clf.clf_value(_s(clf.GRASP, off, grip=0.0))
                  - clf.clf_value(_s(clf.GRASP, off, grip=1.0)))
    credit_ctr = (clf.clf_value(_s(clf.GRASP, TGT, grip=0.0))
                  - clf.clf_value(_s(clf.GRASP, TGT, grip=1.0)))
    assert 0.0 < credit_off < 0.2 * credit_ctr
    # at the achieved-precision band (1.6cm) closing earns MEANINGFUL credit
    near = TGT + [0.016, 0, 0]
    credit_near = (clf.clf_value(_s(clf.GRASP, near, grip=0.0))
                   - clf.clf_value(_s(clf.GRASP, near, grip=1.0)))
    assert credit_near > 0.3 * credit_ctr
    # and centering monotonically increases closing credit
    assert credit_ctr > credit_near > credit_off
