"""Unit tests for the Observer agent's visual servoing + verify/re-grasp.

Pure numpy — no mujoco/torch. The IK/FK are injected analytic fakes: ``fk`` reads
the pinch position straight out of the first three joint slots and ``ik_solve``
writes a target back into them, so the geometry of the Observer's servoing and
re-grasp logic is exercised without a physics engine or the DINOv2/CNN heads.
The perception step is bypassed by handing :meth:`ObserverAgent.servo` a
pre-built :class:`ObserverReading`, so these tests need no torch either.
"""

from __future__ import annotations

import numpy as np

from odyssey.embodiments.ur5e_drugsort.observer import (
    GRASP_PINCH_BELOW_CAP,
    PH_GRASPED,
    PH_LIFTED,
    PH_REACHING,
    ObserverAgent,
    ObserverReading,
    ObserverServoConfig,
    base_to_world,
    bounded_nudge,
    grasp_pinch_target,
    in_box,
    world_to_base,
)

IDENT = np.eye(3)
POCKET = np.array([0.39, -0.174, 0.228])


def _fk(q):
    return np.array([q[0], q[1], q[2]], dtype=float)


def _ik(target, seed):
    t = np.asarray(target, dtype=float).reshape(3)
    return (float(t[0]), float(t[1]), float(t[2]), seed[3], seed[4], seed[5])


def _agent(**cfg):
    return ObserverAgent(
        ik_solve=_ik, fk=_fk, base_pos=np.zeros(3), base_mat=IDENT,
        pocket_world=POCKET, config=ObserverServoConfig(**cfg),
    )


def _reading(cap_world, phase, *, lo=(0.2, -0.45, 0.16), hi=(0.75, 0.45, 0.45)):
    cw = np.asarray(cap_world, float)
    probs = np.zeros(4)
    probs[phase] = 1.0
    return ObserverReading(cap_base=cw.copy(), cap_world=cw, cap_ok=in_box(cw, lo, hi),
                           phase_label=phase, phase_probs=probs)


def _action(*qs, grip):
    return {"chunk_q": list(qs), "chunk_grip": [grip] * len(qs)}


# --- pure helpers -----------------------------------------------------------

def test_base_to_world_inverts_world_to_base():
    base_pos = np.array([0.1, -0.2, 0.0])
    base_mat = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])  # 180deg about z
    p_world = np.array([0.45, 0.15, 0.22])
    back = base_to_world(world_to_base(p_world, base_pos, base_mat), base_pos, base_mat)
    assert np.allclose(back, p_world, atol=1e-9)


def test_grasp_pinch_target_drops_below_cap():
    cap = np.array([0.4, 0.1, 0.30])
    tgt = grasp_pinch_target(cap)
    assert np.allclose(tgt[:2], cap[:2])
    assert abs(tgt[2] - (0.30 - GRASP_PINCH_BELOW_CAP)) < 1e-9


def test_bounded_nudge_clamps_norm():
    step = bounded_nudge(np.zeros(3), np.array([1.0, 0, 0]), gain=1.0, max_step=0.05)
    assert abs(np.linalg.norm(step) - 0.05) < 1e-9
    small = bounded_nudge(np.zeros(3), np.array([0.02, 0, 0]), gain=0.5, max_step=0.05)
    assert np.allclose(small, [0.01, 0.0, 0.0])


# --- visual servoing --------------------------------------------------------

def test_servo_grasp_pulls_pinch_toward_cap():
    ag = _agent()
    cap = np.array([0.40, 0.10, 0.30])
    cmd = (0.43, 0.07, 0.28, 0.0, 0.0, 0.0)          # 3cm off, inside gate + agreement
    arm = (0.43, 0.07, 0.29, 0.0, 0.0, 0.0)
    res = ag.servo(_reading(cap, PH_REACHING), _action(cmd, grip=0.0), arm)
    assert res.mode == "servo_grasp" and res.corrected
    goal = grasp_pinch_target(cap)
    assert np.linalg.norm(np.array(res.chunk_q[0][:3]) - goal) < np.linalg.norm(np.array(cmd[:3]) - goal)


def test_servo_skips_when_perception_disagrees():
    ag = _agent()
    cap = np.array([0.40, 0.10, 0.30])
    cmd = (0.9, 0.9, 0.30, 0, 0, 0)                   # GR00T far from the observed cap
    arm = (0.42, 0.11, 0.31, 0, 0, 0)                 # near cap so the gate opens
    res = ag.servo(_reading(cap, PH_REACHING), _action(cmd, grip=0.0), arm)
    assert res.mode == "passthrough" and res.chunk_q[0] == cmd


def test_far_from_cap_is_passthrough():
    ag = _agent()
    cap = np.array([0.40, 0.10, 0.30])
    cmd = (0.10, 0.60, 0.35, 0, 0, 0)                 # pinch >15cm from cap -> gate closed
    res = ag.servo(_reading(cap, PH_REACHING), _action(cmd, grip=0.0), cmd)
    assert res.mode == "passthrough"


def test_place_servo_only_over_nest():
    ag = _agent()
    cap = np.array([0.40, 0.10, 0.30])
    far = (0.40, 0.20, 0.35, 0, 0, 0)
    assert ag.servo(_reading(cap, PH_LIFTED), _action(far, grip=1.0), far).mode == "passthrough"
    over = (0.40, -0.16, 0.28, 0, 0, 0)
    res = ag.servo(_reading(cap, PH_LIFTED), _action(over, grip=1.0), over)
    assert res.mode == "servo_place" and res.corrected


# --- grasp verification + re-grasp -----------------------------------------

def test_verify_grasp_true_when_holding_and_centered():
    ag = _agent()
    cap = np.array([0.40, 0.10, 0.30])
    arm = (0.40, 0.10, 0.30, 0, 0, 0)                 # pinch under the cap
    assert ag.verify_grasp(_reading(cap, PH_GRASPED), arm) is True
    off = (0.50, 0.10, 0.30, 0, 0, 0)                 # pinch 10cm off in x
    assert ag.verify_grasp(_reading(cap, PH_GRASPED), off) is False
    assert ag.verify_grasp(_reading(cap, PH_REACHING), arm) is False  # not a holding phase


def test_closed_but_reaching_triggers_regrasp_sequence():
    ag = _agent(ticks_per_subphase=1)
    cap = np.array([0.40, 0.10, 0.30])
    arm = (0.40, 0.10, 0.30, 0.0, 0.0, 0.0)
    act = _action((0, 0, 0, 0, 0, 0), grip=1.0)       # closed but classifier says 'reaching'
    rd = _reading(cap, PH_REACHING)

    r1 = ag.servo(rd, act, arm)      # OPEN: gripper commanded open
    assert r1.mode == "regrasp" and all(g == 0.0 for g in r1.chunk_grip)
    r2 = ag.servo(rd, act, arm)      # DESCEND: open, pinch driven to the grasp target
    assert r2.mode == "regrasp" and np.allclose(r2.chunk_q[0][:3], grasp_pinch_target(cap), atol=1e-6)
    r3 = ag.servo(rd, act, arm)      # CLOSE: re-close on the cap
    assert r3.mode == "regrasp" and all(g == 1.0 for g in r3.chunk_grip)
    r4 = ag.servo(rd, act, arm)      # attempt spent -> hand back to GR00T
    assert r4.mode == "passthrough"


def test_regrasp_respects_max_attempts():
    ag = _agent(ticks_per_subphase=1, regrasp_max_attempts=0)
    cap = np.array([0.40, 0.10, 0.30])
    arm = (0.40, 0.10, 0.30, 0, 0, 0)
    res = ag.servo(_reading(cap, PH_REACHING), _action((0, 0, 0, 0, 0, 0), grip=1.0), arm)
    assert res.mode == "passthrough"   # zero budget -> never overrides
