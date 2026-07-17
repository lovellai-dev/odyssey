"""Control Barrier Function (CBF) constraints for the UR5e drug-sort pipeline.

Discrete-time CBFs h(x) >= 0 defining the safe set, with the decay condition
h(x_{k+1}) >= (1 - gamma) * h(x_k) limiting how fast a trajectory may approach
the boundary. Together with the CLF module (clf_reward_ur5e) this yields the
classic CLF-CBF split: **CBF filters** candidate chunks (hard feasibility),
**CLF ranks** the survivors (progress) — one step of sampled, constrained MPC
over the flow-sampler latent, with the frozen GR00T pilot as the actuator
model and the Observer parameterizing both sides online.

Barriers are locked to failures this program has actually produced:
* ``vial_protect`` — approach-speed cap near the ungrasped vial: steered
  rollouts batted the vial 1-3 m across the cell (v0.2 ep002/ep013, dagger
  ep001). A candidate that moves the pinch fast inside the protection radius
  is infeasible, period.
* ``table_clear`` — pinch stays above the table EXCEPT inside a descend cone
  around the Observer target (so the barrier never fights the grasp itself).
* ``workspace`` / ``joint_margin`` / ``step_motion`` — box, limit and per-step
  displacement invariants.

Perception-aware safety: callers scale ``Limits.shrink(conf)`` with Observer
confidence — degraded perception tightens the safe set toward hold-in-place.
Pure numpy; positions in metres, robot base frame (Observer + FK frame).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class Limits:
    """Cell safety envelope (defaults sized for the aseptipack cell)."""
    # Workspace box MEASURED from the expert's FK'd pinch envelope (base frame,
    # 2026-07-17): x [-0.504, 0.503], y [-0.492, 0.476], z [0.205, 0.342];
    # box = envelope + 0.15 m margin. (The first hand-guessed box capped x at
    # +0.25 — inside the task's own transport arc — and rejected ~90% of real
    # candidates including the pure steering mean.)
    workspace_lo: np.ndarray = field(default_factory=lambda: np.array([-0.66, -0.65, 0.05]))
    workspace_hi: np.ndarray = field(default_factory=lambda: np.array([0.66, 0.63, 0.90]))
    table_z: float = 0.16                 # min pinch height outside the descend cone
    descend_cone_xy: float = 0.06         # radius around the target where descent is allowed
    descend_floor_z: float = 0.015        # min pinch height inside the cone
    # Motion caps CALIBRATED against the expert's own decoded trajectories
    # (gate states, 2026-07-17): all-steps p99 = 0.109 m/tick, near-vial p99 =
    # 0.0138 m/tick. Caps = ~1.5x expert p99 — safety bounds AROUND demonstrated
    # behavior. (The first hand-guessed caps, 0.035/0.012, sat below the
    # expert's p99 and HOLD-rejected 86% of real best-of-N queries.)
    step_motion_max: float = 0.16         # m per 50ms control tick
    vial_protect_r: float = 0.10          # protection radius around the ungrasped vial
    vial_near_step_max: float = 0.022     # m/tick at the radius edge
    # Capture refinement (post-first-lift): 6/15 episodes STRUCK the vial during
    # the closing pass. The cap now GRADES with distance — full 0.022 at the
    # radius edge, floored at ~the expert's near-contact p95 (0.68 cm/tick)
    # at touch range — so approach automatically decelerates into contact.
    vial_step_floor: float = 0.006        # m/tick floor at contact range
    joint_lo: np.ndarray = field(default_factory=lambda: np.array([-6.28, -3.14, -2.9, -6.28, -6.28, -6.28]))
    joint_hi: np.ndarray = field(default_factory=lambda: np.array([6.28, 0.0, 2.9, 6.28, 6.28, 6.28]))
    joint_margin: float = 0.05            # rad of required distance to a limit

    def shrink(self, confidence: float) -> "Limits":
        """Perception-aware tightening: low Observer confidence shrinks motion
        allowances toward hold-in-place (conf 1 -> unchanged; conf 0 -> ~frozen)."""
        c = float(np.clip(confidence, 0.0, 1.0))
        out = Limits(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        out.step_motion_max = self.step_motion_max * (0.1 + 0.9 * c)
        out.vial_near_step_max = self.vial_near_step_max * (0.1 + 0.9 * c)
        return out


def barrier_values(s: dict[str, Any], s_prev: dict[str, Any] | None,
                   lim: Limits) -> dict[str, float]:
    """All h_i(s) (>= 0 means safe) for one step.

    s keys: pinch (3,), q (6,) optional, vial (3,), grasp_target (3,),
            grasped (bool). ``s_prev`` supplies the displacement for the
            motion/vial barriers (None on the first step -> those pass).
    """
    pinch = np.asarray(s["pinch"], float)
    h: dict[str, float] = {}
    h["workspace"] = float(min(np.min(pinch - lim.workspace_lo),
                               np.min(lim.workspace_hi - pinch)))
    xy_to_tgt = float(np.linalg.norm(pinch[:2] - np.asarray(s["grasp_target"], float)[:2]))
    floor = lim.descend_floor_z if xy_to_tgt <= lim.descend_cone_xy else lim.table_z
    h["table_clear"] = float(pinch[2] - floor)
    if "q" in s and s["q"] is not None:
        q = np.asarray(s["q"], float)
        h["joint_margin"] = float(min(np.min(q - lim.joint_lo),
                                      np.min(lim.joint_hi - q)) - lim.joint_margin)
    if s_prev is not None:
        step = float(np.linalg.norm(pinch - np.asarray(s_prev["pinch"], float)))
        h["step_motion"] = float(lim.step_motion_max - step)
        if not bool(s.get("grasped")):
            d_vial = float(np.linalg.norm(pinch - np.asarray(s["vial"], float)))
            if d_vial <= lim.vial_protect_r:
                cap = max(lim.vial_step_floor,
                          lim.vial_near_step_max * d_vial / lim.vial_protect_r)
                h["vial_protect"] = float(cap - step)
    return h


DECAY_BARRIERS = frozenset({"workspace", "table_clear", "joint_margin"})


def chunk_feasible(states: Sequence[dict[str, Any]], *, lim: Limits | None = None,
                   gamma: float = 0.4, tol: float = 0.0,
                   decay_barriers: frozenset[str] = DECAY_BARRIERS
                   ) -> tuple[bool, dict[str, Any]]:
    """CBF feasibility of one candidate chunk (the EXECUTED steps only — callers
    must slice to the execution horizon; steps that never run cannot be unsafe).

    Feasible iff every barrier satisfies h_i >= -tol at every step, AND — for
    STATE-DISTANCE barriers only (``decay_barriers``) — the decay condition
    h_i(k+1) >= (1-gamma)*h_i(k) - tol. Input-type barriers (step_motion,
    vial_protect) are per-step caps whose h fluctuates by nature; applying
    decay to them criminalizes variability (first live run: 1030 spurious
    step_motion:decay rejections). Returns (feasible, report); the report is
    the per-decision explainability artifact.
    """
    lim = lim or Limits()
    worst_h: dict[str, float] = {}
    worst_decay: dict[str, float] = {}
    prev_h: dict[str, float] = {}
    for k, s in enumerate(states):
        hs = barrier_values(s, states[k - 1] if k > 0 else None, lim)
        for name, v in hs.items():
            worst_h[name] = min(worst_h.get(name, np.inf), v)
            if name in prev_h and name in decay_barriers:
                slack = v - (1.0 - gamma) * prev_h[name]
                worst_decay[name] = min(worst_decay.get(name, np.inf), slack)
            prev_h[name] = v
    ok = all(v >= -tol for v in worst_h.values()) and \
        all(v >= -tol for v in worst_decay.values())
    return bool(ok), {
        "feasible": bool(ok),
        "worst_h": {k: float(v) for k, v in worst_h.items()},
        "worst_decay_slack": {k: float(v) for k, v in worst_decay.items()},
        "violated": sorted([k for k, v in worst_h.items() if v < -tol] +
                           [f"{k}:decay" for k, v in worst_decay.items() if v < -tol]),
    }


def filter_candidates(cands: Sequence[Sequence[dict[str, Any]]], *,
                      lim: Limits | None = None, gamma: float = 0.4,
                      tol: float = 0.0) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Vector wrapper for best-of-N: returns (feasible mask, per-candidate reports).

    Intended composition (sampled CLF-CBF-MPC over the latent):
        mask, reps = filter_candidates(decoded)          # CBF: hard filter
        best = argmax over mask of clf.score_chunk(...)  # CLF: progress rank
        if not mask.any(): execute HOLD (zero-motion chunk) and escalate.
    """
    reports = []
    mask = np.zeros(len(cands), dtype=bool)
    for i, states in enumerate(cands):
        ok, rep = chunk_feasible(states, lim=lim, gamma=gamma, tol=tol)
        mask[i] = ok
        reports.append(rep)
    return mask, reports
