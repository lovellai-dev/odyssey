"""Control-Lyapunov-Function (CLF) reward for the UR5e drug-sort pipeline.

A per-phase, predicate-gated Lyapunov potential V(s) >= 0 with V(goal) = 0,
rewarding DECREASE (potential-based shaping — policy-invariant, Ng et al. 1999)
rather than raw distance. Design is dictated by two of this program's own
measurements:

* The v0.2 browser A/B episode 7 decreased vial-to-pocket distance to 6.5 cm by
  PUSHING the ungrasped vial — so transport/place potentials contribute ONLY
  while the ``grasped`` predicate holds, and the reach potential is defined on
  the PINCH (not the vial). Distance hacking earns zero by construction.
* The task is hybrid/contact-rich, so a single global smooth CLF does not
  exist; per-phase potentials with predicate-gated transitions (and V resets
  across transitions — no cross-phase jump reward) follow standard hybrid-
  systems practice. Phase completions pay fixed bonuses instead.

Consumers (learning ladder):
  R2 best-of-N   — ``score_chunk`` ranks decoded candidate chunks render-free.
  R3 self-imit.  — ``score_chunk``'s monotone/violation stats filter winners.
  Runtime guard  — ``exp_decrease_margin`` rejects chunks that raise V.
  Offline gate   — report-only reach-phase metrics (wired in fk-gate).

Pure numpy; no torch/mujoco imports. All positions are metres in a single
consistent frame supplied by the caller (Observer targets and FK pinch both
live in the robot base frame in this stack).
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

REACH, GRASP, TRANSPORT, PLACE = 0, 1, 2, 3

# Weights/geometry (metres). HOVER lifts the transport goal above the pocket so
# transport's V=0 sits at the pre-place hover, not inside the rack wall.
GRASP_PINCH_W = 0.5          # keep-pinch-on-target term inside the grasp phase
GRASP_ALIGN_R = 0.012        # closing earns credit ONLY within this centering radius
POCKET_HOVER_M = 0.06
COMPLETION_BONUS = {GRASP: 1.0, TRANSPORT: 0.5, PLACE: 2.0}   # paid on ENTERING
ALPHA_DEFAULT = 0.5          # exponential-decrease rate for the margin check


def clf_value(s: dict[str, Any]) -> float | None:
    """V(s) for state dict s; None when the phase's predicate gate fails.

    s keys: phase (int), pinch (3,), grip (float 0..1), grasped (bool),
            grasp_target (3,), vial (3,), pocket (3,).
    Transport/place REQUIRE grasped=True — an ungrasped vial moving toward the
    pocket contributes nothing (the ep007 pushing exploit earns zero).
    """
    phase = int(s["phase"])
    if phase == REACH:
        d = np.asarray(s["pinch"], float) - np.asarray(s["grasp_target"], float)
        return float(d @ d)
    if phase == GRASP:
        d = np.asarray(s["pinch"], float) - np.asarray(s["grasp_target"], float)
        dist2 = float(d @ d)
        open_frac = 1.0 - float(np.clip(s["grip"], 0.0, 1.0))
        # Capture refinement: the closure term is CENTERING-GATED — while the
        # pinch is outside GRASP_ALIGN_R of the target, the openness cost is
        # frozen at its maximum, so closing off-center earns NOTHING and
        # "center first, then close" is strictly optimal (6/15 first-lift
        # episodes struck the vial with off-center closes).
        close_term = open_frac ** 2 if dist2 <= GRASP_ALIGN_R ** 2 else 1.0
        return float(close_term + GRASP_PINCH_W * dist2)
    if phase == TRANSPORT:
        if not bool(s.get("grasped")):
            return None
        goal = np.asarray(s["pocket"], float) + np.array([0.0, 0.0, POCKET_HOVER_M])
        d = np.asarray(s["vial"], float) - goal
        return float(d @ d)
    if phase == PLACE:
        if not bool(s.get("grasped")):
            return None
        d = np.asarray(s["vial"], float) - np.asarray(s["pocket"], float)
        return float(d @ d)
    raise ValueError(f"unknown phase {phase}")


def clf_reward(prev: dict[str, Any], curr: dict[str, Any]) -> float:
    """Potential-shaping step reward r = V(prev) - V(curr), phase-aware.

    Same phase: the shaped decrease (0 if either V is gated off).
    Phase ADVANCE (curr.phase == prev.phase + 1): fixed completion bonus only —
    V resets across the boundary, so the jump itself is never rewarded.
    Phase REGRESSION (e.g. a drop: grasped lost): the completion bonus of the
    lost phase is charged back (symmetric, so drop-regrasp cycles cannot farm
    bonuses).
    """
    pp, cp = int(prev["phase"]), int(curr["phase"])
    if cp == pp:
        v0, v1 = clf_value(prev), clf_value(curr)
        if v0 is None or v1 is None:
            return 0.0
        return v0 - v1
    if cp > pp:
        return float(sum(COMPLETION_BONUS.get(p, 0.0) for p in range(pp + 1, cp + 1)))
    return -float(sum(COMPLETION_BONUS.get(p, 0.0) for p in range(cp + 1, pp + 1)))


def exp_decrease_margin(Vs: Sequence[float], alpha: float = ALPHA_DEFAULT,
                        dt: float = 1.0) -> float:
    """Worst-case margin of the exponential-decrease condition V' <= -alpha*V.

    Negative margin everywhere == condition satisfied. Returns max over steps of
    (V_{k+1} - V_k)/dt + alpha*V_k  — a runtime guard rejects chunks whose
    margin exceeds a tolerance.
    """
    Vs = [float(v) for v in Vs]
    if len(Vs) < 2:
        return 0.0
    return max((Vs[k + 1] - Vs[k]) / dt + alpha * Vs[k] for k in range(len(Vs) - 1))


def score_chunk(states: Sequence[dict[str, Any]], *, alpha: float = ALPHA_DEFAULT
                ) -> dict[str, float]:
    """Score one candidate chunk (sequence of per-step state dicts).

    Returns:
      total_reward     — shaped decrease + bonuses over the chunk
      total_decrease   — V(first) - V(last) within the final phase segment
      monotone_frac    — fraction of same-phase steps with V non-increasing
      violation_frac   — fraction of same-phase steps with V increasing
      exp_margin       — worst exponential-decrease margin (gate on <= tol)
    Ranking rule for best-of-N: higher total_reward wins; ties break on
    monotone_frac then lower exp_margin.
    """
    if len(states) < 2:
        return {"total_reward": 0.0, "total_decrease": 0.0, "monotone_frac": 1.0,
                "violation_frac": 0.0, "exp_margin": 0.0}
    total = 0.0
    mono = viol = comparable = 0
    seg_Vs: list[float] = []
    worst_margin = -np.inf
    for k in range(len(states) - 1):
        a, b = states[k], states[k + 1]
        total += clf_reward(a, b)
        if int(a["phase"]) == int(b["phase"]):
            va, vb = clf_value(a), clf_value(b)
            if va is not None and vb is not None:
                comparable += 1
                mono += int(vb <= va + 1e-12)
                viol += int(vb > va + 1e-12)
                seg_Vs.append(va)
        else:
            if len(seg_Vs) >= 2:
                worst_margin = max(worst_margin, exp_decrease_margin(seg_Vs, alpha))
            seg_Vs = []
    vlast = clf_value(states[-1])
    if vlast is not None:
        seg_Vs.append(vlast)
    if len(seg_Vs) >= 2:
        worst_margin = max(worst_margin, exp_decrease_margin(seg_Vs, alpha))
    total_decrease = (seg_Vs[0] - seg_Vs[-1]) if len(seg_Vs) >= 2 else 0.0
    return {
        "total_reward": float(total),
        "total_decrease": float(total_decrease),
        "monotone_frac": float(mono / comparable) if comparable else 1.0,
        "violation_frac": float(viol / comparable) if comparable else 0.0,
        "exp_margin": float(worst_margin) if np.isfinite(worst_margin) else 0.0,
    }


def reach_series_metrics(pinch_traj: np.ndarray, grasp_target: np.ndarray,
                         *, alpha: float = ALPHA_DEFAULT) -> dict[str, float]:
    """Convenience for the offline gate: REACH-phase CLF metrics of a decoded
    arm trajectory (pinch positions (T,3) vs a fixed target (3,))."""
    states = [{"phase": REACH, "pinch": p, "grip": 0.0, "grasped": False,
               "grasp_target": grasp_target, "vial": grasp_target,
               "pocket": grasp_target} for p in np.asarray(pinch_traj, float)]
    return score_chunk(states, alpha=alpha)
