"""Best-of-N candidate selection (R2 rung): CBF filters, CLF ranks, HOLD falls back.

Pure logic — the serving layer decodes K candidate chunks from K sampled initial
noises (steering mean + jitter) and hands this module, per candidate:
arm trajectory (16,6) -> pinch positions via an injected FK callable, plus grip
trajectory (16,). This module builds per-step state dicts, applies the CBF
safety filter (cbf_constraints_ur5e), ranks survivors by the CLF progress score
(clf_reward_ur5e), and returns the chosen index — or None, meaning execute the
always-feasible HOLD chunk and escalate.

Selection = one step of sampled, constrained MPC over the flow-sampler latent:
the frozen GR00T pilot remains the only source of actuator values (every
candidate is its own decode); selection among its own proposals preserves the
authenticity invariant. The returned report is the per-decision provenance /
safety certificate.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

import cbf_constraints_ur5e as cbf
import clf_reward_ur5e as clf


def build_candidate_states(pinch_traj: np.ndarray, grip_traj: np.ndarray,
                           q_traj: np.ndarray | None, *, vial: np.ndarray,
                           grasp_target: np.ndarray, pocket: np.ndarray,
                           phase: int, grasped: bool) -> list[dict[str, Any]]:
    """Per-step state dicts consumed by BOTH the CBF filter and the CLF scorer."""
    out = []
    for t in range(len(pinch_traj)):
        out.append({
            "phase": int(phase),
            "pinch": np.asarray(pinch_traj[t], float),
            "grip": float(np.clip(grip_traj[t], 0.0, 1.0)),
            "q": (np.asarray(q_traj[t], float) if q_traj is not None else None),
            "grasped": bool(grasped),
            "vial": np.asarray(vial, float),
            "grasp_target": np.asarray(grasp_target, float),
            "pocket": np.asarray(pocket, float),
        })
    return out


def make_cem_seeds(elite: np.ndarray, k: int, sigma: float,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """CEM refinement seeds: the elite itself + (k-1) samples around it.

    Round-2 of iterated selection at grasp: resample around round-1's winner at
    half the exploration sigma, so search FOCUSES where the image-conditioned
    sampler already showed good behavior. Deterministic slot 0 = the elite, so
    a refinement round can never do worse than its starting point.
    """
    rng = rng or np.random.default_rng()
    seeds = np.repeat(np.asarray(elite, np.float32)[None], k, axis=0)
    if k > 1 and sigma > 0:
        seeds[1:] += (sigma * rng.standard_normal(seeds[1:].shape)).astype(np.float32)
    return seeds


def hold_chunk(q_now: np.ndarray, grip_now: float, horizon: int = 16
               ) -> tuple[np.ndarray, np.ndarray]:
    """The always-feasible fallback: hold pose + hold grip for the horizon."""
    q = np.tile(np.asarray(q_now, float).reshape(1, 6), (horizon, 1))
    g = np.full(horizon, float(np.clip(grip_now, 0.0, 1.0)))
    return q, g


def select(arm_trajs: np.ndarray, grip_trajs: np.ndarray,
           fk: Callable[[np.ndarray], np.ndarray], *,
           vial: Sequence[float], grasp_target: Sequence[float],
           pocket: Sequence[float] | None, phase: int, grasped: bool,
           lim: cbf.Limits | None = None, gamma: float = 0.4,
           tol: float = 0.0, exec_horizon: int = 8
           ) -> tuple[int | None, dict[str, Any]]:
    """Choose among K decoded candidates.

    arm_trajs (K,16,6) absolute joint targets; grip_trajs (K,16) in 0..1;
    fk(q6) -> pinch xyz (robot base frame).
    Returns (index | None, report). None => caller executes hold_chunk().
    Ranking: CBF-feasible candidates by CLF total_reward (ties: monotone_frac,
    then lower exp_margin). Infeasible candidates are never chosen regardless
    of progress — safety is a filter, not a weight.
    """
    K = len(arm_trajs)
    # Safety + progress are judged on the EXECUTED prefix only — chunk steps
    # beyond the execution horizon are never applied, so a wild late-chunk
    # excursion is not a safety event (first live run vetoed ~90% of real
    # candidates on never-executed steps).
    H = min(int(exec_horizon), arm_trajs.shape[1])
    arm_trajs = arm_trajs[:, :H]
    grip_trajs = grip_trajs[:, :H]
    vial = np.asarray(vial, float)
    grasp_target = np.asarray(grasp_target, float)
    pocket = np.asarray(pocket if pocket is not None else grasp_target, float)
    lim = lim or cbf.Limits()

    cand_states = []
    for k in range(K):
        pinch = np.stack([np.asarray(fk(arm_trajs[k, t]), float)
                          for t in range(arm_trajs.shape[1])])
        cand_states.append(build_candidate_states(
            pinch, grip_trajs[k], arm_trajs[k], vial=vial,
            grasp_target=grasp_target, pocket=pocket, phase=phase,
            grasped=grasped))

    mask, cbf_reports = cbf.filter_candidates(cand_states, lim=lim, gamma=gamma,
                                              tol=tol)
    # GRIP-HOLD guard (Move-3 transport safety): while carrying, a candidate
    # that opens the gripper before the PLACE phase would DROP the vial —
    # infeasible regardless of its progress score.
    if grasped and int(phase) < clf.PLACE:
        for k in range(K):
            if mask[k] and bool((np.asarray(grip_trajs[k]) < 0.4).any()):
                mask[k] = False
                cbf_reports[k]["violated"] = sorted(
                    set(cbf_reports[k]["violated"]) | {"grip_hold"})
    scores: list[dict[str, float] | None] = [None] * K
    best_idx, best_key = None, None
    for k in range(K):
        if not mask[k]:
            continue
        s = clf.score_chunk(cand_states[k])
        scores[k] = s
        key = (s["total_reward"], s["monotone_frac"], -s["exp_margin"])
        if best_key is None or key > best_key:
            best_key, best_idx = key, k

    report = {
        "k": int(K),
        "n_feasible": int(mask.sum()),
        "chosen": (None if best_idx is None else int(best_idx)),
        "fallback_hold": best_idx is None,
        "chosen_clf": (None if best_idx is None else scores[best_idx]),
        "clf_reward_range": ([min(s["total_reward"] for s in scores if s),
                              max(s["total_reward"] for s in scores if s)]
                             if any(scores) else None),
        "cbf_rejections": {
            v: int(sum(1 for r_i, m_i in zip(cbf_reports, mask)
                       if not m_i and v in r_i["violated"]))
            for v in sorted({v for r_i, m_i in zip(cbf_reports, mask)
                             if not m_i for v in r_i["violated"]})
        },
    }
    return best_idx, report
