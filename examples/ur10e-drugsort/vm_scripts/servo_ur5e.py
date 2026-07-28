#!/usr/bin/env python3
"""L5 lever — bounded final-centimeter VISUAL SERVO for the UR5e drug-sort pilot.

The program's funnel loses almost everything at reached->centered->closed
capture: the base GR00T fine-tunes REACH to ~100% and CLOSE to ~89%, but LIFT
sits at ~0-13% because the pilot does gross reach + gross gripper and NOT the
sub-cm centering that actually captures the vial (Phase-1: near-miss 6-12 cm,
wrist-attention collapse). L5 adds exactly that missing precision as a BOUNDED
Observer-error correction composed INTO GR00T's own action stream — never a
second actuator authority. GR00T stays the sole emitter of actuator values; the
servo only nudges its output the last few millimetres, like a human micro-
adjusting the hand before closing on a vial.

Design (respecting the platform role law + the committed CBF/CLF patterns):

* **Signal** — the Observer sidecar already localises a sub-cm 3D grasp target in
  the robot base frame per tick (``serve_observer_conditioning.py``). The servo
  error is ``e = observer_target - pinch(a_groot[0])`` where ``pinch`` is the
  injected FK of the ``gr_pinch`` site (``serve_fk_ur5e.py`` / the DLS-IK
  Jacobian in ``embodiments/ur5e_drugsort/ik.py``).
* **Correction** — ONE damped-least-squares step toward the target, the exact
  inner update of :class:`DampedLeastSquaresIK`:
  ``dq = J^T (J J^T + lambda^2 I)^-1 e`` (J is a finite-difference Jacobian of
  the injected FK, so this module is pure-numpy and needs no mujoco). A single
  bounded ``Delta_q`` is composed onto EVERY step of the chosen chunk
  (``a_final = a_groot + Delta_q``), arm dims only — the gripper is untouched.
* **Phase-gated** — active ONLY in REACH/GRASP within a small radius (pinch
  within ``radius`` of the target); ZERO correction in transport/place or when
  far, so it never fights gross motion.
* **Bounded + dominance-gated (authenticity ship-block)** — ``||Delta_q||`` is
  capped both by an absolute per-joint bound AND to a fraction of GR00T's own
  commanded chunk motion, so GR00T always owns the MAJORITY of the motion. The
  per-tick report records both magnitudes and the dominance ratio.
* **CBF-safe** — the corrected chunk is run THROUGH the SAME feasibility check as
  best-of-N (``cbf_constraints_ur5e.chunk_feasible``); if the nudge would breach
  a barrier the correction is shrunk (scale backoff) until feasible. Scale 0 ==
  the untouched GR00T chunk, so the check always terminates on a feasible chunk.

The returned ``report`` is the explainability certificate — magnitudes, the
dominance ratio, the gate decision and the CBF shrink scale — attached to the
service response next to the ``bestofn``/``steer_head`` fields.

Pure numpy; positions in metres, joint angles in radians, robot base frame.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

import cbf_constraints_ur5e as cbf

# Phase ids mirror clf_reward_ur5e / the sidecar PhaseInference (0..3).
REACH, GRASP, TRANSPORT, PLACE = 0, 1, 2, 3

# CBF-shrink backoff: the largest feasible scale on this ladder is kept. 0.0 is
# always present and reproduces the untouched GR00T chunk, so a feasible chunk
# always exists (the selected/hold chunk was already CBF-feasible upstream).
_SHRINK_SCALES: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.1, 0.0)


def _batch_pinch(pinch_fn: Callable[[np.ndarray], np.ndarray],
                 qs: np.ndarray) -> np.ndarray:
    """Evaluate the injected FK on a batch of arm configs -> (M, 3)."""
    out = np.asarray(pinch_fn(np.asarray(qs, dtype=np.float64).reshape(-1, 6)), dtype=np.float64)
    return out.reshape(-1, 3)


def _fd_jacobian(pinch_fn: Callable[[np.ndarray], np.ndarray],
                 q_ref: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Finite-difference Jacobian (3x6) of the pinch FK at ``q_ref``.

    One batched FK call over ``[q_ref, q_ref+eps*e_0 .. q_ref+eps*e_5]``. Returns
    ``(J, pinch_ref)``.
    """
    q_ref = np.asarray(q_ref, dtype=np.float64).reshape(6)
    probes = np.repeat(q_ref[None], 7, axis=0)
    probes[1:] += eps * np.eye(6, dtype=np.float64)
    pts = _batch_pinch(pinch_fn, probes)          # (7, 3)
    pinch_ref = pts[0]
    jac = ((pts[1:] - pts[0]) / eps).T            # (3, 6)
    return jac, pinch_ref


def _dls_step(jac: np.ndarray, err: np.ndarray, damping: float) -> np.ndarray:
    """One damped-least-squares step ``J^T (J J^T + lambda^2 I)^-1 e``.

    Identical to the inner update of
    :class:`odyssey.embodiments.ur5e_drugsort.ik.DampedLeastSquaresIK` (no
    null-space term — a single bounded nudge has no redundancy seed to hold).
    """
    jjt = jac @ jac.T + (damping ** 2) * np.eye(3, dtype=np.float64)
    return jac.T @ np.linalg.solve(jjt, np.asarray(err, dtype=np.float64).reshape(3))


def _feasible(chunk: np.ndarray, pinch_fn: Callable[[np.ndarray], np.ndarray], *,
              observer_target: np.ndarray, grasped: bool, phase: int,
              lim: cbf.Limits, exec_horizon: int, gamma: float, tol: float) -> bool:
    """Run a (corrected) chunk through the SAME CBF filter best-of-N uses.

    Judged on the EXECUTED prefix only (matches ``bestofn_select.select``): chunk
    steps beyond the horizon are never applied and cannot be a safety event.
    """
    H = min(int(exec_horizon), chunk.shape[0])
    prefix = chunk[:H]
    pinch = _batch_pinch(pinch_fn, prefix)        # (H, 3)
    tgt = np.asarray(observer_target, dtype=np.float64).reshape(3)
    states = [
        {"phase": int(phase), "pinch": pinch[t], "q": np.asarray(prefix[t], float),
         "grasped": bool(grasped), "vial": tgt, "grasp_target": tgt}
        for t in range(H)
    ]
    ok, _rep = cbf.chunk_feasible(states, lim=lim, gamma=gamma, tol=tol)
    return bool(ok)


def servo_correction(
    a_groot_chunk: np.ndarray,
    pinch_fn: Callable[[np.ndarray], np.ndarray],
    observer_target: Sequence[float] | None,
    phase: int,
    grasped: bool,
    lim: cbf.Limits | None = None,
    *,
    radius: float = 0.08,
    gain: float = 0.8,
    damping: float = 0.08,
    bound: float = 0.08,
    max_dominance: float = 0.5,
    active_phases: Sequence[int] = (REACH, GRASP),
    exec_horizon: int = 8,
    fd_eps: float = 1.0e-4,
    gamma: float = 0.4,
    tol: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compose a bounded final-centimeter correction onto GR00T's chosen chunk.

    Args:
        a_groot_chunk: (T, 6) absolute arm-joint targets emitted by GR00T (the
            selected best-of-N / hold chunk). Gripper dims are handled elsewhere.
        pinch_fn: batched FK ``(M, 6) -> (M, 3)`` giving ``gr_pinch`` in the base
            frame (e.g. the best-of-N service's FK sidecar call, or a stub).
        observer_target: sub-cm 3D grasp target (base frame) from the Observer.
        phase: inferred phase id (0 REACH .. 3 PLACE).
        grasped: closure latch — drives the CBF vial-protection exemption.
        lim: CBF envelope (defaults to the v4-calibrated ``Limits``).

    Returns:
        ``(a_corrected_chunk, report)``. When the gate is closed the FIRST return
        value is the INPUT array object itself (byte-identical, no copy) so the
        STEER_SERVO-off / gated path is provably unchanged.
    """
    lim = lim or cbf.Limits()
    arm = np.asarray(a_groot_chunk, dtype=np.float64)
    report: dict[str, Any] = {
        "servo": True,
        "applied": False,
        "gated_on": False,
        "phase": int(phase),
        "reason": "off",
        "radius_m": float(radius),
        "bound_rad": float(bound),
        "max_dominance": float(max_dominance),
        "pinch_err_before_cm": None,
        "pinch_err_after_cm": None,
        "delta_q_norm": 0.0,
        "groot_step_norm": 0.0,
        "dominance_ratio": 0.0,
        "cbf_scale": 0.0,
        "cbf_feasible": None,
    }

    # --- trivial rejects: no target / degenerate chunk -----------------------
    if observer_target is None or arm.ndim != 2 or arm.shape[0] < 2 or arm.shape[1] != 6:
        report["reason"] = "no_target_or_degenerate"
        return a_groot_chunk, report

    if int(phase) not in {int(p) for p in active_phases}:
        report["reason"] = "phase_gate"
        return a_groot_chunk, report

    tgt = np.asarray(observer_target, dtype=np.float64).reshape(3)

    # Linearise at the chunk's first commanded pose (~ the current arm config);
    # the correction is a single bounded offset broadcast onto the whole chunk.
    jac, pinch_ref = _fd_jacobian(pinch_fn, arm[0], fd_eps)
    err = tgt - pinch_ref
    err_before = float(np.linalg.norm(err))
    report["pinch_err_before_cm"] = round(err_before * 100.0, 3)

    if err_before > radius:
        report["reason"] = "out_of_radius"
        return a_groot_chunk, report

    # --- one damped-least-squares step, then authenticity caps ----------------
    dq = gain * _dls_step(jac, err, damping)

    # GR00T's own commanded motion over the executed horizon (net joint travel).
    H = min(int(exec_horizon), arm.shape[0])
    groot_step = float(np.linalg.norm(arm[H - 1] - arm[0]))
    report["groot_step_norm"] = round(groot_step, 5)

    # Dominance cap FIRST: the servo may never own the majority of the motion.
    dq_norm = float(np.linalg.norm(dq))
    if groot_step <= 0.0:
        # A frozen pilot has no motion to nudge without dominating it — honest
        # no-op rather than becoming a covert primary actuator.
        dq = np.zeros(6, dtype=np.float64)
    elif dq_norm > max_dominance * groot_step:
        dq = dq * (max_dominance * groot_step / dq_norm)
    # Absolute per-joint bound.
    dq = np.clip(dq, -bound, bound)

    delta_q_norm = float(np.linalg.norm(dq))
    if delta_q_norm == 0.0:
        report["reason"] = "correction_zero"
        report["cbf_feasible"] = True
        return a_groot_chunk, report

    # --- CBF-safe: shrink the nudge until the corrected chunk is feasible ------
    scale_kept, feasible_kept = 0.0, None
    for scale in _SHRINK_SCALES:
        cand = arm + scale * dq[None, :]
        if _feasible(cand, pinch_fn, observer_target=tgt, grasped=grasped,
                     phase=int(phase), lim=lim, exec_horizon=exec_horizon,
                     gamma=gamma, tol=tol):
            scale_kept, feasible_kept = float(scale), True
            break
    if feasible_kept is None:
        # Not even scale 0 (== a_groot) is feasible; keep GR00T's chunk unchanged
        # (never worse than what would have been executed without the servo).
        scale_kept, feasible_kept = 0.0, False

    dq_final = scale_kept * dq
    dqf_norm = float(np.linalg.norm(dq_final))
    report["cbf_scale"] = float(scale_kept)
    report["cbf_feasible"] = bool(feasible_kept)
    report["delta_q_norm"] = round(dqf_norm, 5)
    report["dominance_ratio"] = round(dqf_norm / groot_step, 4) if groot_step > 0 else 0.0

    if dqf_norm == 0.0:
        report["reason"] = "cbf_shrunk_to_zero" if feasible_kept else "cbf_infeasible"
        report["gated_on"] = True
        return a_groot_chunk, report

    corrected = (arm + dq_final[None, :]).astype(np.asarray(a_groot_chunk).dtype, copy=False)
    # Post-correction single-step FK error (how much closer the linearised nudge
    # brought the first-step pinch to the target) for the certificate.
    pinch_after = _batch_pinch(pinch_fn, corrected[0][None])[0]
    report["pinch_err_after_cm"] = round(float(np.linalg.norm(tgt - pinch_after)) * 100.0, 3)
    report["applied"] = True
    report["gated_on"] = True
    report["reason"] = "applied"
    return corrected, report


# ---------------------------------------------------------------------------
# Observer -> IK-oracle HANDOFF (full-authority final-centimetre capture)
# ---------------------------------------------------------------------------
# The servo above is a BOUNDED nudge — GR00T keeps the majority of the motion,
# which is right for a policy that mostly works but caps how far it can fix the
# sub-cm capture GR00T structurally can't do (Phase-1: gross reach + blind grip).
# The handoff is the stronger move: once GR00T's gross reach has put the pinch
# inside a small CAPTURE ZONE and the phase is REACH/GRASP, the IK ORACLE takes
# FULL authority for the last centimetre — multi-step DLS-IK drives the pinch to
# the Observer target and the gripper closes once aligned. This offloads the
# precision to the component that HAS it (physics-exact IK + render-robust
# Observer) instead of hoping the VLA learns sub-cm. Opt-in (STEER_HANDOFF=1);
# byte-identical to the V4 path when off.


def _ik_solve(pinch_fn: Callable[[np.ndarray], np.ndarray], q0: np.ndarray,
              target: np.ndarray, *, max_iters: int, damping: float,
              step_bound: float, fd_eps: float, tol: float) -> tuple[np.ndarray, float, int]:
    """Multi-step damped-least-squares IK: drive pinch(q) -> target in joint space.

    Iterates the same DLS update as :func:`_dls_step` to convergence (unlike the
    servo's single bounded step). Returns ``(q_star, final_err_m, iters)``.
    """
    q = np.asarray(q0, dtype=np.float64).reshape(6).copy()
    tgt = np.asarray(target, dtype=np.float64).reshape(3)
    err_m = 9.99
    for k in range(int(max_iters)):
        jac, pinch = _fd_jacobian(pinch_fn, q, fd_eps)
        err = tgt - pinch
        err_m = float(np.linalg.norm(err))
        if err_m <= tol:
            return q, err_m, k
        dq = np.clip(_dls_step(jac, err, damping), -step_bound, step_bound)
        q = q + dq
    return q, err_m, int(max_iters)


def _retry_offset(attempt: int, jitter: float) -> tuple[float, float]:
    """x-y search offset for grasp attempt ``attempt`` (0-based).

    Attempt 0 sits exactly on the estimate. Later attempts walk an outward square
    spiral so a systematically-biased Observer estimate still gets probed from
    several sides rather than being retried at the same wrong point.
    """
    if attempt <= 0 or jitter <= 0.0:
        return 0.0, 0.0
    ring, idx = (attempt - 1) // 4 + 1, (attempt - 1) % 4
    r = jitter * ring
    return [(r, 0.0), (0.0, r), (-r, 0.0), (0.0, -r)][idx]


def ik_handoff(
    a_groot_chunk: np.ndarray,
    grip_chunk: np.ndarray,
    pinch_fn: Callable[[np.ndarray], np.ndarray],
    observer_target: Sequence[float] | None,
    phase: int,
    grasped: bool,
    lim: cbf.Limits | None = None,
    *,
    capture_zone: float = 0.10,     # x-y proximity that says "GR00T's gross reach is done"
    descend_offset: float = 0.05,   # grasp_z = observer_z - this (correct the ~4cm-high observer + 1cm below vial)
    grasp_z: float | None = None,   # if set, use this fixed FK-frame grasp height instead of observer_z-descend
    grasp_tol: float = 0.015,       # pinch within this of the grasp point => close + hold
    xy_from_groot: bool = True,     # use GR00T's reached x-y (accurate) vs the coarse observer x-y
    max_iters: int = 12,
    damping: float = 0.05,
    step_bound: float = 0.2,
    move_cap: float = 0.4,
    active_phases: Sequence[int] = (REACH, GRASP),
    exec_horizon: int = 8,
    fd_eps: float = 1.0e-4,
    gamma: float = 0.4,
    tol: float = 0.0,
    state: dict[str, Any] | None = None,   # per-episode retry state (owned by caller)
    hold_queries: int = 2,          # queries to hold the grasp pose while the fingers shut
    verify_queries: int = 3,        # queries to wait for lift-off after handing back to GR00T
    max_attempts: int = 4,          # total grasp attempts per episode (1 = no retry)
    retract: float = 0.06,          # how far above the grasp point to withdraw on abort
    retry_jitter: float = 0.012,    # x-y search offset applied on attempts 2..N
    q_now: Sequence[float] | None = None,   # MEASURED arm joints (not GR00T's proposal)
    tgt_settled: bool = True,       # enough close-range Observer reads to trust the median
    pocket: Sequence[float] | None = None,  # STATIC pocket pose (base frame) -> owned place
    place_z_hi: float = 0.34,       # safe transit height above the rack
    place_z_lo: float = 0.28,       # release height above the pocket
    place_tol: float = 0.02,        # waypoint arrival tolerance
    hover: float = 0.06,            # line up this far ABOVE the grasp point first
    align_tol: float = 0.012,       # x-y error under which it is safe to come down
    descend_step: float | None = None,  # cap the per-query terminal approach (m); a
                                    # biased-target graze then NUDGES the vial ~cm
                                    # (recoverable by retry) instead of launching it
                                    # metres (measured 3.4-5.8 m, unrecoverable)
    close_ramp: bool = False,       # stage the close over queries (0.7 then 1.0)
                                    # instead of slamming shut in one step — the
                                    # browser executes only ~15-25% of a chunk, so
                                    # an intra-chunk ramp never actually closes;
                                    # per-query stages are re-commanded each query
    approach_budget: int | None = None,  # applied align/settle/descend queries allowed
                                    # per attempt before the approach is declared
                                    # stalled -> retract + jitter + fresh reads
                                    # (measured stall: 118 align queries churning
                                    # around a nudged/ghost target, no commit,
                                    # and the lift-test retry never engages)
    seat_queries: int = 0,          # queries to hold POSITION with the grip pinned
                                    # 1.0 on carry entry before any motion — welds
                                    # can form from a glancing partial-close contact
                                    # (>0.5N on any finger), and yanking such a weak
                                    # weld into the transit slings the vial metres;
                                    # a stationary full-close first seats the grip
    rise_step: float | None = None, # cap the per-query rise toward wp0 (m) so a
                                    # weld that is going to snap does so at low
                                    # speed near the table (vial drops back cm,
                                    # not flung) instead of during the fast rise
    transit_step: float | None = None,  # cap the per-query step toward wp1/wp2 (m).
                                    # The wp0->wp1 transition otherwise commands the
                                    # full ~30cm rack offset in ONE query (2g ep008:
                                    # dxy=30.3cm, dq=0.64 rad) — that yank is what
                                    # rips weak welds and slings the vial metres.
                                    # The rise is naturally gentle (IK + smoothing);
                                    # transit is where the energy spike lives.
    close_commit_through: bool = False,  # once a close begins, finish close+hold
                                    # against the frozen commit point instead of
                                    # re-checking tolerance mid-close; the 2e
                                    # audit showed the grind risk: fingers pressed
                                    # at a bad point for 3 queries can pop the
                                    # vial out ballistically (penetration energy)
    drop_retry: bool = False,       # mid-carry drop handling. False = 2b-era
                                    # semantics: disengage once the Pilot's phase
                                    # left the active window, otherwise fall
                                    # through to a fresh approach WITHOUT
                                    # spending an attempt/retract. True = spend
                                    # an attempt + retract per drop (measured
                                    # retry-amplifier: repeated re-grasp->sling
                                    # cycles; 2r replicated 2/15 vs 2b's 8/15
                                    # with this behavior unconditional).
    descend_latch: bool = False,    # freeze the x-y target at the align->descend
                                    # boundary: wrist-camera reads DEGRADE on the
                                    # way down (p95_7777d: median |err| 0.85cm at
                                    # z>0.27 vs 1.24cm at z<0.24), so the estimate
                                    # at descend entry is the best this attempt
                                    # will ever have; re-latched fresh per attempt
    carry_mode: str = "owned",      # "owned" = IK waypoint transport (default);
                                    # "pilot" = hybrid: after the owned capture,
                                    # pin the grip and return GR00T's own chunk
                                    # for transport, release handed to the Pilot
                                    # at PLACE (2b-era carry semantics — 2r5
                                    # measured 4/4 lifts seated, zero slings;
                                    # capture stays owned: 8-9/15 lifts vs 4/15
                                    # under pilot-owned capture)
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Grasp-strategy-aware Observer->IK handoff — descend, close, HOLD, carry.

    The playground grasp only binds (MuJoCo weld) when the fingers hold >0.5N for
    5 consecutive steps, and the grasp point is ~1cm BELOW the vial (fingers
    straddle the body). GR00T reaches decent x-y but stops ~4cm TOO HIGH and never
    holds. This mimics the expert FSM:

    * **descend** — once GR00T's pinch x-y is within ``capture_zone`` of the vial,
      IK-drive the pinch straight down to the grasp point ``[x-y, grasp_z]`` with
      the gripper OPEN. ``grasp_z`` is deterministic (vials on the table): default
      = ``observer_z - descend_offset`` (corrects the observer's high z bias), or a
      fixed FK-frame height. x-y = GR00T's reached pinch (more accurate than the
      Observer) by default.
    * **close + HOLD** — once the pinch is within ``grasp_tol`` of the grasp point,
      command the SAME pose (no motion) and CLOSE the gripper, so contact persists
      the >=5 steps the weld needs.
    * **owned lift-test + retry** — closing is committed against the *Observer's*
      estimate, so an Observer error of a few cm shuts the fingers on empty air
      with no way back. After ``hold_queries`` at the grasp pose the handoff LIFTS
      the closed gripper itself (IK, straight up ``retract`` metres from the
      committed grasp point) — actuator authority is never returned to GR00T
      between close and verify, because GR00T's wander (37-92cm proposals) drags
      the gripper off the vial and voids the test. ``grasped`` (the caller's load
      signal — the vial left the table) flipping true exits to carry. If it has
      not flipped within ``verify_queries`` lift queries the grasp is declared
      failed: withdraw with the fingers OPEN, then descend again. The re-descend
      is not a blind repeat — the Observer target is re-read every query, and by
      then the wrist camera sits centimetres from the vial instead of ~20cm away,
      so the second estimate is materially better. ``retry_jitter`` adds a small
      x-y search offset from the 2nd attempt on to break a systematically-biased
      estimate.
    * **carry** — once ``grasped`` (weld formed), keep the gripper CLOSED through
      GRASP/TRANSPORT (don't let GR00T drop the vial); release at PLACE.

    ``state`` is a caller-owned dict holding the retry state for ONE episode; pass a
    fresh dict per episode (or omit it to disable retry bookkeeping across calls).

    Returns ``(arm_chunk, grip_chunk, report)``; unchanged inputs when gated off.
    """
    lim = lim or cbf.Limits()
    arm = np.asarray(a_groot_chunk, dtype=np.float64)
    grip = np.asarray(grip_chunk, dtype=np.float64).reshape(-1).copy()
    st = state if state is not None else {}
    st.setdefault("mode", "descend")     # "descend" | "retract"
    st.setdefault("attempt", 0)          # 0-based grasp attempt index
    st.setdefault("closed_queries", 0)   # consecutive queries held closed with no weld
    report: dict[str, Any] = {
        "handoff": True, "applied": False, "reason": "off", "stage": None,
        "pinch_err_before_cm": None, "grasp_z": None,
        "iters": 0, "grip_close": False, "cbf_scale": 0.0,
        "attempt": int(st["attempt"]), "closed_queries": int(st["closed_queries"]),
    }
    if arm.ndim != 2 or arm.shape[0] < 2 or arm.shape[1] != 6:
        report["reason"] = "degenerate"
        return a_groot_chunk, grip_chunk, report

    # --- CARRY: already grasped -> hold the gripper closed through the lift, ------
    #     let GR00T own the arm; release at PLACE.
    if bool(grasped):
        st["mode"], st["closed_queries"] = "descend", 0   # grasp took: retry state spent
        st["approach_q"] = 0
        st["lift_queries"] = 0
        st.pop("commit_gp", None)
        st.pop("descend_gp", None)
        st["was_grasped"] = True
        if not st.get("carry_live"):
            # first query of THIS carry (fresh grasp or re-grasp after a drop):
            # restart the waypoint ladder and the seat-hold window.
            st["carry_live"] = True
            st["wp"] = 0
            st["seat_q"] = 0
        grip[:] = 1.0
        # --- OWNED CARRY + PLACE -------------------------------------------------
        # Handing the arm to the Pilot for transport was the measured top loss
        # class (4/11 honest failures: verified lifts flung up to 5.2 m mid-carry).
        # The pocket is STATIC KNOWN geometry, so transport and placement need no
        # perception: IK waypoints, move-capped, CBF-checked, grip pinned until
        # the release. If the load signal drops mid-carry the next query falls
        # back into the grasp cycle automatically (grasped=False -> descend on the
        # re-read Observer target).
        if pocket is None:
            report.update(applied=True, stage="carry_hold", reason="grip_hold")
            return a_groot_chunk, grip, report
        q0c = (np.asarray(q_now, dtype=np.float64).reshape(6)
               if q_now is not None else np.asarray(arm[0], dtype=np.float64))
        pin = _batch_pinch(pinch_fn, q0c[None])[0]
        if int(seat_queries) > 0 and int(st.get("seat_q", 0)) < int(seat_queries):
            # SEAT: stationary full-close queries before any carry motion, so a
            # weak glancing-contact weld becomes an enclosed grip (or is at least
            # not yanked at transit speed).
            st["seat_q"] = int(st.get("seat_q", 0)) + 1
            report.update(applied=True, stage="carry_seat", reason="applied")
            hold = np.repeat(q0c[None, :], arm.shape[0], axis=0)
            return hold.astype(np.asarray(a_groot_chunk).dtype, copy=False), grip, report
        if str(carry_mode) == "pilot":
            # HYBRID transport: capture was owned (above); the carry is the
            # Pilot's. Mid-carry drops and the post-release disengage flow
            # through the was_grasped/phase gates below exactly as 2b-era.
            if int(phase) >= PLACE:
                report["reason"] = "place_release"
                return a_groot_chunk, grip_chunk, report
            report.update(applied=True, stage="carry_hold", reason="grip_hold")
            return a_groot_chunk, grip, report
        pk = np.asarray(pocket, dtype=np.float64).reshape(3)
        wps = [np.array([pin[0], pin[1], float(place_z_hi)]),      # rise in place
               np.array([pk[0], pk[1], float(place_z_hi)]),        # transit above pocket
               np.array([pk[0], pk[1], float(place_z_lo)])]        # seat approach
        wp_i = int(st["wp"])
        if wp_i < len(wps) and float(np.linalg.norm(pin - wps[wp_i])) <= float(place_tol):
            wp_i += 1
            st["wp"] = wp_i
        if wp_i >= len(wps):
            # at the release point: open, give the fingers a couple of queries to
            # clear, then retreat and disengage for good.
            st["release_q"] = int(st.get("release_q", 0)) + 1
            grip[:] = 0.0
            target_point = wps[-1] if st["release_q"] <= 2 else                 np.array([pk[0], pk[1], float(place_z_hi)])
            report["stage"] = "place_release" if st["release_q"] <= 2 else "place_retreat"
            if st["release_q"] >= 5:
                st["done"] = True
        else:
            target_point = wps[wp_i]
            report["stage"] = f"carry_wp{wp_i}"
            if wp_i == 0 and rise_step is not None and float(rise_step) > 0.0:
                dvec = target_point - pin
                dist = float(np.linalg.norm(dvec))
                if dist > float(rise_step):
                    target_point = pin + dvec * (float(rise_step) / dist)
            elif wp_i >= 1 and transit_step is not None and float(transit_step) > 0.0:
                dvec = target_point - pin
                dist = float(np.linalg.norm(dvec))
                if dist > float(transit_step):
                    target_point = pin + dvec * (float(transit_step) / dist)
        q_star, _e, iters = _ik_solve(pinch_fn, q0c, target_point, max_iters=max_iters,
                                      damping=damping, step_bound=step_bound,
                                      fd_eps=fd_eps, tol=0.003)
        report["iters"] = iters
        dqt = q_star - q0c
        nrm = float(np.linalg.norm(dqt))
        if nrm > move_cap:
            dqt = dqt * (move_cap / nrm)
        q_cmd = q0c + dqt
        report["pinch_err_before_cm"] = round(
            float(np.linalg.norm(pin[:2] - target_point[:2])) * 100.0, 2)
        report["d_gp_cm"] = round(float(np.linalg.norm(pin - target_point)) * 100.0, 2)
        report["dq_norm"] = round(float(np.linalg.norm(dqt)), 5)
        report["ik_resid_cm"] = round(float(np.linalg.norm(
            _batch_pinch(pinch_fn, q_star[None])[0] - target_point)) * 100.0, 2)
        report["pinch"] = [round(float(v), 4) for v in pin]
        report["grasp_point"] = [round(float(v), 4) for v in target_point]
        T = arm.shape[0]
        ramp = np.linspace(0.0, 1.0, T)[:, None]
        cand, scale_kept = None, 0.0
        for scale in _SHRINK_SCALES:
            c = q0c[None, :] + scale * ramp * (q_cmd - q0c)[None, :]
            if _feasible(c, pinch_fn, observer_target=target_point, grasped=True,
                         phase=int(phase), lim=lim, exec_horizon=exec_horizon,
                         gamma=gamma, tol=tol):
                cand, scale_kept = c, float(scale)
                break
        report["cbf_scale"] = scale_kept
        if cand is None or scale_kept == 0.0:
            report.update(applied=True, reason="carry_cbf_hold")
            hold = np.repeat(q0c[None, :], arm.shape[0], axis=0)
            return hold.astype(np.asarray(a_groot_chunk).dtype, copy=False), grip, report
        report.update(applied=True, reason="applied")
        return cand.astype(np.asarray(a_groot_chunk).dtype, copy=False), grip, report

    # --- pre-grasp gate --------------------------------------------------------
    if st.get("carry_live"):
        # The load signal dropped MID-CARRY: the weld snapped / a marginal grip
        # slipped before any release was commanded. Re-closing at the same point
        # just re-drops it (measured on p95_7777e: 5x close->lift->seat->drop
        # cycles that bypassed the attempt counter entirely), so spend an attempt
        # and withdraw — the jitter search + fresh at-height reads take it from
        # there. If release had begun this was a completed place: latch done.
        st["carry_live"] = False
        if int(st.get("release_q", 0)) > 0:
            st["done"] = True
        elif bool(drop_retry) and int(st["attempt"]) + 1 < int(max_attempts):
            st["attempt"] = int(st["attempt"]) + 1
            st["mode"] = "retract"
            st["retract_q"] = 0
            st["approach_q"] = 0
            st["closed_queries"] = 0
            st.pop("commit_gp", None)
            st.pop("descend_gp", None)
    # Once the vial has been carried and RELEASED the job is done — never
    # re-engage, or the handoff would pick the seated vial back out of the rack.
    # (A drop while the phase machine already reads PLACE is conservatively
    # treated as done too: re-grasping next to the rack risks wrecking it.)
    if st.get("was_grasped") and int(phase) >= PLACE:
        st["done"] = True
    if not bool(drop_retry) and st.get("was_grasped") \
            and int(phase) not in active_phases:
        # 2b-era gate: after any grasp, once the Pilot's phase leaves the
        # active window the handoff disengages for good.
        st["done"] = True
    if st.get("done"):
        report["reason"] = "placed"
        return a_groot_chunk, grip_chunk, report
    # While nothing is held the phase label is not meaningful: the phase machine
    # advances to GRASP/TRANSPORT the moment the policy COMMANDS a close, even if
    # the fingers shut on air — and after a MID-CARRY DROP it reads TRANSPORT
    # while the vial sits back on the table (measured on p95_7777e ep003: the old
    # was_grasped+phase gate handed the arm to GR00T for the whole remaining
    # episode, fingers open). Pre-grasp we gate on proximity alone.
    if observer_target is None:
        report["reason"] = "phase_gate"
        return a_groot_chunk, grip_chunk, report
    tgt = np.asarray(observer_target, dtype=np.float64).reshape(3)
    # Gate and solve from where the arm ACTUALLY is, not from GR00T's proposed
    # first action. Using the proposal means a wild chunk (GR00T regularly flings
    # the target 40-90cm away) reads as "out of the capture zone", the handoff
    # stands down, and that very jump is what gets executed -- so the handoff kept
    # losing the arm right after lining it up over the vial.
    q0 = (np.asarray(q_now, dtype=np.float64).reshape(6)
          if q_now is not None else arm[0].astype(np.float64))
    pinch0 = _batch_pinch(pinch_fn, q0[None])[0]
    # GR00T must have done the gross reach (x-y near the vial); z is expected high.
    dxy = float(np.linalg.norm(pinch0[:2] - tgt[:2]))
    report["pinch_err_before_cm"] = round(dxy * 100.0, 3)
    if dxy > capture_zone:
        report["reason"] = "out_of_capture_zone"
        return a_groot_chunk, grip_chunk, report

    # --- grasp point: accurate x-y + deterministic grasp depth -----------------
    gz = float(grasp_z) if grasp_z is not None else float(tgt[2] - descend_offset)
    # After a failed attempt GR00T's pinch sits at the *bad* spot, so re-using it
    # would repeat the miss: from attempt 2 on always take x-y from the (re-read,
    # much closer) Observer estimate.
    use_groot_xy = bool(xy_from_groot) and int(st["attempt"]) == 0
    gx, gy = (float(pinch0[0]), float(pinch0[1])) if use_groot_xy else (float(tgt[0]), float(tgt[1]))
    ox, oy = _retry_offset(int(st["attempt"]), float(retry_jitter))
    grasp_point = np.array([gx + ox, gy + oy, gz], dtype=np.float64)
    # DESCEND LATCH: freeze the x-y target the moment the wrist is lined up and
    # the median is settled (= descend entry). Near reads only get worse; keep
    # the hover-height estimate through the whole descent. Cleared wherever the
    # approach restarts (retract-complete, carry, lift-abort) so each attempt
    # re-latches from fresh at-height reads.
    if descend_latch:
        if "descend_gp" in st:
            grasp_point = np.asarray(st["descend_gp"], dtype=np.float64)
        elif (float(np.linalg.norm(pinch0[:2] - grasp_point[:2])) <= align_tol
              and bool(tgt_settled)):
            st["descend_gp"] = [float(v) for v in grasp_point]
    # COMMIT LATCH: once the first close fired, the commit point is frozen —
    # a millimetre of median drift must not flip the stage back to align and
    # void the commit (audit of ho8: 15/28 commit windows broken exactly so).
    # Cleared on retract-complete, carry, or lift-abort.
    if "commit_gp" in st and st["mode"] == "descend":
        grasp_point = np.asarray(st["commit_gp"], dtype=np.float64)
    report["grasp_z"] = round(gz, 4)
    d_gp = float(np.linalg.norm(pinch0 - grasp_point))     # dominated by the descend (z) gap
    report["d_gp_cm"] = round(d_gp * 100.0, 2)
    report["pinch"] = [round(float(v), 4) for v in pinch0]
    report["grasp_point"] = [round(float(v), 4) for v in grasp_point]

    # --- lift-test: WE lift, with the grip pinned closed -------------------------
    # Handing the arm back to GR00T for the lift was a measured failure: GR00T
    # wanders (37-92cm proposals), drags the just-closed gripper off the vial and
    # resets the verify window — 19 test-lift events, 0 retries, on ho7. The
    # handoff keeps actuator authority through the whole close->lift->verify arc;
    # the caller's `grasped` load signal flipping true is what exits to carry.
    if st["mode"] == "lift_test":
        lift_from = np.asarray(st.get("commit_gp", grasp_point), dtype=np.float64)
        target_point = lift_from + np.array([0.0, 0.0, float(retract)])
        grip[:] = 1.0                       # never let go mid-test
        st["lift_queries"] = int(st.get("lift_queries", 0)) + 1
        report["stage"] = "lift_test"
        if st["lift_queries"] > int(verify_queries):
            # lifted for the whole window and the load signal never flipped:
            # the fingers are holding air -> abort, withdraw, try again.
            st["lift_queries"] = 0
            st.pop("commit_gp", None)
            st.pop("descend_gp", None)
            if int(st["attempt"]) + 1 < int(max_attempts):
                st["attempt"] = int(st["attempt"]) + 1
                st["mode"] = "retract"
                report["stage"] = "grasp_failed_retry"
            else:
                st["mode"] = "descend"
                report["stage"] = "grasp_failed_exhausted"
            report.update(applied=True, reason="applied", attempt=int(st["attempt"]))
            return a_groot_chunk, grip, report
        q_star, _e, iters = _ik_solve(pinch_fn, q0, target_point, max_iters=max_iters,
                                      damping=damping, step_bound=step_bound,
                                      fd_eps=fd_eps, tol=0.003)
        report["iters"] = iters
        dqt = q_star - q0
        nrm = float(np.linalg.norm(dqt))
        if nrm > move_cap:
            dqt = dqt * (move_cap / nrm)
        q_cmd = q0 + dqt
        T = arm.shape[0]
        ramp = np.linspace(0.0, 1.0, T)[:, None]
        cand, scale_kept = None, 0.0
        for scale in _SHRINK_SCALES:
            c = q0[None, :] + scale * ramp * (q_cmd - q0)[None, :]
            # grasped=True for the CBF: the whole premise of the test is that we
            # may be holding the vial, and the motion is straight up and away.
            if _feasible(c, pinch_fn, observer_target=target_point, grasped=True,
                         phase=int(phase), lim=lim, exec_horizon=exec_horizon,
                         gamma=gamma, tol=tol):
                cand, scale_kept = c, float(scale)
                break
        report["cbf_scale"] = scale_kept
        if cand is None or scale_kept == 0.0:
            report["reason"] = "lift_cbf_infeasible"
            return a_groot_chunk, grip, report   # hold grip closed even on a hold
        report.update(applied=True, reason="applied", attempt=int(st["attempt"]))
        return cand.astype(np.asarray(a_groot_chunk).dtype, copy=False), grip, report

    # --- retract: withdraw above the grasp point with the gripper OPEN, then -----
    #     hand back to `descend` for another (better-informed) attempt.
    if st["mode"] == "retract":
        above = np.asarray(st.get("commit_gp", grasp_point), dtype=np.float64) \
            + np.array([0.0, 0.0, float(retract)])
        # Retract exists ONLY to gain vertical clearance before re-approaching;
        # completing on the full 3-D point stalls when the knocked vial rolls and
        # the observer-anchored target drifts under the arm (measured: 109
        # consecutive retract queries chasing a moving 2 cm tolerance). Complete
        # on HEIGHT alone; align/descend own the xy. Hard cap as a stall fuse.
        st["retract_q"] = int(st.get("retract_q", 0)) + 1
        if pinch0[2] >= gz + 0.8 * float(retract) or st["retract_q"] > 12:
            st["mode"] = "descend"          # clear of the vial -> approach again
            st["retract_q"] = 0
            st["approach_q"] = 0
            st.pop("commit_gp", None)
            st.pop("descend_gp", None)       # next attempt re-commits from a fresh read
        else:
            target_point = above
            grip[:] = 0.0                   # open on the way out
            report["stage"] = "retract"
            q_star, _e, iters = _ik_solve(pinch_fn, q0, target_point, max_iters=max_iters,
                                          damping=damping, step_bound=step_bound,
                                          fd_eps=fd_eps, tol=0.003)
            report["iters"] = iters
            dqt = q_star - q0
            nrm = float(np.linalg.norm(dqt))
            if nrm > move_cap:
                dqt = dqt * (move_cap / nrm)
            q_cmd = q0 + dqt
            T = arm.shape[0]
            ramp = np.linspace(0.0, 1.0, T)[:, None]
            cand, scale_kept = None, 0.0
            for scale in _SHRINK_SCALES:
                c = q0[None, :] + scale * ramp * (q_cmd - q0)[None, :]
                if _feasible(c, pinch_fn, observer_target=target_point, grasped=False,
                             phase=int(phase), lim=lim, exec_horizon=exec_horizon,
                             gamma=gamma, tol=tol):
                    cand, scale_kept = c, float(scale)
                    break
            report["cbf_scale"] = scale_kept
            if cand is None or scale_kept == 0.0:
                st["mode"] = "descend"      # cannot withdraw -> just try again in place
                report["reason"] = "retract_cbf_infeasible"
                return a_groot_chunk, grip_chunk, report
            report.update(applied=True, reason="applied", attempt=int(st["attempt"]))
            return cand.astype(np.asarray(a_groot_chunk).dtype, copy=False), grip, report

    # --- align, THEN descend (the expert's approach->descend, not a diagonal) ---
    # Solving IK straight to the grasp point from wherever GR00T parked drags the
    # gripper down at an angle: it reaches table height while still centimetres
    # off in x-y, wedges against the table and then cannot move laterally at all
    # (the arm is position-controlled, so contact simply stalls it). Line up over
    # the vial at hover height first, then come straight down like the expert.
    dxy_gp = float(np.linalg.norm(pinch0[:2] - grasp_point[:2]))
    if dxy_gp > align_tol:
        target_point = grasp_point + np.array([0.0, 0.0, float(hover)])
        report["stage_hint"] = "align"
    elif not bool(tgt_settled):
        # Lined up but the close-range median is still forming (the caller resets
        # the filter window on first descend so near-vial reads dominate it).
        # Hold at hover: descending onto a 1-2-read target re-creates the punch.
        target_point = grasp_point + np.array([0.0, 0.0, float(hover)])
        report["stage_hint"] = "settle"
    else:
        target_point = grasp_point
        if descend_step is not None and float(descend_step) > 0.0:
            dvec = grasp_point - pinch0
            dist = float(np.linalg.norm(dvec))
            if dist > float(descend_step):
                target_point = pinch0 + dvec * (float(descend_step) / dist)
        report["stage_hint"] = "descend"
    report["dxy_gp_cm"] = round(dxy_gp * 100.0, 2)

    q_star, _e, iters = _ik_solve(pinch_fn, q0, target_point, max_iters=max_iters,
                                  damping=damping, step_bound=step_bound,
                                  fd_eps=fd_eps, tol=0.003)
    report["iters"] = iters
    dqt = q_star - q0
    nrm = float(np.linalg.norm(dqt))
    if nrm > move_cap:
        dqt = dqt * (move_cap / nrm)
    q_cmd = q0 + dqt
    report["dq_norm"] = round(float(np.linalg.norm(dqt)), 5)
    report["ik_resid_cm"] = round(float(np.linalg.norm(
        _batch_pinch(pinch_fn, q_star[None])[0] - grasp_point)) * 100.0, 2)
    T = arm.shape[0]
    ramp = np.linspace(0.0, 1.0, T)[:, None]

    cand, scale_kept = None, 0.0
    for scale in _SHRINK_SCALES:
        c = q0[None, :] + scale * ramp * (q_cmd - q0)[None, :]
        if _feasible(c, pinch_fn, observer_target=target_point, grasped=False, phase=int(phase),
                     lim=lim, exec_horizon=exec_horizon, gamma=gamma, tol=tol):
            cand, scale_kept = c, float(scale)
            break
    report["cbf_scale"] = scale_kept
    if cand is None or scale_kept == 0.0:
        report["reason"] = "cbf_infeasible"
        return a_groot_chunk, grip_chunk, report

    # --- close, then hand off to the OWNED lift-test ----------------------------
    # We only reach here while `grasped` is False (the carry branch returns early),
    # so this is the "fingers are shut but nothing is held yet" window.
    committing = bool(close_commit_through) and int(st["closed_queries"]) > 0
    if committing or (d_gp <= grasp_tol and dxy_gp <= align_tol and bool(tgt_settled)):
        # COMMIT-THROUGH: once the first close fired, finish close+hold against
        # the frozen commit point — do not re-check tolerance mid-close (measured:
        # first-close contact nudges the pinch a few mm, the re-check aborted the
        # close after one query, and the fingers churned half-shut over the vial).
        # A close that genuinely missed is caught by the lift-test -> retry.
        st["approach_q"] = 0
        grip[:] = 1.0
        if close_ramp and int(st["closed_queries"]) == 0:
            # First close query: partial close (flat, re-commanded — an intra-chunk
            # ramp is defeated by the browser executing ~15-25% of a chunk). An
            # off-centre slam launches the vial metres; a soft first contact
            # shoves it centimetres, which retract->re-descend can still capture.
            grip[:] = 0.7
        report["grip_close"] = True
        report["stage"] = "close_hold"
        st["closed_queries"] = int(st["closed_queries"]) + 1
        if st["closed_queries"] == 1:
            # freeze the point we are committing to — lift/retract reference this,
            # not the live (drifting) observer read
            st["commit_gp"] = [float(v) for v in grasp_point]
            st["wp"] = 0
            st["release_q"] = 0
        if st["closed_queries"] >= int(hold_queries):
            # fingers have had time to seat -> WE lift next query (grip pinned);
            # authority is never returned to GR00T between close and verify.
            st["closed_queries"] = 0
            st["lift_queries"] = 0
            st["mode"] = "lift_test"
    else:
        st["closed_queries"] = 0
        report["stage"] = report.get("stage_hint", "descend")
        if approach_budget is not None and int(approach_budget) > 0:
            st["approach_q"] = int(st.get("approach_q", 0)) + 1
            if (st["approach_q"] > int(approach_budget)
                    and int(st["attempt"]) + 1 < int(max_attempts)):
                # Approach-stall fuse: nudged/ghost targets can churn align
                # forever without ever committing, so the lift-test retry never
                # gets its chance. Spend an attempt: withdraw, jitter, re-read.
                st["attempt"] = int(st["attempt"]) + 1
                st["mode"] = "retract"
                st["approach_q"] = 0
                st["retract_q"] = 0
                report["reason"] = "approach_budget_retry"

    report["applied"] = True
    if report.get("reason") != "approach_budget_retry":
        report["reason"] = "applied"
    report["attempt"] = int(st["attempt"])
    report["closed_queries"] = int(st["closed_queries"])
    corrected = cand.astype(np.asarray(a_groot_chunk).dtype, copy=False)
    return corrected, grip, report
