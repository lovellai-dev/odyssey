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


def ik_handoff(
    a_groot_chunk: np.ndarray,
    grip_chunk: np.ndarray,
    pinch_fn: Callable[[np.ndarray], np.ndarray],
    observer_target: Sequence[float] | None,
    phase: int,
    grasped: bool,
    lim: cbf.Limits | None = None,
    *,
    capture_zone: float = 0.15,
    close_thresh: float = 0.02,
    grasp_tol: float = 0.005,
    max_iters: int = 12,
    damping: float = 0.05,
    step_bound: float = 0.2,
    move_cap: float = 0.6,
    active_phases: Sequence[int] = (REACH, GRASP),
    exec_horizon: int = 8,
    fd_eps: float = 1.0e-4,
    gamma: float = 0.4,
    tol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Full-authority Observer->IK handoff for the final centimetre.

    Gated to REACH/GRASP within ``capture_zone`` of the target and only while NOT
    yet grasped. When it fires it REPLACES GR00T's chosen arm chunk with a DLS-IK
    ramp toward the grasp target (per-query move-capped so it converges over
    re-queries, closed-loop), CBF-checked, and CLOSES the gripper once the pinch
    is within ``close_thresh``. Returns ``(arm_chunk, grip_chunk, report)``; when
    the gate is closed both inputs are returned unchanged.
    """
    lim = lim or cbf.Limits()
    arm = np.asarray(a_groot_chunk, dtype=np.float64)
    grip = np.asarray(grip_chunk, dtype=np.float64).reshape(-1).copy()
    report: dict[str, Any] = {
        "handoff": True, "applied": False, "reason": "off",
        "pinch_err_before_cm": None, "pinch_err_after_cm": None,
        "iters": 0, "grip_close": False, "cbf_scale": 0.0,
    }
    if observer_target is None or arm.ndim != 2 or arm.shape[0] < 2 or arm.shape[1] != 6:
        report["reason"] = "no_target_or_degenerate"
        return a_groot_chunk, grip_chunk, report
    if int(phase) not in {int(p) for p in active_phases} or bool(grasped):
        report["reason"] = "phase_or_grasped_gate"
        return a_groot_chunk, grip_chunk, report

    tgt = np.asarray(observer_target, dtype=np.float64).reshape(3)
    q0 = arm[0].astype(np.float64)
    pinch0 = _batch_pinch(pinch_fn, q0[None])[0]
    d0 = float(np.linalg.norm(tgt - pinch0))
    report["pinch_err_before_cm"] = round(d0 * 100.0, 3)
    if d0 > capture_zone:
        report["reason"] = "out_of_capture_zone"        # GR00T still doing gross reach
        return a_groot_chunk, grip_chunk, report

    # --- IK oracle takes over: solve, then move-cap for closed-loop convergence -
    q_star, _err_star, iters = _ik_solve(pinch_fn, q0, tgt, max_iters=max_iters,
                                         damping=damping, step_bound=step_bound,
                                         fd_eps=fd_eps, tol=grasp_tol)
    report["iters"] = iters
    dqt = q_star - q0
    nrm = float(np.linalg.norm(dqt))
    if nrm > move_cap:
        dqt = dqt * (move_cap / nrm)
    q_cmd = q0 + dqt
    T = arm.shape[0]
    ramp = np.linspace(0.0, 1.0, T)[:, None]

    # CBF-safe: keep the largest feasible scale of the ramp toward q_cmd.
    cand, scale_kept = None, 0.0
    for scale in _SHRINK_SCALES:
        c = q0[None, :] + scale * ramp * (q_cmd - q0)[None, :]
        if _feasible(c, pinch_fn, observer_target=tgt, grasped=grasped, phase=int(phase),
                     lim=lim, exec_horizon=exec_horizon, gamma=gamma, tol=tol):
            cand, scale_kept = c, float(scale)
            break
    report["cbf_scale"] = scale_kept
    if cand is None or scale_kept == 0.0:
        report["reason"] = "cbf_infeasible"
        return a_groot_chunk, grip_chunk, report

    # Close the gripper once ALREADY aligned (a prior query drove the pinch in).
    if d0 <= close_thresh:
        grip[:] = 1.0
        report["grip_close"] = True

    report["pinch_err_after_cm"] = round(
        float(np.linalg.norm(tgt - _batch_pinch(pinch_fn, cand[0][None])[0])) * 100.0, 3)
    report["applied"] = True
    report["reason"] = "applied"
    corrected = cand.astype(np.asarray(a_groot_chunk).dtype, copy=False)
    return corrected, grip, report
