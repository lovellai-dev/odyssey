"""Pure verdict logic for the Phase-1 inference-input integrity probe.

Consumes the raw telemetry the probe gathers (per-tick browser records, the
controlled-payload battery, the VM native-vs-bridge diff, expert dataset stats)
and derives the Phase-1 verdict fork of PLAN_MULTIAGENT.md:

    WIRING_BUG          observation path dead/stale (images, state or Observer
                        target never vary or never arrive)
    DECODING_BUG        actions arrive but at the wrong scale vs the expert
                        action distribution (de-normalisation class of bug)
    BRIDGE_PAYLOAD_BUG  the HTTP bridge materially disagrees with a native ZMQ
                        query on the same observation
    POLICY_INPUT_BLIND  obs alive + scale right, but actions do not respond to
                        controlled input changes (mode collapse)
    CAPABILITY_CEILING  everything wired and responsive — the policy is just weak

No torch / network / browser here: pure functions over dicts, unit-testable.
"""
from __future__ import annotations

import math
from typing import Any

# Thresholds (rad / m / ratios) — deliberately loose: this probe flags order-of-
# magnitude pathologies, not fine effects.
STATE_STD_MIN = 1e-4          # joint state must move at least this much across ticks
IMG_UNIQ_MIN = 0.9            # fraction of distinct frame hashes across ticks
ACT_ABS_MIN = 0.01            # rad: minimum action delta that counts as "responded"
SENS_RATIO_MIN = 3.0          # signal (cross-condition delta) vs noise (repeat std)
SCALE_INLIER_MIN = 0.95       # fraction of chunk values inside expert range (+margin)
SCALE_MARGIN = 0.15           # rad margin added around the expert per-dim range
HOME_COLLAPSE_RAD = 0.10      # mean |chunk - home| below this = "commanded to stay home"
TRACK_ERR_MAX = 0.05          # rad: post-chunk |ctrl - qpos| that counts as tracking
BRIDGE_DIFF_MAX = 0.02        # rad: native-vs-bridge mean first-step disagreement
GRIP_SKEW_MIN = 0.10          # mean |cmd - meas| grip divergence worth flagging


def _std(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def obs_liveness(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """Are the images/state/Observer target the policy receives actually varying?"""
    if not ticks:
        return {"images_alive": False, "state_alive": False, "target_alive": False,
                "n_ticks": 0}
    ext = [t["ext_sha"] for t in ticks]
    uniq = len(set(ext)) / len(ext)
    wr = [t.get("wrist_sha") for t in ticks if t.get("wrist_sha")]
    wr_uniq = (len(set(wr)) / len(wr)) if wr else None
    joint_stds = [
        _std([float(t["state_sent"][j]) for t in ticks]) for j in range(6)
    ]
    tgts = [t.get("grasp_target") for t in ticks if t.get("grasp_target")]
    tgt_std = (
        max(_std([float(g[k]) for g in tgts]) for k in range(3)) if len(tgts) > 1 else 0.0
    )
    return {
        "images_alive": uniq >= IMG_UNIQ_MIN and (wr_uniq is None or wr_uniq >= IMG_UNIQ_MIN),
        "image_hash_uniqueness": round(uniq, 3),
        "wrist_hash_uniqueness": (None if wr_uniq is None else round(wr_uniq, 3)),
        "state_alive": max(joint_stds) >= STATE_STD_MIN,
        "state_max_joint_std": round(max(joint_stds), 6),
        "target_alive": len(tgts) > 0 and all(
            all(math.isfinite(float(v)) for v in g) for g in tgts
        ),
        "target_within_tick_std": round(tgt_std, 4),
        "n_ticks": len(ticks),
    }


def target_tracks_scene(per_episode_targets: list[list[float]],
                        per_episode_vials: list[list[float]]) -> dict[str, Any]:
    """Across episodes with different vial poses, does the Observer target move too?

    Loose check: the across-episode spread of the injected target should be at
    least half the across-episode spread of the GT vial position (xy).
    """
    if len(per_episode_targets) < 2:
        return {"target_tracks_scene": None, "reason": "need >=2 episodes"}
    def spread(rows: list[list[float]], dims: int) -> float:
        return max(
            max(r[k] for r in rows) - min(r[k] for r in rows) for k in range(dims)
        )
    tgt_spread = spread(per_episode_targets, 2)
    vial_spread = spread(per_episode_vials, 2)
    if vial_spread < 0.005:
        return {"target_tracks_scene": None, "reason": "vial poses too similar"}
    ok = tgt_spread >= 0.5 * vial_spread
    return {"target_tracks_scene": ok, "target_xy_spread_m": round(tgt_spread, 4),
            "vial_xy_spread_m": round(vial_spread, 4)}


def action_scale(chunks: list[list[list[float]]],
                 expert_min: list[float], expert_max: list[float],
                 home_q: list[float]) -> dict[str, Any]:
    """Are returned arm targets inside the expert action range, or collapsed to home?"""
    vals_per_dim: list[list[float]] = [[] for _ in range(6)]
    for chunk in chunks:
        for row in chunk:
            for j in range(6):
                vals_per_dim[j].append(float(row[j]))
    n = sum(len(v) for v in vals_per_dim)
    if n == 0:
        return {"scale_ok": False, "reason": "no chunks"}
    inliers = 0
    for j in range(6):
        lo, hi = expert_min[j] - SCALE_MARGIN, expert_max[j] + SCALE_MARGIN
        inliers += sum(1 for v in vals_per_dim[j] if lo <= v <= hi)
    frac = inliers / n
    mean_dev_home = _mean([
        abs(v - home_q[j]) for j in range(6) for v in vals_per_dim[j]
    ])
    per_dim_range = [round(max(v) - min(v), 4) if v else 0.0 for v in vals_per_dim]
    return {
        "scale_ok": frac >= SCALE_INLIER_MIN,
        "inlier_fraction": round(frac, 4),
        "mean_abs_dev_from_home_rad": round(mean_dev_home, 4),
        "home_collapse": mean_dev_home < HOME_COLLAPSE_RAD,
        "per_dim_chunk_range_rad": per_dim_range,
    }


def sensitivity(base_actions: list[list[float]],
                cond_actions: list[list[float]]) -> dict[str, Any]:
    """Signal-to-noise of a controlled input change on the FIRST action step.

    base_actions / cond_actions: k repeats of the first 6-dim arm action under the
    base and the perturbed condition. Responds = mean shift beats both repeat
    noise (ratio) and an absolute floor.
    """
    if not base_actions or not cond_actions:
        return {"responds": None, "reason": "missing actions"}
    dims = len(base_actions[0])
    base_mean = [_mean([a[j] for a in base_actions]) for j in range(dims)]
    cond_mean = [_mean([a[j] for a in cond_actions]) for j in range(dims)]
    delta = math.sqrt(sum((b - c) ** 2 for b, c in zip(base_mean, cond_mean)))
    noise = max(
        max(_std([a[j] for a in base_actions]) for j in range(dims)),
        max(_std([a[j] for a in cond_actions]) for j in range(dims)),
        1e-6,
    )
    ratio = delta / noise
    responds = bool(delta >= ACT_ABS_MIN and ratio >= SENS_RATIO_MIN)
    return {"responds": responds, "delta_rad": round(delta, 5),
            "repeat_noise_rad": round(noise, 5), "snr": round(ratio, 2)}


def verdict(liveness: dict[str, Any], scale: dict[str, Any],
            sens: dict[str, dict[str, Any]],
            bridge_diff_rad: float | None,
            across_tick_action_std: float) -> dict[str, Any]:
    """Fold every probe axis into the Phase-1 fork of PLAN_MULTIAGENT.md."""
    obs_ok = bool(liveness.get("images_alive") and liveness.get("state_alive")
                  and liveness.get("target_alive"))
    any_responds = any(
        bool(s.get("responds")) for s in sens.values() if s.get("responds") is not None
    ) or across_tick_action_std >= ACT_ABS_MIN
    checks = {
        "obs_alive": obs_ok,
        "scale_ok": bool(scale.get("scale_ok")),
        "home_collapse": bool(scale.get("home_collapse")),
        "responds_to_inputs": any_responds,
        "bridge_matches_native": (None if bridge_diff_rad is None
                                   else bool(bridge_diff_rad <= BRIDGE_DIFF_MAX)),
    }
    if not obs_ok:
        v = "WIRING_BUG"
        nxt = "fix the observation path (stale/dead input) and re-eval — near-zero cost"
    elif not checks["scale_ok"]:
        v = "DECODING_BUG"
        nxt = "fix action de-normalisation/decoding in the serving path and re-eval"
    elif checks["bridge_matches_native"] is False:
        v = "BRIDGE_PAYLOAD_BUG"
        nxt = "fix the HTTP bridge payload construction (image/state packaging) and re-eval"
    elif not any_responds or checks["home_collapse"]:
        v = "POLICY_INPUT_BLIND"
        nxt = "mode collapse — Phase 3 (segmented SFT + clean retrain) is mandatory"
    else:
        v = "CAPABILITY_CEILING"
        nxt = ("wires clean; the policy is just weak — Phase 3 knowing cheap fixes are "
               "dead; gate Phase 4 on near-miss")
    return {"verdict": v, "next": nxt, "checks": checks}


def grip_skew(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """Quantify the measured-vs-commanded grip divergence (Phase 1b input)."""
    rows = [t["post"] for t in ticks
            if t.get("post") and (t["post"].get("grip_cmd", 0) > 0.05
                                  or t["post"].get("grip_meas", 0) > 0.05)]
    if not rows:
        return {"grip_skew_flagged": False, "n_active_ticks": 0}
    diffs = [abs(float(r["grip_cmd"]) - float(r["grip_meas"])) for r in rows]
    m = _mean(diffs)
    return {"grip_skew_flagged": m >= GRIP_SKEW_MIN,
            "mean_abs_cmd_minus_meas": round(m, 4), "n_active_ticks": len(rows)}


def tracking(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    """Does the arm reach the targets GR00T commands (post-chunk |ctrl - qpos|)?"""
    errs = [max(float(e) for e in t["post"]["track_err"])
            for t in ticks if t.get("post") and t["post"].get("track_err")]
    if not errs:
        return {"tracking_ok": None, "n": 0}
    m = _mean(errs)
    return {"tracking_ok": m <= TRACK_ERR_MAX, "mean_max_track_err_rad": round(m, 4),
            "worst_track_err_rad": round(max(errs), 4), "n": len(errs)}
