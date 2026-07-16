"""Unit tests for the Phase-1 probe verdict logic (probe_analysis.py).

The analyzer is pure-Python (no torch/browser/network), so the verdict fork of
PLAN_MULTIAGENT.md is tested here BEFORE the multi-hour probe run depends on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / "examples" / "ur5e-drugsort" / "browser_capture"))

import probe_analysis as pa  # noqa: E402

HOME = [-1.57, -1.56, 1.58, -1.57, -1.57, 0.0]
EXPERT_MIN = [-2.2, -2.0, 0.5, -2.5, -2.0, -1.0]
EXPERT_MAX = [-0.8, -0.8, 2.2, -0.8, -1.0, 1.0]


def _tick(i: int, *, sha: str | None = None, wrist_sha: str | None = None,
          state: list[float] | None = None,
          target: list[float] | None = None, grip_cmd: float = 0.0,
          grip_meas: float = 0.0, track: float = 0.01) -> dict:
    return {
        "ext_sha": sha if sha is not None else f"h{i:04d}",
        "wrist_sha": wrist_sha if wrist_sha is not None else f"w{i:04d}",
        "state_sent": (state or [HOME[j] + 0.001 * i for j in range(6)]) + [0.0],
        "grasp_target": target if target is not None else [-0.42, -0.01, 0.27],
        "post": {"grip_cmd": grip_cmd, "grip_meas": grip_meas,
                 "track_err": [track] * 6},
    }


# ---- obs_liveness -----------------------------------------------------------

def test_liveness_alive_when_all_vary() -> None:
    ticks = [_tick(i) for i in range(50)]
    out = pa.obs_liveness(ticks)
    assert out["images_alive"] and out["state_alive"] and out["target_alive"]


def test_liveness_dead_images_when_hashes_repeat() -> None:
    ticks = [_tick(i, sha="frozen") for i in range(50)]
    assert pa.obs_liveness(ticks)["images_alive"] is False


def test_liveness_dead_images_when_wrist_frozen() -> None:
    # A frozen wrist feed with a live exterior must still flag the images dead.
    ticks = [_tick(i, wrist_sha="frozen-wrist") for i in range(50)]
    assert pa.obs_liveness(ticks)["images_alive"] is False


def test_liveness_tolerates_missing_wrist_hashes() -> None:
    ticks = [{k: v for k, v in _tick(i).items() if k != "wrist_sha"} for i in range(50)]
    out = pa.obs_liveness(ticks)
    assert out["images_alive"] is True and out["wrist_hash_uniqueness"] is None


def test_liveness_dead_state_when_constant() -> None:
    ticks = [_tick(i, state=list(HOME)) for i in range(50)]
    assert pa.obs_liveness(ticks)["state_alive"] is False


def test_liveness_dead_target_when_nonfinite() -> None:
    ticks = [_tick(i, target=[float("nan")] * 3) for i in range(5)]
    assert pa.obs_liveness(ticks)["target_alive"] is False


# ---- target_tracks_scene ----------------------------------------------------

def test_target_tracks_scene_true_when_it_moves_with_vial() -> None:
    out = pa.target_tracks_scene(
        [[-0.44, -0.03, 0.27], [-0.40, 0.02, 0.27], [-0.42, -0.01, 0.27]],
        [[0.36, -0.05, 0.11], [0.40, 0.01, 0.11], [0.38, -0.02, 0.11]],
    )
    assert out["target_tracks_scene"] is True


def test_target_tracks_scene_false_when_frozen() -> None:
    out = pa.target_tracks_scene(
        [[-0.42, -0.01, 0.27]] * 3,
        [[0.36, -0.05, 0.11], [0.40, 0.01, 0.11], [0.38, -0.02, 0.11]],
    )
    assert out["target_tracks_scene"] is False


# ---- action_scale -----------------------------------------------------------

def test_scale_ok_inside_expert_range() -> None:
    chunk = [[-1.5, -1.5, 1.6, -1.5, -1.5, 0.1]] * 8
    out = pa.action_scale([chunk] * 5, EXPERT_MIN, EXPERT_MAX, HOME)
    assert out["scale_ok"] is True


def test_scale_flags_normalised_actions() -> None:
    # A de-norm bug returning [-1, 1]-ish values: elbow ~1.6 rad expected but 0.1.
    chunk = [[0.1, -0.1, 0.05, 0.0, -0.05, 0.02]] * 8
    out = pa.action_scale([chunk] * 5, EXPERT_MIN, EXPERT_MAX, HOME)
    assert out["scale_ok"] is False


def test_scale_detects_home_collapse() -> None:
    chunk = [[h + 0.01 for h in HOME]] * 8
    out = pa.action_scale([chunk] * 5, EXPERT_MIN, EXPERT_MAX, HOME)
    assert out["home_collapse"] is True


# ---- sensitivity ------------------------------------------------------------

def test_sensitivity_responds_on_clear_shift() -> None:
    base = [[0.0] * 6 for _ in range(5)]
    cond = [[0.08, 0, 0, 0, 0, 0] for _ in range(5)]
    assert pa.sensitivity(base, cond)["responds"] is True


def test_sensitivity_blind_when_identical() -> None:
    base = [[0.0] * 6 for _ in range(5)]
    assert pa.sensitivity(base, list(base))["responds"] is False


def test_sensitivity_blind_when_shift_below_noise() -> None:
    base = [[0.05 * ((i % 2) * 2 - 1)] + [0.0] * 5 for i in range(6)]
    cond = [[0.05 * ((i % 2) * 2 - 1) + 0.01] + [0.0] * 5 for i in range(6)]
    assert pa.sensitivity(base, cond)["responds"] is False


# ---- verdict fork -----------------------------------------------------------

LIVE_OK = {"images_alive": True, "state_alive": True, "target_alive": True}
SCALE_OK = {"scale_ok": True, "home_collapse": False}
SENS_OK = {"target": {"responds": True}}


def test_verdict_wiring_bug() -> None:
    v = pa.verdict({**LIVE_OK, "images_alive": False}, SCALE_OK, SENS_OK, 0.001, 0.05)
    assert v["verdict"] == "WIRING_BUG"


def test_verdict_decoding_bug() -> None:
    v = pa.verdict(LIVE_OK, {"scale_ok": False, "home_collapse": False},
                   SENS_OK, 0.001, 0.05)
    assert v["verdict"] == "DECODING_BUG"


def test_verdict_bridge_payload_bug() -> None:
    v = pa.verdict(LIVE_OK, SCALE_OK, SENS_OK, 0.5, 0.05)
    assert v["verdict"] == "BRIDGE_PAYLOAD_BUG"


def test_verdict_input_blind_on_no_response() -> None:
    v = pa.verdict(LIVE_OK, SCALE_OK, {"target": {"responds": False}}, 0.001, 0.0)
    assert v["verdict"] == "POLICY_INPUT_BLIND"


def test_verdict_input_blind_on_home_collapse() -> None:
    v = pa.verdict(LIVE_OK, {"scale_ok": True, "home_collapse": True},
                   SENS_OK, 0.001, 0.05)
    assert v["verdict"] == "POLICY_INPUT_BLIND"


def test_verdict_capability_ceiling() -> None:
    v = pa.verdict(LIVE_OK, SCALE_OK, SENS_OK, 0.001, 0.05)
    assert v["verdict"] == "CAPABILITY_CEILING"


def test_verdict_no_bridge_diff_available() -> None:
    v = pa.verdict(LIVE_OK, SCALE_OK, SENS_OK, None, 0.05)
    assert v["verdict"] == "CAPABILITY_CEILING"
    assert v["checks"]["bridge_matches_native"] is None


# ---- grip skew + tracking ---------------------------------------------------

def test_grip_skew_flagged() -> None:
    ticks = [_tick(i, grip_cmd=1.0, grip_meas=0.2) for i in range(20)]
    out = pa.grip_skew(ticks)
    assert out["grip_skew_flagged"] is True


def test_grip_skew_ignores_inactive_ticks() -> None:
    ticks = [_tick(i, grip_cmd=0.0, grip_meas=0.0) for i in range(20)]
    assert pa.grip_skew(ticks)["n_active_ticks"] == 0


def test_tracking_flags_untracked_targets() -> None:
    ticks = [_tick(i, track=0.4) for i in range(10)]
    assert pa.tracking(ticks)["tracking_ok"] is False
