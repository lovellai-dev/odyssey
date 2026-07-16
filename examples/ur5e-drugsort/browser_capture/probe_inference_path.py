#!/usr/bin/env python3
"""Phase-1 controlled-payload battery + verdict (PLAN_MULTIAGENT.md Phase 1).

Runs AFTER probe_browser_groot.js has produced live-loop telemetry and dumped two
real frame+state samples. Three jobs:

1. EXPERT STATS — per-dim action min/max/mean/std from the local conditioned
   LeRobot dataset (the distribution GR00T was trained to output). Detects the
   de-normalisation class of bug Gate 0 could not see.
2. CONTROLLED BATTERY — k-repeat queries with one input changed at a time:
     * full deployment path (agent-service -> conditioner -> tunnel -> bridge):
       baseline repeats + a different-scene frame (Observer end-to-end aliveness);
     * direct bridge (tunnel), 10-D state under OUR control: grasp-target shifts
       (+5cm x/y/z), proprio shift, black exterior/wrist/both frames, instruction
       swap. Measures which inputs the policy actually responds to.
3. VERDICT — folds everything (plus the VM native-vs-bridge diff, if present)
   through probe_analysis.verdict into probe_phase1_result.json.

Usage:
  probe_inference_path.py --out <probe_out_dir> --dataset <lerobot dir>
      --agent-url http://127.0.0.1:8032 --bridge-url http://127.0.0.1:5596
      [--vm-diff <vm_diff.json>] [--k 5] --result <probe_phase1_result.json>
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_analysis as pa  # noqa: E402

INSTRUCTION = "pick up the vial and place it in the rack"


# ---- expert stats -----------------------------------------------------------

def expert_action_stats(dataset: Path) -> dict:
    files = sorted((dataset / "data" / "chunk-000").glob("episode_*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {dataset}")
    acts, states = [], []
    for f in files:
        t = pq.read_table(f, columns=["action", "observation.state"])
        acts.append(np.stack(t.column("action").to_pylist()))
        states.append(np.stack(t.column("observation.state").to_pylist()))
    a = np.concatenate(acts)            # (T, 7): 6 joints + grip
    s = np.concatenate(states)          # (T, 10): 6 joints + grip + target xyz
    return {
        "n_samples": int(a.shape[0]),
        "action_min": [float(x) for x in a.min(0)],
        "action_max": [float(x) for x in a.max(0)],
        "action_mean": [float(x) for x in a.mean(0)],
        "action_std": [float(x) for x in a.std(0)],
        "state_grasp_target_min": [float(x) for x in s[:, 7:10].min(0)],
        "state_grasp_target_max": [float(x) for x in s[:, 7:10].max(0)],
    }


# ---- HTTP query helpers ------------------------------------------------------

def _post(url: str, payload: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from None


def _black_frame_b64() -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (0, 0, 0)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def query_k(url: str, payload: dict, k: int) -> dict:
    """k queries -> first-step arm actions + full chunks + grips."""
    firsts, chunks, grips, targets = [], [], [], []
    for _ in range(k):
        res = _post(f"{url}/get_action", payload)
        if "q" not in res:
            raise RuntimeError(f"bad response from {url}: {json.dumps(res)[:200]}")
        firsts.append([float(x) for x in res["q"]])
        chunks.append([[float(x) for x in row] for row in res.get("chunk_q") or [res["q"]]])
        grips.append(float(res.get("grip", 0.0)))
        if res.get("grasp_target"):
            targets.append([float(x) for x in res["grasp_target"]])
        time.sleep(0.15)
    return {"first_actions": firsts, "chunks": chunks, "grips": grips,
            "targets": targets}


# ---- battery -----------------------------------------------------------------

def run_battery(out: Path, agent_url: str, bridge_url: str, k: int) -> dict:
    ext = (out / "frame_q0_ext.b64").read_text().strip()
    wrist = (out / "frame_q0_wrist.b64").read_text().strip()
    state7 = json.loads((out / "frame_q0_state.json").read_text())[:7]
    extB = (out / "frame_q30_ext.b64").read_text().strip() if (out / "frame_q30_ext.b64").exists() else None
    wristB = (out / "frame_q30_wrist.b64").read_text().strip() if (out / "frame_q30_wrist.b64").exists() else None

    battery: dict = {"k": k}

    # -- full deployment path (7-D state; conditioner appends the live target) --
    base_payload = {"image_b64": ext, "image_b64_wrist": wrist, "state": state7,
                    "instruction": INSTRUCTION, "sid": "probe-full"}
    battery["full_baseline"] = query_k(f"{agent_url}/api/groot", base_payload, k)
    if extB and wristB:
        battery["full_frameB"] = query_k(
            f"{agent_url}/api/groot",
            {**base_payload, "image_b64": extB, "image_b64_wrist": wristB,
             "sid": "probe-full-b"}, k)

    # -- direct bridge (10-D state fully under our control) ---------------------
    tgt = battery["full_baseline"]["targets"]
    tgt0 = tgt[0] if tgt else [-0.425, -0.015, 0.272]
    state10 = [float(x) for x in state7] + [float(x) for x in tgt0]
    dbase = {"image_b64": ext, "image_b64_wrist": wrist, "state": state10,
             "instruction": INSTRUCTION}
    battery["bridge_baseline"] = query_k(bridge_url, dbase, k)

    def shifted(dim: int, dv: float) -> list[float]:
        s = list(state10)
        s[dim] += dv
        return s

    battery["bridge_target_x"] = query_k(bridge_url, {**dbase, "state": shifted(7, 0.05)}, k)
    battery["bridge_target_y"] = query_k(bridge_url, {**dbase, "state": shifted(8, 0.05)}, k)
    battery["bridge_target_z"] = query_k(bridge_url, {**dbase, "state": shifted(9, 0.05)}, k)
    battery["bridge_proprio"] = query_k(bridge_url, {**dbase, "state": shifted(0, 0.3)}, k)
    blk = _black_frame_b64()
    battery["bridge_black_ext"] = query_k(bridge_url, {**dbase, "image_b64": blk}, k)
    battery["bridge_black_wrist"] = query_k(bridge_url, {**dbase, "image_b64_wrist": blk}, k)
    battery["bridge_black_both"] = query_k(
        bridge_url, {**dbase, "image_b64": blk, "image_b64_wrist": blk}, k)
    battery["bridge_instruction"] = query_k(
        bridge_url, {**dbase, "instruction": "wave the gripper high in the air"}, k)
    return battery


# ---- assemble the verdict ------------------------------------------------------

def analyse(out: Path, battery: dict, stats: dict, vm_diff: dict | None,
            home_q: list[float]) -> dict:
    ticks: list[dict] = []
    per_ep_targets, per_ep_vials = [], []
    ep_files = sorted(out.glob("probe_ep*.jsonl"),
                      key=lambda p: int(p.stem.split("ep")[1].split("_")[0]))
    for f in ep_files:
        rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
        ticks.extend(rows)
        tg = [r["grasp_target"] for r in rows if r.get("grasp_target")]
        vx = [r["post"]["vial_xyz"] for r in rows if r.get("post")]
        if tg and vx:
            per_ep_targets.append([float(np.median([t[i] for t in tg])) for i in range(3)])
            per_ep_vials.append([float(np.median([v[i] for v in vx])) for i in range(3)])

    liveness = pa.obs_liveness(ticks)
    tracks = pa.target_tracks_scene(per_ep_targets, per_ep_vials)
    if tracks.get("target_tracks_scene") is False:
        liveness["target_alive"] = False
        liveness["target_frozen_across_scenes"] = True

    # Chunks: live-loop chunks (all episodes) + battery baseline chunks.
    chunks: list[list[list[float]]] = list(battery["bridge_baseline"]["chunks"])
    for f in sorted(out.glob("probe_ep*_chunks.json"),
                    key=lambda p: int(p.stem.split("ep")[1].split("_")[0])):
        chunks.extend(json.loads(f.read_text()))
    scale = pa.action_scale(chunks, stats["action_min"][:6], stats["action_max"][:6],
                            home_q)

    base = battery["bridge_baseline"]["first_actions"]
    sens = {
        "target_x": pa.sensitivity(base, battery["bridge_target_x"]["first_actions"]),
        "target_y": pa.sensitivity(base, battery["bridge_target_y"]["first_actions"]),
        "target_z": pa.sensitivity(base, battery["bridge_target_z"]["first_actions"]),
        "proprio": pa.sensitivity(base, battery["bridge_proprio"]["first_actions"]),
        "black_ext": pa.sensitivity(base, battery["bridge_black_ext"]["first_actions"]),
        "black_wrist": pa.sensitivity(base, battery["bridge_black_wrist"]["first_actions"]),
        "black_both": pa.sensitivity(base, battery["bridge_black_both"]["first_actions"]),
        "instruction": pa.sensitivity(base, battery["bridge_instruction"]["first_actions"]),
    }
    if "full_frameB" in battery:
        sens["full_scene_change"] = pa.sensitivity(
            battery["full_baseline"]["first_actions"],
            battery["full_frameB"]["first_actions"])

    # Across-tick action variation in the live loop (first-step arm actions).
    q0s = [t["q0"] for t in ticks if t.get("q0")]
    across_std = float(np.mean(np.std(np.asarray(q0s, dtype=np.float64), axis=0))) if len(q0s) > 2 else 0.0

    # Repeat determinism of the served head (flow-matching may be seeded).
    rep = np.asarray(battery["bridge_baseline"]["first_actions"], dtype=np.float64)
    repeat_std = float(np.mean(np.std(rep, axis=0)))

    # Noise-aware bridge gate: only report a disagreement the VM script itself
    # judged material (diff > 3x within-path repeat noise) — a stochastic flow
    # head must not fabricate BRIDGE_PAYLOAD_BUG out of sampling noise.
    bridge_diff = None
    if vm_diff and "mean_first_step_abs_diff_rad" in vm_diff:
        bridge_diff = (float(vm_diff["mean_first_step_abs_diff_rad"])
                       if vm_diff.get("materially_different") else 0.0)

    # Same discipline for the live-loop signal: across-tick action variation only
    # counts as "responds to inputs" if it clears the head's own sampling noise.
    effective_across = across_std if across_std >= max(pa.ACT_ABS_MIN, 3 * repeat_std) else 0.0

    verdict = pa.verdict(liveness, scale, sens, bridge_diff, effective_across)
    episodes = json.loads((out / "probe_episodes.json").read_text())["episodes"] \
        if (out / "probe_episodes.json").exists() else []

    return {
        "phase": "1-inference-input-integrity-probe",
        "verdict": verdict,
        "obs_liveness": liveness,
        "target_tracks_scene": tracks,
        "action_scale": scale,
        "sensitivity": sens,
        "across_tick_action_std_rad": round(across_std, 5),
        "repeat_noise_rad": round(repeat_std, 5),
        "deterministic_head": repeat_std < 1e-5,
        "bridge_vs_native": vm_diff,
        "grip_semantics_phase1b": pa.grip_skew(ticks),
        "tracking": pa.tracking(ticks),
        "near_miss": {
            "per_episode_min_pad_to_vial_cm": [
                (None if e.get("min_pad_to_vial") is None
                 else round(e["min_pad_to_vial"] * 100, 1)) for e in episodes],
            "note": "Phase-4 gate: residual RL only rescues a near-miss base",
        },
        "expert_stats": {k: v for k, v in stats.items() if k != "n_samples"} | {
            "n_samples": stats["n_samples"]},
        "episodes": episodes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--agent-url", default="http://127.0.0.1:8032")
    ap.add_argument("--bridge-url", default="http://127.0.0.1:5596")
    ap.add_argument("--vm-diff", type=Path, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--result", required=True, type=Path)
    ap.add_argument("--home-q", default="-1.57,-1.56,1.58,-1.57,-1.57,0.0")
    args = ap.parse_args()

    print("[battery] expert stats from", args.dataset, flush=True)
    stats = expert_action_stats(args.dataset)
    print("[battery] action ranges:", [
        f"[{lo:.2f},{hi:.2f}]" for lo, hi in
        zip(stats["action_min"][:6], stats["action_max"][:6])], flush=True)

    print("[battery] controlled payload battery (k=%d)" % args.k, flush=True)
    battery = run_battery(args.out, args.agent_url.rstrip("/"),
                          args.bridge_url.rstrip("/"), args.k)
    (args.out / "battery.json").write_text(json.dumps(battery))

    vm_diff = None
    if args.vm_diff and args.vm_diff.exists():
        vm_diff = json.loads(args.vm_diff.read_text())

    home_q = [float(x) for x in args.home_q.split(",")]
    result = analyse(args.out, battery, stats, vm_diff, home_q)
    args.result.write_text(json.dumps(result, indent=2))
    print("[battery] VERDICT:", json.dumps(result["verdict"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
