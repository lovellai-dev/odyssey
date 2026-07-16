#!/usr/bin/env python3
"""Grip-authority probe: can init-noise steering make the frozen GR00T base
CLOSE THE GRIPPER at grasp states?

Motivation (v0.1 browser A/B): the anti-aliasing steering fixed the home-freeze
(15/15 approaches) but gripMax stayed <= 0.04 in every steered episode — the
gripper never closes. The offline gate scored the ARM end-effector only; grip
fidelity through the noise channel was never measured. This probe answers the
one question that decides v0.2 vs a DAgger detour:

For held-out CLOSING windows (expert grip rises >= 0.5 within the 16-step
chunk), decode through the frozen base:
  (a) oracle    — this state's own inverted w*
  (b) steered   — the steering net's predicted noise
  (c) stock x16 — random seeds (the base's intrinsic noise-conditional
                  grip diversity: the AUTHORITY ceiling)
and report the decoded GRIP trajectory (real dim 6 of 7) vs the expert's.

Verdict fields:
  oracle_grip_ok_frac  — can SOME noise close the gripper? (authority exists)
  steered_grip_ok_frac — does the CURRENT net exploit it?
  stock_grip_max_p95   — do random seeds ever close it?
A window counts "ok" when decoded max grip >= 0.5 while the expert's is >= 0.5.

Run (GR00T venv, VM):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    ~/Isaac-GR00T/.venv/bin/python probe_grip_authority.py \
    --steering-net /home/ubuntu/steering_net_v01.npz --out /home/ubuntu/grip_authority.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REAL_STEPS = 16
GRIP_DIM = 6            # index of grip inside the 7 real action dims
CLOSE = 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="/home/ubuntu/ckpt/ur5e_drugsort_obscond/full/checkpoint-12000")
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--dataset", default="/home/ubuntu/ur5e_drugsort_obscond")
    ap.add_argument("--steering-net", default="/home/ubuntu/steering_net_v01.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--n-seeds", type=int, default=16)
    ap.add_argument("--fp-iters", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/home/ubuntu/grip_authority.json")
    args = ap.parse_args()

    import torch
    from flow_inverter_groot import (
        FlowInverter, enumerate_chunks, read_frames, load_gt_phases, phase_bucket,
        forward_euler_sample,
    )
    import train_steering_ur5e as ts
    from flowdagger_offline_gate_ur5e import load_steering_net, ACTION_HORIZON, ACTION_DIM

    forward, meta = load_steering_net(args.steering_net)
    val_eps = set(meta.get("val_episodes", []))
    inv = FlowInverter(args.model_path, args.embodiment_tag, args.device,
                       fp_iters=args.fp_iters)
    flow = inv.flow
    dataset_dir = Path(args.dataset)
    rng = np.random.default_rng(args.seed)

    # -- find held-out CLOSING windows: expert grip rises to >= CLOSE in-chunk --
    import pyarrow.parquet as pq
    picks = []
    for (ep, fr) in enumerate_chunks(dataset_dir, 4):
        if ep not in val_eps:
            continue
        f = dataset_dir / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        a = np.stack(pq.read_table(f, columns=["action"]).column("action").to_pylist())
        w = a[fr:fr + REAL_STEPS]
        if len(w) < REAL_STEPS:
            continue
        if w[0, GRIP_DIM] < 0.2 and w[:, GRIP_DIM].max() >= CLOSE:
            picks.append((ep, fr))
    if len(picks) > args.n:
        sel = np.sort(rng.choice(len(picks), size=args.n, replace=False))
        picks = [picks[i] for i in sel]
    print(f"[grip] {len(picks)} held-out closing windows", flush=True)
    if not picks:
        json.dump({"error": "no closing windows in val episodes"}, open(args.out, "w"))
        return 1

    gt_cache: dict = {}
    readers: dict = {}
    per = []
    for (ep, fr) in picks:
        frames = read_frames(dataset_dir, [(ep, fr)], readers)
        f0 = frames[0]
        cache, x_target = inv.encode_batch(frames)
        vfn = flow.velocity_fn(cache)
        expert_grip = float(np.max(np.asarray(f0["grip"]).reshape(-1)))

        # steering-net input (v2-aware, same as the offline gate)
        s10 = f0["state10"]
        phases = load_gt_phases(dataset_dir, ep, gt_cache)
        plabel = phase_bucket(phases[fr]) if (phases is not None and fr < len(phases)) else 1
        oh = ts.phase_onehot(np.array([plabel]))[0]
        cols = [s10[:7], s10[7:10], oh]
        if int(meta.get("in_dim", 14)) >= 15:
            n_frames = len(phases) if phases is not None else fr + REAL_STEPS
            cols.append(np.array([min(1.0, fr / max(1.0, float(n_frames)))], dtype=np.float32))
        obs_low = np.concatenate(cols).astype(np.float32)[None]

        def decode_grips(w, use_vfn):
            """(B,40,132) noise -> per-sample max decoded grip (physical 0..1)."""
            wt = torch.as_tensor(np.asarray(w, dtype=np.float32), device=args.device)
            with torch.no_grad():
                x = forward_euler_sample(wt, use_vfn, flow.n_steps, flow.dt)
            phys = flow.decode_to_physical(x)            # {single_arm, gripper}
            return np.asarray(phys["gripper"]).reshape(len(wt), -1).max(axis=1)

        # (a) oracle
        w_or, mse_or = inv.invert_batch(cache, x_target)
        grip_or = float(decode_grips(
            w_or.detach().float().cpu().numpy() if hasattr(w_or, "detach") else w_or,
            vfn)[0])
        # (b) steered
        pred = forward(obs_low)
        w_st = pred.reshape(1, ACTION_HORIZON, ACTION_DIM).astype(np.float32)
        grip_st = float(decode_grips(w_st, vfn)[0])
        # (c) stock seeds (broadcast the backbone cache to the seed batch)
        seeds = rng.standard_normal((args.n_seeds, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
        vfn_n = flow.velocity_fn(flow.broadcast_cache(cache, args.n_seeds))
        grips = decode_grips(seeds, vfn_n)
        per.append({
            "episode": int(ep), "frame": int(fr),
            "oracle_grip_max": grip_or, "steered_grip_max": grip_st,
            "stock_grip_max_best": float(np.max(grips)),
            "stock_grip_max_p95": float(np.percentile(grips, 95)),
            "stock_grip_close_frac": float(np.mean(grips >= CLOSE)),
            "oracle_recon_mse": float(np.asarray(mse_or).reshape(-1)[0]),
        })
        print(f"[grip] ep{ep} fr{fr} oracle={grip_or:.2f} steered={grip_st:.2f} "
              f"stock_best={per[-1]['stock_grip_max_best']:.2f} "
              f"stock_close_frac={per[-1]['stock_grip_close_frac']:.2f}", flush=True)

    ok = lambda k: float(np.mean([p[k] >= CLOSE for p in per]))
    out = {
        "n_windows": len(per),
        "oracle_grip_ok_frac": ok("oracle_grip_max"),
        "steered_grip_ok_frac": ok("steered_grip_max"),
        "stock_any_seed_ok_frac": float(np.mean([p["stock_grip_max_best"] >= CLOSE for p in per])),
        "stock_grip_close_frac_mean": float(np.mean([p["stock_grip_close_frac"] for p in per])),
        "per_window": per,
        "interpretation": {
            "authority_exists": "oracle_grip_ok_frac high => some noise closes the gripper",
            "net_exploits_it": "steered_grip_ok_frac high => the current net finds it",
        },
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print("[grip] HEADLINE", json.dumps({k: out[k] for k in
          ("n_windows", "oracle_grip_ok_frac", "steered_grip_ok_frac",
           "stock_any_seed_ok_frac")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
