#!/usr/bin/env python3
"""A5 — OFFLINE closed-form go/no-go gate for FlowDAgger Stage B.

On the steering net's HELD-OUT episodes (never seen in A4 training), at ~100
held-out states, decode actions three ways through the PATCHED GR00T model:

  (a) steered : init_noise = steering(obs_low_dim)  [real dims + fixed-seed pads]
  (b) stock   : 8 random-seed torch.randn draws (the deployed default)
  (c) oracle  : the state's OWN inverted w*  (the achievable ceiling)

Metrics (real action dims only):
  * real-dim action MSE vs the expert's recorded (normalised) chunk
  * FK end-effector distance at chunk step 8 (gr_pinch) vs the expert EE

GATE (both must hold):
  1. steered beats the stock-seed MEAN real-dim MSE on >= 70% of held-out states
  2. steered cuts the mean EE-distance-to-expert by >= 30% of the stock->oracle gap

Two processes (mirrors the probe):
  decode   — GR00T venv (torch): decode 3 ways, dump step-8 arm6 + MSEs to npz
  fk-gate  — mujoco venv: FK the arm6 -> gr_pinch xyz, compute EE distances, gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

REAL_STEPS = 16
REAL_DIMS = 7
ACTION_HORIZON = 40
ACTION_DIM = 132
STEP8 = 8


# ---------------------------------------------------------------------------
# numpy MLP forward (loads the A4 steering-net npz; no torch needed)
# ---------------------------------------------------------------------------
def load_steering_net(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    W, b = [], []
    for i in (0, 2, 4):  # nn.Sequential linear layer indices
        W.append(z[f"sd.{i}.weight"].astype(np.float32))
        b.append(z[f"sd.{i}.bias"].astype(np.float32))
    def forward(X: np.ndarray) -> np.ndarray:
        h = np.asarray(X, dtype=np.float32)
        for k in range(3):
            h = h @ W[k].T + b[k]
            if k < 2:
                h = np.maximum(h, 0.0)  # ReLU
        return h
    return forward, meta


# ---------------------------------------------------------------------------
# decode mode (GR00T venv)
# ---------------------------------------------------------------------------
def mode_decode(args) -> dict:
    from flow_inverter_groot import (
        FlowInverter, enumerate_chunks, read_frames, load_gt_phases, phase_bucket,
        forward_euler_sample,
    )
    import train_steering_ur5e as ts
    import torch

    forward, meta = load_steering_net(args.steering_net)
    design = meta.get("output_design", "real112")
    val_eps = set(meta.get("val_episodes", []))
    if getattr(args, "val_episodes", None):
        # Override for two-sided gating (e.g. DAgger-source states, whose meta
        # ids carry the 100000*source offset while the raw dataset uses 0..N).
        val_eps = {int(x) for x in str(args.val_episodes).split(",")}
    print(f"[gate] steering net design={design} val_eps={sorted(val_eps)}", flush=True)

    inv = FlowInverter(args.model_path, args.embodiment_tag, args.device, fp_iters=args.fp_iters)
    flow = inv.flow
    dataset_dir = Path(args.dataset)

    # held-out states: frames from val episodes at a denser stride, capped at n
    all_picks = [(e, f) for (e, f) in enumerate_chunks(dataset_dir, args.stride) if e in val_eps]
    rng = np.random.default_rng(args.seed)
    if len(all_picks) > args.n:
        sel = np.sort(rng.choice(len(all_picks), size=args.n, replace=False))
        all_picks = [all_picks[i] for i in sel]
    print(f"[gate] {len(all_picks)} held-out states", flush=True)

    gt_cache: dict = {}
    _dq_cache: dict = {}   # per-episode arm-state arrays for the v3 dq feature
    readers: dict = {}
    fixed_pads = ts.deploy_pads(1)[0]  # (40,132) fixed-seed pad template

    per = []
    steered_arm8, oracle_arm8, expert_arm8 = [], [], []
    stock_arm8 = []  # (N, 8, 6)
    steered_traj, oracle_traj, expert_traj = [], [], []   # (N,16,6) for CLF
    grasp_targets, plabels = [], []
    for (ep, fr) in all_picks:
        frames = read_frames(dataset_dir, [(ep, fr)], readers)
        f0 = frames[0]
        cache, x_target = inv.encode_batch(frames)  # B=1
        vfn = flow.velocity_fn(cache)

        # obs low-dim for the steering net
        s10 = f0["state10"]
        phases = load_gt_phases(dataset_dir, ep, gt_cache)
        plabel = phase_bucket(phases[fr]) if (phases is not None and fr < len(phases)) else 0
        oh = ts.phase_onehot(np.array([plabel]))[0]
        cols = [s10[:7], s10[7:10], oh]
        if int(meta.get("in_dim", 14)) >= 15:
            # v0.1 t_norm feature: frame / (episode length proxy). Matches the
            # trainer's t_norm_feature semantics (max frame + REAL_STEPS).
            n_frames = (len(phases) if phases is not None
                        else fr + REAL_STEPS)
            cols.append(np.array([min(1.0, fr / max(1.0, float(n_frames)))],
                                 dtype=np.float32))
        in_dim = int(meta.get("in_dim", 14))
        if in_dim == 21:
            # v3 recent-motion dq over the trainer's window, from the dataset.
            import pyarrow.parquet as pq
            if ep not in _dq_cache:
                pf = dataset_dir / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
                _dq_cache[ep] = np.stack(
                    pq.read_table(pf, columns=["observation.state"])
                    .column("observation.state").to_pylist())[:, :6]
            sarr = _dq_cache[ep]
            f_now = min(int(fr), len(sarr) - 1)
            dq = (sarr[f_now] - sarr[max(0, f_now - 8)]).astype(np.float32)
            cols.append(dq)
        elif in_dim > 100:
            # v4 image-conditioned head: pooled backbone features from THIS
            # state's own encode cache — same pool_backbone as train + deploy.
            from flow_inverter_groot import pool_backbone
            cols.append(pool_backbone(cache)[0].astype(np.float32))
        obs_low = np.concatenate(cols).astype(np.float32)[None]  # (1, 14|15|21|15+D)

        # (a) steered: build init_noise from the net output per its design.
        pred = forward(obs_low)
        if design == "full5280":
            init_noise = pred.reshape(1, ACTION_HORIZON, ACTION_DIM).astype(np.float32)
        else:  # real112 + fixed-seed pads
            init_noise = ts.assemble_init_noise(
                pred.reshape(1, REAL_STEPS, REAL_DIMS), pads=fixed_pads[None])  # (1,40,132)
        w_steer = torch.as_tensor(init_noise, dtype=torch.float32, device=args.device)
        with torch.no_grad():
            x_steer = forward_euler_sample(w_steer, vfn, flow.n_steps, flow.dt)
        mse_steer = flow.recon_mse_real(x_steer, x_target)

        # (c) oracle: invert this state's expert chunk
        w_oracle, mse_or = inv.invert_batch(cache, x_target)
        with torch.no_grad():
            x_oracle = forward_euler_sample(w_oracle, vfn, flow.n_steps, flow.dt)
        mse_oracle = float(mse_or[0])

        # (b) stock: 8 random seeds
        stock_mses = []
        stock_arms = []
        cache8 = flow.broadcast_cache(cache, args.n_seeds)
        vfn8 = flow.velocity_fn(cache8)
        with torch.no_grad():
            w8 = flow.random_noise(args.n_seeds, seed=args.seed + ep * 1000 + fr)
            x8 = forward_euler_sample(w8, vfn8, flow.n_steps, flow.dt)
            phys8 = flow.decode_to_physical(x8)
        arm8_stock = np.asarray(phys8["single_arm"], dtype=np.float32)[:, STEP8, :]  # (8,6)
        # per-seed real-dim MSE vs the (B=1) target
        xt_real = flow.real_slice(x_target)  # (1,16,7)
        d8 = (flow.real_slice(x8) - xt_real) ** 2  # (8,16,7) broadcast
        stock_mse_each = d8.mean(dim=(1, 2)).detach().float().cpu().numpy()  # (8,)
        stock_mse_mean = float(stock_mse_each.mean())

        # step-8 arm6 physical for steered / oracle / expert
        with torch.no_grad():
            phys_steer = flow.decode_to_physical(x_steer)
            phys_oracle = flow.decode_to_physical(x_oracle)
        steered_arm8.append(np.asarray(phys_steer["single_arm"], dtype=np.float32)[0, STEP8, :])
        oracle_arm8.append(np.asarray(phys_oracle["single_arm"], dtype=np.float32)[0, STEP8, :])
        expert_arm8.append(np.asarray(f0["arm"], dtype=np.float32)[STEP8, :])
        stock_arm8.append(arm8_stock)
        # Full 16-step arm trajectories + the state's grasp target: consumed by
        # the fk-gate's report-only CLF (Lyapunov-decrease) metrics.
        steered_traj.append(np.asarray(phys_steer["single_arm"], dtype=np.float32)[0])
        oracle_traj.append(np.asarray(phys_oracle["single_arm"], dtype=np.float32)[0])
        expert_traj.append(np.asarray(f0["arm"], dtype=np.float32))
        grasp_targets.append(np.asarray(s10[7:10], dtype=np.float32))
        plabels.append(int(plabel))

        per.append({
            "episode": int(ep), "frame": int(fr), "phase": int(plabel),
            "mse_steered": float(mse_steer),
            "mse_stock_mean": stock_mse_mean,
            "mse_oracle": mse_oracle,
            "steered_beats_stock": bool(mse_steer < stock_mse_mean),
        })
        print(f"[gate] ep{ep} fr{fr} steer={mse_steer:.4e} stock={stock_mse_mean:.4e} "
              f"oracle={mse_oracle:.4e} beat={mse_steer < stock_mse_mean}", flush=True)

    np.savez(
        args.decoded_out,
        steered_arm8=np.asarray(steered_arm8, dtype=np.float32),
        oracle_arm8=np.asarray(oracle_arm8, dtype=np.float32),
        expert_arm8=np.asarray(expert_arm8, dtype=np.float32),
        stock_arm8=np.asarray(stock_arm8, dtype=np.float32),
        mse_steered=np.asarray([p["mse_steered"] for p in per], dtype=np.float64),
        mse_stock_mean=np.asarray([p["mse_stock_mean"] for p in per], dtype=np.float64),
        mse_oracle=np.asarray([p["mse_oracle"] for p in per], dtype=np.float64),
        episode=np.asarray([p["episode"] for p in per], dtype=np.int64),
        frame=np.asarray([p["frame"] for p in per], dtype=np.int64),
        steered_traj=np.asarray(steered_traj, dtype=np.float32),
        oracle_traj=np.asarray(oracle_traj, dtype=np.float32),
        expert_traj=np.asarray(expert_traj, dtype=np.float32),
        grasp_target=np.asarray(grasp_targets, dtype=np.float32),
        phase=np.asarray(plabels, dtype=np.int64),
    )
    frac = float(np.mean([p["steered_beats_stock"] for p in per])) if per else 0.0
    out = {
        "n_holdout_states": len(per),
        "steered_beats_stock_frac": frac,
        "mse": {
            "steered_mean": float(np.mean([p["mse_steered"] for p in per])),
            "stock_mean": float(np.mean([p["mse_stock_mean"] for p in per])),
            "oracle_mean": float(np.mean([p["mse_oracle"] for p in per])),
        },
        "decoded_npz": str(args.decoded_out),
        "per_state": per,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_state"}, indent=2), flush=True)
    return out


# ---------------------------------------------------------------------------
# fk-gate mode (mujoco venv) — FK step-8 arm6 -> EE, compute distances, gate
# ---------------------------------------------------------------------------
def mode_fk_gate(args) -> dict:
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco as mj

    sys.path.insert(0, str(Path(args.src) / "scripts"))
    sys.path.insert(0, str(Path(args.src) / "src"))
    from gen_ur5e_drugsort_demos import _build_index

    dec = np.load(args.decoded_npz, allow_pickle=True)
    steered = dec["steered_arm8"]      # (N,6)
    oracle = dec["oracle_arm8"]
    expert = dec["expert_arm8"]
    stock = dec["stock_arm8"]           # (N,8,6)
    mse_steered = dec["mse_steered"]; mse_stock = dec["mse_stock_mean"]; mse_oracle = dec["mse_oracle"]
    N = len(steered)

    model = mj.MjModel.from_xml_path(args.xml)
    data = mj.MjData(model)
    idx = _build_index(mj, model)
    arm_qadr = idx["arm_qadr"]; pinch_site = idx["pinch_site"]
    hk = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")

    def fk(q6):
        if hk >= 0:
            mj.mj_resetDataKeyframe(model, data, hk)
        for a, qv in zip(arm_qadr, q6):
            data.qpos[a] = float(qv)
        mj.mj_forward(model, data)
        return np.array(data.site_xpos[pinch_site], dtype=np.float64)

    per = []
    for k in range(N):
        ee_exp = fk(expert[k])
        ee_steer = fk(steered[k])
        ee_or = fk(oracle[k])
        ee_stock = np.stack([fk(stock[k, j]) for j in range(stock.shape[1])])  # (8,3)
        d_steer = float(np.linalg.norm(ee_steer - ee_exp) * 100.0)
        d_or = float(np.linalg.norm(ee_or - ee_exp) * 100.0)
        d_stock = float(np.mean(np.linalg.norm(ee_stock - ee_exp[None], axis=1)) * 100.0)
        per.append({
            "episode": int(dec["episode"][k]), "frame": int(dec["frame"][k]),
            "ee_dist_steered_cm": d_steer, "ee_dist_stock_mean_cm": d_stock,
            "ee_dist_oracle_cm": d_or,
            "mse_steered": float(mse_steered[k]), "mse_stock_mean": float(mse_stock[k]),
            "mse_oracle": float(mse_oracle[k]),
            "steered_beats_stock_mse": bool(mse_steered[k] < mse_stock[k]),
            "steered_closer_ee": bool(d_steer < d_stock),
        })

    # ---- report-only CLF (Lyapunov-decrease) metrics on REACH-phase states ----
    # V = ||pinch - grasp_target||^2 along the decoded 16-step trajectory; a
    # good candidate DECREASES V monotonically. Purely informational for now —
    # the R2 best-of-N scorer is the primary consumer of clf_reward_ur5e.
    clf_report = None
    if "steered_traj" in dec.files:
        import clf_reward_ur5e as clfmod
        phase_arr = dec["phase"]; tgts = dec["grasp_target"]
        reach_idx = [k for k in range(N) if int(phase_arr[k]) == clfmod.REACH]
        if reach_idx:
            def traj_metrics(traj_key):
                out = []
                for k in reach_idx:
                    pinch = np.stack([fk(q) for q in dec[traj_key][k]])
                    out.append(clfmod.reach_series_metrics(pinch, tgts[k]))
                return {
                    "mean_total_reward": float(np.mean([m["total_reward"] for m in out])),
                    "mean_monotone_frac": float(np.mean([m["monotone_frac"] for m in out])),
                    "mean_violation_frac": float(np.mean([m["violation_frac"] for m in out])),
                }
            clf_report = {"n_reach_states": len(reach_idx),
                          "steered": traj_metrics("steered_traj"),
                          "oracle": traj_metrics("oracle_traj"),
                          "expert": traj_metrics("expert_traj")}

    steered_ee = np.array([p["ee_dist_steered_cm"] for p in per])
    stock_ee = np.array([p["ee_dist_stock_mean_cm"] for p in per])
    oracle_ee = np.array([p["ee_dist_oracle_cm"] for p in per])
    frac_beats = float(np.mean([p["steered_beats_stock_mse"] for p in per]))
    stock_m, steer_m, or_m = float(stock_ee.mean()), float(steered_ee.mean()), float(oracle_ee.mean())
    gap = stock_m - or_m
    improvement = (stock_m - steer_m) / gap if abs(gap) > 1e-9 else float("nan")

    cond1 = frac_beats >= args.frac_thresh          # >=70% beat stock MSE
    cond2 = improvement >= args.improve_thresh        # >=30% EE toward oracle
    passed = bool(cond1 and cond2)

    out = {
        "n_holdout_states": N,
        "steered_beats_stock_frac": frac_beats,
        "ee_dist_cm": {"stock_mean": stock_m, "steered_mean": steer_m, "oracle_mean": or_m},
        "improvement_toward_oracle": improvement,
        "cond1_frac_ge_thresh": bool(cond1), "cond2_ee_improve_ge_thresh": bool(cond2),
        "frac_thresh": args.frac_thresh, "improve_thresh": args.improve_thresh,
        "pass": passed,
        "clf_reach": clf_report,
        "per_state": per,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "per_state"}, indent=2), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    pd = sub.add_parser("decode")
    pd.add_argument("--model-path", default="/home/ubuntu/ckpt/ur5e_drugsort_obscond/full/checkpoint-12000")
    pd.add_argument("--embodiment-tag", default="new_embodiment")
    pd.add_argument("--dataset", default="/home/ubuntu/ur5e_drugsort_obscond")
    pd.add_argument("--steering-net", default="/home/ubuntu/steering_net_v0.npz")
    pd.add_argument("--device", default="cuda")
    pd.add_argument("--stride", type=int, default=8)
    pd.add_argument("--n", type=int, default=100)
    pd.add_argument("--n-seeds", type=int, default=8)
    pd.add_argument("--fp-iters", type=int, default=16)
    pd.add_argument("--seed", type=int, default=0)
    pd.add_argument("--decoded-out", default="/home/ubuntu/flowdagger_gate_decoded.npz")
    pd.add_argument("--val-episodes", default=None,
                    help="comma-separated episode ids to evaluate (overrides the net meta's val split)")
    pd.add_argument("--out", default=None)

    pf = sub.add_parser("fk-gate")
    pf.add_argument("--src", default="/home/ubuntu/odyssey-ur5e")
    pf.add_argument("--xml", default="/home/ubuntu/aseptipack_description/aseptipack.xml")
    pf.add_argument("--decoded-npz", default="/home/ubuntu/flowdagger_gate_decoded.npz")
    pf.add_argument("--frac-thresh", type=float, default=0.70)
    pf.add_argument("--improve-thresh", type=float, default=0.30)
    pf.add_argument("--out", default=None)

    args = ap.parse_args()
    if args.mode == "decode":
        mode_decode(args)
    elif args.mode == "fk-gate":
        mode_fk_gate(args)
    else:  # pragma: no cover
        raise SystemExit(f"unknown mode {args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
