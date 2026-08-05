#!/usr/bin/env python3
"""flow_inverter_pi05.py — π0.5 FlowDAgger noise-target inverter (CLI).

Mirror of ``flow_inverter_groot.py`` for the π0.5 pilot. Given expert action chunks at
dataset states, recover the initial flow-sampler noise ``w*`` such that π0.5's denoiser
decodes ``w*`` back to the (normalized, padded) expert chunk. Those ``w*`` are the regression
targets for the SAME numpy steering net (``train_steering_ur5e.py``) used by the GR00T path.

The heavy lifting (openpi model load, transforms, flow inversion) is in ``pi05_flow.Pi05Flow``;
this CLI reuses the pilot-agnostic dataset readers from ``probe_flow_inversion_groot`` and the
phase helpers from ``flow_inverter_groot``. Runs in openpi's venv (jax) on the GPU box.

Subcommands
-----------
  roundtrip        : invert a few chunks, report real-dim recon MSE (gate: < 1e-3).
  invert-dataset   : invert a full dataset → sharded ``shard_*.npz`` + ``manifest.json``
                     in the schema ``train_steering_ur5e`` consumes.
  encode-features  : pooled π0.5 prefix features per shard → ``feat_shard_*.npz`` (v4 head).

Output shard schema (per-chunk rows), identical keys to the GR00T inverter (only the shapes
differ — H=10, action_dim=32 for π0.5):
  proprio7 (N,7) · grasp_target3 (N,3, zeros for π0.5) · phase_label (N,) ·
  w_star_real (N,H,7) · w_star_full (N,H,32) [--save-full-w] · recon_mse (N,) ·
  episode (N,) · frame_idx (N,)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Pilot-agnostic readers + phase helpers (numpy / pyarrow only; safe to import w/o jax).
from probe_flow_inversion_groot import load_dataset_frame, pick_chunk_frames  # noqa: E402
from flow_inverter_groot import (  # noqa: E402
    enumerate_chunks,
    load_gt_phases,
    phase_bucket,
    heuristic_phase,
    PHASE_TO_BUCKET,
)

DEFAULT_INSTRUCTION = "pick up the vial and place it in the rack"


def expert_chunk_pi05(dataset_dir: Path, episode: int, frame: int, horizon: int,
                      readers: dict) -> np.ndarray:
    """The expert action chunk = next ``horizon`` recorded actions (absolute 7-DoF targets).

    (H, 7) = [6 arm joints, 1 gripper] — exactly what the π0.5 fine-tune supervised here.
    Parametrized by ``horizon`` because π0.5's action_horizon (10) differs from GR00T's (16).
    """
    import pyarrow.parquet as pq

    key = ("tbl", episode)
    if key not in readers:
        readers[key] = pq.read_table(
            dataset_dir / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
        )
    tbl = readers[key]
    acts = np.asarray(
        [tbl.column("action")[frame + k].as_py() for k in range(horizon)],
        dtype=np.float32,
    )
    return acts[:, :7]


def _phase_for(dataset_dir: Path, episode: int, frame: int, state: np.ndarray,
               arm_grip: np.ndarray, gt_cache: dict) -> int:
    phases = load_gt_phases(dataset_dir, episode, gt_cache)
    if phases is not None and frame < len(phases):
        return phase_bucket(phases[frame])
    # Fallback: heuristic needs a 10-D state proxy; pad/truncate the raw state.
    s10 = np.zeros(10, np.float32)
    n = min(10, len(state))
    s10[:n] = np.asarray(state, np.float32)[:n]
    return heuristic_phase(s10, arm_grip)


def _build_flow(args: argparse.Namespace) -> Any:
    from pi05_flow import Pi05Flow

    return Pi05Flow(
        args.config_name,
        args.checkpoint_dir,
        num_denoise_steps=args.denoise_steps,
        method=args.method,
        fp_per_step=args.fp_per_step,
        default_prompt=args.instruction,
    )


# ---------------------------------------------------------------------------
# roundtrip — validation gate before generating shards
# ---------------------------------------------------------------------------
def mode_roundtrip(args: argparse.Namespace) -> dict:
    dataset_dir = Path(args.dataset)
    flow = _build_flow(args)
    H = flow.action_horizon
    picks = pick_chunk_frames(dataset_dir, args.n_chunks, args.seed)
    print(f"[roundtrip] pi05 H={H} D={flow.action_dim} real={flow.real_dims}; "
          f"{len(picks)} chunks", flush=True)
    readers: dict = {}
    per = []
    for j, (ep, fr) in enumerate(picks):
        ext, wrist, state, _ = load_dataset_frame(dataset_dir, ep, fr, readers)
        chunk = expert_chunk_pi05(dataset_dir, ep, fr, H, readers)  # (H,7)
        obs, target = flow.process_frame(
            exterior=ext, wrist=wrist, state=state,
            prompt=args.instruction, action_chunk=chunk,
        )
        assert target is not None, "no action target produced"
        assert tuple(target.shape[1:]) == (H, flow.action_dim), \
            f"unexpected target shape {tuple(target.shape)} != (*,{H},{flow.action_dim})"
        w_star, full_mse = flow.invert(obs, target)               # (H,D)
        recon = flow.decode_internal(obs, w_star)                 # (H,D)
        tgt = np.asarray(target[0])
        real_mse = float(np.mean((recon[:, :flow.real_dims] - tgt[:, :flow.real_dims]) ** 2))
        per.append({"episode": ep, "frame": fr, "real_mse": real_mse,
                    "full_mse": full_mse, "w_abs_p99": float(np.percentile(np.abs(w_star), 99))})
        print(f"  [{j+1}/{len(picks)}] ep{ep} fr{fr}: real_mse={real_mse:.2e} "
              f"full_mse={full_mse:.2e}", flush=True)
    real = np.array([p["real_mse"] for p in per])
    out = {
        "n_chunks": len(per),
        "real_recon_mse": {"mean": float(real.mean()), "p95": float(np.percentile(real, 95)),
                           "worst": float(real.max())},
        "w_abs_p99": float(np.mean([p["w_abs_p99"] for p in per])),
        "method": args.method, "denoise_steps": args.denoise_steps,
        "fp_per_step": args.fp_per_step,
        "pass": bool(real.mean() < args.thresh),
        "thresh": args.thresh,
        "per_chunk": per,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[roundtrip] mean real_mse={real.mean():.3e} pass={out['pass']}", flush=True)
    return out


# ---------------------------------------------------------------------------
# invert-dataset — produce steering-target shards
# ---------------------------------------------------------------------------
def mode_invert_dataset(args: argparse.Namespace) -> dict:
    dataset_dir = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = out_dir / "manifest.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {"shards": {}}

    flow = _build_flow(args)
    H, D, RD = flow.action_horizon, flow.action_dim, flow.real_dims
    chunks = enumerate_chunks(dataset_dir, args.stride)
    print(f"[invert] {len(chunks)} chunks (stride {args.stride}) H={H} D={D}", flush=True)

    readers: dict = {}
    gt_cache: dict = {}
    shard_rows: dict[str, list] = {k: [] for k in
                                   ("proprio7", "grasp_target3", "phase_label",
                                    "w_star_real", "w_star_full", "recon_mse",
                                    "episode", "frame_idx")}
    n_in = n_kept = n_drop = 0
    si = len(manifest["shards"])
    t0 = time.time()

    def flush_shard(idx: int) -> None:
        if not shard_rows["episode"]:
            return
        arr = {}
        for k, v in shard_rows.items():
            if k == "w_star_full" and not args.save_full_w:
                continue
            arr[k] = np.asarray(v)
        np.savez_compressed(out_dir / f"shard_{idx:04d}.npz", **arr)
        manifest["shards"][f"shard_{idx:04d}"] = {"n": len(shard_rows["episode"])}
        manifest_p.write_text(json.dumps(manifest, indent=2))
        for k in shard_rows:
            shard_rows[k] = []

    for (ep, fr) in chunks:
        if (time.time() - t0) / 3600.0 > args.max_gpu_hours:
            print("[invert] max-gpu-hours reached; stopping", flush=True)
            break
        n_in += 1
        ext, wrist, state, _ = load_dataset_frame(dataset_dir, ep, fr, readers)
        chunk = expert_chunk_pi05(dataset_dir, ep, fr, H, readers)
        obs, target = flow.process_frame(exterior=ext, wrist=wrist, state=state,
                                         prompt=args.instruction, action_chunk=chunk)
        w_star, _ = flow.invert(obs, target)
        recon = flow.decode_internal(obs, w_star)
        tgt = np.asarray(target[0])
        real_mse = float(np.mean((recon[:, :RD] - tgt[:, :RD]) ** 2))
        if real_mse > args.quality_thresh:
            n_drop += 1
            continue
        n_kept += 1
        s7 = np.zeros(7, np.float32)
        s7[: min(7, len(state))] = np.asarray(state, np.float32)[:7]
        shard_rows["proprio7"].append(s7)
        shard_rows["grasp_target3"].append(np.zeros(3, np.float32))  # π0.5 drops grasp_target
        shard_rows["phase_label"].append(_phase_for(dataset_dir, ep, fr, state,
                                                     chunk[:, :6], gt_cache))
        shard_rows["w_star_real"].append(w_star[:, :RD].astype(np.float32))
        shard_rows["w_star_full"].append(w_star.astype(np.float32))
        shard_rows["recon_mse"].append(real_mse)
        shard_rows["episode"].append(ep)
        shard_rows["frame_idx"].append(fr)
        if len(shard_rows["episode"]) >= args.shard_size:
            flush_shard(si)
            si += 1
        if n_in % 200 == 0:
            print(f"  in={n_in} kept={n_kept} drop={n_drop} "
                  f"({(time.time()-t0)/3600:.2f}h)", flush=True)
    flush_shard(si)

    manifest["params"] = {"stride": args.stride, "H": H, "action_dim": D, "real_dims": RD,
                          "quality_thresh": args.quality_thresh, "method": args.method,
                          "checkpoint_dir": str(args.checkpoint_dir)}
    manifest["phase_map"] = PHASE_TO_BUCKET
    manifest_p.write_text(json.dumps(manifest, indent=2))
    out = {"n_chunks_in": n_in, "n_kept": n_kept, "n_dropped": n_drop,
           "drop_rate": (n_drop / n_in if n_in else 0.0),
           "gpu_hours": (time.time() - t0) / 3600.0, "out_dir": str(out_dir)}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[invert] kept={n_kept}/{n_in} drop_rate={out['drop_rate']:.4f}", flush=True)
    return out


# ---------------------------------------------------------------------------
# encode-features — pooled π0.5 prefix features for the v4 image-conditioned head
# ---------------------------------------------------------------------------
def mode_encode_features(args: argparse.Namespace) -> dict:
    shards_dir = Path(args.shards)
    dataset_dir = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flow = _build_flow(args)
    readers: dict = {}
    shard_files = sorted(shards_dir.glob("shard_*.npz"))
    print(f"[feats] {len(shard_files)} shards", flush=True)
    total = 0
    for sf in shard_files:
        d = np.load(sf)
        eps, frs = d["episode"], d["frame_idx"]
        feats = []
        for ep, fr in zip(eps.tolist(), frs.tolist()):
            ext, wrist, state, _ = load_dataset_frame(dataset_dir, ep, fr, readers)
            obs, _ = flow.process_frame(exterior=ext, wrist=wrist, state=state,
                                        prompt=args.instruction, action_chunk=None)
            feats.append(flow.pool_backbone(obs).astype(np.float16))
        np.savez_compressed(out_dir / f"feat_{sf.stem}.npz",
                            episode=eps, frame_idx=frs, feats=np.asarray(feats))
        total += len(feats)
        print(f"  {sf.stem}: {len(feats)} feats (dim {feats[0].shape[0]})", flush=True)
    out = {"n_feats": total, "out_dir": str(out_dir)}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return out


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config-name", dest="config_name", default="pi05_ur10e_drugsort")
    p.add_argument("--checkpoint-dir", dest="checkpoint_dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--denoise-steps", dest="denoise_steps", type=int, default=10)
    p.add_argument("--method", default="perstep_fp")
    p.add_argument("--fp-per-step", dest="fp_per_step", type=int, default=5)
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--out", default=None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rt = sub.add_parser("roundtrip")
    _add_common(rt)
    rt.add_argument("--n-chunks", dest="n_chunks", type=int, default=8)
    rt.add_argument("--seed", type=int, default=0)
    rt.add_argument("--thresh", type=float, default=1e-3)
    rt.set_defaults(func=mode_roundtrip)

    iv = sub.add_parser("invert-dataset")
    _add_common(iv)
    iv.add_argument("--out-dir", dest="out_dir", required=True)
    iv.add_argument("--stride", type=int, default=8)
    iv.add_argument("--shard-size", dest="shard_size", type=int, default=256)
    iv.add_argument("--quality-thresh", dest="quality_thresh", type=float, default=1e-3)
    iv.add_argument("--max-gpu-hours", dest="max_gpu_hours", type=float, default=4.0)
    iv.add_argument("--save-full-w", dest="save_full_w", action="store_true")
    iv.set_defaults(func=mode_invert_dataset)

    ef = sub.add_parser("encode-features")
    _add_common(ef)
    ef.add_argument("--shards", required=True)
    ef.add_argument("--out-dir", dest="out_dir", required=True)
    ef.set_defaults(func=mode_encode_features)

    args = ap.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
