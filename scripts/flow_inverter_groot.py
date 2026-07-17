#!/usr/bin/env python3
"""A2 — production flow-inverter for GR00T N1.7 (FlowDAgger noise targets).

Refactors the working inversion code from ``scripts/probe_flow_inversion_groot.py``
into a reusable, batched, resumable module + CLI. Given expert action chunks at
dataset states, recover the initial flow-sampler noise ``w*`` such that the stock
4-step forward-Euler sampler decodes ``w*`` back to the (normalised, padded) expert
chunk. Those ``w*`` are the regression targets for the steering net (A4).

Reuses aggressively from the probe:
  * ``GrootFlow``            — frozen policy + cached backbone + per-step velocity_fn
  * ``invert_perstep_fp``    — per-step fixed-point inverter of the Euler sampler
  * ``forward_euler_sample`` — the stock sampler (for the recon check)
  * ``load_dataset_frame`` / ``expert_chunk_from_dataset`` — LeRobot frame readers

New in this module:
  * BATCHED inversion — collate ``mini_batch`` (>=8) different chunks into ONE
    backbone forward + ONE batched DiT inversion (exact: the DiT is per-sample, no
    cross-sample attention). Big throughput win over the probe's B=1 loop.
  * fp_per_step = 16 default (the setting that hit round-trip MSE 9.1e-5).
  * per-chunk recon-MSE QUALITY FILTER (drop > ``quality_thresh`` = 1e-3, log rate).
  * processor-exact target construction (same collate/normalise path as inference).
  * RESUMABLE over a dataset: writes ``steering_targets/shard_XXXX.npz`` + a
    ``manifest.json``; a rerun skips shards already present in the manifest.
  * bf16-DECODE CHECK: decode a sample of ``w*`` through the deploy-precision bf16
    action-head path and report real-dim MSE vs the fp32 decode — bounds the
    fidelity latent steering can achieve at deploy precision. Flagged if > 1e-3.

Runs in the GR00T venv on the VM (needs torch + gr00t). Import-light at module
scope so the pure helpers stay importable in a torch-less venv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Reuse the probe's verified code. Add the probe's directory to sys.path so this
# module works whether imported or run as a script from anywhere.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from probe_flow_inversion_groot import (  # noqa: E402
    GrootFlow,
    REAL_STEPS,
    REAL_DIMS,
    forward_euler_sample,
    invert_perstep_fp,
    load_dataset_frame,
    expert_chunk_from_dataset,
    DEFAULT_INSTRUCTION,
)

# ---------------------------------------------------------------------------
# Phase labels: the browser GT sidecar records a per-frame scripted-expert phase
# name (10 phases). Bucket them into 4 grasp-mechanics phases for the steering
# net's phase-onehot input. Order per episode:
#   home -> approach -> descend -> close -> lift -> transport -> lower ->
#   release -> retract -> home2  (+ 'finished')
# ---------------------------------------------------------------------------
PHASE_TO_BUCKET: dict[str, int] = {
    # 0 = REACH (move toward vial, gripper open)
    "home": 0, "approach": 0, "descend": 0,
    # 1 = GRASP (close on vial + lift off)
    "close": 1, "lift": 1,
    # 2 = TRANSPORT (carry to rack, lower into pocket)
    "transport": 2, "lower": 2,
    # 3 = PLACE/RETURN (release + retract + home)
    "release": 3, "retract": 3, "home2": 3, "finished": 3,
}
N_PHASES = 4


def phase_bucket(name: str) -> int:
    return PHASE_TO_BUCKET.get(str(name), 0)


def load_gt_phases(dataset_dir: Path, episode: int, _cache: dict) -> list[str] | None:
    """Per-frame scripted-phase names from the GT sidecar (meta/gt/), or None."""
    key = ("gt", episode)
    if key in _cache:
        return _cache[key]
    p = dataset_dir / "meta" / "gt" / f"episode_{episode:06d}.json"
    phases: list[str] | None = None
    if p.exists():
        d = json.loads(p.read_text())
        phases = [f.get("phase", "") for f in d.get("frames", [])]
    _cache[key] = phases
    return phases


def heuristic_phase(state10: np.ndarray, arm_grip_chunk: np.ndarray) -> int:
    """Fallback 4-bucket phase from grip state + grasp-target proximity, used only
    when no GT sidecar is present. state10 = [proprio6, grip, gtx, gty, gtz].
    (Documented fallback; the GT sidecar is the primary source.)"""
    grip = float(state10[6])
    # crude proxy: gripper closed -> grasp/transport; open + near target -> reach
    if grip > 0.5:
        return 2  # holding: transport-ish
    return 0      # open: reaching


# ---------------------------------------------------------------------------
# FlowInverter — batched, cached, quality-filtered inversion
# ---------------------------------------------------------------------------
class FlowInverter:
    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "new_embodiment",
        device: str = "cuda",
        fp_iters: int = 16,
        quality_thresh: float = 1e-3,
    ) -> None:
        self.flow = GrootFlow(model_path, embodiment_tag, device)
        self.torch = self.flow.torch
        self.fp_iters = int(fp_iters)
        self.quality_thresh = float(quality_thresh)
        self.n_steps = self.flow.n_steps
        self.dt = self.flow.dt
        self.action_horizon = self.flow.action_horizon
        self.action_dim = self.flow.action_dim

    # -- batched encode: ONE backbone forward for B different chunks -----------
    def encode_batch(self, frames: list[dict]) -> tuple[dict, Any]:
        """frames: list of dicts {ext, wrist, state10, arm(16,6), grip(16,1)}.

        Returns (cache, x_target) with batch dim B = len(frames). Padding across
        samples is handled by the processor/collate (consistent seq_len + masks),
        so the batched DiT is numerically identical to per-sample B=1 inversion.
        """
        torch = self.torch
        flow = self.flow
        procs = []
        for fr in frames:
            s = np.asarray(fr["state10"], dtype=np.float32).reshape(-1)
            states = {
                "single_arm": s[:6].reshape(1, 6).astype(np.float32),
                "gripper": s[6:7].reshape(1, 1).astype(np.float32),
                "grasp_target": s[7:10].reshape(1, 3).astype(np.float32),
            }
            actions = {
                "single_arm": np.asarray(fr["arm"], dtype=np.float32).reshape(-1, 6),
                "gripper": np.asarray(fr["grip"], dtype=np.float32).reshape(-1, 1),
            }
            vsd = flow._VLAStepData(
                images={
                    "exterior": [np.asarray(fr["ext"], dtype=np.uint8)],
                    "wrist": [np.asarray(fr["wrist"], dtype=np.uint8)],
                },
                states=states, actions=actions,
                text=fr.get("instruction", DEFAULT_INSTRUCTION),
                embodiment=flow.embodiment_tag,
            )
            procs.append(flow.policy.processor(
                [{"type": flow._MessageType.EPISODE_STEP.value, "content": vsd}]
            ))
        collated = flow.policy.collate_fn(procs)["inputs"]
        with torch.no_grad():
            backbone_inputs, action_inputs = flow.model.prepare_input(collated)
            backbone_output = flow.model.backbone(backbone_inputs)
            backbone_output["backbone_features"] = backbone_output["backbone_features"].float()
            action_inputs["state"] = action_inputs["state"].float()
            feats = flow.ah._encode_features(backbone_output, action_inputs)
        cache = {
            "vl_embeds": feats.backbone_features,
            "state_features": feats.state_features,
            "embodiment_id": action_inputs.embodiment_id,
            "image_mask": getattr(backbone_output, "image_mask", None),
            "backbone_attention_mask": getattr(backbone_output, "backbone_attention_mask", None),
            "batch_size": feats.backbone_features.shape[0],
        }
        x_target = action_inputs.action.float()   # (B, H, D)
        return cache, x_target

    # -- batched inversion of the whole mini-batch -----------------------------
    def invert_batch(self, cache: dict, x_target: Any) -> tuple[Any, np.ndarray]:
        torch = self.torch
        flow = self.flow
        vfn = flow.velocity_fn(cache)
        with torch.no_grad():
            w, _res = invert_perstep_fp(x_target, vfn, self.n_steps, self.dt, self.fp_iters)
            x_hat = forward_euler_sample(w, vfn, self.n_steps, self.dt)
            d = (flow.real_slice(x_hat) - flow.real_slice(x_target))  # (B,16,7)
            mse = (d ** 2).mean(dim=(1, 2)).detach().float().cpu().numpy()  # (B,)
        return w, mse

    # -- bf16-decode check: deploy-precision fidelity bound --------------------
    def bf16_decode_check(self, frames: list[dict], w_stars: Any, x_targets: Any) -> dict:
        """Decode ``w_stars`` through the bf16 action-head path (deploy precision)
        and report real-dim MSE vs (i) the fp32 decode and (ii) the expert target.

        We cast the (fp32-inverted) action head + its cached conditioning to
        bfloat16 for the decode, exactly as deploy inference runs. Restored to
        fp32 afterwards so subsequent inversion is unaffected.
        """
        torch = self.torch
        flow = self.flow
        # fp32 decode from the cache re-encoded fresh (isolate this batch's cache).
        cache32, xt32 = self.encode_batch(frames)
        vfn32 = flow.velocity_fn(cache32)
        with torch.no_grad():
            x_fp32 = forward_euler_sample(w_stars, vfn32, self.n_steps, self.dt)
            r_fp32 = flow.real_slice(x_fp32)
            r_tgt = flow.real_slice(xt32)

        # bf16 decode: cast action head + cache + noise to bf16.
        ah = flow.ah
        try:
            ah.to(torch.bfloat16)
            cache_bf16 = dict(cache32)
            for k in ("vl_embeds", "state_features"):
                cache_bf16[k] = cache32[k].to(torch.bfloat16)
            vfn_bf16 = flow.velocity_fn(cache_bf16)
            with torch.no_grad():
                x_bf16 = forward_euler_sample(
                    w_stars.to(torch.bfloat16), vfn_bf16, self.n_steps, self.dt
                )
                r_bf16 = flow.real_slice(x_bf16).float()
        finally:
            ah.float()  # restore fp32 action head for further inversion

        mse_vs_fp32 = float(((r_bf16 - r_fp32) ** 2).mean())
        mse_vs_target = float(((r_bf16 - r_tgt) ** 2).mean())
        mse_fp32_vs_target = float(((r_fp32 - r_tgt) ** 2).mean())
        return {
            "bf16_decode_mse_vs_fp32": mse_vs_fp32,
            "bf16_decode_mse_vs_target": mse_vs_target,
            "fp32_decode_mse_vs_target": mse_fp32_vs_target,
            "n_sampled": int(w_stars.shape[0]),
        }


# ---------------------------------------------------------------------------
# Dataset enumeration
# ---------------------------------------------------------------------------
def enumerate_chunks(dataset_dir: Path, stride: int) -> list[tuple[int, int]]:
    import pyarrow.parquet as pq

    data_dir = dataset_dir / "data" / "chunk-000"
    eps = sorted(int(p.stem.split("_")[-1]) for p in data_dir.glob("episode_*.parquet"))
    picks: list[tuple[int, int]] = []
    for ep in eps:
        n = pq.read_metadata(data_dir / f"episode_{ep:06d}.parquet").num_rows
        last = n - REAL_STEPS - 1
        for fr in range(0, max(1, last), stride):
            picks.append((ep, fr))
    return picks


def _evict_video_readers(readers: dict, keep_eps: set[int]) -> None:
    """Close+drop cached ffmpeg video readers for episodes not in ``keep_eps``.

    ``load_dataset_frame`` caches one imageio ffmpeg reader per (view, episode);
    across 151 episodes that exhausts the process file-descriptor limit (each
    reader holds a live ffmpeg subprocess + pipes). Since we scan episode-ordered,
    we only ever need the current batch's episodes open. Parquet tables ('tbl')
    are cheap and left cached.
    """
    for key in [k for k in readers if k[0] == "vid" and k[2] not in keep_eps]:
        rdr = readers.pop(key)
        try:
            rdr.close()
        except Exception:
            pass


def pool_backbone(cache: dict) -> np.ndarray:
    """Pooled visual conditioning per sample: masked mean of vl_embeds -> (B,D).

    THE canonical pooling for the image-conditioned steering head (v4): the
    encoder pass, the offline gate, and the serving path must all use this
    exact function — the inverted noise targets depend on these features, and
    any pooling mismatch between train and deploy re-creates the target-noise
    floor the head exists to break.
    """
    v = cache["vl_embeds"]                    # (B, S, D) torch fp32
    m = cache.get("backbone_attention_mask")
    if m is not None:
        mm = m.float().unsqueeze(-1)          # (B, S, 1)
        pooled = (v * mm).sum(dim=1) / mm.sum(dim=1).clamp(min=1.0)
    else:
        pooled = v.mean(dim=1)
    return pooled.detach().float().cpu().numpy()


def mode_encode_features(args) -> dict:
    """Re-encoding pass: attach pooled backbone features to EXISTING noise-target
    shards, row-aligned (same episode/frame order), resumable per shard."""
    shards_dir = Path(args.shards)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset)
    inv = FlowInverter(args.model_path, args.embodiment_tag, args.device,
                       fp_iters=1)
    readers: dict = {}
    n_rows = 0
    feat_dim = None
    shard_files = sorted(shards_dir.glob("shard_*.npz"))
    for sf in shard_files:
        outp = out_dir / f"feat_{sf.name}"
        if outp.exists():
            print(f"[feat] {outp.name} exists — skip", flush=True)
            continue
        z = np.load(sf)
        eps = np.asarray(z["episode"], dtype=np.int64)
        frs = np.asarray(z["frame_idx"], dtype=np.int64)
        feats = []
        for i in range(0, len(eps), args.mini_batch):
            batch = [(int(e), int(f)) for e, f in
                     zip(eps[i:i + args.mini_batch], frs[i:i + args.mini_batch])]
            frames = read_frames(dataset_dir, batch, readers)
            cache, _ = inv.encode_batch(frames)
            feats.append(pool_backbone(cache))
        F = np.concatenate(feats).astype(np.float16)
        feat_dim = F.shape[1]
        np.savez(outp, episode=eps, frame_idx=frs, feats=F)
        n_rows += len(eps)
        print(f"[feat] {outp.name}: {len(eps)} rows dim={F.shape[1]}", flush=True)
    out = {"n_rows": int(n_rows), "feat_dim": (int(feat_dim) if feat_dim else None),
           "n_shards": len(shard_files), "out_dir": str(out_dir)}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    print("[feat] DONE", json.dumps(out), flush=True)
    return out


def read_frames(dataset_dir: Path, batch: list[tuple[int, int]], readers: dict) -> list[dict]:
    keep_eps = {ep for ep, _ in batch}
    _evict_video_readers(readers, keep_eps)
    out = []
    for ep, fr in batch:
        ext, wrist, state10, _ = load_dataset_frame(dataset_dir, ep, fr, readers)
        arm, grip = expert_chunk_from_dataset(dataset_dir, ep, fr, readers)
        out.append({"ext": ext, "wrist": wrist, "state10": state10.astype(np.float32),
                    "arm": arm, "grip": grip, "episode": ep, "frame": fr})
    return out


# ---------------------------------------------------------------------------
# CLI: invert-dataset  (the A3 engine — resumable, sharded)
# ---------------------------------------------------------------------------
def mode_invert_dataset(args) -> dict:
    dataset_dir = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "shards": {}, "params": {}, "phase_map": PHASE_TO_BUCKET,
    }

    inv = FlowInverter(args.model_path, args.embodiment_tag, args.device,
                       fp_iters=args.fp_iters, quality_thresh=args.quality_thresh)

    all_picks = enumerate_chunks(dataset_dir, args.stride)
    n_stride8 = len(all_picks)
    print(f"[invert] stride={args.stride} -> {n_stride8} candidate chunks", flush=True)

    # Cost estimate from a warmup batch; widen stride if projected > budget.
    readers: dict = {}
    gt_cache: dict = {}
    mb = args.mini_batch
    warm = read_frames(dataset_dir, all_picks[:mb], readers)
    t0 = time.time()
    c, xt = inv.encode_batch(warm)
    w, mse = inv.invert_batch(c, xt)
    t_warm = time.time() - t0
    t_per_chunk = t_warm / max(1, len(warm))
    proj_h = (t_per_chunk * n_stride8) / 3600.0
    print(f"[invert] warmup {len(warm)} chunks in {t_warm:.1f}s -> {t_per_chunk:.3f}s/chunk; "
          f"projected {proj_h:.2f} GPU-h at stride {args.stride}", flush=True)

    stride = args.stride
    if proj_h > args.max_gpu_hours:
        factor = proj_h / (args.max_gpu_hours * 0.95)
        stride = int(np.ceil(args.stride * factor))
        all_picks = enumerate_chunks(dataset_dir, stride)
        proj_h2 = (t_per_chunk * len(all_picks)) / 3600.0
        print(f"[invert] projected {proj_h:.2f}h > budget {args.max_gpu_hours}h; "
              f"widening stride {args.stride}->{stride} -> {len(all_picks)} chunks "
              f"(~{proj_h2:.2f}h)", flush=True)

    manifest["params"] = {
        "stride": stride, "fp_iters": args.fp_iters, "mini_batch": mb,
        "quality_thresh": args.quality_thresh, "t_per_chunk_s": t_per_chunk,
        "model_path": args.model_path,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Group picks into shards of shard_size, each shard = several mini-batches.
    shard_size = args.shard_size
    shards = [all_picks[i:i + shard_size] for i in range(0, len(all_picks), shard_size)]
    n_total = kept = dropped = 0
    bf16_sample_frames: list[dict] = []
    bf16_sample_w: list = []
    t_start = time.time()
    for si, shard_picks in enumerate(shards):
        shard_name = f"shard_{si:04d}"
        if shard_name in manifest["shards"] and (out_dir / f"{shard_name}.npz").exists():
            m = manifest["shards"][shard_name]
            n_total += m.get("n_in", 0); kept += m.get("n_kept", 0); dropped += m.get("n_dropped", 0)
            print(f"[invert] {shard_name} already done ({m.get('n_kept')} kept) — skip", flush=True)
            continue
        keys = ["proprio7", "grasp_target3", "phase_label", "w_star_real",
                "recon_mse", "episode", "frame_idx"]
        if args.save_full_w:
            keys.append("w_star_full")
        rows: dict[str, list] = {k: [] for k in keys}
        for j in range(0, len(shard_picks), mb):
            batch_picks = shard_picks[j:j + mb]
            frames = read_frames(dataset_dir, batch_picks, readers)
            cache, x_target = inv.encode_batch(frames)
            w, mse = inv.invert_batch(cache, x_target)
            w_real = inv.flow.real_slice(w).detach().float().cpu().numpy()  # (B,16,7)
            w_full = w.detach().float().cpu().numpy() if args.save_full_w else None  # (B,40,132)
            for bi, fr in enumerate(frames):
                n_total += 1
                if not np.isfinite(mse[bi]) or mse[bi] > inv.quality_thresh:
                    dropped += 1
                    continue
                kept += 1
                s10 = fr["state10"]
                phases = load_gt_phases(dataset_dir, fr["episode"], gt_cache)
                if phases is not None and fr["frame"] < len(phases):
                    plabel = phase_bucket(phases[fr["frame"]])
                else:
                    plabel = heuristic_phase(s10, w_real[bi])
                rows["proprio7"].append(s10[:7].astype(np.float32))
                rows["grasp_target3"].append(s10[7:10].astype(np.float32))
                rows["phase_label"].append(np.int64(plabel))
                rows["w_star_real"].append(w_real[bi].astype(np.float32))
                rows["recon_mse"].append(np.float32(mse[bi]))
                rows["episode"].append(np.int64(fr["episode"]))
                rows["frame_idx"].append(np.int64(fr["frame"]))
                if args.save_full_w:
                    rows["w_star_full"].append(w_full[bi].astype(np.float32))
                # collect bf16-check sample (first few kept chunks)
                if len(bf16_sample_frames) < args.bf16_n:
                    bf16_sample_frames.append(fr)
                    bf16_sample_w.append(w[bi:bi + 1].detach())
        npz_path = out_dir / f"{shard_name}.npz"
        save_kw = dict(
            proprio7=np.asarray(rows["proprio7"], dtype=np.float32),
            grasp_target3=np.asarray(rows["grasp_target3"], dtype=np.float32),
            phase_label=np.asarray(rows["phase_label"], dtype=np.int64),
            w_star_real=np.asarray(rows["w_star_real"], dtype=np.float32),
            recon_mse=np.asarray(rows["recon_mse"], dtype=np.float32),
            episode=np.asarray(rows["episode"], dtype=np.int64),
            frame_idx=np.asarray(rows["frame_idx"], dtype=np.int64),
        )
        if args.save_full_w:
            save_kw["w_star_full"] = np.asarray(rows["w_star_full"], dtype=np.float32)
        np.savez(npz_path, **save_kw)
        manifest["shards"][shard_name] = {
            "n_in": len(shard_picks), "n_kept": len(rows["proprio7"]),
            "n_dropped": len(shard_picks) - len(rows["proprio7"]),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        elapsed = time.time() - t_start
        done = si + 1
        eta = (elapsed / done) * (len(shards) - done)
        print(f"[invert] {shard_name} {done}/{len(shards)} kept={len(rows['proprio7'])} "
              f"drop={len(shard_picks) - len(rows['proprio7'])} elapsed={elapsed/60:.1f}m "
              f"eta={eta/60:.1f}m", flush=True)

    # bf16-decode check on the collected sample
    bf16 = {}
    if bf16_sample_w:
        w_cat = inv.torch.cat(bf16_sample_w, dim=0)
        x_t = None  # recomputed inside
        bf16 = inv.bf16_decode_check(bf16_sample_frames, w_cat, x_t)
        bf16["bf16_ok"] = bool(bf16["bf16_decode_mse_vs_fp32"] <= inv.quality_thresh)
        if not bf16["bf16_ok"]:
            print(f"[invert] !!! bf16 DECODE DEGRADES recon: mse_vs_fp32="
                  f"{bf16['bf16_decode_mse_vs_fp32']:.3e} > {inv.quality_thresh:.0e} — "
                  f"bounds deployable fidelity", flush=True)

    gpu_hours = (time.time() - t_start) / 3600.0
    result = {
        "n_chunks_in": n_total, "n_kept": kept, "n_dropped": dropped,
        "drop_rate": (dropped / n_total) if n_total else 0.0,
        "stride": stride, "fp_iters": args.fp_iters, "mini_batch": mb,
        "t_per_chunk_s": t_per_chunk, "gpu_hours": gpu_hours,
        "out_dir": str(out_dir), "bf16_check": bf16,
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("invert-dataset")
    p.add_argument("--model-path", default="/home/ubuntu/ckpt/ur5e_drugsort_obscond/full/checkpoint-12000")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset", default="/home/ubuntu/ur5e_drugsort_obscond")
    p.add_argument("--out-dir", default="/home/ubuntu/steering_targets")
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--fp-iters", type=int, default=16)
    p.add_argument("--mini-batch", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=256)
    p.add_argument("--quality-thresh", type=float, default=1e-3)
    p.add_argument("--max-gpu-hours", type=float, default=4.0)
    p.add_argument("--bf16-n", type=int, default=32)
    p.add_argument("--save-full-w", action="store_true",
                   help="also store the full 40x132 w* (needed only if pad-check -> full5280)")
    p.add_argument("--out", default=None)

    pf = sub.add_parser("encode-features",
                        help="attach pooled backbone features to existing shards (v4 head)")
    pf.add_argument("--model-path", default="/home/ubuntu/ckpt/ur5e_drugsort_obscond/full/checkpoint-12000")
    pf.add_argument("--embodiment-tag", default="new_embodiment")
    pf.add_argument("--device", default="cuda")
    pf.add_argument("--dataset", default="/home/ubuntu/ur5e_drugsort_obscond")
    pf.add_argument("--shards", required=True, help="dir of shard_*.npz to re-encode")
    pf.add_argument("--out-dir", required=True, help="dir for feat_shard_*.npz")
    pf.add_argument("--mini-batch", type=int, default=16)
    pf.add_argument("--out", default=None)

    args = ap.parse_args()
    if args.mode == "invert-dataset":
        out = mode_invert_dataset(args)
    elif args.mode == "encode-features":
        out = mode_encode_features(args)
    else:  # pragma: no cover
        raise SystemExit(f"unknown mode {args.mode}")
    text = json.dumps(out, indent=2)
    if getattr(args, "out", None):
        Path(args.out).write_text(text)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
