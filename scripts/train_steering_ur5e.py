#!/usr/bin/env python3
"""A4 — steering-net v0 + trainer for the FlowDAgger UR5e drug-sort port.

A small MLP that maps a low-dim observation to the flow-sampler's initial noise
``w*`` (the A3 inversion targets), so at deploy we can *steer* the frozen GR00T
pilot by choosing its latent noise instead of sampling ``torch.randn``.

  input  = proprio7 (6 arm + grip) + grasp_target3 (xyz) + phase_onehot4 = 14
  hidden = 256 -> 256 (ReLU)
  output = real dims (16x7 = 112) OR full padded (40x132 = 5280)  [linear head, no tanh]
  loss   = MSE to w_star ; Adam 3e-4 ; batch 256 ; up to 200 epochs
  early stop on a 10% EPISODE-level held-out split (split by episode, not frame)
  logs p99|prediction| per epoch (a manifold sanity check — steered noise should
  look Gaussian, |w*| p99 ~ 3)

OUTPUT DESIGN is measured, not guessed, by the ``pad-check`` mode: decode a
sample of full ``w*`` with (i) its true pad noise and (ii) re-sampled Gaussian
pads; if the real-dim ACTION MSE difference < 1e-4 the pads don't matter -> the
net outputs real dims only (112) and the pads are filled at deploy with a FIXED
seeded Gaussian (``PAD_SEED``); otherwise it outputs the full 5280 tensor.

The pure data-pipeline helpers (``load_shards``, ``build_xy``, ``episode_split``)
are numpy-only and importable in a torch-less venv (unit-tested in
tests/unit/test_steering_targets.py). torch is imported lazily in train/pad-check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

N_PHASES = 4
REAL_STEPS = 16
REAL_DIMS = 7
REAL_OUT = REAL_STEPS * REAL_DIMS          # 112
ACTION_HORIZON = 40
ACTION_DIM = 132
FULL_OUT = ACTION_HORIZON * ACTION_DIM     # 5280
PAD_SEED = 20260716                         # fixed seed for deploy pad noise


# ---------------------------------------------------------------------------
# Pure data pipeline (numpy only — unit-tested)
# ---------------------------------------------------------------------------
SOURCE_EP_OFFSET = 100_000   # keeps episode ids disjoint across shard sources


def load_shards_multi(shard_dirs: list[str | Path]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load + concatenate several shard dirs (Stage C: base + DAgger rounds).

    Returns (arrays, source) where source[i] is the index of the dir sample i
    came from. Episode ids are offset per source (source * SOURCE_EP_OFFSET) so
    episode-level splitting and t_norm grouping never mix sources.
    """
    accs: list[dict[str, np.ndarray]] = []
    sources: list[np.ndarray] = []
    for si, d in enumerate(shard_dirs):
        a = load_shards(d)
        a["episode"] = np.asarray(a["episode"], dtype=np.int64) + si * SOURCE_EP_OFFSET
        accs.append(a)
        sources.append(np.full(len(a["episode"]), si, dtype=np.int64))
    keys = set(accs[0])
    for a in accs[1:]:
        keys &= set(a)
    arrays = {k: np.concatenate([a[k] for a in accs], axis=0) for k in sorted(keys)}
    return arrays, np.concatenate(sources)


def load_feat_shards(feat_dirs: list[str | Path],
                     arrays: dict[str, np.ndarray],
                     source: np.ndarray) -> np.ndarray:
    """Load pooled-backbone feature shards (feat_shard_*.npz), row-aligned with
    the already-loaded target shards; verify alignment via episode/frame.

    ``arrays['episode']`` carries the per-source SOURCE_EP_OFFSET; feat shards
    store raw ids, so alignment is checked modulo the offset.
    """
    ep = np.asarray(arrays["episode"], dtype=np.int64) % SOURCE_EP_OFFSET
    fi = np.asarray(arrays["frame_idx"], dtype=np.int64)
    out: np.ndarray | None = None
    for si, d in enumerate(feat_dirs):
        files = sorted(Path(d).glob("feat_shard_*.npz"))
        if not files:
            raise FileNotFoundError(f"no feat_shard_*.npz in {d}")
        eps, frs, feats = [], [], []
        for f in files:
            z = np.load(f)
            eps.append(np.asarray(z["episode"], dtype=np.int64))
            frs.append(np.asarray(z["frame_idx"], dtype=np.int64))
            feats.append(np.asarray(z["feats"], dtype=np.float32))
        eps, frs = np.concatenate(eps), np.concatenate(frs)
        F = np.concatenate(feats)
        m = source == si
        if not (np.array_equal(eps, ep[m]) and np.array_equal(frs, fi[m])):
            raise ValueError(f"feat shards in {d} misaligned with target shards "
                             f"(source {si}): re-run encode-features")
        if out is None:
            out = np.zeros((len(ep), F.shape[1]), dtype=np.float32)
        out[m] = F
    assert out is not None
    return out


def load_shards(shard_dir: str | Path) -> dict[str, np.ndarray]:
    """Concatenate all ``shard_*.npz`` in a directory into flat arrays."""
    shard_dir = Path(shard_dir)
    files = sorted(shard_dir.glob("shard_*.npz"))
    if not files:
        raise FileNotFoundError(f"no shard_*.npz in {shard_dir}")
    acc: dict[str, list] = {}
    for f in files:
        z = np.load(f)
        for k in z.files:
            acc.setdefault(k, []).append(z[k])
    out = {k: (np.concatenate(v, axis=0) if len(v) else np.array(v)) for k, v in acc.items()}
    return out


def phase_onehot(phase_label: np.ndarray, n: int = N_PHASES) -> np.ndarray:
    pl = np.asarray(phase_label, dtype=np.int64).reshape(-1)
    oh = np.zeros((pl.shape[0], n), dtype=np.float32)
    valid = (pl >= 0) & (pl < n)
    oh[np.arange(pl.shape[0])[valid], pl[valid]] = 1.0
    return oh


def t_norm_feature(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Normalized episode progress per sample: frame_idx / (episode length).

    v0.1 fix for expert-state ALIASING: the scripted expert is time-indexed
    (slow ramps + convergence holds), so 68% of chunks are near-static and a
    time-free net collapses to the "stay" majority — an absorbing fixed point
    at home in the live loop (Stage-B A/B: 13/15 steered home-freezes).
    Episode length proxy = max frame_idx within the episode + REAL_STEPS.
    Deploy-side equivalent: executed_ticks / 900 (see SteeringEngine).
    """
    fi = np.asarray(arrays["frame_idx"], dtype=np.float64).reshape(-1)
    ep = np.asarray(arrays["episode"], dtype=np.int64).reshape(-1)
    ep_len: dict[int, float] = {}
    for e in np.unique(ep):
        ep_len[int(e)] = float(fi[ep == e].max()) + float(REAL_STEPS)
    t = np.array([fi[i] / ep_len[int(ep[i])] for i in range(len(fi))],
                 dtype=np.float32)
    return np.clip(t, 0.0, 1.0)


def motion_weights(motion: np.ndarray, *, floor: float = 0.2,
                   ref: float = 0.05) -> np.ndarray:
    """Per-sample loss weights emphasizing MOVING expert windows.

    weight = floor + (1-floor) * min(1, motion/ref). Static holds (68% of the
    dataset) keep a small floor so legitimate pauses stay learnable (the t_norm
    feature disambiguates them), while the expert's actual skill — the moving
    ~30% — dominates the gradient.
    """
    m = np.asarray(motion, dtype=np.float32).reshape(-1)
    return (floor + (1.0 - floor) * np.clip(m / ref, 0.0, 1.0)).astype(np.float32)


def chunk_motion_from_dataset(arrays: dict[str, np.ndarray], dataset_dir: str | Path
                              ) -> np.ndarray:
    """Endpoint arm motion (max |a[f+15] - a[f]| over the 6 arm dims, rad) of the
    expert chunk at each (episode, frame_idx) sample, from the LeRobot parquet."""
    import pyarrow.parquet as pq
    dataset_dir = Path(dataset_dir)
    ep = np.asarray(arrays["episode"], dtype=np.int64).reshape(-1)
    fi = np.asarray(arrays["frame_idx"], dtype=np.int64).reshape(-1)
    cache: dict[int, np.ndarray] = {}
    out = np.zeros(len(ep), dtype=np.float32)
    for i in range(len(ep)):
        e = int(ep[i])
        if e not in cache:
            f = dataset_dir / "data" / "chunk-000" / f"episode_{e:06d}.parquet"
            cache[e] = np.stack(
                pq.read_table(f, columns=["action"]).column("action").to_pylist()
            )[:, :7]
        a = cache[e]
        s = int(fi[i])
        end = min(s + REAL_STEPS - 1, len(a) - 1)
        # v0.2: motion INCLUDES the grip dim. The v0.1 arm-only definition
        # down-weighted the (arm-static) closing windows to the floor and
        # suppressed the learned grip closure entirely (grip-authority probe:
        # oracle closes at 100% of windows, v0.1-steered at 8%). A 0->1 grip
        # transition now counts as full motion.
        arm_m = float(np.abs(a[end, :6] - a[s, :6]).max())
        grip_m = float(np.abs(a[s:end + 1, 6] - a[s, 6]).max())
        out[i] = max(arm_m, grip_m)
    return out


RECENT_MOTION_WINDOW = 8   # dataset frames == deploy control ticks per query


def recent_motion_from_dataset(arrays: dict[str, np.ndarray],
                               dataset_dir: str | Path,
                               window: int = RECENT_MOTION_WINDOW) -> np.ndarray:
    """Directional recent arm motion dq = qpos[f] - qpos[f-window] per sample (N,6).

    v3 anti-ambiguity feature: at the hover-band poses where DAgger correctives
    live, the EXPERT passing through is moving (dq != 0) while the stuck policy
    is not (dq ~ 0) — the 15-d snapshot input could not represent that, so base
    'keep ramping' and corrective 'start moving' targets collided at the same
    input point (the interference sweep's 34 cm dagger-state EE). Frames before
    ``window`` difference against frame 0 (episode start: both regimes static —
    consistent with the deploy sidecar's zero-initialised per-session tracker).
    """
    import pyarrow.parquet as pq
    dataset_dir = Path(dataset_dir)
    ep = np.asarray(arrays["episode"], dtype=np.int64).reshape(-1)
    fi = np.asarray(arrays["frame_idx"], dtype=np.int64).reshape(-1)
    cache: dict[int, np.ndarray] = {}
    out = np.zeros((len(ep), 6), dtype=np.float32)
    for i in range(len(ep)):
        e = int(ep[i])
        if e not in cache:
            f = dataset_dir / "data" / "chunk-000" / f"episode_{e:06d}.parquet"
            cache[e] = np.stack(pq.read_table(f, columns=["observation.state"])
                                .column("observation.state").to_pylist())[:, :6]
        s = cache[e]
        f_now = min(int(fi[i]), len(s) - 1)
        f_prev = max(0, f_now - window)
        out[i] = (s[f_now] - s[f_prev]).astype(np.float32)
    return out


def build_xy(arrays: dict[str, np.ndarray], output_design: str,
             features: str = "v1", extra_feats: np.ndarray | None = None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X (N,14|15|21), Y (N, 112|5280), episodes (N,)).

    features: 'v1' -> [proprio7, grasp_target3, phase_onehot4]           (14)
              'v2' -> v1 + [t_norm]  (normalized episode progress)       (15)
              'v3' -> v2 + extra_feats (recent-motion dq6 -> 21); caller
                      supplies extra_feats (dataset-dependent).
    output_design: 'real112' -> Y = w_star_real flattened (112);
                   'full5280' -> Y = w_star_full flattened (5280) [requires w_star_full].
    """
    proprio7 = np.asarray(arrays["proprio7"], dtype=np.float32)        # (N,7)
    gt3 = np.asarray(arrays["grasp_target3"], dtype=np.float32)         # (N,3)
    oh = phase_onehot(arrays["phase_label"])                            # (N,4)
    cols = [proprio7, gt3, oh]
    if features in ("v2", "v3", "v4"):
        cols.append(t_norm_feature(arrays)[:, None])                    # (N,1)
    if features in ("v3", "v4"):
        # v3: recent-motion dq6; v4: pooled backbone features (image-conditioned
        # head — the inverted noise targets provably depend on the frames).
        if extra_feats is None:
            raise ValueError(f"features={features!r} requires extra_feats")
        cols.append(np.asarray(extra_feats, dtype=np.float32).reshape(len(proprio7), -1))
    elif features not in ("v1", "v2"):
        raise ValueError(f"unknown features {features!r}")
    X = np.concatenate(cols, axis=1).astype(np.float32)                 # (N,14|15|21)
    if output_design == "real112":
        Y = np.asarray(arrays["w_star_real"], dtype=np.float32).reshape(len(X), REAL_OUT)
    elif output_design == "full5280":
        Y = np.asarray(arrays["w_star_full"], dtype=np.float32).reshape(len(X), FULL_OUT)
    else:
        raise ValueError(f"unknown output_design {output_design!r}")
    episodes = np.asarray(arrays["episode"], dtype=np.int64).reshape(-1)
    return X, Y, episodes


def episode_split(episodes: np.ndarray, val_frac: float = 0.1, seed: int = 0
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices by EPISODE (no episode in both train and val)."""
    episodes = np.asarray(episodes, dtype=np.int64).reshape(-1)
    uniq = np.unique(episodes)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_eps = set(perm[:n_val].tolist())
    is_val = np.array([e in val_eps for e in episodes])
    val_idx = np.nonzero(is_val)[0]
    train_idx = np.nonzero(~is_val)[0]
    return train_idx, val_idx


def deploy_pads(n: int) -> np.ndarray:
    """Fixed-seed Gaussian pad noise for the (40x132) init_noise at deploy, used
    when output_design='real112'. Shape (n, ACTION_HORIZON, ACTION_DIM)."""
    rng = np.random.default_rng(PAD_SEED)
    return rng.standard_normal((n, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)


def assemble_init_noise(real_pred: np.ndarray, pads: np.ndarray | None = None) -> np.ndarray:
    """Place predicted real dims (n,16,7) into a full (n,40,132) tensor; fill the
    rest with fixed-seed Gaussian pads (deploy convention)."""
    real_pred = np.asarray(real_pred, dtype=np.float32).reshape(-1, REAL_STEPS, REAL_DIMS)
    n = real_pred.shape[0]
    full = deploy_pads(n) if pads is None else np.asarray(pads, dtype=np.float32).copy()
    full[:, :REAL_STEPS, :REAL_DIMS] = real_pred
    return full


# ---------------------------------------------------------------------------
# MLP (torch, lazy)
# ---------------------------------------------------------------------------
def build_mlp(in_dim: int, out_dim: int, hidden: int = 256):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),   # linear head, no tanh
    )


def mode_train(args) -> dict:
    import torch

    shard_dirs = [s for s in str(args.shards).split(",") if s]
    arrays, source = load_shards_multi(shard_dirs)
    design = args.output_design
    extra = None
    if args.features == "v4":
        feat_dirs = [d for d in str(args.feat_shards or "").split(",") if d]
        if len(feat_dirs) != len(shard_dirs):
            raise SystemExit("features=v4 requires --feat-shards aligned with --shards")
        extra = load_feat_shards(feat_dirs, arrays, source)
        print(f"[train] v4 pooled-backbone features: dim={extra.shape[1]}", flush=True)
    if args.features == "v3":
        # per-source recent-motion dq (dataset-dependent, like motion weights)
        datasets_v3 = [d for d in str(args.dataset or "").split(",")]
        if len(datasets_v3) == 1 and len(shard_dirs) > 1:
            datasets_v3 = datasets_v3 * len(shard_dirs)
        extra = np.zeros((len(arrays["episode"]), 6), dtype=np.float32)
        for si in range(len(shard_dirs)):
            m = source == si
            ds = datasets_v3[si] if si < len(datasets_v3) else ""
            if not np.any(m) or ds in ("", "none"):
                raise SystemExit("features=v3 requires --dataset for every shard source")
            sub = {k: (np.asarray(v)[m] if len(np.asarray(v)) == len(arrays["episode"]) else v)
                   for k, v in arrays.items()}
            sub["episode"] = np.asarray(sub["episode"]) % SOURCE_EP_OFFSET
            extra[m] = recent_motion_from_dataset(sub, ds)
        print(f"[train] v3 recent-motion: |dq| mean={float(np.abs(extra).mean()):.4f} "
              f"static(<0.005)={float(np.mean(np.abs(extra).max(axis=1) < 0.005)):.1%}", flush=True)
    X, Y, episodes = build_xy(arrays, design, features=args.features, extra_feats=extra)
    train_idx, val_idx = episode_split(episodes, val_frac=args.val_frac, seed=args.seed)
    n_train, n_val = len(train_idx), len(val_idx)
    print(f"[train] N={len(X)} in={X.shape[1]} out={Y.shape[1]} design={design} "
          f"features={args.features} sources={len(shard_dirs)} "
          f"per_source={np.bincount(source).tolist()} | train={n_train} val={n_val} | "
          f"val_eps={len(np.unique(episodes[val_idx]))}", flush=True)

    # v0.1: motion-weighted loss (needs the per-source dataset for chunk motion).
    # Stage C: per-source datasets ('' or 'none' to skip a source) + per-source
    # boost so a small DAgger round is not drowned by the 17k base samples.
    motion = None
    W = np.ones(len(X), dtype=np.float32)
    if args.dataset:
        datasets = [d for d in str(args.dataset).split(",")]
        if len(datasets) == 1 and len(shard_dirs) > 1:
            datasets = datasets * len(shard_dirs)
        motion = np.zeros(len(X), dtype=np.float32)
        for si in range(len(shard_dirs)):
            m = source == si
            ds = datasets[si] if si < len(datasets) else "none"
            if not np.any(m) or ds in ("", "none"):
                motion[m] = 1.0   # unknown motion -> full weight
                continue
            sub = {k: (np.asarray(v)[m] if len(np.asarray(v)) == len(X) else v)
                   for k, v in arrays.items()}
            sub["episode"] = np.asarray(sub["episode"]) % SOURCE_EP_OFFSET
            motion[m] = chunk_motion_from_dataset(sub, ds)
        W = motion_weights(motion, floor=args.weight_floor, ref=args.weight_ref)
        print(f"[train] motion weights: static(<0.02rad)={float(np.mean(motion < 0.02)):.1%} ",
              flush=True)
    boosts = [float(b) for b in str(args.source_boost).split(",")]
    while len(boosts) < len(shard_dirs):
        boosts.append(boosts[-1])
    W = W * np.array([boosts[s] for s in source], dtype=np.float32)
    W = W / W.mean()   # keep the loss scale comparable to unweighted MSE
    print(f"[train] boosts={boosts} w_min={W.min():.3f} w_max={W.max():.3f}", flush=True)

    dev = args.device
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=dev)
    Wt = torch.as_tensor(W, dtype=torch.float32, device=dev)
    tr = torch.as_tensor(train_idx, device=dev)
    va = torch.as_tensor(val_idx, device=dev)

    torch.manual_seed(args.seed)
    net = build_mlp(X.shape[1], Y.shape[1], args.hidden).to(dev)
    # Warm start (interference mitigation): initialize from a prior net (e.g.
    # v0.2) and fine-tune with a low LR so DAgger correctives adjust the
    # mapping instead of re-carving it — the full-round retrain from scratch
    # regressed base-state EE 0.74cm -> 3.64cm.
    if getattr(args, "init_weights", None):
        z = np.load(args.init_weights, allow_pickle=True)
        prior_meta = json.loads(str(z["meta"]))
        assert prior_meta["in_dim"] == X.shape[1] and prior_meta["out_dim"] == Y.shape[1], \
            f"init-weights dims {prior_meta['in_dim']}->{prior_meta['out_dim']} vs data {X.shape[1]}->{Y.shape[1]}"
        sd = {k[len("sd."):]: torch.as_tensor(z[k]) for k in z.files if k.startswith("sd.")}
        net.load_state_dict(sd)
        print(f"[train] warm-start from {args.init_weights} (lr={args.lr})", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lossfn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    patience = args.patience
    bad = 0
    history = []
    for epoch in range(args.epochs):
        net.train()
        perm = tr[torch.randperm(n_train, device=dev)]
        tot = 0.0
        for i in range(0, n_train, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            pred = net(Xt[idx])
            # motion-weighted MSE (Wt=1 when no --dataset given)
            loss = (Wt[idx, None] * (pred - Yt[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        train_mse = tot / n_train

        net.eval()
        with torch.no_grad():
            vpred = net(Xt[va])
            val_mse = float(lossfn(vpred, Yt[va]))
            p99 = float(torch.quantile(vpred.abs().flatten().float(), 0.99))
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse, "p99_pred": p99})
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"[train] ep{epoch:3d} train={train_mse:.5f} val={val_mse:.5f} p99|pred|={p99:.3f}",
                  flush=True)
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[train] early stop at ep{epoch} (best ep{best_epoch} val={best_val:.5f})",
                      flush=True)
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    # final metrics + p99 on val with best weights
    net.eval()
    with torch.no_grad():
        vpred = net(Xt[va])
        final_val = float(lossfn(vpred, Yt[va]))
        final_p99 = float(torch.quantile(vpred.abs().flatten().float(), 0.99))
        tpred = net(Xt[tr])
        final_train = float(lossfn(tpred, Yt[tr]))
        # v0.1 stratified validation: the Stage-B lesson — an unstratified metric
        # is dominated by the 68% static windows and hides an always-stay net.
        val_mse_moving = val_mse_static = None
        if motion is not None:
            vm = torch.as_tensor(motion[val_idx] >= 0.02, device=dev)
            if bool(vm.any()):
                val_mse_moving = float(((vpred[vm] - Yt[va][vm]) ** 2).mean())
            if bool((~vm).any()):
                val_mse_static = float(((vpred[~vm] - Yt[va][~vm]) ** 2).mean())
        val_mse_by_source = {}
        for si in range(len(shard_dirs)):
            sm = torch.as_tensor(source[val_idx] == si, device=dev)
            if bool(sm.any()):
                val_mse_by_source[f"src{si}"] = float(((vpred[sm] - Yt[va][sm]) ** 2).mean())

    ckpt = {
        "state_dict": {k: v.cpu().numpy() for k, v in net.state_dict().items()},
        "in_dim": X.shape[1], "out_dim": Y.shape[1], "hidden": args.hidden,
        "output_design": design, "pad_seed": PAD_SEED,
        "features": args.features,
        "val_episodes": np.unique(episodes[val_idx]).tolist(),
    }
    np.savez(args.model_out, **{f"sd.{k}": v for k, v in ckpt["state_dict"].items()},
             meta=json.dumps({k: v for k, v in ckpt.items() if k != "state_dict"}))
    print(f"[train] saved {args.model_out}", flush=True)

    result = {
        "output_design": ("real112+fixed-pads" if design == "real112" else "full5280"),
        "features": args.features,
        "train_mse": final_train, "val_mse": final_val,
        "val_mse_moving": val_mse_moving, "val_mse_static": val_mse_static,
        "val_mse_by_source": val_mse_by_source,
        "sources": [str(s) for s in shard_dirs], "source_boost": boosts,
        "p99_pred": final_p99,
        "best_epoch": best_epoch, "n_train": n_train, "n_val": n_val,
        "val_episodes": np.unique(episodes[val_idx]).tolist(),
        "model_out": str(args.model_out),
    }
    if args.out:
        Path(args.out).write_text(json.dumps({**result, "history": history}, indent=2))
    return result


# ---------------------------------------------------------------------------
# pad-check (torch + GR00T model, VM) — measure pad influence on real dims
# ---------------------------------------------------------------------------
def mode_pad_check(args) -> dict:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from flow_inverter_groot import FlowInverter, enumerate_chunks, read_frames
    import torch

    inv = FlowInverter(args.model_path, args.embodiment_tag, args.device,
                       fp_iters=args.fp_iters)
    dataset_dir = Path(args.dataset)
    picks = enumerate_chunks(dataset_dir, args.stride)
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(picks), size=min(args.n, len(picks)), replace=False)
    batch = [picks[i] for i in sorted(sel.tolist())]
    readers: dict = {}
    frames = read_frames(dataset_dir, batch, readers)

    flow = inv.flow
    cache, x_target = inv.encode_batch(frames)
    w, mse = inv.invert_batch(cache, x_target)  # full w* (B,40,132)
    vfn = flow.velocity_fn(cache)
    # keep only well-inverted chunks
    keep = np.isfinite(mse) & (mse <= args.fp_thresh)
    from flow_inverter_groot import forward_euler_sample
    with torch.no_grad():
        # (i) decode with TRUE pads
        x_true = forward_euler_sample(w, vfn, flow.n_steps, flow.dt)
        r_true = flow.real_slice(x_true)
        # (ii) re-sample Gaussian pads, keep real dims of w*, decode
        w2 = w.clone()
        pad_mask = torch.ones_like(w2, dtype=torch.bool)
        pad_mask[:, :REAL_STEPS, :REAL_DIMS] = False
        gauss = torch.randn_like(w2)
        w2[pad_mask] = gauss[pad_mask]
        x_pad = forward_euler_sample(w2, vfn, flow.n_steps, flow.dt)
        r_pad = flow.real_slice(x_pad)
    # per-chunk real-dim ACTION MSE difference between the two decodes
    diff = ((r_true - r_pad) ** 2).mean(dim=(1, 2)).detach().float().cpu().numpy()
    diff = diff[keep]
    pad_influence_mse = float(diff.mean()) if len(diff) else float("nan")
    decision = "real112" if pad_influence_mse < args.pad_thresh else "full5280"
    result = {
        "pad_influence_mse": pad_influence_mse,
        "pad_influence_mse_p95": float(np.percentile(diff, 95)) if len(diff) else float("nan"),
        "pad_thresh": args.pad_thresh,
        "n_sampled": int(keep.sum()),
        "output_design_decision": decision,
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--shards", default="/home/ubuntu/steering_targets")
    pt.add_argument("--output-design", default="real112", choices=["real112", "full5280"])
    pt.add_argument("--model-out", default="/home/ubuntu/steering_net_v0.npz")
    pt.add_argument("--hidden", type=int, default=256)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--batch", type=int, default=256)
    pt.add_argument("--epochs", type=int, default=200)
    pt.add_argument("--patience", type=int, default=20)
    pt.add_argument("--val-frac", type=float, default=0.1)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--device", default="cuda")
    pt.add_argument("--out", default=None)
    # v0.1 anti-aliasing options (see t_norm_feature / motion_weights docstrings)
    pt.add_argument("--features", default="v2", choices=["v1", "v2", "v3", "v4"])
    pt.add_argument("--feat-shards", default=None,
                    help="v4: comma-separated feat-shard dirs aligned with --shards")
    pt.add_argument("--dataset", default=None,
                    help="LeRobot dataset dir; enables motion-weighted loss + stratified val")
    pt.add_argument("--weight-floor", type=float, default=0.2)
    pt.add_argument("--weight-ref", type=float, default=0.05)
    pt.add_argument("--source-boost", default="1.0",
                    help="comma-separated per-shard-dir loss-weight multipliers (Stage C dagger boost)")
    pt.add_argument("--init-weights", default=None,
                    help="warm-start from a prior steering .npz (same dims); pair with a low --lr")

    pc = sub.add_parser("pad-check")
    pc.add_argument("--model-path", default="/home/ubuntu/ckpt/ur5e_drugsort_obscond/full/checkpoint-12000")
    pc.add_argument("--embodiment-tag", default="new_embodiment")
    pc.add_argument("--dataset", default="/home/ubuntu/ur5e_drugsort_obscond")
    pc.add_argument("--device", default="cuda")
    pc.add_argument("--stride", type=int, default=8)
    pc.add_argument("--n", type=int, default=48)
    pc.add_argument("--fp-iters", type=int, default=16)
    pc.add_argument("--fp-thresh", type=float, default=1e-3)
    pc.add_argument("--pad-thresh", type=float, default=1e-4)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--out", default=None)

    args = ap.parse_args()
    if args.mode == "train":
        out = mode_train(args)
    elif args.mode == "pad-check":
        out = mode_pad_check(args)
    else:  # pragma: no cover
        raise SystemExit(f"unknown mode {args.mode}")
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
