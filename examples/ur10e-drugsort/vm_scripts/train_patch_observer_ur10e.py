#!/usr/bin/env python3
"""Obs-D: PATCH-TOKEN Observer head (Patch Policy, Zhou et al. 2026, applied).

Our Observer regresses the 3-D grasp point from DINOv2 CLS tokens — a global
summary that the Patch Policy paper shows discards the fine spatial detail
precise manipulation needs. This trains a small cross-attention head over ALL
patch tokens (both views) on the SAME data and SAME val fold as Obs-C2, so the
comparison is one-variable: pooled-CLS MLP vs patch-token attention.
Obs-C2 to beat (identical fold): median 3.86 / mean 4.47 / p90 8.47 cm.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from torch import nn

sys.path.insert(0, "/home/ubuntu/odyssey-ur5e/src")
from odyssey.embodiments.ur5e_drugsort.observer import GraspTargetObserver

DATA = Path("/home/ubuntu/ur10e_percep_mixed")
OUT = Path("/home/ubuntu/ur10e_percep_weights_patch"); OUT.mkdir(exist_ok=True)
CACHE = OUT / "patch_feats_f16.npy"
dev = "cuda"

ext = np.load(DATA / "ext.npy"); wrist = np.load(DATA / "wrist.npy")
tgt = np.load(DATA / "grasp_target_base.npy").astype(np.float32)
ep = np.load(DATA / "episode.npy")
val_eps = [int(v) for v in open(DATA / "val_episodes.txt").read().split(",")]
va = np.where(np.isin(ep, val_eps))[0]; tr = np.where(~np.isin(ep, val_eps))[0]
N = len(tgt); print(f"frames={N} train={len(tr)} val={len(va)}")

obs = GraspTargetObserver(device=dev); obs.load()
bb = obs._backbone; bb.eval()

# ---- extract & cache patch tokens (drop CLS row 0 -> 256 tokens/view) -------
if not CACHE.is_file():
    P, D = 256, int(obs._feat_dim)
    feats = np.lib.format.open_memmap(CACHE, mode="w+", dtype=np.float16,
                                      shape=(N, 2, P, D))
    B = 64
    with torch.no_grad():
        for i in range(0, N, B):
            for v, arr in ((0, ext), (1, wrist)):
                t = obs._prep(arr[i:i+B])
                out = bb(pixel_values=t).last_hidden_state[:, 1:, :]  # (b,256,384)
                feats[i:i+B, v] = out.detach().cpu().numpy().astype(np.float16)
            if (i // B) % 10 == 0:
                print(f"  extract {i}/{N}", flush=True)
    feats.flush(); print("cached", CACHE)
feats = np.load(CACHE, mmap_mode="r")
P, D = feats.shape[2], feats.shape[3]

# ---- patch-attention head ----------------------------------------------------
class PatchHead(nn.Module):
    def __init__(self, d=D, heads=6, out_dim=3):
        super().__init__()
        self.view_emb = nn.Parameter(torch.zeros(2, 1, d))
        self.pos_emb = nn.Parameter(torch.zeros(1, P, d))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.q = nn.Parameter(torch.zeros(1, 1, d)); nn.init.trunc_normal_(self.q, std=0.02)
        self.ln_kv = nn.LayerNorm(d); self.ln_q1 = nn.LayerNorm(d); self.ln_q2 = nn.LayerNorm(d)
        self.attn1 = nn.MultiheadAttention(d, heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, out_dim))
    def forward(self, x):                      # x: (b, 2, P, D)
        b = x.shape[0]
        toks = x + self.view_emb[None, :, 0, :][:, :, None, :] + self.pos_emb[None]
        toks = toks.reshape(b, 2 * P, -1)
        kv = self.ln_kv(toks)
        q = self.q.expand(b, 1, -1)
        q = q + self.attn1(self.ln_q1(q), kv, kv, need_weights=False)[0]
        q = q + self.attn2(self.ln_q2(q), kv, kv, need_weights=False)[0]
        return self.mlp(q[:, 0])

head = PatchHead().to(dev)
print("head params:", sum(p.numel() for p in head.parameters()))
opt = torch.optim.AdamW(head.parameters(), lr=8e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
lossf = nn.SmoothL1Loss(beta=0.02)
tgt_t = torch.tensor(tgt, device=dev)
rng = np.random.default_rng(0)

def evaluate(idx):
    head.eval(); errs = []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            b = idx[i:i+256]
            x = torch.tensor(np.asarray(feats[b], dtype=np.float32), device=dev)
            e = torch.linalg.norm(head(x) - tgt_t[b], dim=1) * 100.0
            errs.append(e.cpu().numpy())
    head.train(); return np.concatenate(errs)

for epoch in range(60):
    perm = tr[rng.permutation(len(tr))]
    for i in range(0, len(perm), 128):
        b = perm[i:i+128]
        x = torch.tensor(np.asarray(feats[b], dtype=np.float32), device=dev)
        opt.zero_grad(); loss = lossf(head(x), tgt_t[b]); loss.backward(); opt.step()
    sched.step()
    if epoch % 5 == 0 or epoch == 59:
        e = evaluate(va)
        print(f"epoch {epoch:3d} val median={np.median(e):.2f} mean={e.mean():.2f} "
              f"p90={np.percentile(e, 90):.2f} cm", flush=True)

e = evaluate(va)
res = {"median_cm": float(np.median(e)), "mean_cm": float(e.mean()),
       "p90_cm": float(np.percentile(e, 90)), "max_cm": float(e.max()),
       "n_val": int(len(va)), "baseline_obsC2": {"median": 3.86, "mean": 4.47, "p90": 8.47}}
torch.save(head.state_dict(), OUT / "patch_head.pt")
json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
print("FINAL", json.dumps(res))
