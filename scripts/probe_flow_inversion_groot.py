#!/usr/bin/env python3
"""FlowDAgger Step 0/1 GPU probe for GR00T N1.7's flow-matching action head.

Decides whether **latent-noise steering** of the frozen UR5e drug-sort pilot is
viable *before* we invest in the full FlowDAgger port. Two measurements:

* **STEP 0 — inversion round-trip** (``roundtrip`` mode). Invert expert action
  chunks through the frozen policy's flow sampler: recover an initial noise
  ``w*`` such that the *stock* 4-step forward-Euler sampler decodes ``w*`` back to
  the (normalised, padded) expert chunk. PASS = reconstruction MSE < 1e-3 on the
  real action dims **and** p99 ``|w*|`` < ~3 (i.e. ``w*`` looks like plausible
  Gaussian latent noise, not an out-of-distribution outlier).

* **STEP 1 — steering authority at miss states** (``collect-miss-states`` +
  ``authority`` + ``fk-spread`` modes). At sim states where the policy misses the
  vial, (a) invert the *scripted expert's* corrective chunk (recon MSE → on/off
  the sampler manifold) and (b) decode 64 random-noise seeds and measure the
  end-effector spread of the resulting actions via forward kinematics — does the
  noise-cloud envelope cover the 6–12 cm correction the expert would make?

Convention (verified against gr00t/model/gr00t_n1d7/gr00t_n1d7.py +
checkpoint config.json):
  * training  : ``noisy = (1-t)*noise + t*action``,  ``velocity = action - noise``
                => t=0 is pure noise, t=1 is the clean action.
  * inference : forward Euler ``x <- x + dt * v_hat(x, t_bucket)`` from pure
                randn noise, ``dt = 1/num_inference_timesteps``.
  * this checkpoint: ``num_inference_timesteps=4`` (dt=0.25),
                ``num_timestep_buckets=1000``  => per-step buckets are
                ``int((i/4)*1000)`` = {0, 250, 500, 750}.
  * ``action_horizon=40``, ``max_action_dim=132`` (padded); the real UR5e chunk is
                16 steps x 7 dims (6 arm + 1 grip). ALL reconstruction MSE and
                ``|w*|`` stats are scored on the real dims only ([:16, :7]).
  * ``use_alternate_vl_dit=True`` => the per-step DiT call also takes
                ``image_mask`` + ``backbone_attention_mask``.

Precision note (honest caveat): the frozen backbone runs in its native bf16 (flash
attention) and its features are cached once per chunk; the per-step action-head
chain (action_encoder -> DiT -> action_decoder) is run in **float32** for the
inversion so the round-trip measures the sampler's true invertibility rather than
bf16 rounding. Deploy inference is bf16 — latent steering in production would have
to tolerate that extra noise (flagged in the result JSON notes).

The pure inverter (``invert_perstep_fp`` / ``forward_euler_sample``) is a
GR00T-agnostic function of a velocity callable ``v(x, step_idx)`` so it is unit
tested against synthetic analytic fields (tests/unit/test_flow_inversion_math.py)
with no GR00T dependency.

Modes
-----
  roundtrip           STEP 0. In-process (GR00T venv, torch, no mujoco).
  collect-miss-states STEP 1a. Two-process: mujoco venv ZMQ-client -> GR00T server.
  authority           STEP 1b. In-process (GR00T venv): invert corrective chunk +
                      decode 64 seeds; emits per-seed step-8 arm targets.
  fk-spread           STEP 1c. mujoco venv: FK the step-8 arm targets -> gr_pinch
                      xyz -> spread / coverage.
  merge               Combine the stage JSONs into the final result JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Pure inverter math — duck-typed over torch tensors OR numpy arrays. No heavy
# imports at module scope so this file is importable in a torch-less venv (the
# unit test guards torch with pytest.importorskip; mujoco modes import mujoco
# lazily).
# ---------------------------------------------------------------------------

VelocityFn = Callable[[Any, int], Any]


def bucket_schedule(n_steps: int, num_buckets: int) -> list[int]:
    """Per-step discretised timestep buckets used by GR00T's inference loop.

    Derived from the source: ``t_cont = i / n_steps`` ; ``t_disc = int(t_cont *
    num_buckets)``. For n_steps=4, num_buckets=1000 this is {0, 250, 500, 750}.
    """
    return [int((i / float(n_steps)) * num_buckets) for i in range(n_steps)]


def _rms(t: Any) -> float:
    """Root-mean-square of a tensor/array (torch or numpy), as a python float."""
    if hasattr(t, "detach"):  # torch.Tensor
        return float((t.detach() ** 2).mean() ** 0.5)
    a = np.asarray(t)
    return float(np.sqrt(np.mean(a * a)))


def forward_euler_sample(w: Any, velocity_fn: VelocityFn, n_steps: int, dt: float) -> Any:
    """Stock forward-Euler flow sampler: x0 = w ; x_{i+1} = x_i + dt * v(x_i, i)."""
    x = w
    for i in range(n_steps):
        x = x + dt * velocity_fn(x, i)
    return x


def invert_perstep_fp(
    x_target: Any,
    velocity_fn: VelocityFn,
    n_steps: int,
    dt: float,
    fp_iters: int = 8,
) -> tuple[Any, list[float]]:
    """Invert the forward-Euler sampler by per-step fixed-point iteration.

    Given the clean chunk ``x_N = x_target`` and the sampler steps
    ``x_{i+1} = x_i + dt * v(x_i, i)``, sweep ``i = N-1 .. 0`` and solve
    ``x_i = x_{i+1} - dt * v(x_i, i)`` for ``x_i`` by fixed point::

        x_i^{0}   = x_{i+1} - dt * v(x_{i+1}, i)          # seed
        x_i^{k+1} = x_{i+1} - dt * v(x_i^{k}, i)

    Returns ``(w, per_step_residuals)`` where ``w = x_0`` is the recovered noise
    and ``per_step_residuals[i]`` is the RMS of the final fixed-point update at
    step ``i`` (a convergence diagnostic; small => the FP has settled).
    """
    x_next = x_target
    residuals: list[float] = [float("nan")] * n_steps
    for i in reversed(range(n_steps)):
        x_i = x_next - dt * velocity_fn(x_next, i)  # seed
        last_delta = None
        for _ in range(fp_iters):
            x_new = x_next - dt * velocity_fn(x_i, i)
            last_delta = x_new - x_i
            x_i = x_new
        residuals[i] = _rms(last_delta) if last_delta is not None else 0.0
        x_next = x_i
    return x_next, residuals


# ===========================================================================
# Everything below needs torch / gr00t / mujoco — imported lazily per mode.
# ===========================================================================

REAL_STEPS = 16          # UR5e action horizon (delta_indices range(0,16))
REAL_DIMS = 7            # 6 arm joints + 1 gripper
STEP8 = 8               # chunk index used for the FK spread measurement
DEFAULT_INSTRUCTION = "pick up the vial and place it in the rack"


def _percentile(a: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(a, dtype=np.float64), q)) if len(a) else float("nan")


# ---------------------------------------------------------------------------
# GR00T flow field wrapper (torch)
# ---------------------------------------------------------------------------
class GrootFlow:
    """Loads the frozen GR00T policy and exposes its flow sampler for inversion.

    Backbone features are computed ONCE per chunk (bf16, cached, cast to fp32);
    the per-step callable ``velocity_fn(cache)`` runs only the action-head chain
    in fp32.
    """

    def __init__(self, model_path: str, embodiment_tag: str, device: str = "cuda") -> None:
        import torch
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.torch = torch
        self.device = device
        t0 = time.time()
        self.policy = Gr00tPolicy(
            model_path=model_path, embodiment_tag=embodiment_tag, device=device
        )
        self.model = self.policy.model
        self.model.eval()
        # Frozen backbone stays bf16 (flash-attn); action head -> fp32 for a clean
        # inversion (its DiT uses SDPA which supports fp32).
        self.model.action_head.float()
        self.ah = self.model.action_head
        cfg = self.ah.config
        self.n_steps = int(cfg.num_inference_timesteps)
        self.num_buckets = int(cfg.num_timestep_buckets)
        self.dt = 1.0 / self.n_steps
        self.buckets = bucket_schedule(self.n_steps, self.num_buckets)
        self.action_horizon = int(self.ah.action_horizon)
        self.action_dim = int(self.ah.action_dim)
        self.use_alt = bool(cfg.use_alternate_vl_dit)
        self.add_pos = bool(cfg.add_pos_embed)
        from gr00t.data.types import MessageType  # noqa

        self._MessageType = MessageType
        self._VLAStepData = __import__(
            "gr00t.data.types", fromlist=["VLAStepData"]
        ).VLAStepData
        self.embodiment_tag = self.policy.embodiment_tag
        print(
            f"[groot] loaded {model_path} in {time.time()-t0:.1f}s | "
            f"n_steps={self.n_steps} dt={self.dt} buckets={self.buckets} "
            f"action_horizon={self.action_horizon} action_dim={self.action_dim} "
            f"use_alt_vl_dit={self.use_alt} model.dtype={self.model.dtype}",
            flush=True,
        )

    # -- build the model batch, optionally injecting an expert action chunk ----
    def _build_batch(
        self,
        ext: np.ndarray,
        wrist: np.ndarray,
        state10: np.ndarray,
        arm: np.ndarray | None = None,
        grip: np.ndarray | None = None,
        instruction: str = DEFAULT_INSTRUCTION,
    ):
        s = np.asarray(state10, dtype=np.float32).reshape(-1)
        states = {
            "single_arm": s[:6].reshape(1, 6).astype(np.float32),
            "gripper": s[6:7].reshape(1, 1).astype(np.float32),
            "grasp_target": s[7:10].reshape(1, 3).astype(np.float32),
        }
        actions: dict[str, np.ndarray] = {}
        if arm is not None:
            actions = {
                "single_arm": np.asarray(arm, dtype=np.float32).reshape(-1, 6),
                "gripper": np.asarray(grip, dtype=np.float32).reshape(-1, 1),
            }
        vsd = self._VLAStepData(
            images={
                "exterior": [np.asarray(ext, dtype=np.uint8)],
                "wrist": [np.asarray(wrist, dtype=np.uint8)],
            },
            states=states,
            actions=actions,
            text=instruction,
            embodiment=self.embodiment_tag,
        )
        proc = self.policy.processor(
            [{"type": self._MessageType.EPISODE_STEP.value, "content": vsd}]
        )
        # collate_fn tokenises vlm_content and wraps the batch as
        # BatchFeature({"inputs": {...}}); model.prepare_input wants the inner dict.
        collated = self.policy.collate_fn([proc])
        return collated["inputs"]

    def encode(self, ext, wrist, state10, arm=None, grip=None, instruction=DEFAULT_INSTRUCTION):
        """Run the backbone once; return a cache for the per-step velocity_fn plus
        (if an action chunk was supplied) the normalised+padded target ``x_N`` and
        its real-dim mask."""
        torch = self.torch
        batch = self._build_batch(ext, wrist, state10, arm, grip, instruction)
        with torch.no_grad():
            backbone_inputs, action_inputs = self.model.prepare_input(batch)
            backbone_output = self.model.backbone(backbone_inputs)
            # Cast the cached conditioning to fp32 for the fp32 action head.
            backbone_output["backbone_features"] = backbone_output["backbone_features"].float()
            action_inputs["state"] = action_inputs["state"].float()
            feats = self.ah._encode_features(backbone_output, action_inputs)
        cache = {
            "vl_embeds": feats.backbone_features,          # fp32, post vlln/self-attn
            "state_features": feats.state_features,        # fp32
            "embodiment_id": action_inputs.embodiment_id,
            "image_mask": getattr(backbone_output, "image_mask", None),
            "backbone_attention_mask": getattr(backbone_output, "backbone_attention_mask", None),
            "batch_size": feats.backbone_features.shape[0],
        }
        x_target = None
        mask = None
        if "action" in action_inputs:
            x_target = action_inputs.action.float()      # [1, action_horizon, action_dim]
            if "action_mask" in action_inputs:
                mask = action_inputs.action_mask.float()
        return cache, x_target, mask

    def broadcast_cache(self, cache: dict, n: int) -> dict:
        """Expand a B=1 cache to batch size ``n`` (for the 64-seed decode)."""
        out = dict(cache)
        for k in ("vl_embeds", "state_features"):
            v = cache[k]
            out[k] = v.expand(n, *v.shape[1:]).contiguous()
        eid = cache["embodiment_id"]
        out["embodiment_id"] = eid.expand(n).contiguous() if eid.ndim == 1 else eid.repeat(n)
        for k in ("image_mask", "backbone_attention_mask"):
            v = cache[k]
            if v is not None and hasattr(v, "expand"):
                out[k] = v.expand(n, *v.shape[1:]).contiguous()
        out["batch_size"] = n
        return out

    def velocity_fn(self, cache: dict) -> VelocityFn:
        torch = self.torch
        ah = self.ah
        vl = cache["vl_embeds"]
        sf = cache["state_features"]
        eid = cache["embodiment_id"]
        img_mask = cache["image_mask"]
        attn_mask = cache["backbone_attention_mask"]
        B = cache["batch_size"]
        dev = vl.device

        def v(x, i):
            t_disc = self.buckets[i]
            ts = torch.full((B,), t_disc, device=dev, dtype=torch.long)
            af = ah.action_encoder(x, ts, eid)
            if self.add_pos:
                pos = torch.arange(af.shape[1], device=dev, dtype=torch.long)
                af = af + ah.position_embedding(pos).unsqueeze(0)
            sa = torch.cat((sf, af), dim=1)
            if self.use_alt:
                mo = ah.model(
                    hidden_states=sa,
                    encoder_hidden_states=vl,
                    timestep=ts,
                    image_mask=img_mask,
                    backbone_attention_mask=attn_mask,
                )
            else:
                mo = ah.model(hidden_states=sa, encoder_hidden_states=vl, timestep=ts)
            pred = ah.action_decoder(mo, eid)
            return pred[:, -self.action_horizon:]

        return v

    def random_noise(self, n: int, seed: int | None = None):
        torch = self.torch
        g = None
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(
            n, self.action_horizon, self.action_dim,
            device=self.device, dtype=torch.float32, generator=g,
        )

    # -- reconstruction metrics on the REAL dims only --------------------------
    @staticmethod
    def real_slice(t):
        return t[:, :REAL_STEPS, :REAL_DIMS]

    def recon_mse_real(self, x_hat, x_target) -> float:
        d = self.real_slice(x_hat) - self.real_slice(x_target)
        return float((d ** 2).mean())

    def decode_to_physical(self, normalized):
        """Un-normalise a [B, action_horizon, action_dim] tensor -> dict of
        {single_arm:(B,16,6), gripper:(B,16,1)} in physical units."""
        arr = normalized.detach().float().cpu().numpy()
        return self.policy.processor.decode_action(arr, self.embodiment_tag, None)


# ---------------------------------------------------------------------------
# Adam fallback (masked): minimise ||decode(w)[real] - x_target[real]||^2
# ---------------------------------------------------------------------------
def adam_refine(flow: GrootFlow, cache, x_target, w_init, iters=200, lr=1e-2, clip=1.0):
    torch = flow.torch
    w = w_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    vfn = flow.velocity_fn(cache)
    tgt = flow.real_slice(x_target).detach()
    for _ in range(iters):
        opt.zero_grad()
        x = w
        for i in range(flow.n_steps):
            x = x + flow.dt * vfn(x, i)
        loss = ((flow.real_slice(x) - tgt) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], clip)
        opt.step()
    return w.detach()


# ---------------------------------------------------------------------------
# One-chunk inversion: FP then optional Adam fallback; honest round-trip metric
# ---------------------------------------------------------------------------
def invert_chunk(flow: GrootFlow, cache, x_target, mask, fp_iters, adam_iters, adam_thresh):
    torch = flow.torch
    vfn = flow.velocity_fn(cache)
    with torch.no_grad():
        w, residuals = invert_perstep_fp(x_target, vfn, flow.n_steps, flow.dt, fp_iters)
        x_hat = forward_euler_sample(w, vfn, flow.n_steps, flow.dt)
        mse = flow.recon_mse_real(x_hat, x_target)
    used_adam = False
    if adam_iters > 0 and mse > adam_thresh:
        w = adam_refine(flow, cache, x_target, w, iters=adam_iters)
        with torch.no_grad():
            x_hat = forward_euler_sample(w, vfn, flow.n_steps, flow.dt)
            mse = flow.recon_mse_real(x_hat, x_target)
        used_adam = True
    w_real = flow.real_slice(w).detach().abs().float().cpu().numpy().reshape(-1)
    return {
        "recon_mse_real": float(mse),
        "w_abs_mean": float(w_real.mean()),
        "w_abs_p99": _percentile(w_real, 99),
        "w_abs_max": float(w_real.max()),
        "fp_residuals": [float(r) for r in residuals],
        "adam_used": used_adam,
        "w": w,
        "w_abs_real": w_real,  # raw real-dim |w*| values for pooled aggregation
    }


# ===========================================================================
# MODE: roundtrip (STEP 0)
# ===========================================================================
def load_dataset_frame(dataset_dir: Path, episode: int, frame: int, _readers: dict):
    """Return (ext_frame, wrist_frame, state10, action7) for one dataset frame."""
    import pyarrow.parquet as pq

    pf = dataset_dir / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    key = ("tbl", episode)
    if key not in _readers:
        _readers[key] = pq.read_table(pf)
    tbl = _readers[key]
    state = np.asarray(tbl.column("observation.state")[frame].as_py(), dtype=np.float32)
    action = np.asarray(tbl.column("action")[frame].as_py(), dtype=np.float32)

    import imageio.v2 as imageio

    def _read_frame(view: str) -> np.ndarray:
        rk = ("vid", view, episode)
        if rk not in _readers:
            mp4 = (dataset_dir / "videos" / "chunk-000"
                   / f"observation.images.{view}" / f"episode_{episode:06d}.mp4")
            _readers[rk] = imageio.get_reader(str(mp4), "ffmpeg")
        img = _readers[rk].get_data(frame)
        if img.shape[:2] != (256, 256):
            from PIL import Image
            img = np.asarray(Image.fromarray(img).resize((256, 256)), dtype=np.uint8)
        return np.asarray(img, dtype=np.uint8)

    return _read_frame("exterior"), _read_frame("wrist"), state, action


def pick_chunk_frames(dataset_dir: Path, n_chunks: int, seed: int) -> list[tuple[int, int]]:
    """Sample (episode, frame) pairs across episodes AND phases (early reach /
    descent / grasp / transport) so the inversion is exercised broadly."""
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    data_dir = dataset_dir / "data" / "chunk-000"
    eps = sorted(int(p.stem.split("_")[-1]) for p in data_dir.glob("episode_*.parquet"))
    picks: list[tuple[int, int]] = []
    # phase fractions across the episode timeline
    phase_fracs = [(0.05, 0.25), (0.30, 0.50), (0.50, 0.70), (0.70, 0.90)]
    ep_choices = rng.choice(eps, size=min(len(eps), max(n_chunks, 8)), replace=False)
    i = 0
    while len(picks) < n_chunks:
        ep = int(ep_choices[i % len(ep_choices)])
        nrows = pq.read_metadata(data_dir / f"episode_{ep:06d}.parquet").num_rows
        lo, hi = phase_fracs[len(picks) % len(phase_fracs)]
        # leave room for the 16-step future chunk
        fr = int(rng.integers(int(lo * nrows), max(int(lo * nrows) + 1, int(hi * nrows))))
        fr = min(fr, nrows - REAL_STEPS - 1)
        if fr < 0:
            i += 1
            continue
        picks.append((ep, fr))
        i += 1
    return picks


def expert_chunk_from_dataset(dataset_dir: Path, episode: int, frame: int, readers: dict):
    """The expert action chunk = the next REAL_STEPS recorded actions (absolute
    joint targets), i.e. exactly what the fine-tune supervised at this frame."""
    import pyarrow.parquet as pq

    pf = dataset_dir / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    key = ("tbl", episode)
    if key not in readers:
        readers[key] = pq.read_table(pf)
    tbl = readers[key]
    acts = np.asarray(
        [tbl.column("action")[frame + k].as_py() for k in range(REAL_STEPS)],
        dtype=np.float32,
    )  # (16, 7)
    return acts[:, :6], acts[:, 6:7]


def mode_roundtrip(args) -> dict:
    dataset_dir = Path(args.dataset)
    flow = GrootFlow(args.model_path, args.embodiment_tag, args.device)
    picks = pick_chunk_frames(dataset_dir, args.n_chunks, args.seed)
    print(f"[roundtrip] {len(picks)} chunks: {picks}", flush=True)
    readers: dict = {}
    per = []
    w_abs_pool: list = []
    for j, (ep, fr) in enumerate(picks):
        ext, wrist, state10, _ = load_dataset_frame(dataset_dir, ep, fr, readers)
        arm, grip = expert_chunk_from_dataset(dataset_dir, ep, fr, readers)
        cache, x_target, mask = flow.encode(ext, wrist, state10, arm, grip)
        assert x_target is not None, "no action target produced"
        assert x_target.shape[1] == flow.action_horizon and x_target.shape[2] == flow.action_dim, \
            f"unexpected target shape {tuple(x_target.shape)}"
        res = invert_chunk(flow, cache, x_target, mask, args.fp_iters,
                           args.adam_iters, args.adam_thresh)
        res.pop("w", None)
        w_abs_pool.append(res.pop("w_abs_real"))
        res.update({"episode": ep, "frame": fr})
        per.append(res)
        print(f"[roundtrip] chunk {j+1}/{len(picks)} ep{ep} fr{fr} "
              f"mse={res['recon_mse_real']:.3e} w_p99={res['w_abs_p99']:.2f} "
              f"adam={res['adam_used']} fp_res={[f'{r:.1e}' for r in res['fp_residuals']]}",
              flush=True)

    mses = np.array([p["recon_mse_real"] for p in per])
    per_step_res = np.array([p["fp_residuals"] for p in per])
    # Pool ALL real-dim |w*| across chunks for the honest p99 (spec: p99 |w*| < ~3).
    w_pool = np.concatenate(w_abs_pool)
    pooled_p99 = _percentile(w_pool, 99)
    passed = bool(mses.mean() < 1e-3 and pooled_p99 < 3.5)
    out = {
        "n_chunks": len(per),
        "recon_mse_real_dims": {
            "mean": float(mses.mean()), "p95": _percentile(mses, 95),
            "worst": float(mses.max()),
        },
        "w_star_abs": {
            "mean": float(w_pool.mean()), "p99": pooled_p99,
            "max": float(w_pool.max()),
        },
        "fp_per_step": args.fp_iters,
        "per_step_fp_residuals": [float(x) for x in per_step_res.mean(axis=0)],
        "adam_fallback_used": bool(any(p["adam_used"] for p in per)),
        "pass": passed,
        "per_chunk": per,
    }
    return out


# ===========================================================================
# MODE: authority (STEP 1b) — invert corrective chunk + decode 64 seeds
# ===========================================================================
def mode_authority(args) -> dict:
    miss = np.load(args.miss_states, allow_pickle=True)
    ext_all = miss["exterior"]        # (N,256,256,3) uint8
    wrist_all = miss["wrist"]
    state10_all = miss["state10"]      # (N,10)
    corr_all = miss["corrective_chunk"]  # (N,16,7)
    needed_cm = miss["needed_correction_cm"] if "needed_correction_cm" in miss else None
    N = len(ext_all)
    flow = GrootFlow(args.model_path, args.embodiment_tag, args.device)
    torch = flow.torch

    per = []
    seed_arm_step8 = []   # (N, 64, 6)
    corr_arm_step8 = []    # (N, 6)
    for k in range(N):
        arm = corr_all[k][:, :6]
        grip = corr_all[k][:, 6:7]
        cache, x_target, mask = flow.encode(ext_all[k], wrist_all[k], state10_all[k], arm, grip)
        res = invert_chunk(flow, cache, x_target, mask, args.fp_iters,
                           args.adam_iters, args.adam_thresh)
        res.pop("w", None)
        # decode 64 random seeds through the stock sampler
        cache64 = flow.broadcast_cache(cache, args.n_seeds)
        vfn = flow.velocity_fn(cache64)
        with torch.no_grad():
            w = flow.random_noise(args.n_seeds, seed=args.seed + k)
            x = forward_euler_sample(w, vfn, flow.n_steps, flow.dt)
            phys = flow.decode_to_physical(x)     # single_arm:(64,16,6)
        arm_seeds = np.asarray(phys["single_arm"], dtype=np.float32)  # (64,16,6)
        seed_arm_step8.append(arm_seeds[:, STEP8, :])
        corr_arm_step8.append(corr_all[k][STEP8, :6].astype(np.float32))
        per.append({
            "corrective_recon_mse": res["recon_mse_real"],
            "corrective_w_p99": res["w_abs_p99"],
            "adam_used": res["adam_used"],
            "needed_correction_cm": (float(needed_cm[k]) if needed_cm is not None else None),
        })
        print(f"[authority] miss {k+1}/{N} corr_mse={res['recon_mse_real']:.3e} "
              f"adam={res['adam_used']}", flush=True)

    np.savez(
        args.decoded_out,
        seed_arm_step8=np.asarray(seed_arm_step8, dtype=np.float32),
        corr_arm_step8=np.asarray(corr_arm_step8, dtype=np.float32),
        corrective_recon_mse=np.asarray([p["corrective_recon_mse"] for p in per], dtype=np.float64),
        needed_correction_cm=(np.asarray([p["needed_correction_cm"] for p in per], dtype=np.float64)
                              if needed_cm is not None else np.array([])),
    )
    mses = np.array([p["corrective_recon_mse"] for p in per])
    out = {
        "n_miss_states": N,
        "corrective_recon_mse": {
            "mean": float(mses.mean()), "p95": _percentile(mses, 95),
            "frac_below_1e3": float((mses < 1e-3).mean()),
        },
        "decoded_npz": str(args.decoded_out),
        "per_state": per,
    }
    return out


# ===========================================================================
# MODE: fk-spread (STEP 1c) — mujoco FK of step-8 arm targets -> gr_pinch
# ===========================================================================
def mode_fk_spread(args) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco as mj

    dec = np.load(args.decoded_npz, allow_pickle=True)
    seed_arm = dec["seed_arm_step8"]     # (N,64,6)
    corr_arm = dec["corr_arm_step8"]      # (N,6)
    recon_mse = dec["corrective_recon_mse"] if "corrective_recon_mse" in dec else None
    needed = dec["needed_correction_cm"] if "needed_correction_cm" in dec else None

    sys.path.insert(0, str(Path(args.src) / "scripts"))
    sys.path.insert(0, str(Path(args.src) / "src"))
    from gen_ur5e_drugsort_demos import _build_index  # noqa: E402
    model = mj.MjModel.from_xml_path(args.xml)
    data = mj.MjData(model)
    # resolve arm qpos addresses + gr_pinch site (reuse the demo indexer)
    idx = _build_index(mj, model)
    arm_qadr = idx["arm_qadr"]
    pinch_site = idx["pinch_site"]
    hk = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")

    def fk(q6: np.ndarray) -> np.ndarray:
        if hk >= 0:
            mj.mj_resetDataKeyframe(model, data, hk)
        for a, qv in zip(arm_qadr, q6):
            data.qpos[a] = float(qv)
        mj.mj_forward(model, data)
        return np.array(data.site_xpos[pinch_site], dtype=np.float64)

    N = len(seed_arm)
    per_state = []
    for k in range(N):
        pts = np.stack([fk(seed_arm[k, j]) for j in range(seed_arm.shape[1])])  # (64,3)
        corr_pt = fk(corr_arm[k])
        # max pairwise distance
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        max_pair = float(d.max())
        per_axis_std = (pts.std(axis=0) * 100.0).tolist()  # cm
        dist_to_corr = float(np.linalg.norm(pts - corr_pt[None, :], axis=1).min())
        covered = bool(dist_to_corr < 0.02)
        per_state.append({
            "spread_max_pairwise_cm": max_pair * 100.0,
            "per_axis_std_cm": per_axis_std,
            "min_dist_to_corrective_cm": dist_to_corr * 100.0,
            "corrective_covered_2cm": covered,
            "corrective_recon_mse": (float(recon_mse[k]) if recon_mse is not None and len(recon_mse) else None),
            "needed_correction_cm": (float(needed[k]) if needed is not None and len(needed) else None),
        })
        print(f"[fk] state {k+1}/{N} spread={max_pair*100:.2f}cm "
              f"cover_dist={dist_to_corr*100:.2f}cm covered={covered}", flush=True)

    spreads = np.array([p["spread_max_pairwise_cm"] for p in per_state])
    covered_frac = float(np.mean([p["corrective_covered_2cm"] for p in per_state]))
    recon = np.array([p["corrective_recon_mse"] for p in per_state
                      if p["corrective_recon_mse"] is not None])
    needed_arr = np.array([p["needed_correction_cm"] for p in per_state
                           if p["needed_correction_cm"] is not None])
    frac_on_manifold = float((recon < 1e-3).mean()) if len(recon) else float("nan")
    # STEP 1 pass = latent noise has real steering authority at the near-grasp states:
    #  (1) the scripted-expert corrective chunks are predominantly ON the sampler
    #      manifold (invertible to in-distribution noise), AND
    #  (2) the 64-seed EE spread ENVELOPE reaches into the 6-12 cm correction band
    #      (mean max-pairwise >= 6 cm), AND
    #  (3) the corrective EE point is inside the reachable noise-cloud (covered
    #      within 2 cm) for a majority of states.
    passed = bool(
        (len(recon) and (recon < 1e-3).mean() >= 0.5)
        and float(spreads.mean()) >= 6.0
        and covered_frac >= 0.5
    )
    out = {
        "n_miss_states": N,
        "ee_spread_cm": {
            "mean_max_pairwise": float(spreads.mean()),
            "median_max_pairwise": float(np.median(spreads)),
            "per_state": spreads.tolist(),
        },
        "corrective_covered_within_2cm_frac": covered_frac,
        "corrective_on_manifold_frac": frac_on_manifold,
        "needed_correction_cm": {"per_state": needed_arr.tolist(),
                                 "mean": float(needed_arr.mean()) if len(needed_arr) else None},
        "pass": passed,
        "per_state": per_state,
    }
    return out


# ===========================================================================
# MODE: collect-miss-states (STEP 1a) — mujoco rollout of the served policy
# ===========================================================================
def mode_collect_miss_states(args) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco as mj

    src = Path(args.src)
    sys.path.insert(0, str(src / "scripts"))
    sys.path.insert(0, str(src / "src"))

    from gen_ur5e_drugsort_demos import _build_index  # noqa
    from eval_ur5e_drugsort_groot import GrootPolicyClient
    from odyssey.embodiments.ur5e_drugsort import embodiment as emb
    from odyssey.embodiments.ur5e_drugsort.ik import DampedLeastSquaresIK
    import dagger_ur5e_drugsort as dg
    from odyssey.embodiments.ur5e_drugsort.policy import GRIP_OPEN, GRIP_CLOSE, SETTLE_STEPS
    GRIP_DRIVER_RANGE = dg.GRIP_DRIVER_RANGE

    model = mj.MjModel.from_xml_path(args.xml)
    data = mj.MjData(model)
    idx = _build_index(mj, model)
    renderer = mj.Renderer(model, height=256, width=256)
    ik = DampedLeastSquaresIK(
        model=model, site_id=idx["pinch_site"],
        arm_qadr=tuple(idx["arm_qadr"]), arm_dofadr=tuple(idx["arm_dofadr"]),
    )

    # static base transform for the grasp-target (cap_base) state channel. This
    # reproduces augment_state_grasp_target.py / gen_ur5e_perception_data.py:
    #   cap_world = vial_xyz + R(vial_quat) @ [0,0,VIAL_CAP_OFFSET_Z]
    #   cap_base  = base_R^T @ (cap_world - base_pos)
    hk = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
    mj.mj_resetDataKeyframe(model, data, hk)
    mj.mj_forward(model, data)
    base_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, args.base_body)
    assert base_body >= 0, f"base body {args.base_body!r} not found in MJCF"
    base_pos = np.array(data.xpos[base_body], dtype=np.float64)
    base_mat = np.array(data.xmat[base_body], dtype=np.float64).reshape(3, 3)
    cap_off = float(args.cap_offset_z)

    def cap_base(vial_qpos7: np.ndarray) -> np.ndarray:
        vxyz = np.asarray(vial_qpos7[:3], dtype=np.float64)
        vmat = np.zeros(9)
        mj.mju_quat2Mat(vmat, np.asarray(vial_qpos7[3:7], dtype=np.float64))
        cap_world = vxyz + vmat.reshape(3, 3) @ np.array([0.0, 0.0, cap_off])
        return base_mat.T @ (cap_world - base_pos)

    arm_act, grip_act = idx["arm_act"], idx["grip_act"]
    arm_qadr, grip_qadr = idx["arm_qadr"], idx["grip_qadr"]
    vq = idx["vial_qadr"]
    VIDEO = dict(emb.VIDEO_KEY_TO_CAMERA)

    client = GrootPolicyClient(args.zmq_host, args.zmq_port, timeout_ms=args.timeout_ms)
    assert client.ping(), "GR00T ZMQ server not answering"
    rng = np.random.default_rng(args.seed)

    def read_arm():
        return np.array([data.qpos[a] for a in arm_qadr], dtype=np.float32)

    def render_views():
        v = {}
        for vk, cam in VIDEO.items():
            renderer.update_scene(data, camera=cam)
            v[vk] = renderer.render().copy()
        return v

    def build_obs10(measured, grip_meas, view, gt):
        return {
            "video": {k: f[None, None].astype(np.uint8) for k, f in view.items()},
            "state": {
                "single_arm": measured[None, None].astype(np.float32),
                "gripper": np.asarray([[[grip_meas]]], dtype=np.float32),
                "grasp_target": np.asarray(gt, dtype=np.float32).reshape(1, 1, 3),
            },
            "language": {"annotation.human.task_description": [[args.instruction]]},
        }

    decim = max(1, round((1.0 / args.fps) / model.opt.timestep))
    collected: list[dict] = []
    for ep in range(args.episodes):
        mj.mj_resetDataKeyframe(model, data, hk)
        dg._randomise_vial(mj, model, data, idx, area_x=args.area_x, area_y=args.area_y,
                           yaw_jitter=args.yaw_jitter, rng=rng)
        home_q = tuple(float(v) for v in read_arm())
        relab, _plan = dg._build_expert(ik, mj, model, data, idx, home_q)
        for a, q in zip(arm_act, read_arm()):
            data.ctrl[a] = q
        data.ctrl[grip_act] = GRIP_OPEN
        for _ in range(SETTLE_STEPS):
            mj.mj_step(model, data)

        z0 = float(data.xpos[idx["vial_body"], 2])
        arm_chunk = grip_chunk = None
        ci = 0
        chunk_len = 0
        tick = 0
        ep_miss = 0
        diag_dists: list[float] = []
        diag_grip_max = 0.0
        while tick < args.max_ticks:
            measured = read_arm()
            grip_meas = float(np.clip(data.qpos[grip_qadr] / GRIP_DRIVER_RANGE, 0.0, 1.0))
            view = render_views()
            vial_qpos = np.array([data.qpos[vq + i] for i in range(7)], dtype=np.float64)
            gt = cap_base(vial_qpos)
            pinch = np.array(data.site_xpos[idx["pinch_site"]], dtype=np.float64)
            vial_xyz = np.array(data.xpos[idx["vial_body"]], dtype=np.float64)
            grasp_pt = np.array([vial_xyz[0], vial_xyz[1],
                                 vial_xyz[2] - dg.GRASP_PINCH_BELOW_VIAL], dtype=np.float64)
            pinch_to_vial = float(np.linalg.norm(pinch - grasp_pt))
            lifted = (float(data.xpos[idx["vial_body"], 2]) - z0) > 0.02
            diag_dists.append(pinch_to_vial)
            diag_grip_max = max(diag_grip_max, grip_meas)

            # miss-state selection: approach/descent, 4-15 cm short, not grasping/lifted
            is_miss = (args.miss_lo <= pinch_to_vial <= args.miss_hi
                       and grip_meas < 0.5 and not lifted)
            if is_miss and ep_miss < args.per_ep_cap and len(collected) < args.max_states:
                corrective = _expert_corrective_chunk(
                    mj, model, data, idx, relab, arm_act, grip_act, arm_qadr, grip_qadr,
                    decim, GRIP_CLOSE, GRIP_DRIVER_RANGE, REAL_STEPS)
                state10 = np.concatenate([measured, [grip_meas], gt]).astype(np.float32)
                collected.append({
                    "exterior": view["exterior"], "wrist": view["wrist"],
                    "state10": state10, "corrective_chunk": corrective,
                    "needed_correction_cm": pinch_to_vial * 100.0,
                    "episode": ep, "tick": tick,
                })
                ep_miss += 1
                print(f"[collect] ep{ep} tick{tick} miss dist={pinch_to_vial*100:.1f}cm "
                      f"(total {len(collected)})", flush=True)

            if arm_chunk is None or ci >= chunk_len:
                action = client.get_action(build_obs10(measured, grip_meas, view, gt))
                arm_chunk = np.asarray(action["single_arm"], dtype=np.float32).reshape(-1, 6)
                grip_chunk = np.asarray(action["gripper"], dtype=np.float32).reshape(-1, 1)
                chunk_len = min(args.n_action_steps, arm_chunk.shape[0])
                ci = 0
            for a, q in zip(arm_act, arm_chunk[ci]):
                data.ctrl[a] = float(q)
            data.ctrl[grip_act] = float(np.clip(grip_chunk[ci, 0], 0.0, 1.0)) * GRIP_CLOSE
            for _ in range(decim):
                mj.mj_step(model, data)
            ci += 1
            tick += 1
        dd = np.array(diag_dists) if diag_dists else np.array([np.nan])
        in_band = int(np.sum((dd >= args.miss_lo) & (dd <= args.miss_hi)))
        print(f"[collect] episode {ep+1}/{args.episodes} done, states so far {len(collected)} "
              f"| pinch-dist cm min={dd.min()*100:.1f} med={np.median(dd)*100:.1f} "
              f"max={dd.max()*100:.1f} | grip_max={diag_grip_max:.2f} "
              f"| ticks_in_band[{args.miss_lo*100:.0f},{args.miss_hi*100:.0f}]={in_band}",
              flush=True)
        if len(collected) >= args.max_states:
            break

    if not collected:
        return {"n_miss_states": 0, "note": "no miss states matched the selection window"}
    np.savez(
        args.miss_out,
        exterior=np.stack([c["exterior"] for c in collected]),
        wrist=np.stack([c["wrist"] for c in collected]),
        state10=np.stack([c["state10"] for c in collected]),
        corrective_chunk=np.stack([c["corrective_chunk"] for c in collected]),
        needed_correction_cm=np.array([c["needed_correction_cm"] for c in collected]),
        episode=np.array([c["episode"] for c in collected]),
        tick=np.array([c["tick"] for c in collected]),
    )
    return {
        "n_miss_states": len(collected),
        "miss_npz": str(args.miss_out),
        "needed_correction_cm": {
            "mean": float(np.mean([c["needed_correction_cm"] for c in collected])),
            "min": float(np.min([c["needed_correction_cm"] for c in collected])),
            "max": float(np.max([c["needed_correction_cm"] for c in collected])),
        },
    }


def mode_collect_dataset_states(args) -> dict:
    """Harvest near-grasp 'miss-regime' states from the IN-DISTRIBUTION dataset.

    The obscond checkpoint is visually out-of-distribution in the MuJoCo aseptipack
    render (a fresh sim rollout base-joint-runs-away; confirmed by a discriminator
    where the served policy reproduces recorded expert actions to <0.1 rad on real
    dataset frames but diverges on sim frames). So rather than harvest miss states
    from an unrepresentative sim rollout, we take dataset frames in the descent /
    approach regime where the pinch is ``[miss_lo, miss_hi]`` from the vial cap and
    the gripper is still open — the same 4-15 cm near-grasp band the deployed policy
    reaches then fails in. The 'corrective chunk' is the recorded expert continuation
    (next 16 absolute joint targets), which reaches the vial. The observation is the
    real dataset frame (in-distribution), so the 64-seed decode in ``authority`` is
    meaningful. MuJoCo is used ONLY for forward kinematics (arm qpos -> gr_pinch)."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco as mj

    sys.path.insert(0, str(Path(args.src) / "scripts"))
    sys.path.insert(0, str(Path(args.src) / "src"))
    from gen_ur5e_drugsort_demos import _build_index

    dataset_dir = Path(args.dataset)
    model = mj.MjModel.from_xml_path(args.xml)
    data = mj.MjData(model)
    idx = _build_index(mj, model)
    arm_qadr = idx["arm_qadr"]
    pinch_site = idx["pinch_site"]
    hk = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
    base_body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, args.base_body)
    mj.mj_resetDataKeyframe(model, data, hk); mj.mj_forward(model, data)
    base_pos = np.array(data.xpos[base_body], dtype=np.float64)
    base_mat = np.array(data.xmat[base_body], dtype=np.float64).reshape(3, 3)

    def pinch_base(arm6: np.ndarray) -> np.ndarray:
        if hk >= 0:
            mj.mj_resetDataKeyframe(model, data, hk)
        for a, q in zip(arm_qadr, arm6):
            data.qpos[a] = float(q)
        mj.mj_forward(model, data)
        pw = np.array(data.site_xpos[pinch_site], dtype=np.float64)
        return base_mat.T @ (pw - base_pos)

    import pyarrow.parquet as pq
    data_dir = dataset_dir / "data" / "chunk-000"
    eps = sorted(int(p.stem.split("_")[-1]) for p in data_dir.glob("episode_*.parquet"))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(eps)
    readers: dict = {}
    collected: list[dict] = []
    diag = []
    for ep in eps:
        if len(collected) >= args.max_states:
            break
        tbl = pq.read_table(data_dir / f"episode_{ep:06d}.parquet")
        state = np.asarray(tbl.column("observation.state").to_pylist(), dtype=np.float32)
        action = np.asarray(tbl.column("action").to_pylist(), dtype=np.float32)
        n = len(state)
        cands = []
        for fr in range(0, n - REAL_STEPS - 1):
            grip = float(state[fr, 6])
            gt = state[fr, 7:10].astype(np.float64)
            d = float(np.linalg.norm(pinch_base(action[fr, :6]) - gt))
            diag.append(d)
            if args.miss_lo <= d <= args.miss_hi and grip < 0.5:
                cands.append((fr, d))
        if not cands:
            continue
        # take up to per_ep_cap spread across the candidate window
        pick_idx = np.linspace(0, len(cands) - 1, min(args.per_ep_cap, len(cands))).astype(int)
        for pi in pick_idx:
            fr, d = cands[pi]
            ext, wrist, s10, _ = load_dataset_frame(dataset_dir, ep, fr, readers)
            corrective = action[fr:fr + REAL_STEPS, :7].astype(np.float32)
            collected.append({
                "exterior": ext, "wrist": wrist, "state10": s10.astype(np.float32),
                "corrective_chunk": corrective, "needed_correction_cm": d * 100.0,
                "episode": ep, "frame": fr,
            })
            print(f"[collect-ds] ep{ep} fr{fr} pinch-to-vial={d*100:.1f}cm "
                  f"(total {len(collected)})", flush=True)
            if len(collected) >= args.max_states:
                break

    dd = np.array(diag)
    print(f"[collect-ds] scanned {len(diag)} frames; pinch-dist cm "
          f"p05={_percentile(dd,5)*100:.1f} p50={_percentile(dd,50)*100:.1f} "
          f"p95={_percentile(dd,95)*100:.1f}; collected {len(collected)}", flush=True)
    if not collected:
        return {"n_miss_states": 0, "note": "no dataset frames in the near-grasp band"}
    np.savez(
        args.miss_out,
        exterior=np.stack([c["exterior"] for c in collected]),
        wrist=np.stack([c["wrist"] for c in collected]),
        state10=np.stack([c["state10"] for c in collected]),
        corrective_chunk=np.stack([c["corrective_chunk"] for c in collected]),
        needed_correction_cm=np.array([c["needed_correction_cm"] for c in collected]),
        episode=np.array([c["episode"] for c in collected]),
        frame=np.array([c["frame"] for c in collected]),
    )
    ncm = np.array([c["needed_correction_cm"] for c in collected])
    return {
        "n_miss_states": len(collected),
        "miss_npz": str(args.miss_out),
        "source": "in_distribution_dataset_near_grasp_band",
        "needed_correction_cm": {"mean": float(ncm.mean()), "min": float(ncm.min()),
                                 "max": float(ncm.max())},
    }


def _expert_corrective_chunk(mj, model, data, idx, relab, arm_act, grip_act, arm_qadr,
                             grip_qadr, decim, GRIP_CLOSE, GRIP_DRIVER_RANGE, n):
    """Roll the scripted IK expert forward ``n`` control ticks FROM the current sim
    state, recording its (absolute joint target, grip) commands. Restores the sim
    + relabeler state afterwards so the outer policy rollout is unperturbed."""
    qpos0 = data.qpos.copy(); qvel0 = data.qvel.copy()
    act0 = data.act.copy() if data.act.size else None
    ctrl0 = data.ctrl.copy(); time0 = data.time
    i0 = relab.i
    chunk = np.zeros((n, 7), dtype=np.float32)
    for k in range(n):
        pinch = np.array(data.site_xpos[idx["pinch_site"]], dtype=np.float64)
        vial = np.array(data.xpos[idx["vial_body"]], dtype=np.float64)
        grip_meas = float(np.clip(data.qpos[grip_qadr] / GRIP_DRIVER_RANGE, 0.0, 1.0))
        tgt_q, tgt_grip = relab.step(pinch, vial, grip_meas)
        chunk[k, :6] = np.asarray(tgt_q, dtype=np.float32)
        chunk[k, 6] = float(tgt_grip)
        for a, q in zip(arm_act, tgt_q):
            data.ctrl[a] = float(q)
        data.ctrl[grip_act] = float(tgt_grip) * GRIP_CLOSE
        for _ in range(decim):
            mj.mj_step(model, data)
    # restore
    data.qpos[:] = qpos0; data.qvel[:] = qvel0
    if act0 is not None:
        data.act[:] = act0
    data.ctrl[:] = ctrl0; data.time = time0
    relab.i = i0
    mj.mj_forward(model, data)
    return chunk


# ===========================================================================
# MODE: merge
# ===========================================================================
def decide_verdict(step0_pass: bool, step1_pass: bool) -> str:
    if not step0_pass:
        return "STEP0_FAIL_DEBUG"
    return "PORT_GO" if step1_pass else "FINETUNE_FIRST"


def mode_merge(args) -> dict:
    step0 = json.loads(Path(args.step0_json).read_text()) if args.step0_json else {}
    auth = json.loads(Path(args.authority_json).read_text()) if args.authority_json else {}
    fk = json.loads(Path(args.fk_json).read_text()) if args.fk_json else {}
    step1 = {
        "n_miss_states": fk.get("n_miss_states", auth.get("n_miss_states", 0)),
        "corrective_recon_mse": auth.get("corrective_recon_mse", {}),
        "ee_spread_cm": fk.get("ee_spread_cm", {}),
        "corrective_covered_within_2cm_frac": fk.get("corrective_covered_within_2cm_frac"),
        "needed_correction_cm": fk.get("needed_correction_cm", {}),
        "pass": bool(fk.get("pass", False)),
    }
    s0pass = bool(step0.get("pass", False))
    s1pass = bool(step1.get("pass", False))
    result = {
        "step0_roundtrip": {k: v for k, v in step0.items() if k != "per_chunk"},
        "step1_authority": {k: v for k, v in step1.items() if k != "per_state"},
        "verdict": decide_verdict(s0pass, s1pass),
        "notes": args.notes or "",
    }
    return result


# ===========================================================================
# CLI
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common_model(p):
        p.add_argument("--model-path", default="/home/ubuntu/ckpt/ur5e_drugsort_obscond/full/checkpoint-12000")
        p.add_argument("--embodiment-tag", default="new_embodiment")
        p.add_argument("--device", default="cuda")
        p.add_argument("--fp-iters", type=int, default=8)
        p.add_argument("--adam-iters", type=int, default=200)
        p.add_argument("--adam-thresh", type=float, default=1e-3)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", default=None, help="write this stage's JSON here")

    p0 = sub.add_parser("roundtrip")
    common_model(p0)
    p0.add_argument("--dataset", default="/home/ubuntu/ur5e_drugsort_obscond")
    p0.add_argument("--n-chunks", type=int, default=16)

    pc = sub.add_parser("collect-miss-states")
    pc.add_argument("--src", default="/home/ubuntu/odyssey-ur5e")
    pc.add_argument("--xml", default="/home/ubuntu/aseptipack_description/aseptipack.xml")
    pc.add_argument("--zmq-host", default="127.0.0.1")
    pc.add_argument("--zmq-port", type=int, default=5558)
    pc.add_argument("--timeout-ms", type=int, default=180000)
    pc.add_argument("--episodes", type=int, default=10)
    pc.add_argument("--max-ticks", type=int, default=140)
    pc.add_argument("--n-action-steps", type=int, default=4)
    pc.add_argument("--fps", type=int, default=20)
    pc.add_argument("--area-x", type=float, default=0.05)
    pc.add_argument("--area-y", type=float, default=0.05)
    pc.add_argument("--yaw-jitter", type=float, default=0.4)
    pc.add_argument("--miss-lo", type=float, default=0.04)
    pc.add_argument("--miss-hi", type=float, default=0.15)
    pc.add_argument("--per-ep-cap", type=int, default=3)
    pc.add_argument("--max-states", type=int, default=20)
    pc.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    pc.add_argument("--base-body", default="base")
    pc.add_argument("--cap-offset-z", type=float, default=0.024)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--miss-out", default="/home/ubuntu/flowdagger_miss_states.npz")
    pc.add_argument("--out", default=None)

    pd = sub.add_parser("collect-dataset-states")
    pd.add_argument("--src", default="/home/ubuntu/odyssey-ur5e")
    pd.add_argument("--xml", default="/home/ubuntu/aseptipack_description/aseptipack.xml")
    pd.add_argument("--dataset", default="/home/ubuntu/ur5e_drugsort_obscond")
    pd.add_argument("--base-body", default="base")
    pd.add_argument("--miss-lo", type=float, default=0.04)
    pd.add_argument("--miss-hi", type=float, default=0.15)
    pd.add_argument("--per-ep-cap", type=int, default=3)
    pd.add_argument("--max-states", type=int, default=18)
    pd.add_argument("--seed", type=int, default=0)
    pd.add_argument("--miss-out", default="/home/ubuntu/flowdagger_miss_states.npz")
    pd.add_argument("--out", default=None)

    pa = sub.add_parser("authority")
    common_model(pa)
    pa.add_argument("--miss-states", default="/home/ubuntu/flowdagger_miss_states.npz")
    pa.add_argument("--n-seeds", type=int, default=64)
    pa.add_argument("--decoded-out", default="/home/ubuntu/flowdagger_authority_decoded.npz")

    pf = sub.add_parser("fk-spread")
    pf.add_argument("--src", default="/home/ubuntu/odyssey-ur5e")
    pf.add_argument("--xml", default="/home/ubuntu/aseptipack_description/aseptipack.xml")
    pf.add_argument("--decoded-npz", default="/home/ubuntu/flowdagger_authority_decoded.npz")
    pf.add_argument("--out", default=None)

    pm = sub.add_parser("merge")
    pm.add_argument("--step0-json", default=None)
    pm.add_argument("--authority-json", default=None)
    pm.add_argument("--fk-json", default=None)
    pm.add_argument("--notes", default="")
    pm.add_argument("--out", default=None)

    args = ap.parse_args()
    if args.mode == "roundtrip":
        out = mode_roundtrip(args)
    elif args.mode == "collect-miss-states":
        out = mode_collect_miss_states(args)
    elif args.mode == "collect-dataset-states":
        out = mode_collect_dataset_states(args)
    elif args.mode == "authority":
        out = mode_authority(args)
    elif args.mode == "fk-spread":
        out = mode_fk_spread(args)
    elif args.mode == "merge":
        out = mode_merge(args)
    else:  # pragma: no cover
        raise SystemExit(f"unknown mode {args.mode}")

    text = json.dumps(out, indent=2)
    if getattr(args, "out", None):
        Path(args.out).write_text(text)
        print(f"[{args.mode}] wrote {args.out}", flush=True)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
