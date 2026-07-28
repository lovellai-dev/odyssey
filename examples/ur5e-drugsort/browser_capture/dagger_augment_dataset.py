#!/usr/bin/env python3
"""Visual data augmentation for a LeRobot-format UR5e browser dataset.

Produces an augmented COPY of every episode (videos transformed, parquet/state/
actions untouched) as a valid sibling dataset, ready for
`dagger_relabel_assemble.py merge --base <src> --add <out>`.

Augmentations target the sim-real gap and are temporally coherent per episode
(params sampled once per episode per camera from a seeded RNG):
  - hue shift, saturation scale, per-channel white-balance gain (colour temp)
  - gamma + gain (exposure / lighting level change)
  - directional lighting gradient (uneven room light)
  - slow sinusoidal brightness flicker (light change over the episode)
  - motion blur (per-frame probabilistic, random direction)
  - optional mild defocus blur (episode-level)
  - gaussian sensor noise (per-frame, temporally white)

Usage:
  python dagger_augment_dataset.py --src ~/ur5e_dagger_browser \
      --out ~/ur5e_dagger_browser_augonly --seed 20260727 --workers 12
"""
import argparse
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_KEYS = ["observation.images.exterior", "observation.images.wrist"]


def sample_params(rng: np.random.Generator) -> dict:
    """One camera-view's episode-coherent augmentation parameters."""
    return {
        "hue_shift": float(rng.uniform(-6, 6)),            # OpenCV hue units (2 deg each)
        "sat_scale": float(rng.uniform(0.7, 1.3)),
        "wb_gain": rng.uniform(0.92, 1.08, size=3).tolist(),  # per-channel (BGR)
        "gamma": float(10 ** rng.uniform(-0.14, 0.14)),    # ~[0.72, 1.38]
        "gain": float(rng.uniform(0.85, 1.15)),
        "grad_amp": float(rng.uniform(0.08, 0.28)),        # lighting gradient strength
        "grad_angle": float(rng.uniform(0, 2 * np.pi)),
        "flicker_amp": float(rng.uniform(0.0, 0.06)),
        "flicker_period": float(rng.uniform(40, 120)),     # frames
        "flicker_phase": float(rng.uniform(0, 2 * np.pi)),
        "mblur_p": float(rng.uniform(0.15, 0.35)),         # per-frame probability
        "mblur_len": int(rng.integers(3, 8)),              # kernel length px
        "defocus_sigma": float(rng.uniform(0.5, 1.0)) if rng.random() < 0.15 else 0.0,
        "noise_sigma": float(rng.uniform(1.5, 5.0)),
        "frame_seed": int(rng.integers(0, 2**31 - 1)),     # for per-frame randomness
    }


def build_static_maps(p: dict, h: int, w: int):
    """Precompute per-episode lookup tables and the lighting-gradient field."""
    lut = np.arange(256, dtype=np.float64) / 255.0
    lut = np.clip((lut ** p["gamma"]) * p["gain"], 0, 1)
    gamma_lut = (lut * 255.0).astype(np.uint8)  # single-channel gamma+gain LUT

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = (xx / w - 0.5) * np.cos(p["grad_angle"]) + (yy / h - 0.5) * np.sin(p["grad_angle"])
    grad = (1.0 + p["grad_amp"] * 2.0 * proj)[..., None]  # multiplicative field around 1.0

    wb = np.asarray(p["wb_gain"], dtype=np.float32).reshape(1, 1, 3)
    return gamma_lut, grad.astype(np.float32), wb


def motion_blur_kernel(length: int, angle_deg: float) -> np.ndarray:
    k = np.zeros((length, length), dtype=np.float32)
    c = (length - 1) / 2
    a = np.deg2rad(angle_deg)
    for i in range(length):
        t = i - c
        x = int(round(c + t * np.cos(a)))
        y = int(round(c + t * np.sin(a)))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    s = k.sum()
    return k / s if s > 0 else k


def augment_frame(img: np.ndarray, p: dict, gamma_lut, grad, wb,
                  t: int, frng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + p["hue_shift"]) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * p["sat_scale"], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    img = cv2.LUT(img, gamma_lut)

    flick = 1.0 + p["flicker_amp"] * np.sin(
        2 * np.pi * t / p["flicker_period"] + p["flicker_phase"])
    f = img.astype(np.float32) * grad * wb * flick
    img = np.clip(f, 0, 255).astype(np.uint8)

    if p["defocus_sigma"] > 0:
        img = cv2.GaussianBlur(img, (0, 0), p["defocus_sigma"])

    if frng.random() < p["mblur_p"]:
        k = motion_blur_kernel(p["mblur_len"], float(frng.uniform(0, 180)))
        img = cv2.filter2D(img, -1, k)

    noise = frng.normal(0, p["noise_sigma"], img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def process_video(src_mp4: Path, dst_mp4: Path, p: dict, fps: int) -> int:
    cap = cv2.VideoCapture(str(src_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src_mp4}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    gamma_lut, grad, wb = build_static_maps(p, h, w)
    frng = np.random.default_rng(p["frame_seed"])

    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
         "-movflags", "+faststart", str(dst_mp4)],
        stdin=subprocess.PIPE)
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = augment_frame(frame, p, gamma_lut, grad, wb, n, frng)
            enc.stdin.write(out.tobytes())
            n += 1
    finally:
        cap.release()
        enc.stdin.close()
        rc = enc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg rc={rc} for {dst_mp4}")
    return n


def process_episode(job):
    src, out, ep_idx, seed, fps = job
    counts = {}
    for key in VIDEO_KEYS:
        # independent per-camera params, deterministic per (seed, episode, camera)
        rng = np.random.default_rng([seed, ep_idx, VIDEO_KEYS.index(key)])
        p = sample_params(rng)
        s = src / "videos" / "chunk-000" / key / f"episode_{ep_idx:06d}.mp4"
        d = out / "videos" / "chunk-000" / key / f"episode_{ep_idx:06d}.mp4"
        counts[key] = process_video(s, d, p, fps)
    return ep_idx, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="only first N episodes (smoke)")
    args = ap.parse_args()

    src, out = Path(args.src).expanduser(), Path(args.out).expanduser()
    info = json.load((src / "meta" / "info.json").open())
    fps = int(info["fps"])
    n_eps = int(info["total_episodes"])
    if args.limit:
        n_eps = min(n_eps, args.limit)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # copy everything except videos (parquet/meta/gt identical — augmentation is visual-only)
    for sub in ("data", "meta", "gt"):
        if (src / sub).exists():
            shutil.copytree(src / sub, out / sub)

    if args.limit:
        # trim meta so the smoke output is itself a consistent dataset
        eps = [json.loads(l) for l in (out / "meta" / "episodes.jsonl").open()][:n_eps]
        with (out / "meta" / "episodes.jsonl").open("w") as f:
            for e in eps:
                f.write(json.dumps(e) + "\n")
        info["total_episodes"] = n_eps
        info["total_frames"] = sum(int(e["length"]) for e in eps)
        info["total_videos"] = n_eps * len(VIDEO_KEYS)
        info["splits"] = {"train": f"0:{n_eps}"}
        (out / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
        for pqf in sorted((out / "data" / "chunk-000").glob("episode_*.parquet")):
            if int(pqf.stem.split("_")[1]) >= n_eps:
                pqf.unlink()

    jobs = [(src, out, i, args.seed, fps) for i in range(n_eps)]
    done = 0
    with mp.Pool(args.workers) as pool:
        for ep_idx, counts in pool.imap_unordered(process_episode, jobs):
            done += 1
            print(f"AUG ep{ep_idx:03d} done {counts} [{done}/{n_eps}]", flush=True)
    print(f"AUG_DATASET_DONE eps={n_eps} out={out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
