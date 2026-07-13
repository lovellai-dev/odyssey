#!/usr/bin/env python3
"""Micro-benchmark for OpenVLA Pilot inference latency.

Isolates VLARuntime.act() from sim/env overhead so you can A/B the same
checkpoint on two branches (e.g. develop vs feat/openvla-inference-latency)
and see whether the FlashAttention/SDPA + inference_mode change moved the
per-step latency.

Run on the GPU VM (latency is hardware-specific), same checkpoint both times:

    # baseline
    git checkout develop
    python bench_openvla_latency.py --checkpoint /path/to/ckpt

    # candidate
    git checkout feat/openvla-inference-latency
    python bench_openvla_latency.py --checkpoint /path/to/ckpt

Compare the reported median ms/action. This script is a throwaway harness —
not committed to the PR unless you decide it's worth keeping.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from odyssey.runners.models.openvla import VLARuntime


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="LoRA adapter or merged model dir")
    p.add_argument("--unnorm-key", default="bridge_orig")
    p.add_argument("--warmup", type=int, default=10, help="calls to discard first")
    p.add_argument("--iters", type=int, default=100, help="timed calls")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--instruction", default="pick up the object")
    args = p.parse_args()

    import torch

    runtime = VLARuntime(args.checkpoint, unnorm_key=args.unnorm_key)
    device = runtime._device  # read-only private access, benchmark only
    is_cuda = device.startswith("cuda")
    print(f"device={device}  iters={args.iters}  warmup={args.warmup}")
    print("(check the load log above for the actual attn_implementation used)")

    # Fixed synthetic frame — content is irrelevant for latency, only shape/dtype.
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(args.height, args.width, 3), dtype=np.uint8)

    def sync() -> None:
        if is_cuda:
            torch.cuda.synchronize()

    # Warm-up: absorbs lazy CUDA init, kernel autotuning, caches.
    for _ in range(args.warmup):
        runtime.act(image, args.instruction)
    sync()

    # Timed loop — synchronize inside each iteration so we measure real
    # GPU execution, not async kernel launches.
    samples_ms: list[float] = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        runtime.act(image, args.instruction)
        sync()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    samples_ms.sort()
    median = statistics.median(samples_ms)
    p90 = samples_ms[int(0.9 * len(samples_ms)) - 1]
    mean = statistics.fmean(samples_ms)
    print("\n--- per-action latency (ms) ---")
    print(f"median : {median:7.2f}")
    print(f"p90    : {p90:7.2f}")
    print(f"mean   : {mean:7.2f}")
    print(f"min    : {samples_ms[0]:7.2f}")
    print(f"max    : {samples_ms[-1]:7.2f}")
    print(f"\nthroughput (median): {1000.0 / median:6.2f} actions/s")


if __name__ == "__main__":
    main()
