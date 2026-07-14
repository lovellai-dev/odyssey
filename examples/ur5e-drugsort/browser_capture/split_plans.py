#!/usr/bin/env python3
"""Split a precomputed plans.json into N shards for PARALLEL browser data-gen.

Each shard keeps the ORIGINAL episode ids (so per-episode raw dirs never collide
across instances) plus the full meta wrapper, so ``browser_harness.js`` can consume
it unchanged. Used by ``gen_parallel.sh`` to run several headless Chrome instances
concurrently (near-linear speedup) — the single-tab harness is ~90 s/episode, so a
full ~300-episode set is ~7.5 h single-tab but ~1.5 h across 5 instances.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="plans_full.json")
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--prefix", default="shard")
    args = ap.parse_args()

    meta = json.loads(Path(args.plans).read_text())
    plans = meta["plans"]
    n = len(plans)
    per = math.ceil(n / args.shards)
    written = []
    for k in range(args.shards):
        chunk = plans[k * per:(k + 1) * per]
        if not chunk:
            continue
        out = dict(meta)
        out["plans"] = chunk
        out["shard_index"] = k
        out["shard_episodes"] = [chunk[0]["episode"], chunk[-1]["episode"]]
        p = f"{args.prefix}_{k}.json"
        Path(p).write_text(json.dumps(out) + "\n")
        written.append((p, len(chunk), chunk[0]["episode"], chunk[-1]["episode"]))
        print(f"[split] {p}: {len(chunk)} eps (ep{chunk[0]['episode']}..ep{chunk[-1]['episode']})")
    print(f"[split] total {n} eps -> {len(written)} shards (<= {per}/shard)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
