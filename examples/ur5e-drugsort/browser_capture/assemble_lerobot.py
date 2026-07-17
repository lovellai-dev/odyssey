#!/usr/bin/env python3
"""Assemble the browser-captured raw episodes into a GR00T LeRobot v2.1 dataset.

Final stage of the browser-frame harness. Reads ``<raw>/epNNN/{meta.json,
exterior/*.png, wrist/*.png}`` produced by ``browser_harness.js`` (frames rendered
by the Playground's REAL Three.js deployment renderer) and writes them through the
EXISTING, unchanged ``odyssey.datasets.LeRobotDatasetWriter`` +
``ur5e_drugsort.embodiment`` — so the on-disk schema is byte-identical to the
headless MuJoCo dataset that already feeds ``gr00t.experiment.launch_finetune`` +
the Observer training. The ONLY difference is the *pixels*: Three.js browser frames
instead of MuJoCo frames. Retraining GR00T + the Observer on this dataset closes
the render gap.

Additionally writes a ground-truth sidecar under ``<out>/meta/gt/episode_XXXXXX.json``
(per-frame vial + nest pose from the MuJoCo-WASM state + episode success/lift/seat)
for the Observer's grasp-target labels. The sidecar lives OUTSIDE the files the
GR00T loader reads, so the core dataset still loads unchanged.

Usage::

    python assemble_lerobot.py --raw ./browser_capture_out/raw --out ./dataset_browser
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
from odyssey.datasets import LeRobotDatasetWriter  # noqa: E402
from odyssey.embodiments.ur5e_drugsort import embodiment as emb  # noqa: E402


def load_frames(d: Path) -> list[np.ndarray]:
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in sorted(d.glob("f*.png"))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="browser_capture_out/raw", help="dir of epNNN/ raw episodes")
    ap.add_argument("--out", default="dataset_browser", help="output LeRobot dataset dir")
    # Base-fix: with trajectory-shape domain randomization a fraction of expert
    # demos miss (grasp/place fails under the jittered shape). Those demos teach
    # the policy the WRONG action for the scene, so keep only SUCCESSFUL episodes
    # by default. --keep-failures restores the old (unfiltered) behaviour.
    ap.add_argument("--success-only", dest="success_only", action="store_true", default=True,
                    help="keep only episodes whose expert grasp+place SUCCEEDED (default)")
    ap.add_argument("--keep-failures", dest="success_only", action="store_false",
                    help="assemble ALL episodes regardless of expert success")
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    ep_dirs = sorted([p for p in raw.iterdir() if p.is_dir() and p.name.startswith("ep")])
    if not ep_dirs:
        print(f"ERROR: no raw episodes under {raw}", file=sys.stderr)
        return 2

    writer = LeRobotDatasetWriter(
        out, modality=emb.modality_json(), info_builder=emb.info_json,
        video_keys=emb.VIDEO_ORIGINAL_KEYS, fps=emb.DEFAULT_FPS,
        width=emb.DEFAULT_WIDTH, height=emb.DEFAULT_HEIGHT,
    )
    gt_dir = out / "meta" / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    n_success = 0
    n_dropped = 0
    for d in ep_dirs:
        meta = json.loads((d / "meta.json").read_text())
        # SUCCESS FILTER (base-fix): a domain-randomized demo that missed teaches
        # the wrong scene->action mapping. Drop it before it reaches the dataset.
        if args.success_only and not meta.get("success"):
            n_dropped += 1
            print(f"[assemble] DROP {d.name}: expert MISS "
                  f"(lifted={meta.get('lifted')} seated={meta.get('seated')} "
                  f"lift={meta.get('lift_height', 0)*100:.1f}cm place={meta.get('place_dist', 0)*100:.1f}cm)")
            continue
        states = np.asarray(meta["states"], dtype=np.float32)
        actions = np.asarray(meta["actions"], dtype=np.float32)
        ext = load_frames(d / "exterior")
        wrist = load_frames(d / "wrist")
        T = states.shape[0]
        assert len(ext) == T and len(wrist) == T and actions.shape[0] == T, (
            f"{d.name}: length mismatch states={T} act={actions.shape[0]} ext={len(ext)} wrist={len(wrist)}"
        )
        assert states.shape[1] == emb.STATE_DIM and actions.shape[1] == emb.ACTION_DIM
        assert ext[0].shape == (emb.DEFAULT_HEIGHT, emb.DEFAULT_WIDTH, 3)
        idx = writer.add_episode(
            states=states, actions=actions,
            frames={"exterior": ext, "wrist": wrist}, task=emb.DEFAULT_INSTRUCTION,
        )
        (gt_dir / f"episode_{idx:06d}.json").write_text(json.dumps({
            "episode_index": idx,
            "success": meta["success"], "lifted": meta["lifted"], "seated": meta["seated"],
            "lift_height_m": meta["lift_height"], "place_dist_m": meta["place_dist"],
            "vial_qpos_init": meta["vial_qpos"], "ik_max_err_mm": meta["ik_max_err_mm"],
            "frames": [{"vial_pose": g["vial"], "pocket_pos": g["pocket"], "phase": g["phase"]} for g in meta["gt"]],
        }))
        n_success += int(meta["success"])
        print(f"[assemble] ep {idx:03d}: T={T} success={meta['success']} "
              f"lift={meta['lift_height']*100:.1f}cm place={meta['place_dist']*100:.1f}cm")

    info = writer.finalize()
    n_kept = len(ep_dirs) - n_dropped
    rate = (n_kept / len(ep_dirs)) if ep_dirs else 0.0
    print(f"[assemble] wrote {info['total_episodes']} episodes / {info['total_frames']} frames -> {out}")
    print(f"[assemble] grasp+place success (browser physics): {n_success}/{n_kept} kept episodes")
    if args.success_only:
        print(f"[assemble] SUCCESS-FILTER: kept {n_kept}/{len(ep_dirs)} "
              f"(dropped {n_dropped} misses) -> filter_rate={rate:.3f}")
    print(f"[assemble] video keys: {list(emb.VIDEO_ORIGINAL_KEYS.values())}")
    # Machine-readable summary for the driver (success-filter rate).
    (out / "assemble_summary.json").write_text(json.dumps({
        "raw_episodes": len(ep_dirs), "kept": n_kept, "dropped": n_dropped,
        "success_filter_rate": rate, "success_only": bool(args.success_only),
    }) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
