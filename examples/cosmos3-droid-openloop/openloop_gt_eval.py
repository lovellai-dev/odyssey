#!/usr/bin/env python3
"""Open-loop ground-truth eval for a Cosmos 3 policy — `evaluation_type: custom`.

The cheapest IN-DISTRIBUTION check of the whole serving + decode path, no
simulator involved: feed the policy server real frames from a DROID LeRobot
episode and compare the predicted action chunk against the actions the robot
actually took (the dataset's ground truth). This is the motivating "open-loop
GT eval" of ``CustomEvalRunner``'s contract (``runners/evals/custom.py``):
metric-only — per-dim MAE + gripper agreement — with NO fabricated
success_rate.

Why it matters for the Cosmos3 integration: the LIBERO smoke validated the
*wiring* out-of-distribution (DROID checkpoint, LIBERO scenes → 0% expected);
this eval validates the *action semantics* in-distribution (right dims, right
scales, sane gripper) — the marked de-risk item before RoboLab / a LIBERO SFT.

Contract (launched by ``CustomEvalRunner``)::

    <eval_python> openloop_gt_eval.py --checkpoint <id> --out-json <path> \
        --dataset_root <lerobot-dir> [--host --port --num_samples ...]

The policy server is EXTERNALLY-SERVED (``action_policy_server_libero``,
HTTP): this script only needs its host:port. Dataset: any LeRobot v3 dir with
DROID-style features (``action.joint_position`` [7] + ``action.gripper_position``
[1], exterior + wrist mp4 videos) — e.g. the tiny 1-episode
``droid_lerobot_example`` shipped in NVIDIA/cosmos cookbooks (see README.md).

Baselines included so the MAE has context: ``zero`` (predict all-zeros) and
``hold`` (persist the last observed action across the horizon). A real policy
should beat both; matching ``zero`` on the joint dims would suggest the decode
or prompt format is wrong, not the model.

Deps beyond the odyssey client venv: pandas + pyarrow (parquet) and opencv
(frame seeks) — see README.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Open-loop GT eval for Cosmos3 policies.")
    # --- CustomEvalRunner contract ---
    ap.add_argument("--checkpoint", required=True,
                    help="Recorded in metrics; the policy SERVER holds the weights.")
    ap.add_argument("--out-json", required=True, dest="out_json")
    # --- data + server (config passthrough) ---
    ap.add_argument("--dataset_root", required=True,
                    help="LeRobot v3 dataset dir (meta/ + data/ + videos/).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--num_samples", type=int, default=8,
                    help="Query points sampled evenly inside the episode(s).")
    ap.add_argument("--horizon", type=int, default=16,
                    help="GT steps compared per query (capped by the returned chunk).")
    ap.add_argument("--image_size", type=int, default=480,
                    help="Server-side target height (DROID recipe trains 480p).")
    # Must be one of cosmos-framework's registered embodiments (get_domain_id):
    # droid_lerobot, libero, bridge_orig_lerobot, fractal, agibotworld, umi, ...
    # An unknown value makes the server answer an EMPTY action chunk (H100 finding).
    ap.add_argument("--domain_name", default="droid_lerobot")
    ap.add_argument("--exterior_key", default="observation.image.exterior_image_1_left")
    ap.add_argument("--wrist_key", default="observation.image.wrist_image_left")
    ap.add_argument("--instruction", default="",
                    help="Prompt override; default is the dataset's own task string.")
    ap.add_argument("--episode", type=int, default=0,
                    help="Episode index (LeRobot v2.x per-episode files).")
    ap.add_argument("--gripper_threshold", type=float, default=0.5,
                    help="Binarization threshold for gripper agreement.")
    return ap


def _fill(template: str, **kwargs) -> str:
    """Format a LeRobot path template, tolerating v2 vs v3 placeholder names."""
    return template.format(**kwargs)


def _data_path(root: Path, info: dict, episode: int) -> Path:
    return root / _fill(info["data_path"], chunk_index=0, file_index=0,
                        episode_chunk=0, episode_index=episode)


def _video_path(root: Path, info: dict, key: str, episode: int) -> Path:
    return root / _fill(info["video_path"], video_key=key, chunk_index=0,
                        file_index=0, episode_chunk=0, episode_index=episode)


def _load_gt_actions(df):
    """GT actions as (T, D): a single ``action`` column (generic LeRobot, e.g.
    the UR drugsort sets) or DROID's split joint/gripper columns."""
    import numpy as np
    if "action" in df.columns:
        return np.stack(df["action"].to_numpy()).astype(np.float64)
    joints = np.stack(df["action.joint_position"].to_numpy())
    gripper = np.stack(df["action.gripper_position"].to_numpy()).reshape(-1, 1)
    return np.concatenate([joints, gripper], axis=1).astype(np.float64)


def _read_frame(cap, index: int):
    import cv2
    import numpy as np
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame_bgr = cap.read()
    if not ok:
        raise RuntimeError(f"could not read frame {index}")
    return np.ascontiguousarray(frame_bgr[:, :, ::-1])  # BGR -> RGB


def _load_task_string(root: Path) -> str:
    """The dataset's task string: v3 keeps it as the tasks.parquet index; v2.x
    ships meta/tasks.jsonl. Empty/separator-only annotations (e.g. " |  | " in
    the DROID cookbook sample) fall back to a neutral instruction.
    """
    task = ""
    parquet = root / "meta" / "tasks.parquet"
    jsonl = root / "meta" / "tasks.jsonl"
    if parquet.is_file():
        import pandas as pd
        tasks = pd.read_parquet(parquet)
        task = str(tasks.index[0]) if len(tasks) else ""
    elif jsonl.is_file():
        first = jsonl.read_text().strip().splitlines()
        task = str(json.loads(first[0]).get("task", "")) if first else ""
    if not task.replace("|", "").strip():
        return "complete the manipulation task"
    return task


def run_eval(args: argparse.Namespace) -> dict:
    import cv2
    import numpy as np
    import pandas as pd

    from odyssey.runners.evals.cosmos3_transforms import (
        build_cosmos3_predict_request,
        cosmos3_chunk_from_response,
    )
    from odyssey.runners.models.cosmos3 import Cosmos3HttpClient

    root = Path(args.dataset_root).expanduser()
    info = json.loads((root / "meta" / "info.json").read_text())
    df = pd.read_parquet(_data_path(root, info, args.episode))

    gt = _load_gt_actions(df)  # (T, D): 8 for DROID (7+gripper), 7 for UR (6+gripper)
    action_dim = gt.shape[1]
    total = gt.shape[0]

    instruction = args.instruction or _load_task_string(root)
    print(f"[openloop] episode={args.episode} frames={total} action_dim={action_dim} "
          f"instruction={instruction!r}", flush=True)

    cap_ext = cv2.VideoCapture(str(_video_path(root, info, args.exterior_key, args.episode)))
    cap_wrist = cv2.VideoCapture(str(_video_path(root, info, args.wrist_key, args.episode)))
    client = Cosmos3HttpClient(host=args.host, port=args.port)

    # Sample query points evenly, leaving room for the comparison horizon.
    horizon = int(args.horizon)
    last_start = total - horizon - 1
    if last_start < 1:
        raise SystemExit(f"episode too short ({total}) for horizon {horizon}")
    starts = np.linspace(0, last_start, num=int(args.num_samples), dtype=int).tolist()

    pred_errs, zero_errs, hold_errs = [], [], []
    grip_hits = grip_total = 0
    used_horizons = []
    per_dim_abs = None
    pred_width = None

    for t in starts:
        request = build_cosmos3_predict_request(
            image=_read_frame(cap_ext, t),
            wrist_image=_read_frame(cap_wrist, t),
            instruction=instruction,
            domain_name=args.domain_name,
            image_size=args.image_size,
        )
        chunk = cosmos3_chunk_from_response(client.infer(request))
        if chunk.size == 0:
            raise SystemExit(
                "server answered an EMPTY action chunk — usually a rejected request "
                "(e.g. unknown --domain_name); check the policy server log."
            )
        h = min(horizon, chunk.shape[0])
        used_horizons.append(h)
        # The server's row width is embodiment-defined and may differ from the
        # dataset's (H100 finding: droid_lerobot answers 7 joint dims, no
        # gripper column, vs the dataset's 7+1). Compare the common prefix.
        pred_width = chunk.shape[1] if pred_width is None else pred_width
        width = min(chunk.shape[1], action_dim)
        if per_dim_abs is None:
            per_dim_abs = np.zeros(width)
        pred = chunk[:h, :width]
        truth = gt[t: t + h, :width]
        hold = np.repeat(gt[max(t - 1, 0)][None, :width], h, axis=0)

        err = np.abs(pred - truth)
        pred_errs.append(err.mean())
        zero_errs.append(np.abs(truth).mean())
        hold_errs.append(np.abs(hold - truth).mean())
        per_dim_abs += err.mean(axis=0)

        if width >= action_dim:  # the prediction covers the gripper column
            thr = args.gripper_threshold
            grip_hits += int(
                ((pred[:, action_dim - 1] > thr) == (truth[:, action_dim - 1] > thr)).sum()
            )
            grip_total += h
        print(f"[openloop] t={t:4d} h={h} width={width} mae={err.mean():.4f} "
              f"(zero={zero_errs[-1]:.4f} hold={hold_errs[-1]:.4f})", flush=True)

    cap_ext.release()
    cap_wrist.release()

    n = len(starts)
    payload = {
        "num_episodes": n,  # per the out-json contract: samples stand in for episodes
        "metrics": {
            "mae": round(float(np.mean(pred_errs)), 5),
            "baseline_zero_mae": round(float(np.mean(zero_errs)), 5),
            "baseline_hold_mae": round(float(np.mean(hold_errs)), 5),
            "mae_per_dim": [round(float(v / n), 5) for v in per_dim_abs],
            "gripper_agreement": (
                round(grip_hits / grip_total, 4) if grip_total else None
            ),
            "action_dim": action_dim,
            "pred_action_width": pred_width,
            "horizon": horizon,
            "chunk_horizons_used": sorted(set(used_horizons)),
            "num_samples": n,
            "episode": args.episode,
            "instruction": instruction,
            "checkpoint": args.checkpoint,
            "dataset_root": str(root),
        },
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"[openloop] wrote {args.out_json}", flush=True)
    return payload


def main() -> None:
    run_eval(build_parser().parse_args())


if __name__ == "__main__":
    main()
