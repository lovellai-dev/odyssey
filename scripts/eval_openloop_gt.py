#!/usr/bin/env python3
"""Open-loop ground-truth eval of a fine-tuned GR00T pilot against the dataset.

This is the *cheap* eval counterpart to ``scripts/eval_ur5e_drugsort_groot.py``
(closed-loop physics success). Instead of rolling the policy out in a simulator,
it replays the **recorded** observations from held-out episodes of the GR00T
LeRobot dataset, asks the served policy for its predicted action chunk at each
sampled tick, and scores it against the **recorded expert action** — i.e. the
ground truth the fine-tune was trained toward.

Metrics per sampled tick (predicted 16-step chunk vs the expert's next 16 steps):
  * per-joint arm MAE (rad) over the chunk horizon
  * gripper open/close agreement (binarised at 0.5) over the horizon
Aggregated (mean) across all sampled ticks and episodes.

Why open-loop: it needs no MuJoCo/browser and no ground-truth physics — only the
dataset (parquet + mp4) and a running GR00T policy server. It correlates with
grasp-phase quality and is the fastest signal that a checkpoint learned the task
shape. It does NOT measure task success (use the closed-loop eval for that).

The GR00T server must already be running (this script does not boot it), same as
the closed-loop eval::

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m gr00t.eval.run_gr00t_server \
        --model-path <ckpt> --embodiment-tag new_embodiment --port 5555 --device cuda

    python scripts/eval_openloop_gt.py \
        --dataset /data/ur10e_drugsort_v0/ur10e_partial_cond_aug \
        --host 127.0.0.1 --port 5555 --held-out 4 --stride 8 --out-json /tmp/openloop.json

Only ``numpy`` is needed to import this module; the CLI additionally uses
``pyarrow`` (parquet), ``imageio`` (mp4) and ``pyzmq``/``msgpack`` (the server
client) — all imported lazily inside ``main`` so the pure metric functions below
stay import-light and unit-testable without those deps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Pure metric functions (numpy-only; unit-tested in test_eval_openloop_gt) ───

ARM_JOINTS = 6


def arm_chunk_mae(pred_arm: np.ndarray, gt_arm: np.ndarray) -> np.ndarray:
    """Per-joint mean-absolute-error between a predicted and an expert arm chunk.

    ``pred_arm`` / ``gt_arm`` are (H, 6) absolute joint targets (rad). The
    comparison is over the overlapping horizon ``min(len(pred), len(gt))`` so a
    truncated tail near the end of an episode is handled cleanly. Returns a
    length-6 array of per-joint MAE (rad); an empty overlap yields all-NaN.
    """
    pred = np.asarray(pred_arm, dtype=np.float64).reshape(-1, ARM_JOINTS)
    gt = np.asarray(gt_arm, dtype=np.float64).reshape(-1, ARM_JOINTS)
    h = min(pred.shape[0], gt.shape[0])
    if h == 0:
        return np.full(ARM_JOINTS, np.nan)
    return np.abs(pred[:h] - gt[:h]).mean(axis=0)


def gripper_agreement(pred_grip: np.ndarray, gt_grip: np.ndarray, thresh: float = 0.5) -> float:
    """Fraction of horizon steps where the binarised gripper command matches.

    Both inputs are closures in [0, 1]; each is thresholded (>= ``thresh`` =>
    closed) and compared step-by-step over the overlapping horizon. Returns NaN
    for an empty overlap.
    """
    pred = (np.asarray(pred_grip, dtype=np.float64).ravel() >= thresh)
    gt = (np.asarray(gt_grip, dtype=np.float64).ravel() >= thresh)
    h = min(pred.size, gt.size)
    if h == 0:
        return float("nan")
    return float((pred[:h] == gt[:h]).mean())


def aggregate(per_joint_maes: list[np.ndarray], grip_accs: list[float]) -> dict:
    """Reduce per-tick records to a run summary (nan-safe means)."""
    maes = np.asarray(per_joint_maes, dtype=np.float64) if per_joint_maes else np.empty((0, ARM_JOINTS))
    accs = np.asarray(grip_accs, dtype=np.float64) if grip_accs else np.empty(0)
    per_joint = np.nanmean(maes, axis=0).tolist() if maes.size else [float("nan")] * ARM_JOINTS
    return {
        "n_ticks": int(maes.shape[0]),
        "arm_mae_per_joint_rad": per_joint,
        "arm_mae_overall_rad": float(np.nanmean(maes)) if maes.size else float("nan"),
        "gripper_agreement": float(np.nanmean(accs)) if accs.size else float("nan"),
    }


# ── Dataset IO + eval loop (heavy deps imported lazily) ────────────────────────

def _read_episode(dataset: Path, ep: int):
    """Return (state[N,10], action[N,7], {video_key: frames[N,H,W,3]}) for one episode."""
    import imageio.v3 as iio  # lazy: mp4 decode (imageio[ffmpeg])
    import pyarrow.parquet as pq  # lazy: parquet

    pqf = dataset / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    t = pq.read_table(pqf)
    state = np.asarray(t.column("observation.state").to_pylist(), dtype=np.float32)
    action = np.asarray(t.column("action").to_pylist(), dtype=np.float32)
    frames: dict[str, np.ndarray] = {}
    for key in ("exterior", "wrist"):
        mp4 = dataset / "videos" / "chunk-000" / f"observation.images.{key}" / f"episode_{ep:06d}.mp4"
        if mp4.is_file():
            frames[key] = np.asarray(iio.imread(mp4, plugin="pyav"))  # (N,H,W,3) uint8
    return state, action, frames


def _build_obs(state_t: np.ndarray, frames_t: dict[str, np.ndarray], instruction: str) -> dict:
    """Nested GR00T observation for one tick (B=1,T=1) — mirrors eval_ur5e build_obs.

    The recorded 10-D state is [arm6, gripper, grasp_target3]; feed the arm +
    gripper as proprio and the grasp_target as its own state modality (present
    only when the state is 10-D, matching the obscond modality config).
    """
    st = np.asarray(state_t, dtype=np.float32)
    state = {
        "single_arm": st[:6][None, None],                       # (1,1,6)
        "gripper": np.asarray([[[st[6]]]], dtype=np.float32),   # (1,1,1)
    }
    if st.shape[0] >= 10:
        state["grasp_target"] = st[7:10].reshape(1, 1, 3).astype(np.float32)
    return {
        "video": {k: f[None, None].astype(np.uint8) for k, f in frames_t.items()},
        "state": state,
        "language": {"annotation.human.task_description": [[instruction]]},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="path to the LeRobot dataset dir (…/ur10e_partial_cond_aug)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="explicit episode indices to eval (overrides --held-out)")
    ap.add_argument("--held-out", type=int, default=4,
                    help="if --episodes is unset, eval the LAST N episodes (fit-eval "
                         "caveat: they are in the train split unless you excluded them "
                         "from training — treat as a sanity signal, not generalization)")
    ap.add_argument("--stride", type=int, default=8,
                    help="sample every Nth tick (8 matches the served n_action_steps)")
    ap.add_argument("--horizon", type=int, default=16, help="action-chunk horizon to compare")
    ap.add_argument("--instruction", default="pick up the vial and place it in the rack")
    ap.add_argument("--timeout-ms", type=int, default=180000)
    ap.add_argument("--out-json", default="", help="write the run summary JSON here")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    if not (dataset / "meta" / "info.json").is_file():
        print(f"ERROR: not a LeRobot dataset dir: {dataset}", file=sys.stderr)
        return 2
    info = json.loads((dataset / "meta" / "info.json").read_text())
    total = int(info["total_episodes"])
    eps = args.episodes if args.episodes is not None else list(range(max(0, total - args.held_out), total))
    print(f"[openloop] dataset={dataset.name} total_eps={total} eval_eps={eps} "
          f"stride={args.stride} horizon={args.horizon}")

    # Lazy client import (pyzmq/msgpack) — reuse the exact wire protocol as the
    # closed-loop eval so no torch/gr00t is needed in this process.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_ur5e_drugsort_groot import GrootPolicyClient  # noqa: E402

    client = GrootPolicyClient(args.host, args.port, timeout_ms=args.timeout_ms)
    if not client.ping():
        print("[openloop] WARNING: server ping failed; is the GR00T server up?", file=sys.stderr)

    all_maes: list[np.ndarray] = []
    all_accs: list[float] = []
    per_ep: list[dict] = []
    for ep in eps:
        state, action, frames = _read_episode(dataset, ep)
        n = state.shape[0]
        ep_maes: list[np.ndarray] = []
        ep_accs: list[float] = []
        for t in range(0, n - 1, args.stride):
            frames_t = {k: v[t] for k, v in frames.items() if t < v.shape[0]}
            obs = _build_obs(state[t], frames_t, args.instruction)
            act = client.get_action(obs)
            pred_arm = np.asarray(act["single_arm"], dtype=np.float32).reshape(-1, ARM_JOINTS)
            pred_grip = np.asarray(act["gripper"], dtype=np.float32).reshape(-1)
            gt = action[t : t + args.horizon]
            ep_maes.append(arm_chunk_mae(pred_arm[: args.horizon], gt[:, :6]))
            ep_accs.append(gripper_agreement(pred_grip[: args.horizon], gt[:, 6]))
        ep_summary = aggregate(ep_maes, ep_accs)
        ep_summary["episode"] = ep
        per_ep.append(ep_summary)
        all_maes.extend(ep_maes)
        all_accs.extend(ep_accs)
        print(f"[openloop] ep {ep:03d}: ticks={ep_summary['n_ticks']:3d} "
              f"arm_MAE={ep_summary['arm_mae_overall_rad']:.4f} rad "
              f"grip_agree={ep_summary['gripper_agreement']*100:5.1f}%")

    summary = aggregate(all_maes, all_accs)
    summary["episodes"] = per_ep
    summary["dataset"] = str(dataset)
    print("\n[openloop] ===== OPEN-LOOP GT SUMMARY =====")
    print(f"[openloop] ticks: {summary['n_ticks']}")
    print(f"[openloop] arm MAE per joint (rad): "
          f"[{', '.join(f'{v:.4f}' for v in summary['arm_mae_per_joint_rad'])}]")
    print(f"[openloop] arm MAE overall: {summary['arm_mae_overall_rad']:.4f} rad")
    print(f"[openloop] gripper agreement: {summary['gripper_agreement']*100:.1f}%")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[openloop] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
