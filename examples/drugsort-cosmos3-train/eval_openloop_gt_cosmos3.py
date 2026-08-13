#!/usr/bin/env python3
"""Open-loop ground-truth eval of a fine-tuned **Cosmos 3** drug-sort pilot.

The Cosmos-3 counterpart to ``scripts/eval_openloop_gt.py`` (which targets a
GR00T zmq server in joint space). Same idea — replay recorded observations from
held-out episodes, ask the policy for its predicted action chunk, score it
against the expert ground truth — but two things differ because Cosmos 3 is a
different animal:

1. **Wire protocol.** Cosmos 3 serves over cosmos-framework's HTTP
   ``action_policy_server_libero`` (base64-PNG ``concat_view`` request, JSON
   ``/predict`` -> ``{"action": [[...], ...]}``), NOT GR00T's zmq/msgpack. We
   reuse Odyssey's own client glue (``Cosmos3HttpClient`` +
   ``build_cosmos3_predict_request`` + ``cosmos3_chunk_from_response``).

2. **Action space.** The drug-sort SFT recipe (``action_policy_drugsort_nano``,
   ``DrugsortURDataset``) emits 10-D ``frame_wise_relative`` cartesian rows
   ``[dpos(3), rot6d(6), gripper(1)]`` — EE-space deltas obtained by FK on the
   arm joints. The recorded expert ``action`` is 7-D **absolute joint targets**
   ``[6 arm joints, gripper]``. So to compare like-with-like we run the SAME
   transform the dataset uses at train time on the GT joint chunk (MuJoCo FK ->
   ``pose_abs_to_rel(rot6d, backward_framewise)`` + gripper invert) and compare
   the two in EE-delta space:
     * translation MAE (metres) over the dpos block
     * rotation geodesic error (degrees) from the rot6d block
     * gripper open/close agreement (binarised at 0.5)

Why open-loop: needs no MuJoCo *rollout*, no browser, no physics GT — only the
dataset (parquet + mp4) and a running policy server. It's the fastest signal
that a checkpoint learned the task shape; it does NOT measure task success (use
the closed-loop harness for that).

Serving: with ``--serve`` (the default) this script BOOTS the cosmos-framework
``action_policy_server_libero`` on ``--checkpoint``, waits for ``GET /info``,
runs, then tears it down — so it satisfies Odyssey's ``custom`` eval-runner
contract (``--checkpoint`` / ``--out-json``, script owns serving). Pass
``--no-serve`` to score against a server you started yourself.

Note the dataset the eval READS is the **LeRobot v2.1** original
(``…/ur10e_partial_cond_aug`` — per-episode ``data/chunk-000/episode_*.parquet``
and ``videos/…/episode_*.mp4``); the v3.0 *conversion* is only what
cosmos-framework's TRAIN loader needs, and its consolidated ``file-000`` layout
is not what ``_read_episode`` below walks. Point ``--dataset`` at the v2.1 dir.

Only ``numpy`` is needed to import the pure metric/transform helpers (unit-tested
in ``tests/unit/test_eval_openloop_gt_cosmos3.py``); the CLI additionally uses
pyarrow, imageio, mujoco + cosmos-framework (FK) and Odyssey's cosmos3 client —
all imported lazily inside ``main``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Pure metric + transform functions (numpy-only; unit-tested) ────────────────

ARM_JOINTS = 6
# Action-row widths we compare. The GT (FK'd from joints) is 10-D
# [dpos(3), rot6d(6), gripper(1)]; the cosmos server returns a DECODED env-native
# 7-D row [dpos(3), axis-angle(3), gripper(1)] for the robomind-ur domain (verified
# on the first GPU rollout). Both decode to (dpos, R, gripper) for a width-agnostic
# comparison.
_ROW_ROT6D = 10
_ROW_AXISANGLE = 7


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt a 6-D rotation representation into a 3x3 rotation matrix.

    ``rot6d`` is the first two columns of R stacked (Zhou et al. 2019): the
    model/dataset convention used by the cosmos LIBERO/drug-sort recipes. Column
    1 is normalised; column 2 is orthogonalised against it; column 3 = c1 x c2.
    """
    v = np.asarray(rot6d, dtype=np.float64).reshape(6)
    a1, a2 = v[:3], v[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2 = a2 - (b1 @ a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Rodrigues: a 3-vector axis-angle (magnitude = angle in rad) -> 3x3 R."""
    v = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(v))
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)


def _row_to_pose(row: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Decode one action row -> (dpos(3), R(3x3), gripper) for either width.

    Width 10 = ``[dpos(3), rot6d(6), gripper(1)]`` (GT); width 7 =
    ``[dpos(3), axis-angle(3), gripper(1)]`` (cosmos server, robomind-ur). A row
    wider than 10 is treated as 10-D + padding; a 7..9 row as axis-angle.
    """
    v = np.asarray(row, dtype=np.float64).ravel()
    dpos = v[:3]
    if v.shape[0] >= _ROW_ROT6D:
        return dpos, rot6d_to_matrix(v[3:9]), float(v[9])
    return dpos, axis_angle_to_matrix(v[3:6]), float(v[6])


def geodesic_deg_mat(rp: np.ndarray, rg: np.ndarray) -> float:
    """Geodesic angle (degrees) between two rotation matrices, clamped."""
    cos = (np.trace(rp.T @ rg) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def cartesian_chunk_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    """Per-chunk (translation MAE m, rotation geodesic deg, gripper agreement).

    ``pred`` / ``gt`` are ``(H, W)`` rows — W may differ between the two (the
    cosmos server returns decoded 7-D axis-angle rows, the FK'd GT is 10-D rot6d);
    each row is decoded to ``(dpos, R, gripper)`` so the comparison is
    width-agnostic. Compared over the overlapping horizon; empty overlap -> NaN.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    pred = pred.reshape(1, -1) if pred.ndim == 1 else pred
    gt = gt.reshape(1, -1) if gt.ndim == 1 else gt
    h = min(pred.shape[0], gt.shape[0])
    if h == 0:
        return float("nan"), float("nan"), float("nan")
    trans, rot, grip = [], [], []
    for t in range(h):
        pp, pr, pgrip = _row_to_pose(pred[t])
        gp, gr, ggrip = _row_to_pose(gt[t])
        trans.append(float(np.abs(pp - gp).mean()))
        rot.append(geodesic_deg_mat(pr, gr))
        grip.append(float((pgrip >= 0.5) == (ggrip >= 0.5)))
    return float(np.mean(trans)), float(np.mean(rot)), float(np.mean(grip))


def aggregate(trans: list[float], rot: list[float], grip: list[float]) -> dict:
    """Reduce per-tick records to a run summary (nan-safe means)."""
    t = np.asarray(trans, dtype=np.float64) if trans else np.empty(0)
    r = np.asarray(rot, dtype=np.float64) if rot else np.empty(0)
    g = np.asarray(grip, dtype=np.float64) if grip else np.empty(0)
    return {
        "n_ticks": int(t.size),
        "trans_mae_m": float(np.nanmean(t)) if t.size else float("nan"),
        "rot_geodesic_deg": float(np.nanmean(r)) if r.size else float("nan"),
        "gripper_agreement": float(np.nanmean(g)) if g.size else float("nan"),
    }


# ── FK GT transform (mirrors DrugsortURDataset._build_raw_action) ──────────────

def gt_ee_delta_chunk(joint_chunk: np.ndarray, fk) -> np.ndarray:
    """Turn a recorded ``(H+1, 7)`` joint-action chunk into ``(H, 10)`` EE deltas.

    ``joint_chunk`` rows are ``[6 arm joints, gripper]`` absolute targets. ``fk``
    is a callable ``arm_q(K,6) -> (positions(K,3), rotations(K,3,3))`` (MuJoCo FK
    on the UR5e). Mirrors the training transform EXACTLY: absolute EE poses ->
    ``pose_abs_to_rel(rot6d, backward_framewise)`` (imported lazily here) and
    gripper inverted ``1 - q_grip``. Returns ``[dpos(3), rot6d(6), gripper(1)]``
    — the same space Cosmos 3 predicts.
    """
    from cosmos_framework.data.generator.action.pose_utils import pose_abs_to_rel

    q = np.asarray(joint_chunk, dtype=np.float32)
    horizon = len(q) - 1
    pos, rot = fk(q[:, :ARM_JOINTS])
    poses_abs = np.tile(np.eye(4, dtype=np.float32), (horizon + 1, 1, 1))
    poses_abs[:, :3, 3] = pos
    poses_abs[:, :3, :3] = rot
    poses_rel = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention="backward_framewise")
    gripper = (1.0 - q[:horizon, ARM_JOINTS:ARM_JOINTS + 1]).astype(np.float32)
    return np.concatenate([np.asarray(poses_rel, dtype=np.float32), gripper], axis=-1)  # (H,10)


# ── Dataset IO (v2.1 layout — heavy deps lazy) ─────────────────────────────────

def _read_episode(dataset: Path, ep: int):
    """Return (state[N,S], action[N,7], {view: frames[N,H,W,3]}) for one episode."""
    import imageio.v3 as iio
    import pyarrow.parquet as pq

    pqf = dataset / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    t = pq.read_table(pqf)
    state = np.asarray(t.column("observation.state").to_pylist(), dtype=np.float32)
    action = np.asarray(t.column("action").to_pylist(), dtype=np.float32)
    frames: dict[str, np.ndarray] = {}
    for key in ("exterior", "wrist"):
        mp4 = dataset / "videos" / "chunk-000" / f"observation.images.{key}" / f"episode_{ep:06d}.mp4"
        if mp4.is_file():
            frames[key] = np.asarray(iio.imread(mp4, plugin="pyav"))
    return state, action, frames


def _make_ur5e_fk():
    """Build the UR5e MuJoCo FK callable by reusing RoboMINDURDataset's kinematics."""
    import mujoco
    from cosmos_framework.data.generator.action.datasets.robomind_ur_dataset import (
        RoboMINDURDataset,
    )

    mj_model, mj_data, ee_site_id = RoboMINDURDataset._init_mujoco()

    def fk(arm_q: np.ndarray):
        arm_q = np.asarray(arm_q, dtype=np.float32)
        k = len(arm_q)
        positions = np.empty((k, 3), dtype=np.float32)
        rotations = np.empty((k, 3, 3), dtype=np.float32)
        for i in range(k):
            mj_data.qpos[:ARM_JOINTS] = arm_q[i]
            mujoco.mj_forward(mj_model, mj_data)
            positions[i] = mj_data.site_xpos[ee_site_id]
            rotations[i] = mj_data.site_xmat[ee_site_id].reshape(3, 3)
        return positions, rotations

    return fk


# ── Optional server lifecycle (custom-runner contract) ─────────────────────────

def _serve_cosmos(checkpoint: str, port: int, log_path: Path):
    """Boot action_policy_server_libero on ``checkpoint``; return the Popen."""
    import subprocess

    cmd = [
        sys.executable, "-m", "cosmos_framework.scripts.action_policy_server_libero",
        "--checkpoint-path", checkpoint, "--port", str(port),
    ]
    log = open(log_path, "w")  # noqa: SIM115 — handle must outlive this fn (Popen stdout)
    print(f"[cosmos-openloop] serving: {' '.join(cmd)} (log {log_path})")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def _wait_info(client, timeout_s: float) -> bool:
    """Poll GET /info until the server answers or ``timeout_s`` elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            client.info()
            return True
        except Exception:
            time.sleep(3.0)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Cosmos3 open-loop GT eval")
    ap.add_argument("--dataset", required=True, help="LeRobot v2.1 dir (…/ur10e_partial_cond_aug)")
    ap.add_argument("--checkpoint", default="", help="export dir to serve (with --serve)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--serve", dest="serve", action="store_true", default=True,
                    help="boot the cosmos server on --checkpoint (default)")
    ap.add_argument("--no-serve", dest="serve", action="store_false",
                    help="score against an already-running server")
    # Flag names use underscores so the odyssey `custom` runner's verbatim
    # `--<config-key> value` passthrough (domain_name / held_out / …) matches.
    ap.add_argument("--domain_name", default="robomind-ur",
                    help="domain tag sent with each /predict (drug-sort trains under robomind-ur)")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--held_out", type=int, default=4,
                    help="eval the LAST N episodes when --episodes is unset (fit-eval caveat: "
                         "they are in the train split — a sanity signal, not generalization)")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--instruction", default="pick up the vial and place it in the rack")
    ap.add_argument("--server-timeout-s", type=float, default=600.0)
    ap.add_argument("--out-json", default="")
    # tolerate the odyssey custom-runner's extra passthrough flags.
    args, _unknown = ap.parse_known_args()

    dataset = Path(args.dataset)
    if not (dataset / "meta" / "info.json").is_file():
        print(f"ERROR: not a LeRobot dataset dir: {dataset}", file=sys.stderr)
        return 2
    info = json.loads((dataset / "meta" / "info.json").read_text())
    total = int(info["total_episodes"])
    eps = args.episodes if args.episodes is not None else list(range(max(0, total - args.held_out), total))

    # Make `odyssey` importable when launched under an arbitrary interpreter: the
    # `custom` eval runner runs this via `eval_python` (e.g. the cosmos-framework
    # venv), which need not have odyssey installed nor PYTHONPATH forwarded to the
    # child. Fall back to the repo's own src/ (this file is examples/<m>/<f>.py).
    _src = Path(__file__).resolve().parents[2] / "src"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from odyssey.runners.evals.cosmos3_transforms import (
        build_cosmos3_predict_request,
        cosmos3_chunk_from_response,
    )
    from odyssey.runners.models.cosmos3 import Cosmos3HttpClient

    proc = None
    try:
        client = Cosmos3HttpClient(host=args.host, port=args.port, timeout_seconds=args.server_timeout_s)
        if args.serve:
            if not args.checkpoint:
                print("ERROR: --serve needs --checkpoint", file=sys.stderr)
                return 2
            log_path = (
                Path(args.out_json).with_suffix(".server.log") if args.out_json
                else Path("/tmp/cosmos_server.log")
            )
            proc = _serve_cosmos(args.checkpoint, args.port, log_path)
            if not _wait_info(client, args.server_timeout_s):
                print("ERROR: cosmos server did not come up (see server log)", file=sys.stderr)
                return 3
        else:
            client.info()  # fail fast if nothing is listening

        fk = _make_ur5e_fk()
        all_t: list[float] = []
        all_r: list[float] = []
        all_g: list[float] = []
        per_ep: list[dict] = []
        for ep in eps:
            state, action, frames = _read_episode(dataset, ep)
            n = state.shape[0]
            ep_t: list[float] = []
            ep_r: list[float] = []
            ep_g: list[float] = []
            for t in range(0, n - 1, args.stride):
                if not all(t < v.shape[0] for v in frames.values()):
                    break
                req = build_cosmos3_predict_request(
                    image=frames["exterior"][t],
                    wrist_image=frames["wrist"][t] if "wrist" in frames else None,
                    instruction=args.instruction,
                    domain_name=args.domain_name,
                    image_size=args.image_size,
                )
                pred = cosmos3_chunk_from_response(client.infer(req))[: args.horizon]
                gt_joint = action[t : t + args.horizon + 1]
                if len(gt_joint) < 2:
                    break
                gt = gt_ee_delta_chunk(gt_joint, fk)
                tm, rm, gm = cartesian_chunk_metrics(pred, gt)
                ep_t.append(tm)
                ep_r.append(rm)
                ep_g.append(gm)
            s = aggregate(ep_t, ep_r, ep_g)
            s["episode"] = ep
            per_ep.append(s)
            all_t.extend(ep_t)
            all_r.extend(ep_r)
            all_g.extend(ep_g)
            print(f"[cosmos-openloop] ep {ep:03d}: ticks={s['n_ticks']:3d} "
                  f"trans_MAE={s['trans_mae_m']:.4f} m  rot={s['rot_geodesic_deg']:.2f} deg  "
                  f"grip={s['gripper_agreement'] * 100:5.1f}%")

        summary = aggregate(all_t, all_r, all_g)
        summary["episodes"] = per_ep
        summary["dataset"] = str(dataset)
        # odyssey custom-runner metric contract: fold flat metrics under "metrics".
        summary["metrics"] = {
            "trans_mae_m": summary["trans_mae_m"],
            "rot_geodesic_deg": summary["rot_geodesic_deg"],
            "gripper_agreement": summary["gripper_agreement"],
        }
        print("\n[cosmos-openloop] ===== SUMMARY =====")
        print(f"[cosmos-openloop] ticks={summary['n_ticks']}  trans_MAE={summary['trans_mae_m']:.4f} m  "
              f"rot={summary['rot_geodesic_deg']:.2f} deg  grip={summary['gripper_agreement'] * 100:.1f}%")
        if args.out_json:
            Path(args.out_json).write_text(json.dumps(summary, indent=2) + "\n")
            print(f"[cosmos-openloop] wrote {args.out_json}")
        return 0
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
