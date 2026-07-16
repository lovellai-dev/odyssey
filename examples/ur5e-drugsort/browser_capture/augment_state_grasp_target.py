#!/usr/bin/env python3
"""Augment the browser LeRobot dataset with an explicit **grasp-target** state channel.

Roadmap Step 2 (Observer-conditioned GR00T). The render-gap close proved the
Observer localises the vial cap to sub-cm on the deployment (Three.js) frames, yet
the browser GR00T pilot still scored 0/15 — it cannot infer *descent depth* from
pixels alone. This script rewrites ``dataset_browser_full`` so every frame's
``observation.state`` carries the 3D grasp target (vial cap, robot **base** frame)
appended after the 7 proprio dims:

    state:  [ arm_0..arm_5, gripper, cap_base_x, cap_base_y, cap_base_z ]   (7 -> 10)

The target per frame is the SAME quantity the Observer's ``predict()`` returns and
the SAME transform ``perception_from_browser.py`` uses for the Observer's training
labels — ``cap_world = vial_xyz + R(vial_quat) @ [0,0,VIAL_CAP_OFFSET_Z]`` then
``cap_base = base_R^T (cap_world - base_pos)`` — so training targets (from the GT
sidecar) and the live deploy signal (from the Observer) live in one frame. At
deploy the Observer supplies this channel live (0.62 cm median).

Everything else (action, videos, pixels, fps, episode split) is byte-identical;
only ``observation.state`` grows by 3 dims and ``meta/{info,modality,stats}.json``
are updated to match. The GT sidecar under ``meta/gt/`` is copied through unchanged.

Usage::
    MUJOCO_GL=egl .venv-ur5e/bin/python augment_state_grasp_target.py \
        --src dataset_browser_full --out dataset_browser_cond
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

# Reuse the reference perception generator's constants (VIAL_CAP_OFFSET_Z, XML) so
# the appended target matches the Observer's label definition EXACTLY (parity).
_gpd_path = _REPO / "scripts" / "gen_ur5e_perception_data.py"
_spec = importlib.util.spec_from_file_location("gpd", _gpd_path)
gpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpd)

GRASP_NAMES = ["grasp_target.x", "grasp_target.y", "grasp_target.z"]


def _base_transform(xml: str):
    """Static robot-base pose (pos + 3x3 rot) from the MJCF ``home`` keyframe."""
    import mujoco as mj

    model = mj.MjModel.from_xml_path(xml)
    data = mj.MjData(model)
    idx = gpd._build_index(mj, model)
    hk = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
    if hk >= 0:
        mj.mj_resetDataKeyframe(model, data, hk)
    mj.mj_forward(model, data)
    bpos = np.array(data.xpos[idx["base_body"]], dtype=np.float64)
    bmat = np.array(data.xmat[idx["base_body"]], dtype=np.float64).reshape(3, 3)
    return bpos, bmat, mj


def _cap_base(vial_pose, bpos, bmat, mj) -> np.ndarray:
    """vial free-joint world pose [x,y,z,qw,qx,qy,qz] -> vial cap in robot base frame."""
    vxyz = np.asarray(vial_pose[:3], dtype=np.float64)
    vmat = np.zeros(9, dtype=np.float64)
    mj.mju_quat2Mat(vmat, np.asarray(vial_pose[3:7], dtype=np.float64))
    vmat = vmat.reshape(3, 3)
    cap_world = vxyz + vmat @ np.array([0.0, 0.0, gpd.VIAL_CAP_OFFSET_Z])
    return bmat.T @ (cap_world - bpos)


def _stats(col: np.ndarray) -> dict:
    """LeRobot per-dimension stats block (matches the existing stats.json shape)."""
    std = np.std(col, axis=0)
    std = np.maximum(std, 1e-6)  # never divide by ~0 during normalisation
    return {
        "mean": np.mean(col, axis=0).tolist(),
        "std": std.tolist(),
        "min": np.min(col, axis=0).tolist(),
        "max": np.max(col, axis=0).tolist(),
        "q01": np.quantile(col, 0.01, axis=0).tolist(),
        "q99": np.quantile(col, 0.99, axis=0).tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="dataset_browser_full")
    ap.add_argument("--out", default="dataset_browser_cond")
    ap.add_argument("--xml", default=gpd.DEFAULT_XML)
    ap.add_argument("--verify", action="store_true",
                    help="cross-check the analytic cap_base against a MuJoCo mj_forward on a few frames")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not (src / "meta" / "info.json").exists():
        print(f"ERROR: no dataset at {src}", file=sys.stderr)
        return 2
    bpos, bmat, mj = _base_transform(args.xml)
    print(f"[augment] base_pos={np.round(bpos, 4).tolist()} cap_offset_z={gpd.VIAL_CAP_OFFSET_Z}")

    if args.verify:
        _verify(src, args.xml, bpos, bmat, mj)

    # --- fresh output tree: copy meta + videos, rewrite data ------------------
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "meta").mkdir(parents=True)
    # videos are pixel-identical -> hardlink for speed/space (fall back to copy).
    for chunk in sorted((src / "videos").glob("chunk-*")):
        dst = out / "videos" / chunk.name
        dst.mkdir(parents=True, exist_ok=True)
        for vk in chunk.iterdir():           # video_key dirs
            (dst / vk.name).mkdir(exist_ok=True)
            for mp4 in vk.glob("*.mp4"):
                try:
                    os.link(mp4, dst / vk.name / mp4.name)
                except OSError:
                    shutil.copy2(mp4, dst / vk.name / mp4.name)
    # passthrough meta files
    for name in ("tasks.jsonl", "episodes.jsonl"):
        shutil.copy2(src / "meta" / name, out / "meta" / name)
    if (src / "meta" / "gt").exists():
        shutil.copytree(src / "meta" / "gt", out / "meta" / "gt")

    # --- rewrite every episode's observation.state (7 -> 10) ------------------
    all_state = []
    n_frames = 0
    for chunk in sorted((src / "data").glob("chunk-*")):
        (out / "data" / chunk.name).mkdir(parents=True, exist_ok=True)
        for pf in sorted(chunk.glob("episode_*.parquet")):
            ep = int(pf.stem.split("_")[-1])
            gt = json.loads((src / "meta" / "gt" / f"episode_{ep:06d}.json").read_text())
            frames = gt["frames"]
            tbl = pq.read_table(pf)
            state7 = np.asarray(tbl.column("observation.state").to_pylist(), dtype=np.float64)
            fidx = np.asarray(tbl.column("frame_index").to_pylist(), dtype=np.int64)
            assert state7.shape[1] == 7, f"{pf.name}: expected 7-D state, got {state7.shape}"
            assert len(frames) == state7.shape[0], (
                f"{pf.name}: gt frames {len(frames)} != rows {state7.shape[0]}")
            tgt = np.stack([_cap_base(frames[int(f)]["vial_pose"], bpos, bmat, mj) for f in fidx])
            state10 = np.concatenate([state7, tgt], axis=1).astype(np.float32)
            new_col = pa.array([row.tolist() for row in state10],
                               type=pa.list_(pa.float32()))
            tbl = tbl.set_column(tbl.schema.get_field_index("observation.state"),
                                 "observation.state", new_col)
            pq.write_table(tbl, out / "data" / chunk.name / pf.name)
            all_state.append(state10)
            n_frames += state10.shape[0]
    all_state = np.concatenate(all_state, axis=0)
    print(f"[augment] rewrote {n_frames} frames across "
          f"{sum(1 for _ in (out / 'data').rglob('*.parquet'))} episodes")

    # --- meta/info.json : observation.state 7 -> 10 ---------------------------
    info = json.loads((src / "meta" / "info.json").read_text())
    st = info["features"]["observation.state"]
    st["shape"] = [10]
    st["names"] = st["names"][:7] + GRASP_NAMES
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    # --- meta/modality.json : add grasp_target slot 7:10 ----------------------
    mod = json.loads((src / "meta" / "modality.json").read_text())
    mod["state"]["grasp_target"] = {"start": 7, "end": 10}
    (out / "meta" / "modality.json").write_text(json.dumps(mod, indent=4))

    # --- meta/stats.json : recompute observation.state (10-D) -----------------
    stats = json.loads((src / "meta" / "stats.json").read_text())
    stats["observation.state"] = _stats(all_state)
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    gm = _stats(all_state[:, 7:10])
    print(f"[augment] grasp_target stats  mean={np.round(gm['mean'], 4).tolist()} "
          f"min={np.round(gm['min'], 4).tolist()} max={np.round(gm['max'], 4).tolist()}")
    print(f"[augment] DONE -> {out}  (state dim 7 -> 10)")
    return 0


def _verify(src: Path, xml: str, bpos, bmat, mj) -> None:
    """Confirm analytic cap_base == the perception pipeline's mj_forward cap_base."""
    import mujoco as mjm

    model = mjm.MjModel.from_xml_path(xml)
    data = mjm.MjData(model)
    idx = gpd._build_index(mjm, model)
    hk = mjm.mj_name2id(model, mjm.mjtObj.mjOBJ_KEY, "home")
    vb, vq = idx["vial_body"], idx["vial_qadr"]
    gt = json.loads((src / "meta" / "gt" / "episode_000000.json").read_text())
    worst = 0.0
    # Length-relative sample indices: episodes vary (base ~948 frames, DAgger
    # rollouts 900) — hardcoded indices broke on the first Stage-C mini-round.
    n_f = len(gt["frames"])
    for fi in sorted({0, n_f // 8, n_f // 2, (3 * n_f) // 4, n_f - 1}):
        vp = gt["frames"][fi]["vial_pose"]
        mjm.mj_resetDataKeyframe(model, data, hk)
        for i in range(7):
            data.qpos[vq + i] = vp[i]
        mjm.mj_forward(model, data)
        vxyz = np.array(data.xpos[vb]); vmat = np.array(data.xmat[vb]).reshape(3, 3)
        cw = vxyz + vmat @ np.array([0.0, 0.0, gpd.VIAL_CAP_OFFSET_Z])
        mj_cap = bmat.T @ (cw - bpos)
        an_cap = _cap_base(vp, bpos, bmat, mj)
        worst = max(worst, float(np.linalg.norm(mj_cap - an_cap)))
    print(f"[augment][verify] analytic vs mj_forward cap_base max diff = {worst:.2e} m")
    assert worst < 1e-6, "analytic cap_base diverges from the perception pipeline"


if __name__ == "__main__":
    raise SystemExit(main())
