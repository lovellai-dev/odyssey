#!/usr/bin/env python3
"""Convert BROWSER-captured raw episodes -> the Observer's .npy perception dataset.

Closes the render gap for the Observer (DINOv2 grasp-target head + success
classifier). ``scripts/gen_ur5e_perception_data.py`` renders its own MuJoCo pixels;
this instead takes the Three.js frames captured in-browser by ``browser_harness.js``
and pairs them with GT labels reconstructed from the SAME sim state, so the on-disk
layout is byte-identical to that generator's and ``scripts/train_ur5e_perception.py``
consumes it UNCHANGED — only the pixels are Three.js (the deployment renderer).

Per kept frame it rebuilds the exact sim pose (arm qpos from the recorded
``states``, gripper from ``states[6]``, vial free-joint from the GT vial pose),
``mj_forward``s, and reads pinch/base/pocket to compute the identical labels the
reference generator writes:
  * ``grasp_target_base`` (N,3) — vial cap point in the robot base frame (m)
  * ``labels`` (N,) — phase class {0 reaching,1 grasped,2 lifted,3 seated}
  * ``episode`` (N,), ``success`` (N,)  + ``ext.npy``/``wrist.npy`` (N,224,224,3)
Reuses the reference generator's helpers/constants (importlib-loaded) for parity.

Usage::
    MUJOCO_GL=egl .venv-ur5e/bin/python perception_from_browser.py \
        --raw out_full/raw_all --out percep_browser --keep-every 4 \
        --max-frames-per-ep 70 --max-total-frames 15000
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "src"))

# importlib-load the reference generator to reuse its helpers + thresholds (parity).
_gpd_path = _REPO / "scripts" / "gen_ur5e_perception_data.py"
_spec = importlib.util.spec_from_file_location("gpd", _gpd_path)
gpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gpd)

from odyssey.embodiments.ur5e_drugsort import embodiment as emb  # noqa: E402

GRIP_DRIVER_RANGE = 0.8
DEFAULT_XML = gpd.DEFAULT_XML


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w:
        return img
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR), dtype=np.uint8)


def load_png(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, help="dir of epNNN/ browser raw episodes")
    ap.add_argument("--out", required=True)
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--keep-every", type=int, default=4)
    ap.add_argument("--max-frames-per-ep", type=int, default=70)
    ap.add_argument("--max-total-frames", type=int, default=15000)
    args = ap.parse_args()

    import mujoco as mj

    model = mj.MjModel.from_xml_path(str(args.xml))
    data = mj.MjData(model)
    idx = gpd._build_index(mj, model)
    home_key = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
    arm_qadr, grip_qadr, vq = idx["arm_qadr"], idx["grip_qadr"], idx["vial_qadr"]
    base_body, vial_body, pinch_site, pocket_site = (
        idx["base_body"], idx["vial_body"], idx["pinch_site"], idx["pocket_site"])

    raw = Path(args.raw)
    ep_dirs = sorted([p for p in raw.iterdir() if p.is_dir() and p.name.startswith("ep")])
    if not ep_dirs:
        print(f"ERROR: no raw episodes under {raw}", file=sys.stderr)
        return 2

    all_ext, all_wrist, all_tgt, all_lab, all_ep, all_succ = [], [], [], [], [], []
    all_de, all_dw, all_uv_e, all_uv_w = [], [], [], []
    total = 0
    for d in ep_dirs:
        meta = json.loads((d / "meta.json").read_text())
        states = np.asarray(meta["states"], dtype=np.float64)
        gt = meta["gt"]
        success = int(meta["success"])
        ep_id = int(meta["episode"])
        T = states.shape[0]
        ext_files = sorted((d / "exterior").glob("f*.png"))
        wr_files = sorted((d / "wrist").glob("f*.png"))
        if not (len(ext_files) == len(wr_files) == T == len(gt)):
            print(f"[skip] {d.name}: length mismatch T={T} gt={len(gt)} "
                  f"ext={len(ext_files)} wr={len(wr_files)}")
            continue
        z0 = float(gt[0]["vial"][2])
        kept = 0
        # Sample frames UNIFORMLY across the whole episode so every phase
        # (reach -> descend -> close -> lift -> transport -> place -> retract) is
        # represented — the grasp/lift/seat frames live in the latter half, so a
        # first-N subsample would be all-reaching. keep_every still thins dense runs.
        n_take = min(args.max_frames_per_ep, (T + args.keep_every - 1) // args.keep_every)
        take_idx = sorted(set(int(round(x)) for x in np.linspace(0, T - 1, n_take)))
        for f in take_idx:
            if kept >= args.max_frames_per_ep:
                break
            # Rebuild the exact sim pose and read GT (identical to the reference generator).
            mj.mj_resetDataKeyframe(model, data, home_key)
            for i, a in enumerate(arm_qadr):
                data.qpos[a] = states[f][i]
            data.qpos[grip_qadr] = float(states[f][6]) * GRIP_DRIVER_RANGE
            for i in range(7):
                data.qpos[vq + i] = gt[f]["vial"][i]
            mj.mj_forward(model, data)

            vxyz = np.array(data.xpos[vial_body], dtype=np.float64)
            vmat = np.array(data.xmat[vial_body], dtype=np.float64).reshape(3, 3)
            cap_world = vxyz + vmat @ np.array([0.0, 0.0, gpd.VIAL_CAP_OFFSET_Z])
            pinch_xyz = np.array(data.site_xpos[pinch_site], dtype=np.float64)
            pkt_xyz = np.array(data.site_xpos[pocket_site], dtype=np.float64)
            bpos = np.array(data.xpos[base_body], dtype=np.float64)
            bmat = np.array(data.xmat[base_body], dtype=np.float64).reshape(3, 3)
            cap_base = bmat.T @ (cap_world - bpos)

            grip_meas = float(np.clip(states[f][6], 0.0, 1.0))
            near = float(np.linalg.norm(vxyz - pinch_xyz)) < gpd.NEAR_PINCH_THRESH
            lift_h = float(vxyz[2] - z0)
            place_xy = float(np.linalg.norm(vxyz[:2] - pkt_xyz[:2]))
            lab = gpd.label_phase(
                grip_meas=grip_meas, near_pinch=near, lift_height=lift_h,
                place_dist_xy=place_xy, vial_z=float(vxyz[2]), phase_name=gt[f]["phase"])

            cams = idx["cams"]
            ue, ve, de, _ = gpd.project(
                data.cam_xpos[cams["exterior"]], data.cam_xmat[cams["exterior"]],
                float(model.cam_fovy[cams["exterior"]]), cap_world, args.width, args.height)
            uw, vw, dw, _ = gpd.project(
                data.cam_xpos[cams["wrist"]], data.cam_xmat[cams["wrist"]],
                float(model.cam_fovy[cams["wrist"]]), cap_world, args.width, args.height)

            all_ext.append(resize(load_png(ext_files[f]), args.width, args.height))
            all_wrist.append(resize(load_png(wr_files[f]), args.width, args.height))
            all_tgt.append(cap_base.astype(np.float32))
            all_uv_e.append(np.array([ue, ve], dtype=np.float32))
            all_uv_w.append(np.array([uw, vw], dtype=np.float32))
            all_de.append(np.float32(de)); all_dw.append(np.float32(dw))
            all_lab.append(np.int64(lab)); all_ep.append(np.int64(ep_id))
            all_succ.append(np.int64(success))
            kept += 1
            total += 1
        counts = np.bincount(np.array([all_lab[i] for i in range(len(all_lab) - kept, len(all_lab))],
                                      dtype=int), minlength=4).tolist() if kept else [0, 0, 0, 0]
        print(f"[convert] {d.name} (ep{ep_id:03d}): kept {kept:3d}/{T} labels(r/g/l/s)={counts} "
              f"success={success} total={total}")
        if total >= args.max_total_frames:
            print(f"[convert] hit max-total-frames={args.max_total_frames}; stopping")
            break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ext = np.stack(all_ext).astype(np.uint8)
    wrist = np.stack(all_wrist).astype(np.uint8)
    np.save(out / "ext.npy", ext)
    np.save(out / "wrist.npy", wrist)
    np.save(out / "grasp_target_base.npy", np.stack(all_tgt).astype(np.float32))
    np.save(out / "cap_uv_ext.npy", np.stack(all_uv_e).astype(np.float32))
    np.save(out / "cap_uv_wrist.npy", np.stack(all_uv_w).astype(np.float32))
    np.save(out / "depth_ext.npy", np.array(all_de, dtype=np.float32))
    np.save(out / "depth_wrist.npy", np.array(all_dw, dtype=np.float32))
    np.save(out / "labels.npy", np.array(all_lab, dtype=np.int64))
    np.save(out / "episode.npy", np.array(all_ep, dtype=np.int64))
    np.save(out / "success.npy", np.array(all_succ, dtype=np.int64))
    counts = np.bincount(np.array(all_lab, dtype=int), minlength=4).tolist()
    meta = {
        "num_frames": int(ext.shape[0]), "width": args.width, "height": args.height,
        "class_names": ["reaching", "grasped", "lifted", "seated"], "class_counts": counts,
        "cap_offset_z_m": gpd.VIAL_CAP_OFFSET_Z, "target_frame": "robot_base",
        "cameras": list(emb.VIDEO_KEY_TO_CAMERA.keys()), "source": "browser_threejs",
        "num_episodes_run": int(len(set(all_ep))),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[convert] wrote {ext.shape[0]} frames ({len(set(all_ep))} eps) -> {out}")
    print(f"[convert] class counts (reaching/grasped/lifted/seated): {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
