#!/usr/bin/env python3
"""Auto-label a perception dataset for the UR5e / Robotiq drug-sorting cell.

Runs the **adaptive IK expert** (:mod:`odyssey.embodiments.ur5e_drugsort`) in
headless MuJoCo with vial-pose + visual domain randomisation, renders the
``exterior`` and ``wrist`` cameras, and — crucially — reads the sim
**ground truth** for free to auto-label every rendered frame with:

* ``grasp_target_base`` — the 3D vial cap/neck point in the robot **base** frame
  (metres). This is the Observer's regression target and the deployable
  grasp-target conditioning input that fixes the "closes ~7 cm short" depth
  failure (a single overhead view can't resolve descent depth; the absolute 3D
  target can).
* ``cap_uv_exterior`` / ``cap_uv_wrist`` — the same physical point projected into
  each camera (pixel keypoint), via the MuJoCo pinhole model.
* ``label`` — the phase class in ``{0 reaching, 1 grasped, 2 lifted, 3 seated}``
  derived from GT (gripper closure, vial-to-pinch distance, lift height,
  place distance). The success classifier distils this signal into a fast CNN.
* ``success`` — episode-level GT success (vial lifted AND seated in the pocket).

No manual annotation and no GPU are needed to *generate* the labels (only an
EGL/OpenGL context for offscreen rendering). Output is a directory of ``.npy``
shards + ``meta.json`` (camera intrinsics/extrinsics + the base transform),
consumed by ``scripts/train_ur5e_perception.py``.

Usage::

    MUJOCO_GL=egl .venv-ur5e/bin/python scripts/gen_ur5e_perception_data.py \
        --out /path/to/percep_data --num-episodes 40 --visual-dr 1.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from odyssey.embodiments.ur5e_drugsort import embodiment as emb
from odyssey.embodiments.ur5e_drugsort.ik import (
    DampedLeastSquaresIK,
    build_adaptive_phases,
)
from odyssey.embodiments.ur5e_drugsort.policy import (
    ARM_ACTUATOR_NAMES,
    GRIP_OPEN,
    GRIPPER_ACTUATOR_NAME,
    SETTLE_STEPS,
    PickPlaceController,
)

DEFAULT_XML = (
    "/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/"
    "aseptipack_description/aseptipack.xml"
)
GRIP_DRIVER_JOINT = "gr_right_driver_joint"
GRIP_DRIVER_RANGE = 0.8
ARM_BODIES = frozenset(
    {"base", "shoulder_link", "upper_arm_link", "forearm_link",
     "wrist_1_link", "wrist_2_link", "wrist_3_link"}
)
# Vial cap/neck offset (m) along the vial's local +z: the red cap geom sits at
# local z=+0.024 (MJCF ``vial_0``). This is the physical keypoint the gripper
# must descend onto, and the point projected into both cameras.
VIAL_CAP_OFFSET_Z = 0.024

# --- Phase-label thresholds (GT-derived) -----------------------------------
GRIP_CLOSED_THRESH = 0.30     # normalised closure above which the gripper is "closed"
NEAR_PINCH_THRESH = 0.055     # vial-to-pinch distance (m) for "gripper on vial"
LIFT_THRESH = 0.03            # vial rise above start (m) counting as "lifted"
SEATED_XY_THRESH = 0.05       # vial-to-pocket xy distance (m) for "seated"
SEATED_Z_MIN = 0.20           # vial height (m) confirming it rests in the nest


def _build_index(mj, model):  # type: ignore[no-untyped-def]
    """Resolve actuator / joint / body / site / camera ids we drive + observe."""
    m2i = mj.mj_name2id
    obj = mj.mjtObj
    arm_act = [m2i(model, obj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATOR_NAMES]
    grip_act = m2i(model, obj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME)
    arm_qadr, arm_dofadr = [], []
    for n in ARM_ACTUATOR_NAMES:
        j = m2i(model, obj.mjOBJ_JOINT, f"{n}_joint")
        arm_qadr.append(int(model.jnt_qposadr[j]))
        arm_dofadr.append(int(model.jnt_dofadr[j]))
    grip_j = m2i(model, obj.mjOBJ_JOINT, GRIP_DRIVER_JOINT)
    grip_qadr = int(model.jnt_qposadr[grip_j])
    pinch_site = m2i(model, obj.mjOBJ_SITE, "gr_pinch")
    vial_body = m2i(model, obj.mjOBJ_BODY, "vial_0")
    vial_qadr = -1
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == vial_body and model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
            vial_qadr = int(model.jnt_qposadr[j])
            break
    if vial_qadr < 0:
        raise RuntimeError("could not find the vial_0 free joint")
    pocket_site = m2i(model, obj.mjOBJ_SITE, "pocket_0")
    base_body = m2i(model, obj.mjOBJ_BODY, "base")
    cams = {vk: m2i(model, obj.mjOBJ_CAMERA, cam) for vk, cam in emb.VIDEO_KEY_TO_CAMERA.items()}
    if any(a < 0 for a in arm_act) or grip_act < 0 or pinch_site < 0 or base_body < 0:
        raise RuntimeError("could not resolve required actuators/sites/bodies in the MJCF")
    if any(c < 0 for c in cams.values()):
        raise RuntimeError(f"could not resolve cameras {emb.VIDEO_KEY_TO_CAMERA}")
    return {
        "arm_act": arm_act, "grip_act": grip_act, "arm_qadr": arm_qadr,
        "arm_dofadr": arm_dofadr, "grip_qadr": grip_qadr, "pinch_site": pinch_site,
        "vial_body": vial_body, "vial_qadr": vial_qadr, "pocket_site": pocket_site,
        "base_body": base_body, "cams": cams,
    }


def _free_arm_collision(mj, model) -> int:  # type: ignore[no-untyped-def]
    freed = 0
    for g in range(model.ngeom):
        bn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        gn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or ""
        if (bn in ARM_BODIES or bn.startswith("gr_")) and "pad" not in gn:
            model.geom_contype[g] = 0
            model.geom_conaffinity[g] = 0
            freed += 1
    return freed


def _snapshot_visual(mj, model):  # type: ignore[no-untyped-def]
    room_cam = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "room")
    snap = {
        "room_cam": room_cam,
        "cam_pos": model.cam_pos.copy(), "cam_fovy": model.cam_fovy.copy(),
        "geom_rgba": model.geom_rgba.copy(),
        "hl_diffuse": np.array(model.vis.headlight.diffuse, dtype=np.float64),
        "hl_ambient": np.array(model.vis.headlight.ambient, dtype=np.float64),
        "hl_specular": np.array(model.vis.headlight.specular, dtype=np.float64),
        "mat_rgba": model.mat_rgba.copy() if model.nmat else None,
        "light_pos": model.light_pos.copy() if model.nlight else None,
        "light_diffuse": model.light_diffuse.copy() if model.nlight else None,
        "light_ambient": model.light_ambient.copy() if model.nlight else None,
    }
    return snap


def _apply_visual_dr(model, snap, rng, strength=1.0):  # type: ignore[no-untyped-def]
    model.cam_pos[:] = snap["cam_pos"]
    model.cam_fovy[:] = snap["cam_fovy"]
    model.geom_rgba[:] = snap["geom_rgba"]
    if snap["mat_rgba"] is not None:
        model.mat_rgba[:] = snap["mat_rgba"]
    if snap["light_pos"] is not None:
        model.light_pos[:] = snap["light_pos"]
        model.light_diffuse[:] = snap["light_diffuse"]
        model.light_ambient[:] = snap["light_ambient"]
    model.vis.headlight.diffuse[:] = snap["hl_diffuse"]
    model.vis.headlight.ambient[:] = snap["hl_ambient"]
    model.vis.headlight.specular[:] = snap["hl_specular"]
    if strength <= 0:
        return
    s = float(strength)
    cid = snap["room_cam"]
    if cid >= 0:
        model.cam_pos[cid] = snap["cam_pos"][cid] + rng.uniform(-0.05, 0.05, 3) * s
        model.cam_fovy[cid] = float(snap["cam_fovy"][cid] + rng.uniform(-2.5, 2.5) * s)
    diff = float(np.clip(1.0 + rng.uniform(-0.4, 0.4) * s, 0.2, 1.8))
    amb = float(np.clip(1.0 + rng.uniform(-0.5, 0.6) * s, 0.2, 2.0))
    warm = 1.0 + rng.uniform(-0.12, 0.12, 3) * s
    model.vis.headlight.diffuse[:] = np.clip(snap["hl_diffuse"] * diff * warm, 0.0, 1.0)
    model.vis.headlight.ambient[:] = np.clip(snap["hl_ambient"] * amb, 0.0, 1.0)
    if snap["light_pos"] is not None and snap["light_pos"].size:
        model.light_pos[:] = snap["light_pos"] + rng.uniform(
            -0.35, 0.35, snap["light_pos"].shape) * s
        model.light_diffuse[:] = np.clip(snap["light_diffuse"] * diff, 0.0, 1.0)
        model.light_ambient[:] = np.clip(snap["light_ambient"] * amb, 0.0, 1.0)
    tint = np.clip(1.0 + rng.uniform(-0.18, 0.18, 3) * s, 0.4, 1.6)
    if snap["mat_rgba"] is not None and model.nmat:
        base = snap["mat_rgba"][:, :3]
        model.mat_rgba[:, :3] = np.clip(
            base * tint * (1.0 + rng.uniform(-0.10, 0.10, base.shape) * s), 0.0, 1.0)
    gbase = snap["geom_rgba"][:, :3]
    model.geom_rgba[:, :3] = np.clip(
        gbase * tint * (1.0 + rng.uniform(-0.10, 0.10, gbase.shape) * s), 0.0, 1.0)


def project(cam_pos, cam_mat, fovy_deg, p_world, width, height):  # type: ignore[no-untyped-def]
    """MuJoCo pinhole projection of ``p_world`` into image pixels ``(u, v)``.

    MuJoCo cameras look along their local **-z**; ``cam_mat`` columns are the
    camera x/y/z axes in world coords. Returns ``(u, v, depth_m, in_front)``.
    """
    r = np.asarray(cam_mat, dtype=np.float64).reshape(3, 3)
    p_cam = r.T @ (np.asarray(p_world, dtype=np.float64) - np.asarray(cam_pos, dtype=np.float64))
    depth = -float(p_cam[2])  # positive when in front of the camera
    f = 0.5 * height / math.tan(0.5 * math.radians(fovy_deg))
    cx, cy = width / 2.0, height / 2.0
    if abs(p_cam[2]) < 1e-9:
        return cx, cy, depth, False
    u = cx + f * (p_cam[0] / depth)
    v = cy - f * (p_cam[1] / depth)
    return float(u), float(v), depth, depth > 0.0


def label_phase(*, grip_meas, near_pinch, lift_height, place_dist_xy, vial_z,
                phase_name):  # type: ignore[no-untyped-def]
    """GT phase class: 0 reaching, 1 grasped, 2 lifted, 3 seated."""
    grip_closed = grip_meas > GRIP_CLOSED_THRESH
    seated = (place_dist_xy < SEATED_XY_THRESH and vial_z > SEATED_Z_MIN
              and phase_name in ("release", "retract", "home2", "finished"))
    if seated:
        return 3
    if grip_closed and lift_height > LIFT_THRESH:
        return 2
    if grip_closed and near_pinch:
        return 1
    return 0


def _run_episode(mj, model, data, renderer, idx, ik, *, fps, area_x, area_y, yaw_jitter,
                 rng, max_steps, width, height, keep_every, max_frames,
                 visual_dr=None, dr_strength=1.0):  # type: ignore[no-untyped-def]
    """Roll out one adaptive pick-place; return kept frames + auto GT labels."""
    key = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
    mj.mj_resetDataKeyframe(model, data, key)
    vq = idx["vial_qadr"]
    if area_x > 0:
        data.qpos[vq + 0] += float(rng.uniform(-area_x, area_x))
    if area_y > 0:
        data.qpos[vq + 1] += float(rng.uniform(-area_y, area_y))
    if yaw_jitter > 0:
        yaw = float(rng.uniform(-yaw_jitter, yaw_jitter))
        data.qpos[vq + 3] = math.cos(yaw / 2.0)
        data.qpos[vq + 4] = 0.0
        data.qpos[vq + 5] = 0.0
        data.qpos[vq + 6] = math.sin(yaw / 2.0)
    mj.mj_forward(model, data)
    if visual_dr is not None:
        _apply_visual_dr(model, visual_dr, rng, strength=dr_strength)

    arm_act, grip_act = idx["arm_act"], idx["grip_act"]
    arm_qadr, grip_qadr = idx["arm_qadr"], idx["grip_qadr"]
    base_body = idx["base_body"]

    def read_arm() -> tuple[float, ...]:
        return tuple(float(data.qpos[a]) for a in arm_qadr)

    home_q = read_arm()
    vial_xyz = np.array(data.xpos[idx["vial_body"]], dtype=np.float64)
    pocket_xyz = np.array(data.site_xpos[idx["pocket_site"]], dtype=np.float64)
    plan = build_adaptive_phases(ik, vial_xyz=vial_xyz, pocket_xyz=pocket_xyz, home_q=home_q)
    ctrl = PickPlaceController(phases=plan.phases)

    for a, q in zip(arm_act, read_arm(), strict=True):
        data.ctrl[a] = q
    data.ctrl[grip_act] = GRIP_OPEN
    for _ in range(SETTLE_STEPS):
        mj.mj_step(model, data)
    ctrl.reset(read_arm())
    decim = max(1, round((1.0 / fps) / model.opt.timestep))

    z0 = float(data.xpos[idx["vial_body"], 2])
    z_max = z0
    cams = idx["cams"]

    ext_frames, wrist_frames = [], []
    tgt_base, uv_ext, uv_wrist, labels = [], [], [], []
    depth_ext, depth_wrist = [], []
    frame_i = 0
    step = 0
    while step < max_steps:
        measured = read_arm()
        out = ctrl.step(measured)
        for a, q in zip(arm_act, out.arm_ctrl, strict=True):
            data.ctrl[a] = q
        data.ctrl[grip_act] = out.grip_ctrl
        mj.mj_step(model, data)
        step += 1
        z_max = max(z_max, float(data.xpos[idx["vial_body"], 2]))

        if step % decim == 0:
            take = (frame_i % keep_every == 0) and (len(labels) < max_frames)
            frame_i += 1
            if take:
                # --- GT geometry (free) ---
                vxyz = np.array(data.xpos[idx["vial_body"]], dtype=np.float64)
                vmat = np.array(data.xmat[idx["vial_body"]], dtype=np.float64).reshape(3, 3)
                cap_world = vxyz + vmat @ np.array([0.0, 0.0, VIAL_CAP_OFFSET_Z])
                pinch_xyz = np.array(data.site_xpos[idx["pinch_site"]], dtype=np.float64)
                pkt_xyz = np.array(data.site_xpos[idx["pocket_site"]], dtype=np.float64)
                grip_meas = float(np.clip(data.qpos[grip_qadr] / GRIP_DRIVER_RANGE, 0.0, 1.0))
                near = float(np.linalg.norm(vxyz - pinch_xyz)) < NEAR_PINCH_THRESH
                lift_h = float(data.xpos[idx["vial_body"], 2] - z0)
                place_xy = float(np.linalg.norm(vxyz[:2] - pkt_xyz[:2]))
                lab = label_phase(
                    grip_meas=grip_meas, near_pinch=near, lift_height=lift_h,
                    place_dist_xy=place_xy, vial_z=float(vxyz[2]),
                    phase_name=out.phase_name,
                )
                # --- cap point in robot base frame (deployable target) ---
                bpos = np.array(data.xpos[base_body], dtype=np.float64)
                bmat = np.array(data.xmat[base_body], dtype=np.float64).reshape(3, 3)
                cap_base = bmat.T @ (cap_world - bpos)
                # --- projections into both cameras ---
                ue, ve, de, _ = project(
                    data.cam_xpos[cams["exterior"]], data.cam_xmat[cams["exterior"]],
                    float(model.cam_fovy[cams["exterior"]]), cap_world, width, height)
                uw, vw, dw, _ = project(
                    data.cam_xpos[cams["wrist"]], data.cam_xmat[cams["wrist"]],
                    float(model.cam_fovy[cams["wrist"]]), cap_world, width, height)
                # --- render both views ---
                renderer.update_scene(data, camera="room")
                ext_frames.append(renderer.render().copy())
                renderer.update_scene(data, camera="wrist")
                wrist_frames.append(renderer.render().copy())
                tgt_base.append(cap_base.astype(np.float32))
                uv_ext.append(np.array([ue, ve], dtype=np.float32))
                uv_wrist.append(np.array([uw, vw], dtype=np.float32))
                depth_ext.append(np.float32(de))
                depth_wrist.append(np.float32(dw))
                labels.append(np.int64(lab))
        if out.finished or len(labels) >= max_frames:
            break

    vxyz = np.array(data.xpos[idx["vial_body"]], dtype=np.float64)
    pkt_xyz = np.array(data.site_xpos[idx["pocket_site"]], dtype=np.float64)
    place_dist = float(np.linalg.norm(vxyz[:2] - pkt_xyz[:2]))
    lifted = (z_max - z0) > 0.02
    seated = place_dist < 0.05 and vxyz[2] > 0.20
    success = bool(lifted and seated)
    return {
        "ext": ext_frames, "wrist": wrist_frames, "tgt_base": tgt_base,
        "uv_ext": uv_ext, "uv_wrist": uv_wrist, "depth_ext": depth_ext,
        "depth_wrist": depth_wrist, "labels": labels, "success": success,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-episodes", type=int, default=40)
    ap.add_argument("--fps", type=int, default=emb.DEFAULT_FPS)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--area-x", type=float, default=0.05)
    ap.add_argument("--area-y", type=float, default=0.07)
    ap.add_argument("--yaw-jitter", type=float, default=0.08)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--noslip-iterations", type=int, default=20)
    ap.add_argument("--visual-dr", type=float, default=1.0)
    ap.add_argument("--keep-every", type=int, default=3,
                    help="keep every Nth decimated frame (subsample the long reaching phase)")
    ap.add_argument("--max-frames-per-ep", type=int, default=90)
    ap.add_argument("--max-total-frames", type=int, default=3600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import mujoco as mj

    xml_path = Path(args.xml)
    if not xml_path.is_file():
        print(f"ERROR: MJCF not found: {xml_path}", file=sys.stderr)
        return 2
    model = mj.MjModel.from_xml_path(str(xml_path))
    if args.noslip_iterations > 0:
        model.opt.noslip_iterations = args.noslip_iterations
    data = mj.MjData(model)
    idx = _build_index(mj, model)
    freed = _free_arm_collision(mj, model)
    renderer = mj.Renderer(model, args.height, args.width)
    ik = DampedLeastSquaresIK(
        model=model, site_id=idx["pinch_site"],
        arm_qadr=tuple(idx["arm_qadr"]), arm_dofadr=tuple(idx["arm_dofadr"]),
    )
    print(f"[percep] loaded {xml_path.name}: freed {freed} arm geoms; "
          f"render {args.width}x{args.height}; visual_dr={args.visual_dr}")
    visual_snap = _snapshot_visual(mj, model) if args.visual_dr > 0 else None
    rng = np.random.default_rng(args.seed)

    all_ext, all_wrist = [], []
    all_tgt, all_uv_e, all_uv_w, all_de, all_dw, all_lab, all_ep, all_succ = (
        [], [], [], [], [], [], [], [])
    total = 0
    try:
        for ep in range(args.num_episodes):
            nominal = ep == 0
            res = _run_episode(
                mj, model, data, renderer, idx, ik, fps=args.fps,
                area_x=0.0 if nominal else args.area_x,
                area_y=0.0 if nominal else args.area_y,
                yaw_jitter=0.0 if nominal else args.yaw_jitter,
                rng=rng, max_steps=args.max_steps, width=args.width, height=args.height,
                keep_every=args.keep_every, max_frames=args.max_frames_per_ep,
                visual_dr=None if nominal else visual_snap, dr_strength=args.visual_dr,
            )
            n = len(res["labels"])
            all_ext.extend(res["ext"])
            all_wrist.extend(res["wrist"])
            all_tgt.extend(res["tgt_base"])
            all_uv_e.extend(res["uv_ext"])
            all_uv_w.extend(res["uv_wrist"])
            all_de.extend(res["depth_ext"])
            all_dw.extend(res["depth_wrist"])
            all_lab.extend(res["labels"])
            all_ep.extend([ep] * n)
            all_succ.extend([int(res["success"])] * n)
            total += n
            counts = np.bincount(np.array(res["labels"], dtype=int), minlength=4).tolist()
            print(f"[percep] ep {ep:03d}: kept {n:3d} frames  labels(r/g/l/s)={counts}  "
                  f"success={res['success']}  total={total}")
            if total >= args.max_total_frames:
                print(f"[percep] hit max-total-frames={args.max_total_frames}; stopping")
                break
    finally:
        renderer.close()

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

    lab_arr = np.array(all_lab, dtype=int)
    counts = np.bincount(lab_arr, minlength=4).tolist()
    meta = {
        "num_frames": int(ext.shape[0]),
        "width": args.width, "height": args.height,
        "class_names": ["reaching", "grasped", "lifted", "seated"],
        "class_counts": counts,
        "cap_offset_z_m": VIAL_CAP_OFFSET_Z,
        "target_frame": "robot_base",
        "cameras": list(emb.VIDEO_KEY_TO_CAMERA.keys()),
        "visual_dr": args.visual_dr,
        "num_episodes_run": int(max(all_ep) + 1) if all_ep else 0,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[percep] wrote {ext.shape[0]} frames to {out}")
    print(f"[percep] class counts (reaching/grasped/lifted/seated): {counts}")
    print(f"[percep] cap keypoint depth (exterior) mean={np.mean(all_de):.3f}m "
          f"(wrist) mean={np.mean(all_dw):.3f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
