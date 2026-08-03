#!/usr/bin/env python3
"""Generate GR00T demonstration data for the UR5e / Robotiq drug-sorting cell.

Loads the AseptiPack fill-finish MJCF in headless MuJoCo, drives the **adaptive
IK expert** (:mod:`odyssey.embodiments.ur5e_drugsort.ik`) with domain
randomisation of the vial pose, renders the single ``exterior`` camera offscreen,
and records each episode into the GR00T-flavoured LeRobot v2.1 dataset format via
:class:`odyssey.datasets.LeRobotDatasetWriter`.

Unlike the fixed scripted policy (``--fixed-waypoints`` — kept as the
zero-variance baseline), the IK expert re-solves the arm joint targets to the
*actual* randomised vial / pocket pose each episode, so the recorded
action/proprio trajectories genuinely vary with the scene (the first-approach
joint-angle variance is logged + written to ``ik_adaptivity.json`` as proof).
The FSM's interpolation, convergence and gripper open/close timing are unchanged
(:class:`odyssey.embodiments.ur5e_drugsort.policy.PickPlaceController`); only the
joint *targets* adapt.

No GPU is required to *generate* data (only an OpenGL/EGL context for offscreen
rendering); the resulting dataset feeds ``gr00t.experiment.launch_finetune`` on
a GPU box (see ``examples/ur5e-drugsort/RUNBOOK.md``).

Usage::

    MUJOCO_GL=egl python scripts/gen_ur5e_drugsort_demos.py \
        --xml /path/to/aseptipack.xml --out /tmp/ur5e_drugsort --num-episodes 64

The first episode is always nominal (no randomisation) to guarantee a clean
reference demonstration; the rest are domain-randomised over the reachable tray.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# Ensure an offscreen GL backend is selected before mujoco is imported. EGL
# works on headless GPU hosts; override with MUJOCO_GL=glfw on a desktop.
os.environ.setdefault("MUJOCO_GL", "egl")

# Make the package importable when run straight from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from odyssey.datasets import LeRobotDatasetWriter
from odyssey.embodiments.ur5e_drugsort import embodiment as emb
from odyssey.embodiments.ur5e_drugsort.ik import (
    DampedLeastSquaresIK,
    build_adaptive_phases,
)
from odyssey.embodiments.ur5e_drugsort.policy import (
    ARM_ACTUATOR_NAMES,
    GRIP_CLOSE,
    GRIP_OPEN,
    GRIPPER_ACTUATOR_NAME,
    SETTLE_STEPS,
    PickPlaceController,
)

DEFAULT_XML = (
    "/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/"
    "aseptipack_description/aseptipack.xml"
)
# Gripper driver joint travel (rad) used to normalise measured closure to [0, 1].
GRIP_DRIVER_JOINT = "gr_right_driver_joint"
GRIP_DRIVER_RANGE = 0.8
# Arm links + gripper (non-pad) geoms have their collisions freed so the arm
# swings freely over the worktop; only the finger pads collide (to grip).
ARM_BODIES = frozenset(
    {"base", "shoulder_link", "upper_arm_link", "forearm_link",
     "wrist_1_link", "wrist_2_link", "wrist_3_link"}
)


def _build_index(mj, model):  # type: ignore[no-untyped-def]
    """Resolve the actuator / joint / body / camera ids we drive and observe."""
    m2i = mj.mj_name2id
    obj = mj.mjtObj
    arm_act = [m2i(model, obj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATOR_NAMES]
    grip_act = m2i(model, obj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME)
    arm_qadr = []
    arm_dofadr = []
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
    if any(a < 0 for a in arm_act) or grip_act < 0:
        raise RuntimeError("could not resolve arm/gripper actuators in the MJCF")
    if pinch_site < 0:
        raise RuntimeError("could not resolve the gr_pinch site in the MJCF")
    return {
        "arm_act": arm_act, "grip_act": grip_act, "arm_qadr": arm_qadr,
        "arm_dofadr": arm_dofadr, "grip_qadr": grip_qadr, "pinch_site": pinch_site,
        "vial_body": vial_body, "vial_qadr": vial_qadr, "pocket_site": pocket_site,
    }


def _free_arm_collision(mj, model) -> int:  # type: ignore[no-untyped-def]
    """Disable collisions on arm links + non-pad gripper geoms (port of the JS)."""
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
    """Copy the render-only model fields we domain-randomise so each episode can
    reset to nominal before re-jittering (jitter stays centred on nominal)."""
    room_cam = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "room")
    snap = {
        "room_cam": room_cam,
        "cam_pos": model.cam_pos.copy(),
        "cam_fovy": model.cam_fovy.copy(),
        "geom_rgba": model.geom_rgba.copy(),
        "hl_diffuse": np.array(model.vis.headlight.diffuse, dtype=np.float64),
        "hl_ambient": np.array(model.vis.headlight.ambient, dtype=np.float64),
        "hl_specular": np.array(model.vis.headlight.specular, dtype=np.float64),
    }
    snap["mat_rgba"] = model.mat_rgba.copy() if model.nmat else None
    snap["light_pos"] = model.light_pos.copy() if model.nlight else None
    snap["light_diffuse"] = model.light_diffuse.copy() if model.nlight else None
    snap["light_ambient"] = model.light_ambient.copy() if model.nlight else None
    return snap


def _apply_visual_dr(model, snap, rng, strength=1.0):  # type: ignore[no-untyped-def]
    """Randomise render-only fields (camera pose/FOV, lighting, material colour).

    None of this touches physics or the recorded state/action — only the rendered
    ``exterior`` frames vary, so a policy trained on them is robust to appearance
    shifts (crucially: the Playground's Three.js render, which is lit/shaded
    differently from MuJoCo's). Reset to nominal first so jitter stays centred.
    """
    # --- reset to nominal ---
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
    # --- camera: viewpoint + zoom jitter (robust to the deploy camera being off) ---
    cid = snap["room_cam"]
    if cid >= 0:
        model.cam_pos[cid] = snap["cam_pos"][cid] + rng.uniform(-0.05, 0.05, 3) * s
        model.cam_fovy[cid] = float(snap["cam_fovy"][cid] + rng.uniform(-2.5, 2.5) * s)
    # --- lighting: global intensity + colour temperature + light position ---
    diff = float(np.clip(1.0 + rng.uniform(-0.4, 0.4) * s, 0.2, 1.8))
    amb = float(np.clip(1.0 + rng.uniform(-0.5, 0.6) * s, 0.2, 2.0))
    warm = 1.0 + rng.uniform(-0.12, 0.12, 3) * s  # per-channel colour-temperature tilt
    model.vis.headlight.diffuse[:] = np.clip(snap["hl_diffuse"] * diff * warm, 0.0, 1.0)
    model.vis.headlight.ambient[:] = np.clip(snap["hl_ambient"] * amb, 0.0, 1.0)
    if snap["light_pos"] is not None and snap["light_pos"].size:
        model.light_pos[:] = snap["light_pos"] + rng.uniform(-0.35, 0.35, snap["light_pos"].shape) * s
        model.light_diffuse[:] = np.clip(snap["light_diffuse"] * diff, 0.0, 1.0)
        model.light_ambient[:] = np.clip(snap["light_ambient"] * amb, 0.0, 1.0)
    # --- material / geom colour: a global tint + small per-element jitter ---
    tint = np.clip(1.0 + rng.uniform(-0.18, 0.18, 3) * s, 0.4, 1.6)
    if snap["mat_rgba"] is not None and model.nmat:
        base = snap["mat_rgba"][:, :3]
        jit = 1.0 + rng.uniform(-0.10, 0.10, base.shape) * s
        model.mat_rgba[:, :3] = np.clip(base * tint * jit, 0.0, 1.0)
    gbase = snap["geom_rgba"][:, :3]
    gjit = 1.0 + rng.uniform(-0.10, 0.10, gbase.shape) * s
    model.geom_rgba[:, :3] = np.clip(gbase * tint * gjit, 0.0, 1.0)


def _run_episode(mj, model, data, renderer, idx, ik, *, fps, area_x, area_y, yaw_jitter, rng, max_steps,
                 visual_dr=None, dr_strength=1.0):  # type: ignore[no-untyped-def]
    """Simulate one adaptive pick-and-place; return recorded arrays + success.

    ``ik`` is a :class:`DampedLeastSquaresIK` solver (or ``None`` to replay the
    fixed scripted waypoints). ``area_x``/``area_y`` are the vial xy
    randomisation half-ranges (m); ``yaw_jitter`` the vial yaw half-range (rad).
    Pass all zero for a nominal (unrandomised) reference episode.
    """
    key = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
    mj.mj_resetDataKeyframe(model, data, key)

    # --- Domain randomisation: place the vial anywhere in the reachable box --
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

    # --- Visual domain randomisation (render-only; no physics / obs-action change) ---
    if visual_dr is not None:
        _apply_visual_dr(model, visual_dr, rng, strength=dr_strength)

    arm_act, grip_act = idx["arm_act"], idx["grip_act"]
    arm_qadr, grip_qadr = idx["arm_qadr"], idx["grip_qadr"]

    def read_arm() -> tuple[float, ...]:
        return tuple(float(data.qpos[a]) for a in arm_qadr)

    # --- Build this episode's controller ------------------------------------
    # The FSM (interpolation / convergence / gripper timing) is unchanged; only
    # the joint *targets* adapt. With IK we re-solve each Cartesian anchor to the
    # randomised vial / pocket pose; without it we replay the fixed scripted set.
    home_q = read_arm()
    if ik is not None:
        vial_xyz = np.array(data.xpos[idx["vial_body"]], dtype=np.float64)
        pocket_xyz = np.array(data.site_xpos[idx["pocket_site"]], dtype=np.float64)
        plan = build_adaptive_phases(
            ik, vial_xyz=vial_xyz, pocket_xyz=pocket_xyz, home_q=home_q
        )
        ctrl = PickPlaceController(phases=plan.phases)
        approach_q = plan.approach_q
        ik_max_err = plan.max_pos_error
        ik_converged = plan.all_converged
    else:
        ctrl = PickPlaceController()
        approach_q = ctrl.phases[1].q
        ik_max_err = 0.0
        ik_converged = True

    # Settle at the home pose (gripper open) before starting the script.
    for a, q in zip(arm_act, read_arm(), strict=True):
        data.ctrl[a] = q
    data.ctrl[grip_act] = GRIP_OPEN
    for _ in range(SETTLE_STEPS):
        mj.mj_step(model, data)

    ctrl.reset(read_arm())
    decim = max(1, round((1.0 / fps) / model.opt.timestep))

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    frames: dict[str, list[np.ndarray]] = {k: [] for k in emb.VIDEO_KEY_TO_CAMERA}

    z0 = float(data.xpos[idx["vial_body"], 2])
    z_max = z0
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
            grip_meas = float(np.clip(data.qpos[grip_qadr] / GRIP_DRIVER_RANGE, 0.0, 1.0))
            state = np.array([*measured, grip_meas], dtype=np.float32)
            action = np.array(
                [*out.arm_ctrl, out.grip_ctrl / GRIP_CLOSE], dtype=np.float32
            )
            states.append(state)
            actions.append(action)
            for vkey, cam in emb.VIDEO_KEY_TO_CAMERA.items():
                renderer.update_scene(data, camera=cam)
                frames[vkey].append(renderer.render().copy())
        if out.finished:
            break

    # --- Success: vial lifted during transport AND seated in the nest pocket ---
    vial_xyz = np.array(data.xpos[idx["vial_body"]], dtype=np.float64)
    pocket_xyz = np.array(data.site_xpos[idx["pocket_site"]], dtype=np.float64)
    place_dist = float(np.linalg.norm(vial_xyz[:2] - pocket_xyz[:2]))
    lifted = (z_max - z0) > 0.02
    seated = place_dist < 0.05 and vial_xyz[2] > 0.20
    success = bool(lifted and seated)

    states_arr = np.asarray(states, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.float32)
    return {
        "states": states_arr, "actions": actions_arr, "frames": frames,
        "success": success, "lifted": lifted, "seated": seated,
        "lift_height": float(z_max - z0), "place_dist": place_dist,
        "num_frames": len(states), "sim_steps": step,
        "approach_q": approach_q, "ik_max_err": ik_max_err,
        "ik_converged": ik_converged,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default=DEFAULT_XML, help="path to aseptipack.xml")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--fps", type=int, default=emb.DEFAULT_FPS)
    ap.add_argument("--width", type=int, default=emb.DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=emb.DEFAULT_HEIGHT)
    ap.add_argument("--area-x", type=float, default=0.05,
                    help="vial x randomisation half-range (m) over the reachable tray")
    ap.add_argument("--area-y", type=float, default=0.07,
                    help="vial y randomisation half-range (m) over the reachable tray")
    ap.add_argument("--yaw-jitter", type=float, default=0.08,
                    help="vial yaw randomisation half-range (rad) — visual jitter")
    ap.add_argument("--jitter", type=float, default=None,
                    help="DEPRECATED: if set, overrides --area-x/--area-y with this "
                         "single xy half-range (m)")
    ap.add_argument("--fixed-waypoints", action="store_true",
                    help="replay the fixed scripted joint waypoints instead of the "
                         "adaptive IK expert (for the zero-variance baseline)")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--noslip-iterations", type=int, default=20,
                    help="MuJoCo noslip solver iterations — stabilises the grasp so "
                         "the lifted vial is not ejected (0 disables; 20 is robust)")
    ap.add_argument("--visual-dr", type=float, default=0.0,
                    help="visual domain-randomisation strength (0 disables; 1.0 is the "
                         "recommended range). Randomises the exterior camera pose/FOV, "
                         "lighting intensity/colour and material colour per episode "
                         "(render-only; no obs/action change) so the policy is robust "
                         "to the Playground's Three.js appearance. Episode 0 stays "
                         "nominal (clean reference).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instruction", default=emb.DEFAULT_INSTRUCTION)
    args = ap.parse_args()

    area_x, area_y = args.area_x, args.area_y
    if args.jitter is not None:
        area_x = area_y = args.jitter

    import mujoco as mj  # imported here so the module stays import-safe without GL

    xml_path = Path(args.xml)
    if not xml_path.is_file():
        print(f"ERROR: MJCF not found: {xml_path}", file=sys.stderr)
        return 2
    model = mj.MjModel.from_xml_path(str(xml_path))
    # Enable MuJoCo's noslip friction refinement: without it the stiff pad/vial
    # contact ejects the grasped vial the instant the arm accelerates on lift
    # (both the scripted and IK experts blow up otherwise). This is an in-memory
    # solver option only — it changes no MJCF file and no recorded obs/action.
    if args.noslip_iterations > 0:
        model.opt.noslip_iterations = args.noslip_iterations
    data = mj.MjData(model)
    idx = _build_index(mj, model)
    freed = _free_arm_collision(mj, model)
    print(f"[gen] loaded {xml_path.name}: nq={model.nq} nu={model.nu} "
          f"ncam={model.ncam}; freed {freed} arm-collision geoms; "
          f"noslip_iters={model.opt.noslip_iterations}; "
          f"timestep={model.opt.timestep} decim={max(1, round((1.0/args.fps)/model.opt.timestep))}")

    renderer = mj.Renderer(model, args.height, args.width)
    # The adaptive IK expert re-solves the arm targets to the randomised vial /
    # pocket pose each episode; --fixed-waypoints falls back to the scripted set.
    ik = None
    if not args.fixed_waypoints:
        ik = DampedLeastSquaresIK(
            model=model,
            site_id=idx["pinch_site"],
            arm_qadr=tuple(idx["arm_qadr"]),
            arm_dofadr=tuple(idx["arm_dofadr"]),
        )
    print(f"[gen] expert: {'FIXED scripted waypoints' if ik is None else 'ADAPTIVE IK'}"
          f"; vial randomisation half-range xy=({area_x:.3f},{area_y:.3f})m "
          f"yaw={args.yaw_jitter:.3f}rad (episode 0 nominal)")
    # Nominal snapshot of the render-only fields for visual domain randomisation.
    visual_snap = _snapshot_visual(mj, model) if args.visual_dr > 0 else None
    _dr_msg = ("OFF" if visual_snap is None
               else f"strength={args.visual_dr:.2f} "
                    "(camera/lighting/material jitter per episode; ep0 nominal)")
    print(f"[gen] visual DR: {_dr_msg}")
    rng = np.random.default_rng(args.seed)

    writer = LeRobotDatasetWriter(
        args.out,
        modality=emb.modality_json(),
        info_builder=emb.info_json,
        video_keys=emb.VIDEO_ORIGINAL_KEYS,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )

    n_success = 0
    approach_qs: list[np.ndarray] = []
    per_ep_action_var: list[float] = []
    worst_ik_err = 0.0
    n_ik_nonconv = 0
    try:
        for ep in range(args.num_episodes):
            nominal = ep == 0
            result = _run_episode(
                mj, model, data, renderer, idx, ik,
                fps=args.fps,
                area_x=0.0 if nominal else area_x,
                area_y=0.0 if nominal else area_y,
                yaw_jitter=0.0 if nominal else args.yaw_jitter,
                rng=rng, max_steps=args.max_steps,
                visual_dr=None if nominal else visual_snap,
                dr_strength=args.visual_dr,
            )
            n_success += int(result["success"])
            approach_qs.append(np.asarray(result["approach_q"], dtype=np.float64))
            per_ep_action_var.append(float(np.var(result["actions"])))
            worst_ik_err = max(worst_ik_err, float(result["ik_max_err"]))
            n_ik_nonconv += int(not result["ik_converged"])
            writer.add_episode(
                states=result["states"], actions=result["actions"],
                frames=result["frames"], task=args.instruction,
            )
            aq = np.asarray(result["approach_q"], dtype=np.float64)
            print(
                f"[gen] ep {ep:03d}: frames={result['num_frames']:4d} "
                f"sim_steps={result['sim_steps']:6d} "
                f"lift={result['lift_height']*100:5.1f}cm "
                f"place_dist={result['place_dist']*100:5.1f}cm "
                f"ik_err={result['ik_max_err']*1000:4.1f}mm "
                f"approach_q=[{', '.join(f'{v:+.3f}' for v in aq)}] "
                f"{'SUCCESS' if result['success'] else 'FAIL'} "
                f"(lifted={result['lifted']} seated={result['seated']})"
            )
    finally:
        renderer.close()

    info = writer.finalize()
    print(f"[gen] wrote {info['total_episodes']} episodes / {info['total_frames']} "
          f"frames to {args.out}")
    print(f"[gen] grasp+place success: {n_success}/{args.num_episodes}")

    # --- Adaptivity proof: variance of the first-approach joint angles ---------
    approach_arr = np.asarray(approach_qs, dtype=np.float64)
    approach_std = approach_arr.std(axis=0)
    approach_range = approach_arr.max(axis=0) - approach_arr.min(axis=0)
    action_var_mean = float(np.mean(per_ep_action_var))
    proof = {
        "expert": "fixed_waypoints" if ik is None else "adaptive_ik",
        "num_episodes": int(args.num_episodes),
        "success": int(n_success),
        "randomisation": {
            "area_x_m": area_x, "area_y_m": area_y, "yaw_rad": args.yaw_jitter,
        },
        "approach_joint_std_rad": approach_std.tolist(),
        "approach_joint_range_rad": approach_range.tolist(),
        "approach_joint_std_max_rad": float(approach_std.max()),
        "mean_per_episode_action_variance": action_var_mean,
        "ik_worst_pos_error_mm": worst_ik_err * 1000.0,
        "ik_nonconverged_episodes": int(n_ik_nonconv),
    }
    (Path(args.out) / "ik_adaptivity.json").write_text(
        json.dumps(proof, indent=2) + "\n"
    )
    print("[gen] ADAPTIVITY PROOF (first-approach joint angles across episodes):")
    print(f"[gen]   per-joint std (rad): [{', '.join(f'{v:.4f}' for v in approach_std)}]")
    print(f"[gen]   per-joint range (rad): [{', '.join(f'{v:.4f}' for v in approach_range)}]")
    print(f"[gen]   max joint std: {approach_std.max():.4f} rad "
          f"({'ADAPTIVE — targets vary with the vial' if approach_std.max() > 1e-3 else 'FIXED — zero variance'})")
    print(f"[gen]   mean per-episode action variance: {action_var_mean:.5f}")
    if ik is not None:
        print(f"[gen]   IK worst pos error: {worst_ik_err*1000:.2f} mm; "
              f"non-converged episodes: {n_ik_nonconv}/{args.num_episodes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
