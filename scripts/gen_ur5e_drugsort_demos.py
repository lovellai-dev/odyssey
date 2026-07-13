#!/usr/bin/env python3
"""Generate GR00T demonstration data for the UR5e / Robotiq drug-sorting cell.

Loads the AseptiPack fill-finish MJCF in headless MuJoCo, replays the scripted
pick-and-place policy (a Python port of the browser controller — see
``odyssey.embodiments.ur5e_drugsort.policy``) with domain randomisation of the
vial pose, renders the ``front`` + ``wrist`` cameras offscreen, and records each
episode into the GR00T-flavoured LeRobot v2.1 dataset format via
:class:`odyssey.datasets.LeRobotDatasetWriter`.

No GPU is required to *generate* data (only an OpenGL/EGL context for offscreen
rendering); the resulting dataset feeds ``gr00t.experiment.launch_finetune`` on
a GPU box (see ``examples/ur5e-drugsort/RUNBOOK.md``).

Usage::

    MUJOCO_GL=egl python scripts/gen_ur5e_drugsort_demos.py \
        --xml /path/to/aseptipack.xml --out /tmp/ur5e_drugsort --num-episodes 3

The first episode is always nominal (no jitter) to guarantee a clean reference
demonstration; the rest are domain-randomised.
"""

from __future__ import annotations

import argparse
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
    for n in ARM_ACTUATOR_NAMES:
        j = m2i(model, obj.mjOBJ_JOINT, f"{n}_joint")
        arm_qadr.append(int(model.jnt_qposadr[j]))
    grip_j = m2i(model, obj.mjOBJ_JOINT, GRIP_DRIVER_JOINT)
    grip_qadr = int(model.jnt_qposadr[grip_j])

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
    return {
        "arm_act": arm_act, "grip_act": grip_act, "arm_qadr": arm_qadr,
        "grip_qadr": grip_qadr, "vial_body": vial_body, "vial_qadr": vial_qadr,
        "pocket_site": pocket_site,
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


def _run_episode(mj, model, data, renderer, idx, ctrl, *, fps, jitter, rng, max_steps):  # type: ignore[no-untyped-def]
    """Simulate one scripted pick-and-place; return recorded arrays + success."""
    key = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
    mj.mj_resetDataKeyframe(model, data, key)

    # --- Domain randomisation: jitter the vial's initial pose ---------------
    vq = idx["vial_qadr"]
    if jitter > 0:
        data.qpos[vq + 0] += float(rng.uniform(-jitter, jitter))
        data.qpos[vq + 1] += float(rng.uniform(-jitter, jitter))
        yaw = float(rng.uniform(-0.08, 0.08))
        data.qpos[vq + 3] = math.cos(yaw / 2.0)
        data.qpos[vq + 4] = 0.0
        data.qpos[vq + 5] = 0.0
        data.qpos[vq + 6] = math.sin(yaw / 2.0)
    mj.mj_forward(model, data)

    arm_act, grip_act = idx["arm_act"], idx["grip_act"]
    arm_qadr, grip_qadr = idx["arm_qadr"], idx["grip_qadr"]

    def read_arm() -> tuple[float, ...]:
        return tuple(float(data.qpos[a]) for a in arm_qadr)

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
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default=DEFAULT_XML, help="path to aseptipack.xml")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--num-episodes", type=int, default=3)
    ap.add_argument("--fps", type=int, default=emb.DEFAULT_FPS)
    ap.add_argument("--width", type=int, default=emb.DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=emb.DEFAULT_HEIGHT)
    ap.add_argument("--jitter", type=float, default=0.006,
                    help="vial xy jitter half-range (m); episode 0 is always nominal")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instruction", default=emb.DEFAULT_INSTRUCTION)
    args = ap.parse_args()

    import mujoco as mj  # imported here so the module stays import-safe without GL

    xml_path = Path(args.xml)
    if not xml_path.is_file():
        print(f"ERROR: MJCF not found: {xml_path}", file=sys.stderr)
        return 2
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    idx = _build_index(mj, model)
    freed = _free_arm_collision(mj, model)
    print(f"[gen] loaded {xml_path.name}: nq={model.nq} nu={model.nu} "
          f"ncam={model.ncam}; freed {freed} arm-collision geoms; "
          f"timestep={model.opt.timestep} decim={max(1, round((1.0/args.fps)/model.opt.timestep))}")

    renderer = mj.Renderer(model, args.height, args.width)
    ctrl = PickPlaceController()
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
    try:
        for ep in range(args.num_episodes):
            jitter = 0.0 if ep == 0 else args.jitter
            result = _run_episode(
                mj, model, data, renderer, idx, ctrl,
                fps=args.fps, jitter=jitter, rng=rng, max_steps=args.max_steps,
            )
            n_success += int(result["success"])
            writer.add_episode(
                states=result["states"], actions=result["actions"],
                frames=result["frames"], task=args.instruction,
            )
            print(
                f"[gen] ep {ep:03d}: frames={result['num_frames']:4d} "
                f"sim_steps={result['sim_steps']:6d} "
                f"lift={result['lift_height']*100:5.1f}cm "
                f"place_dist={result['place_dist']*100:5.1f}cm "
                f"{'SUCCESS' if result['success'] else 'FAIL'} "
                f"(lifted={result['lifted']} seated={result['seated']})"
            )
    finally:
        renderer.close()

    info = writer.finalize()
    print(f"[gen] wrote {info['total_episodes']} episodes / {info['total_frames']} "
          f"frames to {args.out}")
    print(f"[gen] grasp+place success: {n_success}/{args.num_episodes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
