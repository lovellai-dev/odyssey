#!/usr/bin/env python3
"""Precompute per-episode adaptive-IK expert plans for the BROWSER data-gen harness.

This is the Python half of the **browser-frame (Three.js) data-generation harness**
(the counterpart to ``scripts/gen_ur5e_drugsort_demos.py``, which renders with
MuJoCo's ``Renderer``). Its whole purpose is to close the *render gap*: GR00T and
the Observer are trained on MuJoCo frames but deploy on the Lovell AI Robot
Playground's **Three.js** renderer, so both degrade in-browser. To retrain on the
deployment appearance we must first *capture* frames from the Playground's real
renderer — which is what ``browser_harness.js`` does, replaying the plans emitted
here.

This module reuses the EXACT headless expert (``odyssey.embodiments.ur5e_drugsort.ik``
+ ``policy``): for each episode it resets the AseptiPack MJCF to the ``home``
keyframe, domain-randomises the vial free-joint pose (same scheme/ranges as the
headless generator), reads the vial / pocket poses, and solves the damped-least-
squares Jacobian IK to produce the adaptive joint-target waypoints. It writes
``plans.json`` — one entry per episode carrying the vial qpos to set in the
browser + the 10 adaptive waypoints + home_q — so the browser harness replays a
byte-identical expert against MuJoCo-WASM physics while the Playground's Three.js
renderer captures the exterior + wrist frames.

No rendering / physics rollout happens here (that is the browser's job); this only
runs the vial randomisation + IK, which needs the Python ``mujoco`` model.

Usage::

    python precompute_plans.py --xml /path/to/aseptipack.xml \
        --out plans.json --num-episodes 5 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # only for the IK solver's scratch MjData; no rendering here
import numpy as np

# Resolve the odyssey source tree relative to this file (examples/ur5e-drugsort/browser_capture/).
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from odyssey.embodiments.ur5e_drugsort.ik import DampedLeastSquaresIK, build_adaptive_phases  # noqa: E402
from odyssey.embodiments.ur5e_drugsort.policy import ARM_ACTUATOR_NAMES  # noqa: E402

# The AseptiPack MJCF ships in the lai-agent Playground tree (same file the browser loads).
DEFAULT_XML = os.environ.get(
    "ASEPTIPACK_XML",
    "/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/"
    "aseptipack_description/aseptipack.xml",
)


def build_index(mj, model):
    m2i, obj = mj.mj_name2id, mj.mjtObj
    arm_qadr, arm_dofadr = [], []
    for n in ARM_ACTUATOR_NAMES:
        j = m2i(model, obj.mjOBJ_JOINT, f"{n}_joint")
        arm_qadr.append(int(model.jnt_qposadr[j]))
        arm_dofadr.append(int(model.jnt_dofadr[j]))
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
    return {
        "arm_qadr": arm_qadr, "arm_dofadr": arm_dofadr, "pinch_site": pinch_site,
        "vial_body": vial_body, "vial_qadr": vial_qadr, "pocket_site": pocket_site,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default=DEFAULT_XML, help="path to aseptipack.xml (the file the browser loads)")
    ap.add_argument("--out", default="plans.json")
    ap.add_argument("--num-episodes", type=int, default=5)
    ap.add_argument("--area-x", type=float, default=0.05, help="vial x randomisation half-range (m)")
    ap.add_argument("--area-y", type=float, default=0.07, help="vial y randomisation half-range (m)")
    ap.add_argument("--yaw-jitter", type=float, default=0.08, help="vial yaw randomisation half-range (rad)")
    ap.add_argument("--seed", type=int, default=0)
    # Trajectory-shape DR (opt-in; for diverse SFT data, NOT for eval poses)
    ap.add_argument("--randomize", action="store_true",
                    help="domain-randomize the expert trajectory SHAPE (diverse SFT data)")
    ap.add_argument("--dr-clearance", type=float, default=0.04, help="hover-clearance jitter (m)")
    ap.add_argument("--dr-approach-xy", type=float, default=0.03, help="angled-approach column offset (m)")
    ap.add_argument("--dr-grasp-z", type=float, default=0.003, help="grasp-depth jitter (m)")
    ap.add_argument("--dr-carry-z", type=float, default=0.03, help="transport-carry-height jitter (m)")
    ap.add_argument("--dr-vel-lo", type=float, default=0.7, help="min per-episode velocity scale")
    ap.add_argument("--dr-vel-hi", type=float, default=1.4, help="max per-episode velocity scale")
    args = ap.parse_args()

    import mujoco as mj

    if not Path(args.xml).is_file():
        print(f"ERROR: MJCF not found: {args.xml}", file=sys.stderr)
        return 2
    model = mj.MjModel.from_xml_path(args.xml)
    data = mj.MjData(model)
    idx = build_index(mj, model)
    home_key = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
    ik = DampedLeastSquaresIK(
        model=model, site_id=idx["pinch_site"],
        arm_qadr=tuple(idx["arm_qadr"]), arm_dofadr=tuple(idx["arm_dofadr"]),
    )
    rng = np.random.default_rng(args.seed)
    vq = idx["vial_qadr"]

    plans = []
    for ep in range(args.num_episodes):
        nominal = ep == 0  # episode 0 is always nominal (clean reference), matching the headless generator
        ax = 0.0 if nominal else args.area_x
        ay = 0.0 if nominal else args.area_y
        yj = 0.0 if nominal else args.yaw_jitter

        mj.mj_resetDataKeyframe(model, data, home_key)
        if ax > 0:
            data.qpos[vq + 0] += float(rng.uniform(-ax, ax))
        if ay > 0:
            data.qpos[vq + 1] += float(rng.uniform(-ay, ay))
        if yj > 0:
            yaw = float(rng.uniform(-yj, yj))
            data.qpos[vq + 3] = math.cos(yaw / 2.0)
            data.qpos[vq + 4] = 0.0
            data.qpos[vq + 5] = 0.0
            data.qpos[vq + 6] = math.sin(yaw / 2.0)
        mj.mj_forward(model, data)

        home_q = tuple(float(data.qpos[a]) for a in idx["arm_qadr"])
        vial_qpos = [float(data.qpos[vq + i]) for i in range(7)]
        vial_xyz = np.array(data.xpos[idx["vial_body"]], dtype=np.float64)
        pocket_xyz = np.array(data.site_xpos[idx["pocket_site"]], dtype=np.float64)
        # Trajectory-shape domain randomization (opt-in): breaks the single-shape
        # stereotypy so the SFT base must actually use the target/scene rather
        # than memorize one wrist-pixels->action mapping (the attention collapse).
        dr = None
        if args.randomize and not nominal:
            dr = {"rng": rng, "clearance": args.dr_clearance,
                  "approach_xy": args.dr_approach_xy, "grasp_z": args.dr_grasp_z,
                  "carry_z": args.dr_carry_z, "vel_scale": (args.dr_vel_lo, args.dr_vel_hi)}
        plan = build_adaptive_phases(ik, vial_xyz=vial_xyz, pocket_xyz=pocket_xyz,
                                     home_q=home_q, dr=dr)

        plans.append({
            "episode": ep, "nominal": nominal,
            "vial_qpos": vial_qpos, "home_q": list(home_q),
            "vial_xyz": vial_xyz.tolist(), "pocket_xyz": pocket_xyz.tolist(),
            "phases": [
                {"name": w.name, "q": list(w.q), "grip": w.grip,
                 "hold": w.hold, "move_steps": w.move_steps}
                for w in plan.phases
            ],
            "ik_max_err_mm": plan.max_pos_error * 1000.0,
            "ik_converged": bool(plan.all_converged),
        })
        print(f"[precompute] ep {ep:02d} {'NOMINAL' if nominal else 'rand'} "
              f"vial=[{vial_qpos[0]:.3f},{vial_qpos[1]:.3f}] "
              f"ik_err={plan.max_pos_error*1000:.2f}mm conv={plan.all_converged}")

    meta = {
        "xml": args.xml, "num_episodes": args.num_episodes, "seed": args.seed,
        "randomisation": {"area_x": args.area_x, "area_y": args.area_y, "yaw": args.yaw_jitter},
        "arm_actuator_names": list(ARM_ACTUATOR_NAMES),
        "plans": plans,
    }
    Path(args.out).write_text(json.dumps(meta, indent=2) + "\n")
    approach = np.array([p["phases"][1]["q"] for p in plans])  # the 'approach' waypoint
    print(f"[precompute] wrote {len(plans)} plans -> {args.out}")
    print(f"[precompute] approach-joint std across eps (adaptivity proof): "
          f"max={approach.std(axis=0).max():.4f} rad "
          f"({'ADAPTIVE' if approach.std(axis=0).max() > 1e-3 else 'FIXED'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
