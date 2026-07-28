"""Headless closed-loop executor for precomputed drug-sort plans.

Replays each plan with the exact browser_harness.js semantics (ease interpolation,
MOVE=800, hold='conv' -> CONV/CONV_MAX) in python MuJoCo and scores lift + seating.
Usage: exec_plans_headless.py <xml> <plans.json> [--render out.mp4]
       --render: additionally record episode 0 from the 'room' camera at 20 Hz
                 (needs offscreen GL + the ffmpeg binary; skipped with a warning
                 if either is unavailable — scoring is unaffected).
"""
import json
import subprocess
import sys

import mujoco as mj
import numpy as np

XML, PLANS = sys.argv[1], sys.argv[2]
RENDER_MP4 = sys.argv[sys.argv.index("--render") + 1] if "--render" in sys.argv else None
P = json.load(open(PLANS))
m = mj.MjModel.from_xml_path(XML)
MOVE, CONV, CONV_MAX = 800, 6e-3, 2200
ARM = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

ease = lambda t: 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2
aid = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM]
gid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, "gr_fingers_actuator")
jq = [m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n + "_joint")] for n in ARM]

sid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "gr_pinch")

# Deployment semantics (browser_harness.js SETUP): free arm-link + non-pad gripper
# collisions — the arm swings free above the worktop; only the pads grip.
ARM_BODIES = {"base", "shoulder_link", "upper_arm_link", "forearm_link",
              "wrist_1_link", "wrist_2_link", "wrist_3_link"}
freed = 0
for g in range(m.ngeom):
    bn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
    gn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or ""
    if (bn in ARM_BODIES or bn.startswith("gr_")) and "pad" not in gn:
        m.geom_contype[g] = 0
        m.geom_conaffinity[g] = 0
        freed += 1
print("freed %d arm/gripper collision geoms" % freed)

renderer, ffmpeg = None, None
if RENDER_MP4:
    try:
        renderer = mj.Renderer(m, 480, 640)
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", "640x480", "-framerate", "20", "-i", "-", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", RENDER_MP4],
            stdin=subprocess.PIPE)
    except Exception as e:  # no offscreen GL / no ffmpeg — score without video
        print("render disabled (%s: %s)" % (type(e).__name__, e))
        renderer = None

ok = 0
for ep in P["plans"]:
    d = mj.MjData(m)
    mj.mj_resetDataKeyframe(m, d, 0)
    d.qpos[14:21] = ep["vial_qpos"]
    mj.mj_forward(m, d)
    phases, phase, s, step, zmax = ep["phases"], 0, 0, 0, 0.0
    prev = list(ep["home_q"])
    ph_report = []
    while step < 30000 and phase < len(phases):
        meas = [d.qpos[j] for j in jq]
        ph = phases[phase]
        same = all(abs(ph["q"][i] - prev[i]) < 1e-3 for i in range(6))
        mdur = 0 if same else (ph["move_steps"] if ph.get("move_steps") is not None else MOVE)
        armc = [prev[i] + (ph["q"][i] - prev[i]) * ease(s / mdur) for i in range(6)] if s < mdur else ph["q"]
        s += 1
        if s >= mdur:
            held = s - mdur
            if ph["hold"] == "conv":
                done = max(abs(ph["q"][i] - meas[i]) for i in range(6)) < CONV or held > CONV_MAX
            else:
                done = held >= ph["hold"]
            if done:
                pin, vial = d.site_xpos[sid], d.qpos[14:17]
                ph_report.append("%s(err %.0fmm, pin-vial %.1f/%.1fcm)" % (
                    ph["name"],
                    max(abs(ph["q"][i] - meas[i]) for i in range(6)) * 1000,
                    np.linalg.norm(pin[:2] - vial[:2]) * 100, (pin[2] - vial[2]) * 100))
                prev = list(ph["q"])
                phase += 1
                s = 0
        for i, a in enumerate(aid):
            d.ctrl[a] = armc[i]
        d.ctrl[gid] = ph["grip"]
        mj.mj_step(m, d)
        step += 1
        zmax = max(zmax, d.qpos[16])
        if renderer is not None and ep is P["plans"][0] and step % 25 == 0:
            renderer.update_scene(d, camera="room")
            ffmpeg.stdin.write(np.ascontiguousarray(renderer.render()).tobytes())
    vial, pocket = d.qpos[14:17], np.array(ep["pocket_xyz"])
    quat = d.qpos[17:21]
    up = 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)
    dxy = float(np.linalg.norm(vial[:2] - pocket[:2]))
    lifted = zmax > ep["vial_xyz"][2] + 0.05
    seated = dxy < 0.025 and abs(vial[2] - pocket[2]) < 0.02 and up > 0.9
    print("ep%02d: steps=%d lifted=%s zmax=%.3f dxy=%.1fcm up=%.2f seated=%s"
          % (ep["episode"], step, lifted, zmax, dxy * 100, up, seated))
    print("   phases: " + " | ".join(ph_report))
    ok += int(seated)
if renderer is not None:
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print("RENDERED %s" % RENDER_MP4)
print("EXPERT_CLOSED_LOOP %d/%d seated" % (ok, len(P["plans"])))
