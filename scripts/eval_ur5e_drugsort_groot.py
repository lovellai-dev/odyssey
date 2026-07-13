#!/usr/bin/env python3
"""Headless closed-loop evaluation of a **served GR00T policy** in the UR5e /
Robotiq drug-sorting cell — the honest grasp+place success number.

This is the eval counterpart to :mod:`scripts.gen_ur5e_drugsort_demos`. It reuses
that harness's MuJoCo plumbing verbatim — the AseptiPack MJCF loader, the
actuator/joint/vial index (:func:`_build_index`), the arm-collision freeing
(:func:`_free_arm_collision`) and the MuJoCo ``noslip`` grasp-stabilisation — but
**replaces the adaptive IK expert with live queries to a running GR00T policy
server** (``gr00t.eval.run_gr00t_server``, ZMQ/msgpack).

Each control tick it renders the single ``exterior`` camera (256x256) and reads
the 6 arm joint positions + normalised gripper closure — exactly the observation
the fine-tune was trained on and the Playground bridge publishes — sends them to
GR00T, and applies the returned ``{q: [6] rad, grip: [0,1]}`` action chunk
through the *same* actuator mapping the expert used (arm targets in radians,
gripper ``grip * GRIP_CLOSE``). Success is scored identically to the data
generator: the vial must be lifted (>2cm) during transport **and** seated in the
nest pocket (xy < 5cm, z > 20cm) at the end.

The GR00T server must already be running (this script does not boot it)::

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m gr00t.eval.run_gr00t_server \
        --model-path <ckpt> --embodiment-tag new_embodiment --port 5555 --device cuda

    MUJOCO_GL=egl python scripts/eval_ur5e_drugsort_groot.py \
        --xml aseptipack.xml --host 127.0.0.1 --port 5555 \
        --num-episodes 20 --video-dir /tmp/groot_rollouts

Only ``numpy``/``mujoco``/``pyzmq``/``msgpack``/``msgpack_numpy`` are needed (no
torch): the policy runs entirely in the separate server process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Offscreen GL before mujoco is imported (EGL on a headless GPU host).
os.environ.setdefault("MUJOCO_GL", "egl")

# Make the package + sibling gen harness importable from a source checkout.
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))
sys.path.insert(0, str(_SCRIPTS))

import msgpack  # noqa: E402
import msgpack_numpy as mnp  # noqa: E402
import zmq  # noqa: E402

from odyssey.embodiments.ur5e_drugsort import embodiment as emb  # noqa: E402
from odyssey.embodiments.ur5e_drugsort.policy import (  # noqa: E402
    GRIP_CLOSE,
    GRIP_OPEN,
    SETTLE_STEPS,
)

# Reuse the data-gen harness's MuJoCo indexing / collision / grip-normalisation
# so the eval scene is byte-for-byte the one the demonstrations were recorded in.
from gen_ur5e_drugsort_demos import (  # noqa: E402
    GRIP_DRIVER_RANGE,
    _build_index,
    _free_arm_collision,
)

EXTERIOR_CAMERA = emb.VIDEO_KEY_TO_CAMERA["exterior"]  # MJCF "room" camera


class GrootPolicyClient:
    """Minimal ZMQ/msgpack client for ``gr00t.policy.server_client.PolicyServer``.

    Speaks the same ``msgpack_numpy`` wire protocol as the upstream
    ``PolicyClient`` (REQ socket, ``{"endpoint": ..., "data": ...}`` requests)
    so the eval process needs neither torch nor the ``gr00t`` package.
    """

    def __init__(self, host: str, port: int, timeout_ms: int = 180000) -> None:
        self._ctx = zmq.Context.instance()
        self._host, self._port, self._timeout_ms = host, port, timeout_ms
        self._connect()

    def _connect(self) -> None:
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._sock.connect(f"tcp://{self._host}:{self._port}")

    def _call(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        req: dict = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        try:
            self._sock.send(msgpack.packb(req, default=mnp.encode))
            msg = self._sock.recv()
        except zmq.error.Again:
            self._connect()  # REQ socket wedged after a timeout — reset it
            raise
        if msg == b"ERROR":
            raise RuntimeError("GR00T server returned ERROR (wrong policy server?)")
        resp = msgpack.unpackb(msg, object_hook=mnp.decode, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"GR00T server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self._call("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._connect()
            return False

    def get_action(self, observation: dict) -> dict:
        action, _info = self._call(
            "get_action", {"observation": observation, "options": None}
        )
        return action


def build_obs(arm_qpos6: np.ndarray, grip_norm: float, frame_rgb: np.ndarray,
              instruction: str) -> dict:
    """Assemble the nested GR00T observation for one tick (B=1, T=1)."""
    return {
        "video": {"exterior": frame_rgb[None, None].astype(np.uint8)},          # (1,1,H,W,3)
        "state": {
            "single_arm": arm_qpos6[None, None].astype(np.float32),             # (1,1,6)
            "gripper": np.asarray([[[grip_norm]]], dtype=np.float32),           # (1,1,1)
        },
        "language": {"annotation.human.task_description": [[instruction]]},      # (1,1)
    }


def _randomise_vial(mj, model, data, idx, *, area_x, area_y, yaw_jitter, rng):
    """Domain-randomise the vial free-joint pose (matches the data generator)."""
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


def run_episode(mj, model, data, renderer, idx, client, *, fps, area_x, area_y,
                yaw_jitter, rng, max_ticks, n_action_steps, instruction,
                save_frames):
    """Roll out the served GR00T policy for one randomised episode."""
    key = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
    mj.mj_resetDataKeyframe(model, data, key)
    _randomise_vial(mj, model, data, idx, area_x=area_x, area_y=area_y,
                    yaw_jitter=yaw_jitter, rng=rng)

    arm_act, grip_act = idx["arm_act"], idx["grip_act"]
    arm_qadr, grip_qadr = idx["arm_qadr"], idx["grip_qadr"]

    def read_arm() -> np.ndarray:
        return np.array([data.qpos[a] for a in arm_qadr], dtype=np.float32)

    # Settle at the home pose (gripper open) before handing control to GR00T.
    for a, q in zip(arm_act, read_arm(), strict=True):
        data.ctrl[a] = q
    data.ctrl[grip_act] = GRIP_OPEN
    for _ in range(SETTLE_STEPS):
        mj.mj_step(model, data)

    decim = max(1, round((1.0 / fps) / model.opt.timestep))
    z0 = float(data.xpos[idx["vial_body"], 2])
    z_max = z0
    grip_max = 0.0
    frames: list[np.ndarray] = []
    n_queries = 0
    query_t = 0.0

    tick = 0
    while tick < max_ticks:
        measured = read_arm()
        grip_meas = float(np.clip(data.qpos[grip_qadr] / GRIP_DRIVER_RANGE, 0.0, 1.0))
        renderer.update_scene(data, camera=EXTERIOR_CAMERA)
        frame = renderer.render().copy()
        if save_frames:
            frames.append(frame)

        obs = build_obs(measured, grip_meas, frame, instruction)
        t0 = time.time()
        action = client.get_action(obs)
        query_t += time.time() - t0
        n_queries += 1

        arm_chunk = np.asarray(action["single_arm"], dtype=np.float32).reshape(-1, 6)
        grip_chunk = np.asarray(action["gripper"], dtype=np.float32).reshape(-1, 1)
        n = min(n_action_steps, arm_chunk.shape[0])
        for i in range(n):
            for a, q in zip(arm_act, arm_chunk[i], strict=True):
                data.ctrl[a] = float(q)
            g = float(np.clip(grip_chunk[i, 0], 0.0, 1.0))
            grip_max = max(grip_max, g)
            data.ctrl[grip_act] = g * GRIP_CLOSE
            for _ in range(decim):
                mj.mj_step(model, data)
            z_max = max(z_max, float(data.xpos[idx["vial_body"], 2]))
            tick += 1
            if tick >= max_ticks:
                break

    vial_xyz = np.array(data.xpos[idx["vial_body"]], dtype=np.float64)
    pocket_xyz = np.array(data.site_xpos[idx["pocket_site"]], dtype=np.float64)
    place_dist = float(np.linalg.norm(vial_xyz[:2] - pocket_xyz[:2]))
    lifted = (z_max - z0) > 0.02
    seated = place_dist < 0.05 and vial_xyz[2] > 0.20
    success = bool(lifted and seated)
    return {
        "success": success, "lifted": bool(lifted), "seated": bool(seated),
        "lift_height": float(z_max - z0), "place_dist": place_dist,
        "grip_max": grip_max, "ticks": tick, "n_queries": n_queries,
        "mean_query_s": query_t / max(1, n_queries), "frames": frames,
    }


def _save_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default="/home/ubuntu/aseptipack_description/aseptipack.xml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--num-episodes", type=int, default=20)
    ap.add_argument("--fps", type=int, default=emb.DEFAULT_FPS)
    ap.add_argument("--width", type=int, default=emb.DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=emb.DEFAULT_HEIGHT)
    ap.add_argument("--area-x", type=float, default=0.05)
    ap.add_argument("--area-y", type=float, default=0.07)
    ap.add_argument("--yaw-jitter", type=float, default=0.08)
    ap.add_argument("--max-ticks", type=int, default=1000,
                    help="control frames per episode (demos are ~890 frames).")
    ap.add_argument("--n-action-steps", type=int, default=4,
                    help="action-chunk steps replayed per GR00T query before "
                         "re-observing (chunk is 16). A sweep on this checkpoint "
                         "found 4 best: >=8 lets the base joint overshoot into an "
                         "off-distribution runaway; <=2 under-progresses and stalls "
                         "at home. 4 lets the policy approach + attempt the grasp.")
    ap.add_argument("--noslip-iterations", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--instruction", default=emb.DEFAULT_INSTRUCTION)
    ap.add_argument("--video-dir", default="", help="if set, write one mp4 per saved episode.")
    ap.add_argument("--save-videos", type=int, default=3,
                    help="save renders for the first N episodes (0 disables).")
    ap.add_argument("--timeout-ms", type=int, default=180000)
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
    decim = max(1, round((1.0 / args.fps) / model.opt.timestep))
    print(f"[eval] loaded {xml_path.name}: nq={model.nq} nu={model.nu} "
          f"ncam={model.ncam}; freed {freed} arm-collision geoms; "
          f"noslip_iters={model.opt.noslip_iterations}; timestep={model.opt.timestep} "
          f"decim={decim} fps={args.fps}")

    renderer = mj.Renderer(model, args.height, args.width)

    client = GrootPolicyClient(args.host, args.port, timeout_ms=args.timeout_ms)
    if not client.ping():
        print(f"ERROR: GR00T server not reachable at {args.host}:{args.port}", file=sys.stderr)
        return 3
    print(f"[eval] connected to GR00T server {args.host}:{args.port}; "
          f"instruction={args.instruction!r}; episodes={args.num_episodes} "
          f"n_action_steps={args.n_action_steps} max_ticks={args.max_ticks}")

    rng = np.random.default_rng(args.seed)
    video_dir = Path(args.video_dir) if args.video_dir else None

    n_success = n_lifted = n_seated = 0
    results = []
    try:
        for ep in range(args.num_episodes):
            save = video_dir is not None and ep < args.save_videos
            r = run_episode(
                mj, model, data, renderer, idx, client,
                fps=args.fps, area_x=args.area_x, area_y=args.area_y,
                yaw_jitter=args.yaw_jitter, rng=rng, max_ticks=args.max_ticks,
                n_action_steps=args.n_action_steps, instruction=args.instruction,
                save_frames=save,
            )
            n_success += int(r["success"])
            n_lifted += int(r["lifted"])
            n_seated += int(r["seated"])
            print(
                f"[eval] ep {ep:03d}: {'SUCCESS' if r['success'] else 'FAIL   '} "
                f"lift={r['lift_height']*100:5.1f}cm place_dist={r['place_dist']*100:5.1f}cm "
                f"grip_max={r['grip_max']:.2f} (lifted={r['lifted']} seated={r['seated']}) "
                f"queries={r['n_queries']} mean_query={r['mean_query_s']*1000:.0f}ms",
                flush=True,
            )
            if save:
                mp4 = video_dir / f"rollout_ep{ep:03d}_{'success' if r['success'] else 'fail'}.mp4"
                _save_mp4(mp4, r["frames"], args.fps)
                print(f"[eval]   saved render -> {mp4}", flush=True)
            r.pop("frames", None)
            results.append({"episode": ep, **r})
    finally:
        renderer.close()

    rate = n_success / args.num_episodes if args.num_episodes else 0.0
    print("=" * 72)
    print(f"[eval] CLOSED-LOOP GR00T SUCCESS: {n_success}/{args.num_episodes} = {rate*100:.1f}%")
    print(f"[eval]   lifted (grasped+raised>2cm): {n_lifted}/{args.num_episodes}")
    print(f"[eval]   seated (placed in nest):     {n_seated}/{args.num_episodes}")
    print("=" * 72)

    if video_dir is not None:
        summary = {
            "success_rate": rate, "n_success": n_success,
            "n_episodes": args.num_episodes, "n_lifted": n_lifted,
            "n_seated": n_seated, "n_action_steps": args.n_action_steps,
            "max_ticks": args.max_ticks, "results": results,
        }
        (video_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[eval] wrote summary -> {video_dir / 'eval_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
