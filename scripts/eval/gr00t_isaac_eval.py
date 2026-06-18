#!/usr/bin/env python3
"""Blessed GR00T evaluation recipe for Odyssey's Isaac Lab runner (odyssey#17).

This is the *eval script* that ``odyssey``'s subprocess ``IsaacLabRunner``
launches under Isaac Sim's bundled Python (``isaaclab.sh -p``). The runner owns
the launch, the ODYSSEY_* stdout protocol, cancellation and scoring; this script
owns the GR00T <-> Isaac recipe: boot the env, drive a GR00T policy server, and
report each episode.

Launch contract (built by ``odyssey.runners.isaac_lab.build_isaac_lab_argv``)::

    isaaclab.sh -p gr00t_isaac_eval.py \
        --task <env_id> --num_episodes <N> --checkpoint <path> --headless \
        [--host H --port P --instruction "..." --pos_scale .. --rot_scale ..]

It connects to a GR00T policy server (``run_gr00t_server.py``, raw nested obs)
and prints, per the runner's protocol::

    ODYSSEY_EPISODE {"index": 1, "total": 10, "success": true, "return": 1.0}
    ODYSSEY_RESULT  {"success_rate": 0.1, "performance_score": 0.1, "metrics": {}}

Env support: the obs extraction targets the DROID-style Franka *visuomotor*
envs (dual cameras + eef/joint state), e.g.
``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0``. The ``--checkpoint`` is
informational here — GR00T weights live on the server.

Heavy deps (numpy, gr00t_transforms, isaaclab, gymnasium, gr00t, torch) are
imported lazily in the run path so the module imports under the bare stdlib and
its argv + protocol surface stay unit-testable on a CPU box without Isaac.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

log = logging.getLogger("gr00t_isaac_eval")

_EPISODE_PREFIX = "ODYSSEY_EPISODE "
_RESULT_PREFIX = "ODYSSEY_RESULT "


# ---------------------------------------------------------------------------
# Launch contract + ODYSSEY_* protocol  (stdlib only -> unit-testable anywhere)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Argv per the Isaac Lab launch contract + GR00T passthrough config.

    Contract flags (always sent by the runner): --task, --num_episodes,
    --checkpoint, --headless. The rest come from ``task.config`` verbatim
    (snake_case), so they are declared with underscores to match.
    """
    ap = argparse.ArgumentParser(description="GR00T policy eval in Isaac Lab.")
    # --- contract flags ---
    ap.add_argument("--task", required=True, help="Isaac Lab env id (gym registered).")
    ap.add_argument("--num_episodes", type=int, default=10)
    ap.add_argument("--checkpoint", default="", help="Informational; weights live on the GR00T server.")
    ap.add_argument("--headless", action="store_true")
    # --- GR00T server + recipe config (task.config passthrough) ---
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--timeout_ms", type=int, default=600000,
                    help="client recv timeout; CPU inference is slow (~30s/query).")
    ap.add_argument("--instruction", default="stack the red cube on the green cube")
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n_action_steps", type=int, default=16,
                    help="steps replayed per GR00T chunk before re-querying.")
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--pos_scale", type=float, default=1.0)
    ap.add_argument("--rot_scale", type=float, default=1.0)
    ap.add_argument("--translation_only", action="store_true",
                    help="de-risk: zero rotation + fixed-open gripper (CLI/debug only).")
    ap.add_argument("--video_dir", default="", help="if set, write one mp4 per episode here.")
    return ap


def episode_line(*, index: int, total: int, success: bool, ret: float) -> str:
    """One ``ODYSSEY_EPISODE`` protocol line (consumed by the runner's collector)."""
    return _EPISODE_PREFIX + json.dumps(
        {"index": int(index), "total": int(total),
         "success": bool(success), "return": float(ret)})


def result_line(*, success_rate: float, performance_score: float,
                metrics: dict | None = None) -> str:
    """The optional ``ODYSSEY_RESULT`` summary line."""
    return _RESULT_PREFIX + json.dumps(
        {"success_rate": float(success_rate),
         "performance_score": float(performance_score),
         "metrics": dict(metrics or {})})


def _emit(line: str) -> None:
    # The runner reads stdout line-by-line; flush so episodes stream in real
    # time (progress events + responsive cancellation).
    print(line, flush=True)


# ---------------------------------------------------------------------------
# Env-coupled obs/action glue (numpy + gr00t_transforms; imported lazily)
# ---------------------------------------------------------------------------

def _np(x):
    """Torch (possibly CUDA) tensor -> numpy; pass-through otherwise."""
    import numpy as np
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _policy_group(obs: dict) -> dict:
    return obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs


def _frames(obs: dict):
    import numpy as np
    p = _policy_group(obs)
    table = _np(p["table_cam"]).squeeze(0).astype(np.uint8)  # (H,W,3)
    wrist = _np(p["wrist_cam"]).squeeze(0).astype(np.uint8)
    return table, wrist


def _state(obs: dict):
    p = _policy_group(obs)
    eef_pos = _np(p["eef_pos"]).reshape(-1)[:3]
    eef_quat = _np(p["eef_quat"]).reshape(-1)[:4]   # wxyz (FrameTransformer.target_quat_w)
    gripper = float(_np(p["gripper_pos"]).reshape(-1)[0])  # (1,2) finger joints; take one
    joints = _np(p["joint_pos"]).reshape(-1)[:7]    # 9 joints -> 7 arm
    return eef_pos, eef_quat, gripper, joints


def _build_obs(obs: dict, hist, instruction: str) -> dict:
    import numpy as np
    from gr00t_transforms import build_gr00t_obs
    hist.append(_frames(obs))
    cur = hist[-1]
    past = hist[-16] if len(hist) >= 16 else hist[0]   # video delta_indices [-15, 0]
    eef_pos, eef_quat, gripper, joints = _state(obs)
    return build_gr00t_obs(
        exterior_seq=np.stack([past[0], cur[0]]),
        wrist_seq=np.stack([past[1], cur[1]]),
        eef_pos=eef_pos, eef_quat_wxyz=eef_quat,
        gripper=gripper, arm_joints=joints, instruction=instruction,
    )


def _read_success(env) -> bool:
    """Read the ``success`` (cubes_stacked) term at the terminal step — tells a
    real stack from a cube-drop termination. IsaacLab 2.3.2:
    ``termination_manager.get_term(name)`` -> per-env bool buffer, with a
    ``_last_episode_dones`` fallback robust to in-step auto-reset."""
    import numpy as np
    tm = env.unwrapped.termination_manager
    try:
        v = tm.get_term("success")
        if bool(np.asarray(v.detach().cpu()).reshape(-1)[0]):
            return True
    except Exception as e:
        log.warning("get_term('success') failed: %s", e)
    try:
        if "success" in tm.active_terms:
            idx = tm.active_terms.index("success")
            return bool(np.asarray(tm._last_episode_dones[:, idx].detach().cpu()).reshape(-1)[0])
    except Exception:
        pass
    return False


def _save_video(frames, path: str, fps: int = 24) -> None:
    if not frames:
        return
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, fps=fps)
        log.info("wrote video -> %s (%d frames)", path, len(frames))
    except Exception as e:
        log.warning("could not write video %s: %s", path, e)


def _video_frame(env, obs):
    """A frame for the demo video: prefer Isaac's viewport render (higher-res,
    3rd-person); fall back to the up-scaled policy table-cam if render is unavailable."""
    import numpy as np
    try:
        r = env.render()
        if r is not None:
            a = np.asarray(r)
            if a.ndim == 3 and a.shape[0] >= 128:
                return a.astype(np.uint8)
    except Exception:
        pass
    f = _frames(obs)[0]
    try:
        from PIL import Image
        return np.asarray(Image.fromarray(f).resize((600, 600), Image.LANCZOS), np.uint8)
    except Exception:
        return f


# ---------------------------------------------------------------------------
# Run path (heavy imports live here)
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace) -> dict:
    import collections

    from gr00t_transforms import gr00t_action_to_isaac

    # Isaac Sim kit-python stack.
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=args.headless, enable_cameras=True).app
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401  registers Isaac-* envs
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    # GR00T client — light deps (msgpack/pyzmq); add Isaac-GR00T to path.
    g = os.environ.get("ISAAC_GR00T_DIR", os.path.expanduser("~/Isaac-GR00T"))
    if g not in sys.path:
        sys.path.insert(0, g)
    from gr00t.policy.server_client import PolicyClient

    if args.checkpoint:
        log.info("checkpoint (informational; weights on server): %s", args.checkpoint)
    client = PolicyClient(host=args.host, port=args.port, timeout_ms=args.timeout_ms)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    dev = getattr(env.unwrapped, "device", "cpu")

    successes, returns = 0, []
    frames = [] if args.video_dir else None  # one combined clip across all episodes
    try:
        for ep in range(args.num_episodes):
            obs, _ = env.reset()
            hist: collections.deque = collections.deque(maxlen=16)
            done, t, ep_success, ep_return = False, 0, False, 0.0
            while not done and t < args.max_steps:
                result = client.get_action(_build_obs(obs, hist, args.instruction))
                chunk = result[0] if isinstance(result, tuple) else result
                for k in range(args.n_action_steps):
                    a = gr00t_action_to_isaac(
                        chunk, k, pos_scale=args.pos_scale,
                        rot_scale=0.0 if args.translation_only else args.rot_scale)
                    if args.translation_only:
                        a[6] = 1.0  # fixed-open gripper
                    a_t = torch.as_tensor(a, dtype=torch.float32, device=dev).unsqueeze(0)
                    obs, reward, terminated, truncated, _ = env.step(a_t)
                    ep_return += float(_np(reward).reshape(-1)[0])
                    if frames is not None:
                        frames.append(_video_frame(env, obs))
                    if bool(_np(terminated).reshape(-1)[0]):
                        ep_success = _read_success(env)  # capture pre auto-reset
                    done = bool(_np(terminated).reshape(-1)[0]) or bool(_np(truncated).reshape(-1)[0])
                    t += 1
                    if done:
                        break
            successes += int(ep_success)
            returns.append(ep_return)
            log.info("episode %d/%d: %s (steps=%d, return=%.3f)",
                     ep + 1, args.num_episodes, "SUCCESS" if ep_success else "fail", t, ep_return)
            _emit(episode_line(index=ep + 1, total=args.num_episodes,
                               success=ep_success, ret=ep_return))
        if frames:
            os.makedirs(args.video_dir, exist_ok=True)
            _save_video(frames, os.path.join(args.video_dir, "rollout.mp4"))
    finally:
        try:
            env.close()
        finally:
            simulation_app.close()

    n = max(args.num_episodes, 1)
    success_rate = successes / n
    summary = {
        "success_rate": success_rate,
        "performance_score": success_rate,
        "metrics": {
            "successes": successes,
            "episode_returns": [round(r, 4) for r in returns],
            "benchmark": args.task,
        },
    }
    _emit(result_line(**summary))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_eval(build_parser().parse_args())


if __name__ == "__main__":
    main()
