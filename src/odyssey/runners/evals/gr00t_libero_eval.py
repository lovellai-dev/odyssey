#!/usr/bin/env python3
"""GR00T evaluation recipe for the LIBERO benchmark (single-agent pilot).

Sibling of ``gr00t_isaac_eval.py``: same subprocess+server pattern, but the env
layer is **LIBERO** (robosuite/MuJoCo) instead of Isaac Lab. ``LiberoRunner``
launches this script (``pilot: gr00t``) under the Odyssey interpreter, owns the
``ODYSSEY_*`` stdout protocol, cancellation and scoring; this script owns the
GR00T <-> LIBERO recipe: build the env, drive a GR00T policy server, execute the
returned action *chunk* open-loop, and report each episode.

The GR00T model runs in a **separate policy server** (``gr00t.eval.run_gr00t_server``,
booted here in closed-loop ``--serve_checkpoint`` mode or connected to externally).
This process only holds the lightweight ``PolicyClient`` + the LIBERO env, so its
dependency world is LIBERO — not the GR00T model stack.

Launch contract (built by ``LiberoRunner`` for ``pilot: gr00t``)::

    python gr00t_libero_eval.py \
        --task <suite> --num_episodes <N> --checkpoint <path> \
        --task_id 0 --host H --port P --embodiment_tag <tag> \
        --serve_checkpoint true --n_action_steps 16 --pos_scale 1.0 [...]

It prints, per the runner's protocol::

    ODYSSEY_EPISODE {"index": 1, "total": 10, "success": true, "return": 1.0}
    ODYSSEY_RESULT  {"success_rate": 0.1, "performance_score": 0.1, "metrics": {}}

Heavy deps (numpy, libero, gr00t client, the sibling transforms) are imported
lazily in the run path so the module imports under the bare stdlib and its
argv + protocol surface stay unit-testable on a CPU box.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# When launched as a script (python …/runners/evals/gr00t_libero_eval.py), sys.path[0]
# is THIS file's directory — which also holds libero.py (the odyssey LiberoRunner).
# That shadows the real LIBERO namespace package, so `from libero.libero import …`
# fails with "'libero' is not a package". Drop this dir; odyssey's own modules still
# import via the installed (editable) package, not the script dir.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

log = logging.getLogger("gr00t_libero_eval")

# Same ODYSSEY_* stdout protocol the IsaacLabRunner's EvalProtocolCollector parses
# (src/odyssey/runners/evals/isaac_lab.py). Kept as small local emitters — each
# eval recipe owns its protocol output (mirrors gr00t_isaac_eval.py).
_EPISODE_PREFIX = "ODYSSEY_EPISODE "
_RESULT_PREFIX = "ODYSSEY_RESULT "


def _bool(value: str) -> bool:
    """argparse type for booleans forwarded as ``--flag <value>`` strings.

    The runner forwards every ``task.config`` key verbatim as ``--key value``,
    so a ``store_true`` flag would choke on the trailing value.
    """
    return str(value).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Launch contract + ODYSSEY_* protocol  (stdlib only -> unit-testable anywhere)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Argv per the LIBERO launch contract + GR00T passthrough config."""
    ap = argparse.ArgumentParser(description="GR00T policy eval on LIBERO.")
    # --- contract flags ---
    ap.add_argument("--task", required=True, help="LIBERO suite (benchmark_name).")
    ap.add_argument("--num_episodes", type=int, default=10)
    ap.add_argument("--checkpoint", default="",
                    help="GR00T checkpoint. Closed-loop (--serve_checkpoint): the "
                         "server is started on THIS path (its <suite> subdir if present).")
    # --- LIBERO recipe config (task.config passthrough) ---
    ap.add_argument("--task_id", type=int, default=0, help="task index within the suite.")
    ap.add_argument("--instruction", default="", help="override; default is the suite's own.")
    ap.add_argument("--task_instruction", default="", help="declared instruction (contract check).")
    ap.add_argument("--strict_instruction", type=_bool, default=False)
    ap.add_argument("--image_key", default="agentview_image")
    ap.add_argument("--wrist_image_key", default="robot0_eye_in_hand_image")
    ap.add_argument("--flip_images", type=_bool, default=True,
                    help="180° flip (LIBERO offscreen frames are stored rotated).")
    ap.add_argument("--camera_height", type=int, default=256)
    ap.add_argument("--camera_width", type=int, default=256)
    ap.add_argument("--max_steps_per_episode", type=int, default=520)
    ap.add_argument("--num_warmup_steps", type=int, default=10)
    ap.add_argument("--video_dir", default="", help="if set, write one mp4 per episode here.")
    # --- GR00T server + action-chunk config ---
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--timeout_ms", type=int, default=600000,
                    help="client recv timeout; inference can be slow.")
    ap.add_argument("--n_action_steps", type=int, default=16,
                    help="steps replayed per GR00T chunk before re-querying.")
    ap.add_argument("--pos_scale", type=float, default=1.0)
    ap.add_argument("--rot_scale", type=float, default=1.0)
    ap.add_argument("--translation_only", type=_bool, default=False,
                    help="de-risk: zero rotation + fixed-open gripper.")
    # --- closed-loop auto-serve ---
    ap.add_argument("--serve_checkpoint", type=_bool, default=False,
                    help="boot a GR00T policy server on the checkpoint here.")
    ap.add_argument("--served_model_path", default="",
                    help="explicit path to serve; overrides --checkpoint/<suite>.")
    ap.add_argument("--embodiment_tag", default="",
                    help="embodiment tag for the served checkpoint (required with "
                         "--serve_checkpoint; the server can't infer it). "
                         "GR00T-N1.7-LIBERO uses LIBERO_PANDA.")
    ap.add_argument("--sim_policy_wrapper", type=_bool, default=True,
                    help="serve with --use-sim-policy-wrapper (the GR00T LIBERO sim "
                         "recipe; matches NVIDIA's own eval). Off = raw policy server.")
    ap.add_argument("--modality_config_path", default="")
    ap.add_argument("--server_python", default="",
                    help="interpreter that has gr00t installed (auto-served server).")
    ap.add_argument("--server_device", default="cuda:0")
    ap.add_argument("--server_ready_timeout", type=int, default=900)
    ap.add_argument("--server_denoising_steps", type=int, default=0)
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
    # The runner reads stdout line-by-line; flush so episodes stream in real time.
    print(line, flush=True)


def _resolve_served_path(checkpoint: str, suite: str) -> str:
    """GR00T-N1.7-LIBERO ships one subdir per suite (``libero_object/`` ...).

    When ``<checkpoint>/<suite>`` exists on disk, serve that; otherwise serve the
    checkpoint as-is (a suite-specific local dir or an HF id the server resolves).
    """
    sub = os.path.join(str(checkpoint), str(suite))
    return sub if os.path.isdir(sub) else str(checkpoint)


# ---------------------------------------------------------------------------
# Env-coupled obs/action glue (numpy + odyssey helpers; imported lazily)
# ---------------------------------------------------------------------------

def _transforms():
    from odyssey.runners.evals import gr00t_transforms as t
    return t


def _server():
    from odyssey.runners.evals import _gr00t_server as s
    return s


def _frame(obs, key: str, *, flip: bool):
    import numpy as np
    img = np.asarray(obs[key])
    if flip:
        img = img[::-1, ::-1]
    return np.ascontiguousarray(img).astype(np.uint8)


def _build_obs(obs, instruction, *, image_key, wrist_image_key, flip):
    """Build the FLAT GR00T LIBERO obs (video.*/state.*/annotation.*) from a LIBERO obs.

    The GR00T-N1.7-LIBERO checkpoint (embodiment ``LIBERO_PANDA``) uses a single
    video frame + a flat dotted-key schema — NOT the nested T=2 DROID layout. State
    is eef_pos(3) + eef_quat(xyzw)->axis-angle(3) + the 2 gripper finger qpos, matching
    NVIDIA's ``libero_env._process_observation``. robosuite obs keys: ``robot0_eef_pos``
    (3), ``robot0_eef_quat`` (xyzw, 4), ``robot0_gripper_qpos`` (2 finger joints).
    """
    import numpy as np
    t = _transforms()
    return t.build_gr00t_libero_obs(
        image=_frame(obs, image_key, flip=flip),
        wrist_image=_frame(obs, wrist_image_key, flip=flip),
        eef_pos=np.asarray(obs["robot0_eef_pos"], np.float64).reshape(-1)[:3],
        eef_quat_xyzw=np.asarray(obs["robot0_eef_quat"], np.float64).reshape(-1)[:4],
        gripper_qpos=np.asarray(obs["robot0_gripper_qpos"], np.float64).reshape(-1)[:2],
        instruction=instruction,
    )


# ---------------------------------------------------------------------------
# Run path (heavy imports live here)
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace) -> dict:
    from odyssey.runners.evals.libero import (
        _make_libero_env,
        _resolve_libero_instruction,
    )
    from odyssey.runners.video import save_rollout_video, to_uint8_frame

    t = _transforms()
    cfg = {
        "camera_height": args.camera_height,
        "camera_width": args.camera_width,
        "task_instruction": args.task_instruction or None,
        "strict_instruction": args.strict_instruction,
    }
    # robosuite's internal horizon must exceed the full per-episode step budget
    # (warmup + max_steps) or it terminates mid-rollout — mirror LiberoRunner.
    env_horizon = args.num_warmup_steps + args.max_steps_per_episode + 100
    env, task, init_states = _make_libero_env(
        args.task, args.task_id, cfg, horizon=env_horizon
    )
    instruction = args.instruction or _resolve_libero_instruction(task, cfg)
    log.info("LIBERO suite=%s task_id=%d instruction=%r", args.task, args.task_id, instruction)

    client = _server().connect_policy_client(
        host=args.host, port=args.port, timeout_ms=args.timeout_ms
    )

    successes, returns = 0, []
    video_dir = args.video_dir or None
    dummy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # no-op, gripper open (physics settle)

    for ep in range(1, args.num_episodes + 1):
        obs = env.reset()
        env.set_init_state(init_states[(ep - 1) % len(init_states)])
        for _ in range(args.num_warmup_steps):
            obs, _, _, _ = env.step(dummy)

        ep_return, success = 0.0, False
        frames: list = []
        step = 0
        try:
            while step < args.max_steps_per_episode and not success:
                observation = _build_obs(
                    obs, instruction,
                    image_key=args.image_key,
                    wrist_image_key=args.wrist_image_key,
                    flip=args.flip_images,
                )
                result = client.get_action(observation)
                chunk = result[0] if isinstance(result, tuple) else result
                for k in range(args.n_action_steps):
                    if step >= args.max_steps_per_episode:
                        break
                    action = t.gr00t_action_to_libero(
                        chunk, k, translation_only=args.translation_only,
                    )
                    obs, reward, done, _info = env.step(action.tolist())
                    ep_return += float(reward)
                    if video_dir is not None:
                        frame = to_uint8_frame(obs[args.image_key])
                        if frame is not None:
                            frames.append(frame)
                    step += 1
                    if done:  # LIBERO sets done=True when the task is solved
                        success = True
                        break
        except Exception as ep_exc:
            # A flaky get_action()/env.step() must not abort the whole sweep (that
            # would drop the remaining episodes AND the final ODYSSEY_RESULT line).
            log.warning("episode %d/%d aborted (%s) — recording as fail, continuing.",
                        ep, args.num_episodes, ep_exc, exc_info=True)

        successes += int(success)
        returns.append(ep_return)
        log.info("episode %d/%d: %s (steps=%d, return=%.3f)",
                 ep, args.num_episodes, "SUCCESS" if success else "fail", step, ep_return)
        _emit(episode_line(index=ep, total=args.num_episodes, success=success, ret=ep_return))

        if video_dir and frames:
            os.makedirs(video_dir, exist_ok=True)
            tag = "PASS" if success else "FAIL"
            save_rollout_video(frames, Path(video_dir) / f"episode_{ep:02d}_{tag}.mp4", 24)

    env.close()
    n = max(args.num_episodes, 1)
    success_rate = successes / n
    summary = {
        "success_rate": success_rate,
        "performance_score": success_rate,
        "metrics": {
            "successes": successes,
            "episode_returns": [round(r, 4) for r in returns],
            "benchmark": f"{args.task}[task={args.task_id}]",
            "instruction": instruction,
        },
    }
    _emit(result_line(**summary))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.serve_checkpoint:
        served = args.served_model_path or _resolve_served_path(args.checkpoint, args.task)
        if not served:
            raise SystemExit("--serve_checkpoint requires --served_model_path or --checkpoint")
        if not args.embodiment_tag:
            raise SystemExit("--serve_checkpoint requires --embodiment_tag (server can't infer it)")
        log.info("closed-loop: serving GR00T checkpoint %s (embodiment=%s)",
                 served, args.embodiment_tag)
        with _server().serve_checkpoint(
            checkpoint=served,
            embodiment_tag=args.embodiment_tag,
            host=args.host,
            port=args.port,
            server_python=args.server_python or None,
            device=args.server_device,
            modality_config_path=args.modality_config_path or None,
            denoising_steps=args.server_denoising_steps,
            ready_timeout=args.server_ready_timeout,
            sim_policy_wrapper=args.sim_policy_wrapper,
        ):
            run_eval(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
