#!/usr/bin/env python3
"""MolmoAct2 (Ai2) evaluation recipe for the LIBERO benchmark (single-agent pilot).

Sibling of ``pi05_libero_eval.py`` / ``molmoact2_libero_eval.py``: same subprocess
+ policy-server pattern and the same LIBERO (robosuite/MuJoCo) env layer, but the
pilot is **MolmoAct2** served by a FastAPI ``/act`` server (Ai2's json_numpy
protocol). ``LiberoRunner`` launches this script (``pilot: molmoact2``) under the
Odyssey interpreter, owns the ``ODYSSEY_*`` stdout protocol, cancellation and
scoring; this script owns the MolmoAct2 <-> LIBERO recipe: build the env, drive
the pilot, replay the returned action *chunk* open-loop, and report each episode.

MolmoAct2 is **externally-served**: start the server yourself in a GPU env ::

    uv run python examples/quickstart-molmoact2/host_server_libero.py \
        --repo-id allenai/MolmoAct2-LIBERO --port 8000

and point this recipe at ``--host``/``--port``. ``--checkpoint`` is recorded in
the summary but NOT loaded here — the server holds the weights and owns
normalization + flow sampling. ``--n_action_steps 0`` (the default) adopts the
server's ``action_horizon`` from ``GET /act``.

Launch contract (built by ``LiberoRunner`` for ``pilot: molmoact2``)::

    python molmoact2_libero_eval.py \
        --task <suite> --num_episodes <N> --checkpoint <path> \
        --task_id 0 --host H --port P [--n_action_steps 10 --num_steps 10 ...]

It prints, per the runner's protocol::

    ODYSSEY_EPISODE {"index": 1, "total": 10, "success": true, "return": 1.0}
    ODYSSEY_RESULT  {"success_rate": 0.1, "performance_score": 0.1, "metrics": {}}

Heavy deps (numpy, libero, the sibling transforms) are imported lazily in the
run path so the module imports under the bare stdlib and its argv + protocol
surface stay unit-testable on a CPU box.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# When launched as a script (python …/runners/evals/molmoact2_libero_eval.py),
# sys.path[0] is THIS file's directory — which also holds libero.py (the odyssey
# LiberoRunner). That shadows the real LIBERO namespace package, so
# `from libero.libero import …` fails with "'libero' is not a package". Drop this
# dir; odyssey's own modules still import via the installed (editable) package,
# not the script dir. (Mirrors pi05_libero_eval.py / gr00t_libero_eval.py.)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]

log = logging.getLogger("molmoact2_libero_eval")

# Same ODYSSEY_* stdout protocol the runner's EvalProtocolCollector parses
# (src/odyssey/runners/evals/isaac_lab.py). Kept as small local emitters — each
# eval recipe owns its protocol output (mirrors pi05_libero_eval.py).
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
    """Argv per the LIBERO launch contract + MolmoAct2 passthrough config."""
    ap = argparse.ArgumentParser(description="MolmoAct2 policy eval on LIBERO.")
    # --- contract flags ---
    ap.add_argument("--task", required=True, help="LIBERO suite (benchmark_name).")
    ap.add_argument("--num_episodes", type=int, default=10)
    ap.add_argument("--checkpoint", default="",
                    help="MolmoAct2 checkpoint id/path. Recorded in the summary only — "
                         "the policy server (pre-started) already holds the weights.")
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
    # --- MolmoAct2 policy server + action-chunk config ---
    ap.add_argument("--host", default="127.0.0.1", help="MolmoAct2 /act server host.")
    ap.add_argument("--port", type=int, default=8000, help="MolmoAct2 /act server port.")
    ap.add_argument("--n_action_steps", type=int, default=0,
                    help="steps replayed per chunk before re-querying; 0 adopts the "
                         "server's action_horizon from GET /act (10 for MolmoAct2-LIBERO).")
    ap.add_argument("--num_steps", type=int, default=10,
                    help="flow-matching solver steps per query (checkpoint default 10).")
    ap.add_argument("--invert_gripper", type=_bool, default=False,
                    help="flip gripper polarity (escape hatch; LIBERO -1 = open).")
    ap.add_argument("--translation_only", type=_bool, default=False,
                    help="de-risk: zero rotation + fixed-open gripper.")
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


# ---------------------------------------------------------------------------
# Env-coupled obs glue (numpy; imported lazily)
# ---------------------------------------------------------------------------

def _frame(obs, key: str, *, flip: bool):
    import numpy as np
    img = np.asarray(obs[key])
    if flip:
        img = img[::-1, ::-1]
    return np.ascontiguousarray(img).astype(np.uint8)


def _raw_obs(obs, *, image_key, wrist_image_key, flip) -> dict:
    """Shape a LIBERO env observation into ``make_molmoact2_pilot``'s ``raw_obs`` kwargs.

    ``ChunkPilotAdapter``'s injected ``observation_builder`` calls
    ``build_molmoact2_libero_request(instruction=..., **raw_obs)``, so the keys
    here MUST match its signature: image / wrist_image / eef_pos / eef_quat_xyzw /
    gripper_qpos. Images are pre-flipped 180° — the same rotation LeRobot's
    ``LiberoProcessorStep`` (Ai2's LIBERO eval stack) applies.
    """
    import numpy as np
    return {
        "image": _frame(obs, image_key, flip=flip),
        "wrist_image": _frame(obs, wrist_image_key, flip=flip),
        "eef_pos": np.asarray(obs["robot0_eef_pos"], np.float64).reshape(-1)[:3],
        "eef_quat_xyzw": np.asarray(obs["robot0_eef_quat"], np.float64).reshape(-1)[:4],
        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], np.float64).reshape(-1)[:2],
    }


# ---------------------------------------------------------------------------
# Run path (heavy imports live here)
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace) -> dict:
    from odyssey.runners.evals.libero import (
        _make_libero_env,
        _resolve_libero_instruction,
    )
    from odyssey.runners.models.molmoact2 import MolmoAct2HttpClient, make_molmoact2_pilot
    from odyssey.runners.video import save_rollout_video, to_uint8_frame

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

    # Externally-served: the MolmoAct2 /act server is pre-started and already
    # holds the weights; make_molmoact2_pilot wraps its HTTP client in the
    # ChunkPilotAdapter (n_action_steps=None -> adopt the server's action_horizon).
    log.info("connecting to MolmoAct2 policy server at %s:%d (checkpoint=%r served externally)",
             args.host, args.port, args.checkpoint)
    # Fail fast if the server isn't up: without this probe every episode would
    # swallow the connection error as a plain "fail" and the sweep would exit 0
    # with a misleading 0% success_rate. Raises with the serve hint.
    client = MolmoAct2HttpClient(host=args.host, port=args.port)
    server_info = client.info()
    log.info("MolmoAct2 server info: %s", server_info)
    pilot = make_molmoact2_pilot(
        client=client,
        n_action_steps=args.n_action_steps or None,
        num_steps=args.num_steps,
        invert_gripper=args.invert_gripper,
        translation_only=args.translation_only,
    )
    log.info("chunk replay horizon: %d steps", pilot.n_action_steps)

    successes, returns = 0, []
    aborted: list[str] = []  # per-episode exceptions (flaky /act, env.step errors)
    video_dir = args.video_dir or None
    dummy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # no-op, gripper open (physics settle)

    for ep in range(1, args.num_episodes + 1):
        obs = env.reset()
        env.set_init_state(init_states[(ep - 1) % len(init_states)])
        for _ in range(args.num_warmup_steps):
            obs, _, _, _ = env.step(dummy)

        ep_return, success = 0.0, False
        frames: list = []
        pilot.reset()  # drop the chunk buffer so the first act re-queries this episode
        step = 0
        try:
            while step < args.max_steps_per_episode and not success:
                # One act per env step; the ChunkPilotAdapter buffers the chunk and
                # only re-queries the server when it drains (every n_action_steps).
                action = pilot.act(
                    _raw_obs(
                        obs,
                        image_key=args.image_key,
                        wrist_image_key=args.wrist_image_key,
                        flip=args.flip_images,
                    ),
                    instruction,
                )
                obs, reward, done, _info = env.step(action.tolist())
                ep_return += float(reward)
                if video_dir is not None:
                    # match the pilot's orientation: LIBERO's agentview is stored
                    # 180°-rotated, so flip the video frame too (else it's upside down).
                    frame = to_uint8_frame(_frame(obs, args.image_key, flip=args.flip_images))
                    if frame is not None:
                        frames.append(frame)
                step += 1
                if done:  # LIBERO sets done=True when the task is solved
                    success = True
                    break
        except Exception as ep_exc:
            # A flaky /act round-trip or env.step() must not abort the whole sweep (that
            # would drop the remaining episodes AND the final ODYSSEY_RESULT line).
            log.warning("episode %d/%d aborted (%s) — recording as fail, continuing.",
                        ep, args.num_episodes, ep_exc, exc_info=True)
            aborted.append(f"{type(ep_exc).__name__}: {ep_exc}")

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
    if args.num_episodes > 0 and len(aborted) == args.num_episodes:
        # Not a 0% policy: the pilot never got to act (e.g. the server died after
        # the startup probe). Exit non-zero so the task FAILS instead of scoring.
        raise RuntimeError(
            f"all {args.num_episodes} episodes aborted with errors; last: {aborted[-1]}"
        )
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
    run_eval(args)


if __name__ == "__main__":
    main()
