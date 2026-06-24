#!/usr/bin/env python3
"""Blessed GR00T evaluation recipe for Odyssey's Isaac Lab runner (odyssey#17).

This is the *eval script* that ``odyssey``'s subprocess ``IsaacLabRunner``
launches under Isaac Sim's bundled Python (``isaaclab.sh -p``). The runner owns
the launch, the ODYSSEY_* stdout protocol, cancellation and scoring; this script
owns the GR00T <-> Isaac recipe: boot the env, drive a GR00T policy server, and
report each episode.

Lives in ``odyssey.runners.evals`` alongside the runner it serves (relocated
from the top-level ``scripts/`` folder). It is launched by *path*, so its
sibling ``gr00t_transforms`` import works whether run as a script (sibling on
``sys.path[0]``) or imported as a package module — see ``_transforms()``.

Launch contract (built by ``odyssey.runners.evals.isaac_lab.build_isaac_lab_argv``)::

    isaaclab.sh -p gr00t_isaac_eval.py \
        --task <env_id> --num_episodes <N> --checkpoint <path> --headless \
        [--host H --port P --instruction "..." --pos_scale .. --rot_scale ..]

It connects to a GR00T policy server (``run_gr00t_server.py``, raw nested obs)
and prints, per the runner's protocol::

    ODYSSEY_EPISODE {"index": 1, "total": 10, "success": true, "return": 1.0}
    ODYSSEY_RESULT  {"success_rate": 0.1, "performance_score": 0.1, "metrics": {}}

Two server modes:
  * **external** (default): connect to a GR00T server someone already started
    at ``--host/--port`` (the #24 zero-shot recipe).
  * **closed-loop auto-serve** (``--serve_checkpoint true``): boot a GR00T policy
    server on ``--checkpoint`` *here*, wait until it is ready, run the eval
    against it, then tear it down. ``--checkpoint`` is the upstream training
    task's output, so the eval scores the **just-finetuned** policy — the
    train -> deploy -> grade loop, end to end, through one ``odyssey run``.
    The served checkpoint's ``--embodiment_tag`` MUST match the tag it was
    finetuned with (the server can't infer it).

Env support: the obs extraction targets the DROID-style Franka *visuomotor*
envs (dual cameras + eef/joint state), e.g.
``Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0``.

Heavy deps (numpy, gr00t_transforms, isaaclab, gymnasium, gr00t, torch) are
imported lazily in the run path so the module imports under the bare stdlib and
its argv + protocol surface stay unit-testable on a CPU box without Isaac.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys

log = logging.getLogger("gr00t_isaac_eval")

_EPISODE_PREFIX = "ODYSSEY_EPISODE "
_RESULT_PREFIX = "ODYSSEY_RESULT "
# The upstream GR00T policy-server entry point (closed-loop auto-serve).
_SERVER_ENTRY = "gr00t.eval.run_gr00t_server"


def _bool(value: str) -> bool:
    """argparse type for booleans forwarded as ``--flag <value>`` strings.

    The runner forwards every ``task.config`` key verbatim as ``--key value``,
    so a ``store_true`` flag would choke on the trailing value. A value-style
    bool ("true"/"1"/"yes"/"on") passes through that path cleanly.
    """
    return str(value).strip().lower() in ("1", "true", "yes", "on")


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
    ap.add_argument("--checkpoint", default="",
                    help="finetuned weights. External mode: informational (weights "
                         "live on the server). Closed-loop (--serve_checkpoint): the "
                         "GR00T server is started on THIS path.")
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
    # --- closed-loop auto-serve: deploy the finetuned --checkpoint ---
    ap.add_argument("--serve_checkpoint", type=_bool, default=False,
                    help="boot a GR00T policy server on the trained checkpoint here "
                         "(and tear it down after) instead of connecting to an external "
                         "one — closes the train -> deploy -> grade loop.")
    ap.add_argument("--served_model_path", default="",
                    help="explicit path to the checkpoint to serve. Overrides "
                         "--checkpoint for the server (used when the runner's "
                         "--checkpoint is a stub, e.g. a Command-Center-delegated eval "
                         "where training ran as a separate task). Defaults to --checkpoint.")
    ap.add_argument("--embodiment_tag", default="",
                    help="embodiment tag for the served checkpoint; MUST match the "
                         "tag it was finetuned with (required with --serve_checkpoint).")
    ap.add_argument("--modality_config_path", default="",
                    help="modality config for the served checkpoint (needed for the "
                         "new_embodiment tag unless baked into the checkpoint).")
    ap.add_argument("--server_python", default="",
                    help="interpreter that has gr00t installed, used to launch the "
                         "auto-served server; defaults to the current interpreter.")
    ap.add_argument("--server_device", default="cuda:0",
                    help="device for the auto-served GR00T policy server.")
    ap.add_argument("--server_ready_timeout", type=int, default=900,
                    help="seconds to wait for the auto-served server to accept "
                         "connections before failing the eval.")
    ap.add_argument("--server_denoising_steps", type=int, default=0,
                    help="optional flow-matching denoising steps for the server "
                         "(0 -> server default).")
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
# Closed-loop auto-serve: deploy the finetuned checkpoint as a policy server
# (pure argv builder + a socket-based readiness wait — unit-testable on CPU).
# ---------------------------------------------------------------------------

def build_server_command(
    *,
    checkpoint: str,
    embodiment_tag: str,
    port: int,
    server_python: str | None = None,
    device: str = "cuda:0",
    modality_config_path: str | None = None,
    denoising_steps: int = 0,
) -> list[str]:
    """argv to serve a (finetuned) GR00T checkpoint as a policy server.

    Mirrors the manual ``run_gr00t_server`` launch documented in
    ``isaac_eval_mission.yaml`` but points ``--model-path`` at the *trained*
    checkpoint so the eval scores the finetuned policy. Deliberately omits
    ``--use-sim-policy-wrapper``: this recipe builds raw nested observations
    (``_build_obs``), which is exactly what the un-wrapped server expects.
    """
    py = server_python or sys.executable
    argv = [
        py, "-m", _SERVER_ENTRY,
        "--model-path", str(checkpoint),
        "--embodiment-tag", str(embodiment_tag),
        "--port", str(int(port)),
        "--device", str(device),
    ]
    if modality_config_path:
        argv += ["--modality-config-path", str(modality_config_path)]
    if denoising_steps and int(denoising_steps) > 0:
        argv += ["--denoising-steps", str(int(denoising_steps))]
    return argv


def _wait_for_server(host: str, port: int, timeout_s: int, proc=None) -> bool:
    """Block until ``(host, port)`` accepts a TCP connection (server ready).

    Returns ``True`` once reachable, ``False`` on timeout. Fails fast if the
    server ``proc`` exits during startup (so we surface its error instead of
    waiting out the whole timeout).
    """
    import socket
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False  # server died during startup
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            time.sleep(2.0)
    return False


@contextlib.contextmanager
def serve_checkpoint(args: argparse.Namespace):
    """Start a GR00T policy server on ``args.checkpoint``, wait until ready,
    yield, then tear it down. The body runs the eval against ``args.host:port``.
    """
    import ctypes
    import signal
    import subprocess
    import tempfile

    argv = build_server_command(
        checkpoint=(args.served_model_path or args.checkpoint),
        embodiment_tag=args.embodiment_tag,
        port=args.port,
        server_python=args.server_python or None,
        device=args.server_device,
        modality_config_path=args.modality_config_path or None,
        denoising_steps=args.server_denoising_steps,
    )
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")       # gated Cosmos backbone -> offline cache
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    log.info("auto-serve: launching GR00T server: %s", " ".join(argv))

    # DEADLOCK FIX: the server MUST NOT inherit this recipe's stdout/stderr — those are the
    # pipe the Odyssey runner (and Command Center) read. run_eval()'s finally calls
    # simulation_app.close(), which HARD-EXITS this recipe (so the teardown `finally` below
    # never runs and the server is orphaned). If the orphan still held the stdout pipe's write
    # end, the runner's read would never see EOF and the eval would wedge until CC's timeout
    # (observed live: server idle at 0% GPU, ~40-min hang, ODYSSEY_RESULT never consumed).
    # (1) redirect the server's output to a log file so it never holds the orchestrator pipe;
    # (2) PR_SET_PDEATHSIG so the kernel SIGKILLs the server if we hard-exit (no GPU leak).
    server_log_path = os.path.join(tempfile.gettempdir(), f"gr00t_server_{args.port}.log")
    server_log = open(server_log_path, "wb")  # noqa: SIM115  handle outlives this scope (Popen + finally)

    def _die_with_parent():  # Linux: reap this server when the recipe process dies
        with contextlib.suppress(Exception):
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0)

    proc = subprocess.Popen(
        argv, env=env,
        stdout=server_log, stderr=subprocess.STDOUT,
        preexec_fn=_die_with_parent,
    )
    log.info("auto-serve: GR00T server pid=%s (log: %s)", proc.pid, server_log_path)
    try:
        if not _wait_for_server(args.host, args.port, args.server_ready_timeout, proc):
            raise RuntimeError(
                f"GR00T server failed to become ready on {args.host}:{args.port} "
                f"within {args.server_ready_timeout}s (server rc={proc.poll()}; see {server_log_path})"
            )
        log.info("auto-serve: GR00T server ready on %s:%d", args.host, args.port)
        yield
    finally:
        log.info("auto-serve: tearing down GR00T server (pid=%s)", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        with contextlib.suppress(Exception):
            server_log.close()


# ---------------------------------------------------------------------------
# Env-coupled obs/action glue (numpy + gr00t_transforms; imported lazily)
# ---------------------------------------------------------------------------

def _transforms():
    """Import the sibling ``gr00t_transforms`` whether this file is launched as
    a script (sibling on ``sys.path[0]``) or imported as a package module
    (``odyssey.runners.evals.gr00t_transforms``)."""
    try:
        import gr00t_transforms as t  # script launch: isaaclab.sh -p <path>
    except ImportError:
        from odyssey.runners.evals import gr00t_transforms as t  # package import
    return t


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
    build_gr00t_obs = _transforms().build_gr00t_obs
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

    gr00t_action_to_isaac = _transforms().gr00t_action_to_isaac

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
    # This recipe drives a SINGLE-env rollout — it reads env 0 only
    # (_np(reward).reshape(-1)[0]) and steps one action (.unsqueeze(0)).
    # num_envs > 1 would run but silently grade just one env, giving misleading
    # numbers. Clamp to 1 with a warning rather than score a fraction of the batch.
    if args.num_envs != 1:
        log.warning(
            "num_envs=%d ignored — this recipe is single-env (grades env 0 only); "
            "clamping to 1.", args.num_envs,
        )
        args.num_envs = 1
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    dev = getattr(env.unwrapped, "device", "cpu")

    successes, returns = 0, []
    frames = [] if args.video_dir else None  # one combined clip across all episodes
    summary: dict = {"success_rate": 0.0, "performance_score": 0.0, "metrics": {}}
    try:
        for ep in range(args.num_episodes):
            done, t, ep_success, ep_return = False, 0, False, 0.0
            try:
                obs, _ = env.reset()
                hist: collections.deque = collections.deque(maxlen=16)
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
            except Exception as ep_exc:
                # A single flaky get_action()/env.step() must not abort the whole
                # sweep — that would drop the remaining episodes AND the final
                # ODYSSEY_RESULT line. Record the episode as a failure and continue.
                log.warning(
                    "episode %d/%d aborted mid-rollout (%s) — recording as fail, "
                    "continuing.", ep + 1, args.num_episodes, ep_exc, exc_info=True,
                )
                returns.append(ep_return)
                _emit(episode_line(index=ep + 1, total=args.num_episodes,
                                   success=False, ret=ep_return))
                continue
        if frames:
            os.makedirs(args.video_dir, exist_ok=True)
            _save_video(frames, os.path.join(args.video_dir, "rollout.mp4"))
        # Emit ODYSSEY_RESULT *before* tearing down the sim: simulation_app.close()
        # ends the process, so anything after the `finally` never runs (this is why
        # the result line was being lost on real Isaac runs — caught by an e2e run).
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
    finally:
        try:
            env.close()
        finally:
            simulation_app.close()
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.serve_checkpoint:
        served = args.served_model_path or args.checkpoint
        if not served:
            raise SystemExit("--serve_checkpoint requires --served_model_path or --checkpoint")
        if not args.embodiment_tag:
            raise SystemExit("--serve_checkpoint requires --embodiment_tag (server can't infer it)")
        log.info("closed-loop: serving finetuned checkpoint %s (embodiment=%s)",
                 served, args.embodiment_tag)
        with serve_checkpoint(args):
            run_eval(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
