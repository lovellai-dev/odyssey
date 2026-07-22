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
from typing import Any

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
    # --- multi-agent coordination (opt-in; needs a SPECIALIST in the loadout) ---
    ap.add_argument("--coordination", default="",
                    help="'' (single-agent) | planning | delegation. Drives the "
                         "SPECIALIST + a chunk-aware Planned/DelegatedEvalRuntime.")
    ap.add_argument("--specialist_model", default="",
                    help="SPECIALIST HF model id (e.g. google/gemma-4-E2B-it).")
    ap.add_argument("--specialist_quantization", default="")
    ap.add_argument("--specialist_python", default="",
                    help="interpreter with the SPECIALIST stack (ODYSSEY_SPECIALIST_PYTHON).")
    ap.add_argument("--phase_strategy", default="fixed_steps",
                    help="fixed_steps | timeout | completion_gated (planning arm).")
    ap.add_argument("--steps_per_phase", type=int, default=50)
    ap.add_argument("--phase_check_every", type=int, default=10)
    ap.add_argument("--phase_max_steps", type=int, default=100)
    ap.add_argument("--max_phases", type=int, default=8,
                    help="orchestration: cap on routed sub-tasks per episode.")
    ap.add_argument("--check_image_key", default="",
                    help="frame for the completion check (default: --image_key / agentview; "
                         "the wrist cam is top-down + occluded at grasp time, worse for judging).")
    ap.add_argument("--check_crop", type=float, default=0.0,
                    help="center-crop this fraction of the check frame then upscale — the "
                         "objects are ~20px in the 256px agentview, too small for the VLM. "
                         "0 = off; e.g. 0.6 zooms into the central 60%.")
    ap.add_argument("--check_upscale", type=int, default=384,
                    help="upscale the cropped check frame to NxN (more pixels on the object).")
    ap.add_argument("--check_debug", type=_bool, default=False,
                    help="dump the frames the completion check sees to "
                         "/tmp/gr00t_check_frames (to inspect what the VLM judges).")
    ap.add_argument("--trace", type=_bool, default=False,
                    help="verbose per-action trace: log each PILOT/SPECIALIST/"
                         "ORCHESTRATOR action as it happens.")
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

    Return the ``<checkpoint>/<suite>`` subdir when it exists — resolving an HF repo
    id to its local cache snapshot first (offline). This lets the shipped mission use
    ``checkpoint: nvidia/GR00T-N1.7-LIBERO`` and still serve the per-suite subdir: the
    server runs ``HF_HUB_OFFLINE`` and the repo ROOT has no config (the model lives in
    the suite subdir), so a bare repo id would fail. Falls back to the checkpoint
    as-is (a local dir, or an id the server resolves).
    """
    local = str(checkpoint)
    if not os.path.isdir(local):
        try:  # maybe an HF repo id already cached — resolve its snapshot dir offline
            from huggingface_hub import snapshot_download

            local = snapshot_download(local, local_files_only=True)
        except Exception:
            return str(checkpoint)
    sub = os.path.join(local, str(suite))
    return sub if os.path.isdir(sub) else local


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


def _zoom_frame(frame: Any, crop_frac: float, upscale: int):
    """Center-crop ``crop_frac`` of the frame, then upscale to ``upscale``x``upscale``.

    The manipulated object is only ~20px in the 256px agentview — too small for the
    int4 VLM completion check. Cropping the central region (where the workspace sits)
    and upscaling makes the object fill the frame ("zoom" + more pixels on target).
    Pure image op on the existing render — no camera/sim change. ``crop_frac<=0`` or
    ``>=1`` disables the crop; ``upscale<=0`` keeps the crop's native size.
    """
    import numpy as np
    from PIL import Image

    arr = np.asarray(frame)
    h, w = arr.shape[:2]
    if 0.0 < crop_frac < 1.0:
        ch, cw = max(1, int(h * crop_frac)), max(1, int(w * crop_frac))
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        arr = arr[y0:y0 + ch, x0:x0 + cw]
    if upscale and upscale > 0:
        arr = np.asarray(
            Image.fromarray(arr.astype(np.uint8)).resize((upscale, upscale), Image.LANCZOS)
        )
    return np.ascontiguousarray(arr).astype(np.uint8)


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

_AGENT_TRACE = logging.getLogger("odyssey.agents.trace")


class _TracingAgent:
    """Wraps the out-of-process agent to log each plan/ground/check/route call.

    Enabled by ``--trace``; delegates to the real ``RemotePlanner`` and logs the
    actor (SPECIALIST / ORCHESTRATOR) + result, so a run shows *who acted when*.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def plan(self, task_instruction: str, image: Any = None) -> Any:
        steps = self._inner.plan(task_instruction, image)
        _AGENT_TRACE.info(
            "[SPECIALIST] plan(%r) -> %d step(s): %s", task_instruction, len(steps), steps
        )
        return steps

    def check_done(self, instruction: str, image: Any) -> bool:
        done = bool(self._inner.check_done(instruction, image))
        _AGENT_TRACE.info(
            "[SPECIALIST] check(%r) -> %s", instruction, "YES" if done else "no"
        )
        return done

    def ground(self, target_query: str, image: Any) -> str:
        target = self._inner.ground(target_query, image)
        _AGENT_TRACE.info("[SPECIALIST] ground(%r) -> %r", target_query, target)
        return target

    def route(self, task_instruction: str, image: Any, history: list) -> Any:
        decision = self._inner.route(task_instruction, image, history)
        _AGENT_TRACE.info(
            "[ORCHESTRATOR] route -> %r (done=%s)", decision.subtask, decision.done
        )
        return decision

    def close(self) -> None:
        closer = getattr(self._inner, "close", None)
        if callable(closer):
            closer()


class _CheckFrameDetector:
    """Run the completion check on a possibly DIFFERENT frame than route/ground/plan.

    Lets the check use a chosen camera (``--check_image_key``) instead of the scene
    frame the runtime passes. Defaults to the scene frame (agentview); the wrist cam
    was tried and is worse (top-down, occluded at grasp time — see
    ``docs/multiagent-execution-flow.md``). The recipe stashes the current check frame
    each step via ``set_frame``.
    """

    def __init__(self, inner: Any, *, debug_dir: str | None = None) -> None:
        self._inner = inner
        self._frame: Any = None
        self._debug_dir = debug_dir  # if set, dump the frames the check actually sees
        self._saved = 0

    def set_frame(self, frame: Any) -> None:
        self._frame = frame

    def check_done(self, instruction: str, image: Any) -> bool:
        frame = self._frame if self._frame is not None else image
        if self._debug_dir is not None and frame is not None and self._saved < 60:
            self._dump(frame)
        return bool(self._inner.check_done(instruction, frame))

    def _dump(self, frame: Any) -> None:
        try:  # never break the eval for a debug dump
            import numpy as _np
            from PIL import Image

            os.makedirs(self._debug_dir, exist_ok=True)  # type: ignore[arg-type]
            path = os.path.join(self._debug_dir, f"check_{self._saved:03d}.png")  # type: ignore[arg-type]
            Image.fromarray(_np.asarray(frame, dtype=_np.uint8)).save(path)
            self._saved += 1
        except Exception as e:
            log.warning("check frame dump failed: %s", e)


def _build_coordination_runtime(args, client, instruction):
    """Build a chunk-aware multi-agent runtime (planning|delegation) over GR00T.

    The GR00T ``PolicyClient`` is wrapped in a ``ChunkPilotAdapter`` (satisfies
    ``PilotRuntime`` — buffers a chunk, drains per step, re-queries on phase change);
    the SPECIALIST is the out-of-process ``RemotePlanner`` (plan/check/ground over one
    loaded model). Returns ``(runtime, adapter)``; the caller drives
    ``runtime.get_action(image)`` and pushes each raw obs via ``adapter.set_obs(obs)``.
    """
    import functools

    from odyssey.runners.agents.remote_planner import RemotePlanner
    from odyssey.runners.evals.chunk_pilot_adapter import ChunkPilotAdapter

    t = _transforms()
    adapter = ChunkPilotAdapter(
        client,
        obs_builder=functools.partial(
            _build_obs,
            image_key=args.image_key,
            wrist_image_key=args.wrist_image_key,
            flip=args.flip_images,
        ),
        action_mapper=functools.partial(
            t.gr00t_action_to_libero, translation_only=args.translation_only
        ),
        n_action_steps=args.n_action_steps,
    )
    if not args.specialist_model:
        raise SystemExit("--coordination requires --specialist_model (a SPECIALIST agent).")
    specialist: Any = RemotePlanner(
        args.specialist_model,
        args.specialist_quantization or None,
        python_path=args.specialist_python or None,
    )
    if args.trace:
        _AGENT_TRACE.setLevel(logging.INFO)
        specialist = _TracingAgent(specialist)  # log each SPECIALIST/ORCHESTRATOR call
    # The completion check can use a different camera than route/ground/plan via
    # --check_image_key; defaults to the scene frame (agentview).
    check_detector = _CheckFrameDetector(
        specialist,
        debug_dir="/tmp/gr00t_check_frames" if args.check_debug else None,
    )
    cfg = {
        "phase_strategy": args.phase_strategy,
        "steps_per_phase": args.steps_per_phase,
        "phase_check_every": args.phase_check_every,
        "phase_max_steps": args.phase_max_steps,
        "max_phases": args.max_phases,
    }
    coordination = args.coordination.strip().lower()
    if coordination == "delegation":
        from odyssey.runners.agents.delegated import DelegatedEvalRuntime, DelegationConfig

        runtime = DelegatedEvalRuntime(
            pilot=adapter,
            grounder=specialist,
            config=DelegationConfig.from_config(cfg),
            detector=check_detector,
            task_fallback=instruction,
        )
    elif coordination == "planning":
        from odyssey.runners.agents.planned import PhaseConfig, PlannedEvalRuntime

        runtime = PlannedEvalRuntime(
            pilot=adapter,
            planner=specialist,
            phase_config=PhaseConfig.from_config(cfg),
            detector=check_detector,
            fallback_instruction=instruction,
        )
    elif coordination == "orchestration":
        from odyssey.runners.agents.orchestrated import (
            OrchestratedEvalRuntime,
            OrchestrationConfig,
        )

        # Regime D: the SPECIALIST/ORCHESTRATOR (same Gemma) routes the next
        # sub-instruction dynamically + gates hand-back via check_done.
        runtime = OrchestratedEvalRuntime(
            pilot=adapter,
            orchestrator=specialist,
            config=OrchestrationConfig.from_config(cfg),
            detector=check_detector,
            task_fallback=instruction,
        )
    else:
        raise SystemExit(
            f"unknown coordination {args.coordination!r}; "
            "allowed: planning, delegation, orchestration"
        )
    return runtime, adapter, check_detector


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

    coordination = (args.coordination or "").strip().lower()
    runtime = adapter = check_detector = None
    check_key = args.check_image_key or args.image_key
    if coordination:
        runtime, adapter, check_detector = _build_coordination_runtime(
            args, client, instruction
        )
        log.info("multi-agent coordination=%s (SPECIALIST=%s, check_frame=%s)",
                 coordination, args.specialist_model, check_key)

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
            if runtime is not None:
                # Multi-agent: the runtime feeds phase sub-instructions to the chunk
                # adapter (which buffers/flushes chunks); one env.step per get_action.
                runtime.begin_episode(
                    instruction, _frame(obs, args.image_key, flip=args.flip_images)
                )
                adapter.reset()
                while step < args.max_steps_per_episode and not success:
                    adapter.set_obs(obs)
                    if check_detector is not None:  # completion check sees the agentview cam
                        cframe = _frame(obs, check_key, flip=args.flip_images)
                        if args.check_crop > 0 or args.check_upscale > 0:
                            cframe = _zoom_frame(cframe, args.check_crop, args.check_upscale)
                        check_detector.set_frame(cframe)
                    action = runtime.get_action(
                        _frame(obs, args.image_key, flip=args.flip_images)
                    )
                    obs, reward, done, _info = env.step(action.tolist())
                    ep_return += float(reward)
                    for ev in runtime.drain_phase_events():
                        log.info("episode %d phase: %s", ep, ev)
                    if video_dir is not None:
                        frame = to_uint8_frame(_frame(obs, args.image_key, flip=args.flip_images))
                        if frame is not None:
                            frames.append(frame)
                    step += 1
                    if done:  # LIBERO sets done=True when the task is solved
                        success = True
                        break
            else:
                # Single-agent: replay each GR00T chunk open-loop.
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
                            # match the pilot's orientation: LIBERO's agentview is stored
                            # 180°-rotated, so flip the video frame too (else upside down).
                            frame = to_uint8_frame(_frame(obs, args.image_key, flip=args.flip_images))
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

    if runtime is not None:
        runtime.close()  # shut down the SPECIALIST subprocess (reused across episodes)
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
            "coordination": coordination or "single-agent",
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
