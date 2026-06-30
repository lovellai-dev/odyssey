"""LIBERO evaluation runner.

LIBERO (https://libero-project.github.io/) is a robosuite-based simulation
benchmark suite of language-conditioned manipulation tasks on a **Franka Panda** —
"put the X in/on the Y" style tasks across four suites (spatial / object / goal /
long). It is the standard sim benchmark that open VLAs publish checkpoints for, so
it's the fastest path to seeing a VLA pilot actually *succeed* in sim (the published
``openvla/openvla-7b-finetuned-libero-*`` checkpoints score ~70-90%).

This runner mirrors ``RobosuiteRunner`` (same episode loop, video capture,
``build_eval_summary``, single-agent OpenVLA policy + multi-agent ``PlannedEvalRuntime``)
and only swaps the environment layer for LIBERO. The env/obs/action handling follows
OpenVLA's reference ``experiments/robot/libero/run_libero_eval.py``.

Mission wiring (eval-only, no training task needed):

    evaluation_type: libero
    benchmark_name: libero_object          # the LIBERO task suite
    config:
      checkpoint: openvla/openvla-7b-finetuned-libero-object   # HF repo id or local path
      unnorm_key: libero_object            # must match the checkpoint/suite
      task_id: 0                           # which task within the suite (default 0)

⚠️ FIRST-RUN VALIDATION: the LIBERO package pins specific robosuite/robomimic
versions — confirm it co-installs with the OpenVLA stack (see the example setup.sh).
The image orientation (``_libero_image``) and gripper action convention
(``_libero_action``) mirror OpenVLA's reference eval; if the arm behaves inverted on
the first run, those are the two knobs to check against ``run_libero_eval.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from odyssey.runners.base import Runner, TaskContext
from odyssey.runners.evals._common import build_eval_summary, resolve_eval_checkpoint
from odyssey.runners.video import save_rollout_video, to_uint8_frame
from odyssey.spec.tasks import EvaluationTask, EvaluationType, TaskKind

logger = logging.getLogger(__name__)


def _make_libero_env(suite_name: str, task_id: int, cfg: dict[str, Any]) -> Any:
    """Create a LIBERO offscreen env for ``task_id`` in ``suite_name``.

    Returns ``(env, task, init_states)``. Mirrors the env construction in
    OpenVLA's ``run_libero_eval.py``.
    """
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:
        raise RuntimeError(
            "LIBERO eval requires the 'libero' package. Install LIBERO into the "
            "OpenVLA env (see examples/franka-libero/setup.sh):\n"
            "  pip install 'libero @ git+https://github.com/Lifelong-Robot-Learning/LIBERO.git'"
        ) from e

    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(
            f"Unknown LIBERO suite {suite_name!r}. Available: "
            f"{sorted(benchmark_dict)}. Set benchmark_name to one of these."
        )
    suite = benchmark_dict[suite_name]()
    n_tasks = suite.n_tasks
    if not (0 <= task_id < n_tasks):
        raise ValueError(
            f"task_id {task_id} out of range for suite {suite_name!r} "
            f"(has {n_tasks} tasks 0..{n_tasks - 1})."
        )
    task = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)

    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    height = int(cfg.get("camera_height", 256))
    width = int(cfg.get("camera_width", 256))
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=height,
        camera_widths=width,
    )
    return env, task, init_states


def _libero_image(obs: Any, image_key: str) -> Any:
    """Pull the agentview RGB frame and orient it for the policy.

    LIBERO's offscreen agentview is stored rotated 180° relative to what OpenVLA
    was trained on, so ``run_libero_eval.py`` flips it both ways (``[::-1, ::-1]``).
    """
    img = obs[image_key]
    return img[::-1, ::-1]


def _libero_action(action: Any) -> Any:
    """Map an OpenVLA 7-DoF action to LIBERO's expected action.

    Mirrors OpenVLA's reference eval: the predicted gripper (last dim, ~[0, 1]) is
    rescaled to [-1, 1], binarized, then inverted to match LIBERO's gripper sign
    convention (positive = close upstream → LIBERO wants the opposite).
    """
    import numpy as np

    a = np.asarray(action, dtype=np.float64).copy()
    g = 2.0 * a[-1] - 1.0          # [0,1] -> [-1,1]
    g = 1.0 if g > 0.0 else -1.0   # binarize
    a[-1] = -g                     # invert for LIBERO
    return a


class LiberoRunner(Runner):
    """Evaluation runner for ``evaluation_type: libero`` tasks."""

    def __init__(self, *, max_steps_per_episode: int = 600):
        # LIBERO tasks are longer-horizon than robosuite Lift; 600 is a safe default
        # (long suite needs more). Override via config.max_steps_per_episode.
        self._max_steps = max_steps_per_episode

    @property
    def name(self) -> str:
        return "libero"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.EVALUATION}

    @property
    def supported_types(self) -> set[str]:
        return {EvaluationType.LIBERO.value}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        spec = context.task.spec
        if not isinstance(spec, EvaluationTask):
            raise TypeError(
                f"LiberoRunner expects EvaluationTask, got {type(spec).__name__}"
            )

        cfg = spec.config or {}
        suite_name = spec.benchmark_name
        task_id = int(cfg.get("task_id", 0))
        image_key = cfg.get("image_key", "agentview_image")
        max_steps = int(cfg.get("max_steps_per_episode", self._max_steps))
        warmup_steps = int(cfg.get("num_warmup_steps", 10))

        checkpoint = resolve_eval_checkpoint(context)
        await context.emit_progress(
            "model_loading", step="load_policy", step_label=str(checkpoint)
        )

        # Video capture (opt-in) — same near-free pattern as RobosuiteRunner.
        capture_video = bool(cfg.get("capture_video", False))
        video_fps = int(cfg.get("video_fps", 24))
        video_format = str(cfg.get("video_format", "mp4")).lstrip(".")
        video_dir: Path | None = None
        if capture_video and context.output_dir is None:
            logger.warning(
                "capture_video set but TaskContext.output_dir is None — disabling video"
            )
            capture_video = False
        elif capture_video:
            video_dir = context.output_dir / "videos"  # type: ignore[operator]

        # Build the LIBERO env first — the task's natural-language instruction comes
        # from the benchmark (per task), not a default table.
        await context.emit_progress(
            "executing",
            step="env_construct",
            step_label=f"suite={suite_name} task_id={task_id}",
        )
        env, task, init_states = _make_libero_env(suite_name, task_id, cfg)
        instruction = str(getattr(task, "language", None) or cfg.get("task_instruction") or "complete the task")
        logger.info("LIBERO task %d instruction: %r", task_id, instruction)

        # Single-agent (OpenVLA policy) vs multi-agent (PlannedEvalRuntime + Gemma).
        use_planned = _has_specialist(context)
        if use_planned:
            runtime = _build_planned_runtime(context, checkpoint, cfg, instruction)
            await context.emit_progress(
                "model_loading",
                step="load_specialist",
                step_label="PlannedEvalRuntime (PILOT + SPECIALIST)",
            )
            policy = None
        else:
            from odyssey.runners.models.openvla import make_openvla_policy

            # Bake the LIBERO task's own instruction + the suite unnorm_key.
            policy_cfg = dict(cfg)
            policy_cfg["task_instruction"] = instruction
            policy_cfg.setdefault("image_key", image_key)
            policy = make_openvla_policy(
                checkpoint, config=policy_cfg, benchmark_name=suite_name
            )
            runtime = None

        num_episodes = spec.num_episodes
        successes = 0
        episode_returns: list[float] = []
        loop = asyncio.get_running_loop()
        encode_tasks: list[asyncio.Future[Any]] = []
        video_paths: list[str] = []

        for ep in range(1, num_episodes + 1):
            if context.cancelled():
                logger.info("LIBERO task %s cancelled at episode %d", context.task.id, ep)
                break

            obs = env.reset()
            # Deterministic init state per episode (cycle through the task's set).
            env.set_init_state(init_states[(ep - 1) % len(init_states)])

            # Let physics settle: a few no-op steps (gripper open) before acting.
            dummy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            for _ in range(warmup_steps):
                obs, _, _, _ = env.step(dummy)

            episode_return = 0.0
            success = False
            frames: list[Any] = []

            if runtime:
                first_image = _libero_image(obs, image_key)
                plan = runtime.begin_episode(instruction, first_image)
                logger.info("Episode %d plan: %s", ep, plan)
                await context.emit_progress(
                    "executing",
                    step="episode_plan",
                    step_index=ep,
                    step_total=num_episodes,
                    step_label=f"episode {ep}: {len(plan)} phase(s): {plan}",
                )

            for _step in range(max_steps):
                if context.cancelled():
                    break
                image = _libero_image(obs, image_key)
                if capture_video:
                    frame = to_uint8_frame(image)
                    if frame is not None:
                        frames.append(frame)

                if runtime:
                    raw_action = runtime.get_action(image)
                else:
                    assert policy is not None
                    raw_action = policy({image_key: image})

                action = _libero_action(raw_action)
                obs, reward, done, _info = env.step(action.tolist())
                episode_return += float(reward)
                if done:
                    # LIBERO sets done=True when the task is solved.
                    success = True
                    break

            if success:
                successes += 1
            episode_returns.append(episode_return)
            await context.emit_progress(
                "executing",
                step="episode_complete",
                step_index=ep,
                step_total=num_episodes,
                step_label=f"episode {ep}: {'PASS' if success else 'FAIL'} return={episode_return:.3f}",
            )

            if capture_video and frames and video_dir is not None:
                tag = "PASS" if success else "FAIL"
                out_path = video_dir / f"episode_{ep:02d}_{tag}.{video_format}"
                encode_tasks.append(
                    loop.run_in_executor(None, save_rollout_video, frames, out_path, video_fps)
                )

        if runtime is not None:
            runtime.close()
        env.close()

        if encode_tasks:
            video_paths = [str(p) for p in await asyncio.gather(*encode_tasks) if p is not None]
            if video_paths:
                await context.emit_progress(
                    "executing",
                    step="videos_saved",
                    step_label=f"{len(video_paths)} rollout video(s)",
                    metadata={"videos": video_paths},
                )

        attempted = len(episode_returns)
        summary = build_eval_summary(
            num_episodes=attempted,
            successes=successes,
            episode_returns=episode_returns,
            benchmark_name=f"{suite_name}[task={task_id}]",
            checkpoint_path=checkpoint,
        )
        summary["metrics"]["instruction"] = instruction
        summary["artifacts"] = {"videos": video_paths}
        return summary


# ---------------------------------------------------------------------------
# Multi-agent helpers.
# NOTE: these mirror the equivalents in robosuite.py. They're duplicated here to
# avoid a fragile cross-runner private import (the exact coupling PR #41 removed);
# consolidating the shared planner/specialist helpers into _common.py is a tracked
# follow-up.
# ---------------------------------------------------------------------------

def _has_specialist(context: TaskContext) -> bool:
    from odyssey.spec.agents import AgentRole

    for agent in context.agents or context.mission.spec.robot.agents:
        if agent.role == AgentRole.SPECIALIST:
            return True
    return False


def _find_specialist_model(context: TaskContext) -> tuple[str, str | None]:
    from odyssey.spec.agents import AgentRole
    from odyssey.spec.refs import HFModelRef

    for agent in context.agents or context.mission.spec.robot.agents:
        if agent.role == AgentRole.SPECIALIST:
            model = agent.model
            if not isinstance(model, HFModelRef):
                raise ValueError(
                    f"SPECIALIST agent {agent.id!r} uses a non-HuggingFace model."
                )
            return model.base, model.quantization
    raise ValueError("No SPECIALIST agent found in the loadout")


def _build_planned_runtime(
    context: TaskContext,
    checkpoint: Path,
    cfg: dict[str, Any],
    instruction: str,
) -> Any:
    """Compose the PILOT (OpenVLA) + out-of-process SPECIALIST (Gemma) planner."""
    from odyssey.runners.agents.planned import PhaseConfig, PlannedEvalRuntime
    from odyssey.runners.agents.remote_planner import RemotePlanner
    from odyssey.runners.models.openvla import VLARuntime

    unnorm_key = cfg.get("unnorm_key", "bridge_orig")
    pilot = VLARuntime(checkpoint, unnorm_key=unnorm_key)

    model_base, quantization = _find_specialist_model(context)
    specialist_python = os.getenv("ODYSSEY_SPECIALIST_PYTHON")
    if not specialist_python:
        raise RuntimeError(
            "Multi-agent eval requires the out-of-process SPECIALIST: set "
            "ODYSSEY_SPECIALIST_PYTHON to the specialist venv's python (the multimodal "
            "Gemma 4 planner cannot load in the OpenVLA-pinned env)."
        )
    logger.info("SPECIALIST out-of-process: model=%s via %s", model_base, specialist_python)
    planner = RemotePlanner(model_base, quantization, python_path=specialist_python)

    phase_config = PhaseConfig(steps_per_phase=int(cfg.get("steps_per_phase", 50)))
    return PlannedEvalRuntime(
        pilot=pilot,
        planner=planner,
        phase_config=phase_config,
        fallback_instruction=instruction,
    )
