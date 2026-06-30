"""Robosuite evaluation runner — Scope B skeleton.

What this DOES today:
  * Walks the robot's loadout (today exactly one agent) and pulls the
    latest completed training task's checkpoint for it via
    ``mission.latest_checkpoint_for(agent.id)``.
  * Constructs a robosuite environment via a pluggable
    ``env_factory`` (defaults to real ``robosuite.make``).
  * Runs ``num_episodes`` rollouts, querying a pluggable ``policy``
    for actions per step, tallying successes.
  * Returns the result_summary shape lai-trainer's
    ``update_evaluation_results`` expects (``success_rate``,
    ``performance_score``, ``letter_grade``, ``passed``, ``metrics``).

What this DOES NOT do today (the integration gap):
  * Ship a built-in policy that translates an OpenVLA-loaded checkpoint
    into robosuite-shaped actions. That requires:
      - Loading the OpenVLA adapter (HF + peft)
      - Converting robosuite observations (dict of arrays) into the
        prompt + image inputs OpenVLA expects
      - Decoding OpenVLA outputs into 7-DoF (or task-specific) robosuite
        actions
    Each of these is a non-trivial integration that ships in v0.2.x.
    Until then, callers must supply a ``policy`` callable to the
    constructor; the default raises ``NotImplementedError`` with a
    pointer to docs.

What this lets us claim today: the eval-runner *plumbing* works
end-to-end. Pair it with a custom policy and you get real numbers.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from odyssey.runners.base import Runner, TaskContext
from odyssey.runners.evals._common import (
    build_eval_summary,
    resolve_eval_checkpoint,
)
from odyssey.runners.video import save_rollout_video, to_uint8_frame
from odyssey.spec.mission import RobotSpec
from odyssey.spec.tasks import EvaluationTask, EvaluationType, TaskKind

logger = logging.getLogger(__name__)


Policy = Callable[[dict[str, Any]], Any]
"""Maps one robosuite observation dict to one action.

Action shape is robosuite-task-specific (typically a 7-element
ndarray for end-effector control).
"""

PolicyFactory = Callable[[Path], Policy]
"""Builds a Policy from a local checkpoint path. Called once per task
before the episode loop."""

EnvFactory = Callable[[str, str | None], Any]
"""Builds a robosuite environment from ``(benchmark_name, robosuite_robot)``.

``robosuite_robot`` is the PascalCase name Robosuite expects
(``Panda``, ``Sawyer``, ...). None means "let Robosuite pick its
per-env default" — used when the mission's robot is a URDF or a
hosted Lovell embodiment whose name isn't in
``ROBOSUITE_ROBOT_NAMES``.
"""


# Odyssey embodiment → Robosuite robot model. The map deliberately
# mirrors the trimmed LocalRobotProvider.KNOWN_EMBODIMENTS so the same
# names that pass spec validation also run end-to-end.
#
# Anything missing here either has no Robosuite equivalent (Unitree
# quadrupeds, Tiago, Stretch3) or hasn't been wired yet. For those,
# operators can use ``urdf:`` and we'll fall through to Robosuite's
# default robot for the benchmark.
ROBOSUITE_ROBOT_NAMES: dict[str, str] = {
    "franka_panda": "Panda",
    "panda": "Panda",
    "sawyer": "Sawyer",
    "iiwa": "IIWA",
    "jaco": "Jaco",
    "kinova_gen3": "Kinova3",
    "ur5e": "UR5e",
    "baxter": "Baxter",
}


def _default_env_factory(benchmark_name: str, robosuite_robot: str | None) -> Any:
    try:
        import robosuite
    except ImportError as e:
        raise RuntimeError(
            "Robosuite eval requires the 'robosuite' extra. "
            "Install with: pip install 'lovell-odyssey[robosuite]'"
        ) from e
    kwargs: dict[str, Any] = {
        "env_name": benchmark_name,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
    }
    if robosuite_robot is not None:
        # Robosuite accepts either a string ("Panda") or a list
        # (["Panda"], ["Panda", "Sawyer"] for dual-arm envs). We pass a
        # string so single-arm envs work; dual-arm envs (TwoArmLift,
        # etc.) will need an explicit list form when we wire them.
        kwargs["robots"] = robosuite_robot
    return robosuite.make(**kwargs)


def _make_eval_env(
    benchmark_name: str,
    robosuite_robot: str | None,
    config: dict[str, Any] | None = None,
) -> Any:
    """Create a camera-enabled robosuite env for policy evaluation."""
    try:
        import robosuite
    except ImportError as e:
        raise RuntimeError(
            "Robosuite eval requires the 'robosuite' extra. "
            "Install with: pip install 'lovell-odyssey[robosuite]'"
        ) from e
    cfg = config or {}
    camera_names = cfg.get("camera_names", "agentview")
    camera_height = cfg.get("camera_height", 256)
    camera_width = cfg.get("camera_width", 256)
    kwargs: dict[str, Any] = {
        "env_name": benchmark_name,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "camera_names": camera_names,
        "camera_heights": camera_height,
        "camera_widths": camera_width,
    }
    if robosuite_robot is not None:
        kwargs["robots"] = robosuite_robot
    return robosuite.make(**kwargs)


def _default_policy_factory(checkpoint_path: Path) -> Policy:
    raise NotImplementedError(
        "RobosuiteRunner has no built-in policy for v0.1.0-alpha. "
        "Pass policy_factory=... to RobosuiteRunner(...) with a function "
        f"that loads {checkpoint_path!r} and returns a callable mapping "
        "robosuite observations to actions. See "
        "https://github.com/lovell/odyssey/blob/main/docs/runners/robosuite.md"
    )


class RobosuiteRunner(Runner):
    """Evaluation runner for ``evaluation_type: robosuite`` tasks."""

    def __init__(
        self,
        *,
        env_factory: EnvFactory | None = None,
        policy_factory: PolicyFactory | None = None,
        max_steps_per_episode: int = 500,
    ):
        self._env_factory = env_factory or _default_env_factory
        self._policy_factory = policy_factory or _default_policy_factory
        self._max_steps = max_steps_per_episode

    @property
    def name(self) -> str:
        return "robosuite"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.EVALUATION}

    @property
    def supported_types(self) -> set[str]:
        return {EvaluationType.ROBOSUITE.value}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        spec = context.task.spec
        if not isinstance(spec, EvaluationTask):
            raise TypeError(
                f"RobosuiteRunner expects EvaluationTask, got {type(spec).__name__}"
            )

        checkpoint = resolve_eval_checkpoint(context)
        await context.emit_progress(
            "model_loading",
            step="load_policy",
            step_label=str(checkpoint),
        )

        cfg = spec.config or {}
        image_key = cfg.get("image_key", "agentview_image")
        # Opt-in rollout video. The camera-enabled eval env already renders the
        # frame each step (the policy uses it), so capture is near-free: a list
        # append per step plus one mp4 encode per episode, offloaded to a thread.
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
        use_planned = (
            self._policy_factory is _default_policy_factory
            and _has_specialist(context)
        )

        if use_planned:
            runtime = _build_planned_runtime(context, checkpoint, spec)
            await context.emit_progress(
                "model_loading",
                step="load_specialist",
                step_label="PlannedEvalRuntime (PILOT + SPECIALIST)",
            )
            policy = None
        elif self._policy_factory is _default_policy_factory:
            from odyssey.runners.models.openvla import make_openvla_policy

            policy = make_openvla_policy(
                checkpoint,
                config=spec.config,
                benchmark_name=spec.benchmark_name,
            )
            runtime = None
        else:
            policy = self._policy_factory(checkpoint)
            runtime = None

        robosuite_robot = _resolve_robosuite_robot(context.mission.spec.robot)
        await context.emit_progress(
            "executing",
            step="env_construct",
            step_label=(
                f"benchmark={spec.benchmark_name} "
                f"robot={robosuite_robot or 'robosuite-default'}"
            ),
        )

        # Env: use camera-enabled env when no custom factory was injected
        if self._env_factory is _default_env_factory:
            env = _make_eval_env(spec.benchmark_name, robosuite_robot, spec.config)
        else:
            env = self._env_factory(spec.benchmark_name, robosuite_robot)

        num_episodes = spec.num_episodes
        successes = 0
        episode_returns: list[float] = []

        # Video encoding runs in a thread so it overlaps the next episode's
        # rollout instead of blocking it; we await all encodes after the loop.
        # FUTURE (deferred — see feat/rollout-video-capture notes): the episodes
        # themselves are independent and could be parallelized, but on a single
        # 24GB GPU one OpenVLA-7B already saturates VRAM, so true episode
        # parallelism needs vectorized envs + batched policy inference (verify
        # OpenVLA's action head batches) or multi-GPU. Not worth it here.
        loop = asyncio.get_running_loop()
        encode_tasks: list[asyncio.Future[Any]] = []
        video_paths: list[str] = []

        # Resolve the task instruction for PlannedEvalRuntime
        task_instruction = _resolve_task_instruction(spec) if runtime else ""

        for ep in range(1, num_episodes + 1):
            if context.cancelled():
                logger.info(
                    "Robosuite task %s cancelled at episode %d/%d",
                    context.task.id,
                    ep,
                    num_episodes,
                )
                break

            obs = env.reset()
            episode_return = 0.0
            success = False
            frames: list[Any] = []  # rebound fresh each episode; in-flight encodes keep their own list

            if runtime:
                # Plan from the first frame so a multimodal SPECIALIST can
                # ground its plan in the scene (text planners ignore it).
                first_image = _extract_image(obs, image_key)
                plan = runtime.begin_episode(task_instruction, first_image)
                logger.info("Episode %d plan: %s", ep, plan)
                # Surface the plan in telemetry too — the CLI runs at the root
                # WARNING level, so the INFO line above is invisible. A 1-phase
                # plan means the SPECIALIST fell back to single-step.
                await context.emit_progress(
                    "executing",
                    step="episode_plan",
                    step_index=ep,
                    step_total=num_episodes,
                    step_label=f"episode {ep}: {len(plan)} phase(s): {plan}",
                )

            for _step in range(self._max_steps):
                if context.cancelled():
                    break

                if runtime:
                    image = _extract_image(obs, image_key)
                    if capture_video:
                        _capture_frame(frames, image)
                    action = runtime.get_action(image)
                else:
                    assert policy is not None
                    if capture_video:
                        _capture_frame(frames, _extract_image(obs, image_key))
                    action = policy(obs if isinstance(obs, dict) else {"observation": obs})

                step_result = env.step(action)
                # robosuite returns (obs, reward, done, info)
                obs, reward, done, info = step_result
                episode_return += float(reward)
                if done:
                    success = bool(
                        info.get("success", reward > 0)
                        if isinstance(info, dict)
                        else reward > 0
                    )
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

        # Tear down the planner once rollouts finish — closes the
        # out-of-process RemotePlanner subprocess (no-op for in-process /
        # the single-agent policy path). atexit covers the exception path.
        if runtime is not None:
            runtime.close()

        # Drain the in-flight video encodes (no-op when capture is off). Failed
        # encodes return None — they're dropped, never fatal to the eval.
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
            benchmark_name=spec.benchmark_name,
            checkpoint_path=checkpoint,
        )
        # Attach rollout videos (empty list when capture is off) as an eval artifact.
        summary["artifacts"] = {"videos": video_paths}
        return summary


def _resolve_robosuite_robot(robot: RobotSpec) -> str | None:
    """Translate the mission's robot spec into a Robosuite robot name.

    Returns the PascalCase string Robosuite expects, or None when no
    override should be passed (URDF specs, hosted Lovell ids, or no
    embodiment at all — in which case Robosuite picks its per-env
    default robot).

    Raises ValueError when ``embodiment`` is set but has no Robosuite
    equivalent. That's preferable to silently swapping in Robosuite's
    default and producing eval numbers under a robot the operator
    never asked for.
    """
    if robot.embodiment is None:
        return None
    name = ROBOSUITE_ROBOT_NAMES.get(robot.embodiment)
    if name is None:
        raise ValueError(
            f"Robosuite has no built-in robot for embodiment "
            f"{robot.embodiment!r}. Supported: "
            f"{sorted(ROBOSUITE_ROBOT_NAMES)}. Either pick one of these, "
            "or supply ``urdf:`` and a custom env_factory."
        )
    return name


def _has_specialist(context: TaskContext) -> bool:
    """Check whether the loadout includes a SPECIALIST agent."""
    from odyssey.spec.agents import AgentRole

    for agent in context.agents or context.mission.spec.robot.agents:
        if agent.role == AgentRole.SPECIALIST:
            return True
    return False


def _find_specialist_model(context: TaskContext) -> tuple[str, str | None]:
    """Return ``(model_base, quantization)`` for the first SPECIALIST."""
    from odyssey.spec.agents import AgentRole
    from odyssey.spec.refs import HFModelRef

    for agent in context.agents or context.mission.spec.robot.agents:
        if agent.role == AgentRole.SPECIALIST:
            model = agent.model
            if not isinstance(model, HFModelRef):
                raise ValueError(
                    f"SPECIALIST agent {agent.id!r} uses a non-HuggingFace model. "
                    "Only HuggingFace models are supported for SPECIALIST inference."
                )
            return model.base, model.quantization
    raise ValueError("No SPECIALIST agent found in the loadout")


def _extract_image(obs: Any, image_key: str) -> Any:
    """Pull the RGB camera frame out of a robosuite observation.

    Prefers ``image_key`` (e.g. ``agentview_image``); falls back to the first
    value when the obs isn't the expected dict shape.
    """
    obs_dict = obs if isinstance(obs, dict) else {"observation": obs}
    return obs_dict.get(image_key, next(iter(obs_dict.values())))


def _capture_frame(frames: list[Any], image: Any) -> None:
    """Append one rollout frame to ``frames`` for the demo video.

    Robosuite's offscreen renderer stores frames bottom-to-top, so we flip
    vertically to get an upright clip. Non-image observations (the
    ``_extract_image`` fallback) are silently skipped.
    """
    frame = to_uint8_frame(image)
    if frame is not None:
        frames.append(frame[::-1])


def _resolve_task_instruction(spec: EvaluationTask) -> str:
    """Resolve the natural-language task instruction for the benchmark."""
    from odyssey.runners.models.openvla import _DEFAULT_INSTRUCTIONS

    cfg = spec.config or {}
    return str(
        cfg.get("task_instruction")
        or _DEFAULT_INSTRUCTIONS.get(spec.benchmark_name, "complete the task")
    )


def _build_planned_runtime(
    context: TaskContext,
    checkpoint: Path,
    spec: EvaluationTask,
) -> Any:
    """Build a PlannedEvalRuntime from the mission's PILOT + SPECIALIST.

    The SPECIALIST is a multimodal Gemma 4 task planner that runs **out of
    process**: ``ODYSSEY_SPECIALIST_PYTHON`` must point at a separate venv's
    python whose modern ``transformers`` + ``torchvision`` can host Gemma 4,
    free of OpenVLA's pinned ``transformers==4.40.1`` in this venv. The
    SPECIALIST is inference-only (it runs its base checkpoint to plan); only the
    PILOT is trained.
    """
    from odyssey.runners.agents.planned import PhaseConfig, PlannedEvalRuntime
    from odyssey.runners.agents.remote_planner import RemotePlanner
    from odyssey.runners.models.openvla import VLARuntime

    cfg = spec.config or {}
    unnorm_key = cfg.get("unnorm_key", "bridge_orig")

    # Load the PILOT as a VLARuntime (per-call instruction)
    pilot = VLARuntime(checkpoint, unnorm_key=unnorm_key)

    # The SPECIALIST planner runs out-of-process: Gemma 4 needs a modern
    # transformers/torchvision incompatible with OpenVLA's pin in this venv.
    model_base, quantization = _find_specialist_model(context)
    specialist_python = os.getenv("ODYSSEY_SPECIALIST_PYTHON")
    if not specialist_python:
        raise RuntimeError(
            "Multi-agent eval requires the out-of-process SPECIALIST: set "
            "ODYSSEY_SPECIALIST_PYTHON to the specialist venv's python (see the "
            "README 'Multi-agent evaluation' section). The multimodal Gemma 4 "
            "planner cannot load in this venv, which pins transformers==4.40.1 "
            "for OpenVLA."
        )

    logger.info(
        "SPECIALIST out-of-process: model=%s via %s", model_base, specialist_python
    )
    planner = RemotePlanner(
        model_base,
        quantization,
        python_path=specialist_python,
    )

    # Phase config from mission config
    steps_per_phase = cfg.get("steps_per_phase", 50)
    phase_config = PhaseConfig(steps_per_phase=steps_per_phase)

    task_instruction = _resolve_task_instruction(spec)

    return PlannedEvalRuntime(
        pilot=pilot,
        planner=planner,
        phase_config=phase_config,
        fallback_instruction=task_instruction,
    )
