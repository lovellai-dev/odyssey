"""Tests for the rollout-video helper and its wiring into the Robosuite runner.

GPU-free: the helper is exercised with synthetic numpy frames encoded to a GIF
(pillow plugin, no ffmpeg), and the runner path uses an injected image-shaped
fake env. The mp4 path and a real rollout are validated on a GPU eval run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as iio
import numpy as np
import pytest

from odyssey.engine.lifecycle import TaskStatus
from odyssey.engine.records import MissionRun, TaskRun
from odyssey.runners.base import TaskContext
from odyssey.runners.evals import robosuite as robosuite_runner
from odyssey.runners.evals.robosuite import RobosuiteRunner, _resolve_cameras
from odyssey.runners.video import save_rollout_video, to_uint8_frame
from odyssey.spec import (
    AgentRole,
    AgentSpec,
    EvaluationTask,
    EvaluationType,
    HFModelRef,
    Mission,
    MissionMetadata,
    RobotSpec,
    TrainingTask,
    TrainingType,
)
from odyssey.telemetry import EventPublisher

# ---------------------------------------------------------------------------
# Helper: to_uint8_frame
# ---------------------------------------------------------------------------

def test_to_uint8_frame_passes_through_uint8() -> None:
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    out = to_uint8_frame(arr)
    assert out.dtype == np.uint8
    assert out.shape == (8, 8, 3)


def test_to_uint8_frame_normalizes_float_0_1() -> None:
    out = to_uint8_frame(np.ones((4, 4, 3), dtype=np.float32))
    assert out.dtype == np.uint8
    assert int(out.max()) == 255


def test_to_uint8_frame_drops_alpha_channel() -> None:
    out = to_uint8_frame(np.zeros((4, 4, 4), dtype=np.uint8))
    assert out.shape == (4, 4, 3)


def test_to_uint8_frame_rejects_non_image() -> None:
    assert to_uint8_frame(np.array([0.0, 1.0])) is None


# ---------------------------------------------------------------------------
# Helper: save_rollout_video
# ---------------------------------------------------------------------------

def test_save_rollout_video_writes_gif(tmp_path: Path) -> None:
    frames = [np.full((16, 16, 3), i * 20, dtype=np.uint8) for i in range(5)]
    out = tmp_path / "clip.gif"
    result = save_rollout_video(frames, out, fps=10)
    assert result == out
    assert out.exists() and out.stat().st_size > 0


def test_save_rollout_video_empty_returns_none(tmp_path: Path) -> None:
    assert save_rollout_video([], tmp_path / "x.gif") is None


def test_save_rollout_video_bad_path_returns_none_not_raises(tmp_path: Path) -> None:
    # Unknown suffix → encoder errors; helper must swallow it (best-effort).
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]
    assert save_rollout_video(frames, tmp_path / "clip.notaformat") is None


# ---------------------------------------------------------------------------
# Runner wiring: capture_video -> artifacts.videos
# ---------------------------------------------------------------------------

class _NullPublisher(EventPublisher):
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        return


class _ImageFakeEnv:
    """Robosuite-shaped env that returns a camera frame under ``agentview_image``."""

    def __init__(self, *, steps_per_episode: int = 3, success: bool = True):
        self.steps_per_episode = steps_per_episode
        self.success = success
        self._step = 0

    def _obs(self) -> dict[str, Any]:
        return {"agentview_image": np.zeros((16, 16, 3), dtype=np.uint8)}

    def reset(self) -> dict[str, Any]:
        self._step = 0
        return self._obs()

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._step += 1
        done = self._step >= self.steps_per_episode
        reward = 1.0 if done and self.success else 0.0
        info = {"success": self.success} if done else {}
        return self._obs(), reward, done, info


def _mission_with_video() -> MissionRun:
    spec = Mission(
        metadata=MissionMetadata(name="msn-video"),
        objective="o",
        acceptance_criteria="a",
        robot=RobotSpec(
            embodiment="franka_panda",
            agents=[
                AgentSpec(
                    id="pilot",
                    role=AgentRole.PILOT,
                    model=HFModelRef(base="openvla/openvla-7b"),
                ),
            ],
        ),
        tasks=[
            TrainingTask(
                name="train",
                training_type=TrainingType.DEMONSTRATION,
                agent_id="pilot",
            ),
            EvaluationTask(
                name="eval",
                evaluation_type=EvaluationType.ROBOSUITE,
                benchmark_name="Lift",
                num_episodes=2,
                config={
                    "capture_video": True,
                    "video_format": "gif",  # no ffmpeg in CI
                    "image_key": "agentview_image",
                },
            ),
        ],
    )
    mission = MissionRun.from_spec(spec)
    mission.tasks[0].status = TaskStatus.COMPLETED
    mission.tasks[0].result_summary = {
        "checkpoint_path": "/tmp/checkpoint",
        "agent_id": "pilot",
    }
    return mission


def _ctx(mission: MissionRun, eval_task: TaskRun, tmp_path: Path) -> TaskContext:
    return TaskContext(
        task=eval_task,
        mission=mission,
        publisher=_NullPublisher(),
        output_dir=tmp_path,
    )


async def test_capture_video_records_artifacts(tmp_path: Path) -> None:
    mission = _mission_with_video()
    eval_task = mission.tasks[1]
    runner = RobosuiteRunner(
        env_factory=lambda _b, _r: _ImageFakeEnv(steps_per_episode=3, success=True),
        policy_factory=lambda _: lambda obs: [0.0],
    )
    result = await runner.run(_ctx(mission, eval_task, tmp_path))

    videos = result["artifacts"]["videos"]
    assert len(videos) == 2  # one per episode
    for v in videos:
        assert v.endswith(".gif")
        assert Path(v).exists() and Path(v).stat().st_size > 0


async def test_no_capture_leaves_videos_empty(tmp_path: Path) -> None:
    mission = _mission_with_video()
    eval_task = mission.tasks[1]
    eval_task.spec.config = {"image_key": "agentview_image"}  # capture_video off
    runner = RobosuiteRunner(
        env_factory=lambda _b, _r: _ImageFakeEnv(),
        policy_factory=lambda _: lambda obs: [0.0],
    )
    result = await runner.run(_ctx(mission, eval_task, tmp_path))
    assert result["artifacts"]["videos"] == []


# ---------------------------------------------------------------------------
# Dedicated video camera, decoupled from the policy camera
# ---------------------------------------------------------------------------

def test_resolve_cameras_defaults_unchanged_without_video_camera() -> None:
    kwargs, capture_key = _resolve_cameras({"capture_video": True})
    assert kwargs == {
        "camera_names": "agentview",
        "camera_heights": 256,
        "camera_widths": 256,
    }
    assert capture_key is None


def test_resolve_cameras_inert_when_capture_video_off() -> None:
    # A video camera without capture would be pure render cost — stay on defaults.
    kwargs, capture_key = _resolve_cameras({"video_camera": "frontview"})
    assert kwargs["camera_names"] == "agentview"
    assert capture_key is None


def test_resolve_cameras_appends_video_camera() -> None:
    kwargs, capture_key = _resolve_cameras(
        {
            "capture_video": True,
            "video_camera": "frontview",
            "video_height": 480,
            "video_width": 640,
        }
    )
    assert kwargs == {
        "camera_names": ["agentview", "frontview"],
        "camera_heights": [256, 480],
        "camera_widths": [256, 640],
    }
    assert capture_key == "frontview_image"


def test_resolve_cameras_video_resolution_defaults_512() -> None:
    kwargs, _ = _resolve_cameras({"capture_video": True, "video_camera": "frontview"})
    assert kwargs["camera_heights"] == [256, 512]
    assert kwargs["camera_widths"] == [256, 512]


def test_resolve_cameras_handles_policy_camera_list() -> None:
    kwargs, _ = _resolve_cameras(
        {
            "capture_video": True,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "video_camera": "frontview",
        }
    )
    assert kwargs["camera_names"] == ["agentview", "robot0_eye_in_hand", "frontview"]
    assert kwargs["camera_heights"] == [256, 256, 512]
    assert kwargs["camera_widths"] == [256, 256, 512]


def test_resolve_cameras_rejects_policy_camera_collision() -> None:
    with pytest.raises(ValueError, match="already a policy camera"):
        _resolve_cameras({"capture_video": True, "video_camera": "agentview"})


class _DualCamFakeEnv(_ImageFakeEnv):
    """Fake env with a 16x16 policy camera and a 32x32 video camera."""

    def _obs(self) -> dict[str, Any]:
        return {
            "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
            "frontview_image": np.zeros((32, 32, 3), dtype=np.uint8),
        }


def _gif_frame_shape(path: str) -> tuple[int, int]:
    frame = iio.mimread(path)[0]
    return frame.shape[0], frame.shape[1]


async def test_video_camera_records_dedicated_camera(tmp_path: Path) -> None:
    mission = _mission_with_video()
    eval_task = mission.tasks[1]
    eval_task.spec.config["video_camera"] = "frontview"
    runner = RobosuiteRunner(
        env_factory=lambda _b, _r: _DualCamFakeEnv(steps_per_episode=3, success=True),
        policy_factory=lambda _: lambda obs: [0.0],
    )
    result = await runner.run(_ctx(mission, eval_task, tmp_path))

    videos = result["artifacts"]["videos"]
    assert len(videos) == 2
    for v in videos:
        assert _gif_frame_shape(v) == (32, 32)  # video cam, not the policy frame


def _specialist_mission_with_video() -> MissionRun:
    """Mission whose loadout adds a SPECIALIST, so the runner takes the
    planned-runtime (multi-agent) path."""
    spec = Mission(
        metadata=MissionMetadata(name="msn-video-multiagent"),
        objective="o",
        acceptance_criteria="a",
        robot=RobotSpec(
            embodiment="franka_panda",
            agents=[
                AgentSpec(
                    id="pilot",
                    role=AgentRole.PILOT,
                    model=HFModelRef(base="openvla/openvla-7b"),
                ),
                AgentSpec(
                    id="planner",
                    role=AgentRole.SPECIALIST,
                    model=HFModelRef(base="google/gemma-4-E2B-it"),
                ),
            ],
        ),
        tasks=[
            TrainingTask(
                name="train",
                training_type=TrainingType.DEMONSTRATION,
                agent_id="pilot",
            ),
            EvaluationTask(
                name="eval",
                evaluation_type=EvaluationType.ROBOSUITE,
                benchmark_name="Lift",
                num_episodes=1,
                config={
                    "capture_video": True,
                    "video_format": "gif",  # no ffmpeg in CI
                    "image_key": "agentview_image",
                    "video_camera": "frontview",
                },
            ),
        ],
    )
    mission = MissionRun.from_spec(spec)
    mission.tasks[0].status = TaskStatus.COMPLETED
    mission.tasks[0].result_summary = {
        "checkpoint_path": "/tmp/checkpoint",
        "agent_id": "pilot",
    }
    return mission


class _FakeRuntime:
    """Stands in for PlannedEvalRuntime; records the pilot's input frames."""

    def __init__(self) -> None:
        self.frames_seen: list[tuple[int, ...]] = []

    def begin_episode(self, instruction: str, first_image: Any = None) -> list[str]:
        return ["do the task"]

    def get_action(self, image: Any) -> Any:
        self.frames_seen.append(tuple(image.shape))
        return [0.0]

    def close(self) -> None:
        return


async def test_video_camera_multiagent_keeps_policy_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The planned-runtime path must extract TWO frames: the pilot keeps the
    policy camera while the video records the dedicated camera."""
    mission = _specialist_mission_with_video()
    eval_task = mission.tasks[1]
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(
        robosuite_runner, "_build_planned_runtime", lambda *a, **k: fake_runtime
    )
    runner = RobosuiteRunner(env_factory=lambda _b, _r: _DualCamFakeEnv())
    result = await runner.run(_ctx(mission, eval_task, tmp_path))

    assert set(fake_runtime.frames_seen) == {(16, 16, 3)}  # policy input untouched
    videos = result["artifacts"]["videos"]
    assert len(videos) == 1
    assert _gif_frame_shape(videos[0]) == (32, 32)
