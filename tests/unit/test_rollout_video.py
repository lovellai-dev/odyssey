"""Tests for the rollout-video helper and its wiring into the Robosuite runner.

GPU-free: the helper is exercised with synthetic numpy frames encoded to a GIF
(pillow plugin, no ffmpeg), and the runner path uses an injected image-shaped
fake env. The mp4 path and a real rollout are validated on a GPU eval run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from odyssey.engine.lifecycle import TaskStatus
from odyssey.engine.records import MissionRun, TaskRun
from odyssey.runners.base import TaskContext
from odyssey.runners.evals.robosuite import RobosuiteRunner
from odyssey.runners.evals.video import save_rollout_video, to_uint8_frame
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
