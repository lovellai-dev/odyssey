"""Writer for GR00T-flavoured LeRobot v2.1 datasets.

Emits exactly the on-disk layout ``gr00t.experiment.launch_finetune`` expects
(verified against ``Isaac-GR00T/demo_data/cube_to_bowl_5``)::

    <root>/
      meta/
        info.json          modality.json   tasks.jsonl
        episodes.jsonl     stats.json
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/<video_key>/episode_000000.mp4

Per-step observations/actions are stored as ``list<float32>`` parquet columns
(``action`` + ``observation.state``); camera streams are h264 mp4 files under
``videos/``. Normalisation statistics (mean/std/min/max/q01/q99) are computed
across every recorded frame at :meth:`finalize`.

Dependencies are deliberately light: numpy + pyarrow for the tabular data, and
the system ``ffmpeg`` binary (via :mod:`subprocess`) for video encoding — no
imageio / torchvision. The writer is embodiment-agnostic; the UR5e layout comes
from :mod:`odyssey.embodiments.ur5e_drugsort.embodiment`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:  # pragma: no cover - environment guard
        raise RuntimeError(
            "ffmpeg not found on PATH — required to encode LeRobot video streams. "
            "Install ffmpeg (e.g. `apt-get install ffmpeg`)."
        )
    return exe


def encode_mp4(path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
    """Encode a list of HxWx3 uint8 RGB frames to an h264/yuv420p mp4.

    Pipes raw ``rgb24`` bytes to a system ``ffmpeg``; h264 with an even frame
    size (guaranteed by the 256x256 default) decodes everywhere GR00T's LeRobot
    loader runs.
    """
    if not frames:
        raise ValueError(f"no frames to encode for {path}")
    first = np.asarray(frames[0])
    height, width = int(first.shape[0]), int(first.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg_bin(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", str(fps),
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for frame in frames:
            arr = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
            proc.stdin.write(arr.tobytes())
    finally:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0:  # pragma: no cover - environment guard
        raise RuntimeError(f"ffmpeg exited {rc} while encoding {path}")


def _stats_for(matrix: np.ndarray) -> dict[str, list[float]]:
    """Per-column mean/std/min/max/q01/q99 for an (N, D) float matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return {
        "mean": matrix.mean(axis=0).tolist(),
        "std": matrix.std(axis=0).tolist(),
        "min": matrix.min(axis=0).tolist(),
        "max": matrix.max(axis=0).tolist(),
        "q01": np.quantile(matrix, 0.01, axis=0).tolist(),
        "q99": np.quantile(matrix, 0.99, axis=0).tolist(),
    }


class LeRobotDatasetWriter:
    """Accumulate episodes, then write a complete GR00T LeRobot v2.1 dataset."""

    def __init__(
        self,
        root: str | Path,
        *,
        modality: Mapping[str, Any],
        info_builder: Callable[..., dict[str, Any]],
        video_keys: Mapping[str, str],
        fps: int,
        width: int,
        height: int,
    ) -> None:
        """
        Args:
            root: dataset directory (created; must not already contain episodes).
            modality: the ``meta/modality.json`` dict.
            info_builder: callable ``(*, total_episodes, total_frames,
                total_tasks, fps, width, height) -> dict`` producing
                ``meta/info.json`` (see ``embodiment.info_json``).
            video_keys: ``{friendly_key: original_key}`` — original_key is the
                ``observation.images.*`` column the mp4 is stored under.
            fps: control/record frequency the videos are encoded at.
            width, height: camera resolution (must match rendered frames).
        """
        self.root = Path(root)
        self.modality = dict(modality)
        self.info_builder = info_builder
        self.video_keys = dict(video_keys)
        self.fps = fps
        self.width = width
        self.height = height

        self._tasks: dict[str, int] = {}
        self._episodes: list[dict[str, Any]] = []
        self._state_rows: list[np.ndarray] = []
        self._action_rows: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._global_index = 0

        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        (self.root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    def _task_index(self, task: str) -> int:
        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)
        return self._tasks[task]

    def add_episode(
        self,
        *,
        states: np.ndarray,
        actions: np.ndarray,
        frames: Mapping[str, Sequence[np.ndarray]],
        task: str,
    ) -> int:
        """Record one episode. Returns its episode index.

        Args:
            states: ``(T, state_dim)`` float array (observation.state per step).
            actions: ``(T, action_dim)`` float array (action per step).
            frames: ``{friendly_key: [HxWx3 uint8, ...]}`` — one list per camera,
                each of length ``T``.
            task: the natural-language task/instruction for this episode.
        """
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        length = int(states.shape[0])
        if length == 0:
            raise ValueError("cannot add an empty episode")
        if actions.shape[0] != length:
            raise ValueError(
                f"states/actions length mismatch: {length} vs {actions.shape[0]}"
            )
        for key in self.video_keys:
            if key not in frames:
                raise ValueError(f"missing frames for video key {key!r}")
            if len(frames[key]) != length:
                raise ValueError(
                    f"frames[{key!r}] has {len(frames[key])} frames, expected {length}"
                )

        episode_index = len(self._episodes)
        task_index = self._task_index(task)

        timestamps = (np.arange(length, dtype=np.float64) / self.fps).astype(np.float32)
        frame_index = np.arange(length, dtype=np.int64)
        index = np.arange(self._global_index, self._global_index + length, dtype=np.int64)
        self._global_index += length

        table = pa.table(
            {
                "action": pa.array(list(actions), type=pa.list_(pa.float32())),
                "observation.state": pa.array(
                    list(states), type=pa.list_(pa.float32())
                ),
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "frame_index": pa.array(frame_index, type=pa.int64()),
                "episode_index": pa.array(
                    np.full(length, episode_index, dtype=np.int64), type=pa.int64()
                ),
                "index": pa.array(index, type=pa.int64()),
                "task_index": pa.array(
                    np.full(length, task_index, dtype=np.int64), type=pa.int64()
                ),
            }
        )
        parquet_path = (
            self.root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        pq.write_table(table, parquet_path)

        for key, original_key in self.video_keys.items():
            video_path = (
                self.root
                / "videos"
                / "chunk-000"
                / original_key
                / f"episode_{episode_index:06d}.mp4"
            )
            encode_mp4(video_path, list(frames[key]), self.fps)

        self._state_rows.append(states.astype(np.float64))
        self._action_rows.append(actions.astype(np.float64))
        self._timestamps.extend(timestamps.astype(np.float64).tolist())
        self._episodes.append(
            {"episode_index": episode_index, "tasks": [task], "length": length}
        )
        return episode_index

    def finalize(self) -> dict[str, Any]:
        """Write all ``meta/*`` files. Returns the info.json dict."""
        if not self._episodes:
            raise RuntimeError("no episodes recorded — nothing to finalize")

        total_episodes = len(self._episodes)
        total_frames = sum(ep["length"] for ep in self._episodes)
        total_tasks = len(self._tasks)

        info = self.info_builder(
            total_episodes=total_episodes,
            total_frames=total_frames,
            total_tasks=total_tasks,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )
        self._write_json(self.root / "meta" / "info.json", info)
        self._write_json(self.root / "meta" / "modality.json", self.modality)

        self._write_jsonl(
            self.root / "meta" / "tasks.jsonl",
            [
                {"task_index": idx, "task": task}
                for task, idx in sorted(self._tasks.items(), key=lambda kv: kv[1])
            ],
        )
        self._write_jsonl(self.root / "meta" / "episodes.jsonl", self._episodes)

        all_states = np.concatenate(self._state_rows, axis=0)
        all_actions = np.concatenate(self._action_rows, axis=0)
        all_ts = np.asarray(self._timestamps, dtype=np.float64).reshape(-1, 1)
        stats = {
            "action": _stats_for(all_actions),
            "observation.state": _stats_for(all_states),
            "timestamp": _stats_for(all_ts),
        }
        self._write_json(self.root / "meta" / "stats.json", stats)
        return info

    @staticmethod
    def _write_json(path: Path, obj: Any) -> None:
        path.write_text(json.dumps(obj, indent=4) + "\n")

    @staticmethod
    def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
