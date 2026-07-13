"""Tests for the GR00T LeRobot v2.1 dataset writer (schema/dims/meta).

Video encoding is stubbed so these tests do not require ffmpeg; a dedicated,
skip-if-missing test exercises the real encoder.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from odyssey.datasets import lerobot_writer as lw
from odyssey.datasets.lerobot_writer import LeRobotDatasetWriter
from odyssey.embodiments.ur5e_drugsort import embodiment as emb


@pytest.fixture
def stub_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the ffmpeg-backed encoder with a file-touch so tests are hermetic."""

    def _fake(path: Path, frames: object, fps: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00")

    monkeypatch.setattr(lw, "encode_mp4", _fake)


def _writer(root: Path) -> LeRobotDatasetWriter:
    return LeRobotDatasetWriter(
        root,
        modality=emb.modality_json(),
        info_builder=emb.info_json,
        video_keys=emb.VIDEO_ORIGINAL_KEYS,
        fps=emb.DEFAULT_FPS,
        width=emb.DEFAULT_WIDTH,
        height=emb.DEFAULT_HEIGHT,
    )


def _episode(length: int) -> dict[str, object]:
    rng = np.random.default_rng(0)
    frame = np.zeros((emb.DEFAULT_HEIGHT, emb.DEFAULT_WIDTH, 3), dtype=np.uint8)
    return {
        "states": rng.standard_normal((length, emb.STATE_DIM)).astype(np.float32),
        "actions": rng.standard_normal((length, emb.ACTION_DIM)).astype(np.float32),
        "frames": {"exterior": [frame] * length},
        "task": emb.DEFAULT_INSTRUCTION,
    }


def test_writes_full_dataset_layout(tmp_path: Path, stub_encode: None) -> None:
    w = _writer(tmp_path)
    w.add_episode(**_episode(10))  # type: ignore[arg-type]
    w.add_episode(**_episode(7))  # type: ignore[arg-type]
    info = w.finalize()

    # Layout
    assert (tmp_path / "meta" / "info.json").is_file()
    assert (tmp_path / "meta" / "modality.json").is_file()
    assert (tmp_path / "meta" / "tasks.jsonl").is_file()
    assert (tmp_path / "meta" / "episodes.jsonl").is_file()
    assert (tmp_path / "meta" / "stats.json").is_file()
    assert (tmp_path / "data" / "chunk-000" / "episode_000000.parquet").is_file()
    assert (tmp_path / "data" / "chunk-000" / "episode_000001.parquet").is_file()
    assert (
        tmp_path / "videos" / "chunk-000" / "observation.images.exterior"
        / "episode_000000.mp4"
    ).is_file()

    assert info["total_episodes"] == 2
    assert info["total_frames"] == 17


def test_parquet_schema_and_dims(tmp_path: Path, stub_encode: None) -> None:
    w = _writer(tmp_path)
    w.add_episode(**_episode(5))  # type: ignore[arg-type]
    w.finalize()

    table = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    cols = table.column_names
    assert cols == [
        "action", "observation.state", "timestamp",
        "frame_index", "episode_index", "index", "task_index",
    ]
    # list<float> columns, scalar meta columns
    assert str(table.schema.field("action").type) == "list<element: float>"
    assert str(table.schema.field("observation.state").type) == "list<element: float>"
    assert str(table.schema.field("timestamp").type) == "float"
    assert str(table.schema.field("frame_index").type) == "int64"

    row0 = table.to_pylist()[0]
    assert len(row0["action"]) == emb.ACTION_DIM
    assert len(row0["observation.state"]) == emb.STATE_DIM
    assert row0["frame_index"] == 0
    assert row0["episode_index"] == 0


def test_stats_have_correct_dims(tmp_path: Path, stub_encode: None) -> None:
    w = _writer(tmp_path)
    w.add_episode(**_episode(12))  # type: ignore[arg-type]
    w.finalize()
    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    for key in ("action", "observation.state"):
        for stat in ("mean", "std", "min", "max", "q01", "q99"):
            assert len(stats[key][stat]) == emb.STATE_DIM
    assert len(stats["timestamp"]["mean"]) == 1


def test_index_is_global_and_monotonic(tmp_path: Path, stub_encode: None) -> None:
    w = _writer(tmp_path)
    w.add_episode(**_episode(3))  # type: ignore[arg-type]
    w.add_episode(**_episode(4))  # type: ignore[arg-type]
    w.finalize()
    t0 = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    t1 = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000001.parquet")
    idx0 = t0.column("index").to_pylist()
    idx1 = t1.column("index").to_pylist()
    assert idx0 == [0, 1, 2]
    assert idx1 == [3, 4, 5, 6]  # continues globally across episodes


def test_tasks_and_episodes_jsonl(tmp_path: Path, stub_encode: None) -> None:
    w = _writer(tmp_path)
    w.add_episode(**_episode(2))  # type: ignore[arg-type]
    w.finalize()
    tasks = [
        json.loads(line)
        for line in (tmp_path / "meta" / "tasks.jsonl").read_text().splitlines()
    ]
    assert tasks == [{"task_index": 0, "task": emb.DEFAULT_INSTRUCTION}]
    episodes = [
        json.loads(line)
        for line in (tmp_path / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    assert episodes == [
        {"episode_index": 0, "tasks": [emb.DEFAULT_INSTRUCTION], "length": 2}
    ]


def test_add_episode_validates_lengths(tmp_path: Path, stub_encode: None) -> None:
    w = _writer(tmp_path)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="length mismatch"):
        w.add_episode(
            states=np.zeros((5, emb.STATE_DIM), dtype=np.float32),
            actions=np.zeros((4, emb.ACTION_DIM), dtype=np.float32),
            frames={"exterior": [frame] * 5},
            task="x",
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_encode_mp4_roundtrip(tmp_path: Path) -> None:
    frames = [
        np.full((16, 16, 3), i * 10, dtype=np.uint8) for i in range(8)
    ]
    out = tmp_path / "clip.mp4"
    lw.encode_mp4(out, frames, fps=10)
    assert out.is_file()
    assert out.stat().st_size > 0
