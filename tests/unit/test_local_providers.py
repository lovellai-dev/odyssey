"""Tests for the local-mode providers (robots + datasets)."""

from __future__ import annotations

from pathlib import Path

import pytest

from odyssey.providers.local import (
    KNOWN_EMBODIMENTS,
    LocalDatasetProvider,
    LocalRobotProvider,
)
from odyssey.spec.mission import RobotSpec
from odyssey.spec.refs import DatasetRef, DatasetSource

# ---------------------------------------------------------------------------
# LocalRobotProvider
# ---------------------------------------------------------------------------

async def test_robot_resolves_known_embodiment() -> None:
    provider = LocalRobotProvider()
    resolved = await provider.resolve(RobotSpec(embodiment="franka_panda"))
    assert resolved.provider == "local"
    assert resolved.embodiment == "franka_panda"
    assert resolved.name == "franka_panda"


async def test_robot_rejects_unknown_embodiment() -> None:
    provider = LocalRobotProvider()
    with pytest.raises(ValueError, match="Unknown embodiment"):
        await provider.resolve(RobotSpec(embodiment="made_up_robot_99"))


async def test_robot_resolves_existing_urdf(tmp_path: Path) -> None:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot/>")
    provider = LocalRobotProvider()
    resolved = await provider.resolve(RobotSpec(urdf=str(urdf)))
    assert resolved.urdf_path == str(urdf)
    assert resolved.name == "robot"


async def test_robot_rejects_missing_urdf(tmp_path: Path) -> None:
    provider = LocalRobotProvider()
    with pytest.raises(FileNotFoundError):
        await provider.resolve(RobotSpec(urdf=str(tmp_path / "no.urdf")))


def test_known_embodiments_includes_design_doc_examples() -> None:
    # The design doc and the publication plan both name these — the
    # local provider must accept them or quickstart specs break.
    assert "franka_panda" in KNOWN_EMBODIMENTS
    assert "ur5e" in KNOWN_EMBODIMENTS
    assert "unitree_go2" in KNOWN_EMBODIMENTS


# ---------------------------------------------------------------------------
# LocalDatasetProvider
# ---------------------------------------------------------------------------

async def test_dataset_resolves_directory(tmp_path: Path) -> None:
    (tmp_path / "ep1.json").write_text("{}")
    (tmp_path / "ep2.json").write_text("{}")
    provider = LocalDatasetProvider()
    resolved = await provider.resolve(
        DatasetRef(source=DatasetSource.LOCAL, ref=str(tmp_path))
    )
    assert resolved.identifier == str(tmp_path)
    assert resolved.content_hash is not None
    assert resolved.content_hash.startswith("sha256:")


async def test_dataset_resolves_single_file(tmp_path: Path) -> None:
    f = tmp_path / "only.json"
    f.write_text("{}")
    provider = LocalDatasetProvider()
    resolved = await provider.resolve(
        DatasetRef(source=DatasetSource.LOCAL, ref=str(f))
    )
    assert resolved.identifier == str(f)


async def test_dataset_rejects_missing_path(tmp_path: Path) -> None:
    provider = LocalDatasetProvider()
    with pytest.raises(FileNotFoundError):
        await provider.resolve(
            DatasetRef(source=DatasetSource.LOCAL, ref=str(tmp_path / "nope"))
        )


async def test_dataset_streams_per_file(tmp_path: Path) -> None:
    for i in range(3):
        (tmp_path / f"ep{i}.json").write_text("{}")
    provider = LocalDatasetProvider()
    resolved = await provider.resolve(
        DatasetRef(source=DatasetSource.LOCAL, ref=str(tmp_path))
    )
    episodes = [ep async for ep in provider.stream_episodes(resolved)]
    assert len(episodes) == 3
    assert all("path" in e and "index" in e for e in episodes)


async def test_dataset_stream_honors_max_episodes(tmp_path: Path) -> None:
    for i in range(5):
        (tmp_path / f"ep{i}.json").write_text("{}")
    provider = LocalDatasetProvider()
    resolved = await provider.resolve(
        DatasetRef(source=DatasetSource.LOCAL, ref=str(tmp_path))
    )
    episodes = [
        ep async for ep in provider.stream_episodes(resolved, max_episodes=2)
    ]
    assert len(episodes) == 2


def test_dataset_supports_only_local_source() -> None:
    provider = LocalDatasetProvider()
    assert provider.supported_sources == {DatasetSource.LOCAL}
