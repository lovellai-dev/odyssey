"""Tests for the local-mode providers (robots + datasets)."""

from __future__ import annotations

from pathlib import Path

import pytest

from odyssey.providers.local import (
    KNOWN_EMBODIMENTS,
    LocalDatasetProvider,
    LocalRobotProvider,
)
from odyssey.spec.agents import AgentRole, AgentSpec
from odyssey.spec.mission import RobotSpec
from odyssey.spec.refs import DatasetRef, DatasetSource, HFModelRef


def _agents() -> list[AgentSpec]:
    """Loadout for tests that only care about the embodiment-layer
    behavior of the provider."""
    return [
        AgentSpec(
            id="pilot",
            role=AgentRole.PILOT,
            model=HFModelRef(base="openvla/openvla-7b"),
        ),
    ]

# ---------------------------------------------------------------------------
# LocalRobotProvider
# ---------------------------------------------------------------------------

async def test_robot_resolves_known_embodiment() -> None:
    provider = LocalRobotProvider()
    resolved = await provider.resolve(
        RobotSpec(embodiment="franka_panda", agents=_agents())
    )
    assert resolved.provider == "local"
    assert resolved.embodiment == "franka_panda"
    assert resolved.name == "franka_panda"


async def test_robot_resolves_ur10e_embodiment() -> None:
    # ur10e is a GR00T NEW_EMBODIMENT arm (finetune target); it must resolve
    # through the local provider so a `robot.embodiment: ur10e` spec validates.
    provider = LocalRobotProvider()
    resolved = await provider.resolve(
        RobotSpec(embodiment="ur10e", agents=_agents())
    )
    assert resolved.provider == "local"
    assert resolved.embodiment == "ur10e"
    assert resolved.name == "ur10e"


async def test_ur10e_resolves_then_robosuite_refuses_it() -> None:
    # Pins the behaviour change this PR introduces: on develop, `ur10e` was
    # rejected at spec/catalog validation; now it resolves cleanly and only a
    # Robosuite eval task refuses it — loudly (ValueError), never a silent
    # default-Panda substitution. ur10e is the first real catalog name that can
    # reach `_resolve_robosuite_robot`, so no `model_construct` is needed here.
    from odyssey.runners.evals.robosuite import _resolve_robosuite_robot

    spec = RobotSpec(embodiment="ur10e", agents=_agents())
    resolved = await LocalRobotProvider().resolve(spec)
    assert resolved.embodiment == "ur10e"
    with pytest.raises(ValueError, match="Robosuite has no built-in robot"):
        _resolve_robosuite_robot(spec)


async def test_robot_rejects_unknown_embodiment() -> None:
    provider = LocalRobotProvider()
    # model_construct bypasses spec validation — needed because we're
    # exercising the provider's check, not the spec's catalog check.
    with pytest.raises(ValueError, match="Unknown embodiment"):
        await provider.resolve(
            RobotSpec.model_construct(
                embodiment="made_up_robot_99", agents=_agents()
            )
        )


async def test_robot_resolves_existing_urdf(tmp_path: Path) -> None:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot/>")
    provider = LocalRobotProvider()
    resolved = await provider.resolve(
        RobotSpec(urdf=str(urdf), agents=_agents())
    )
    assert resolved.urdf_path == str(urdf)
    assert resolved.name == "robot"


async def test_robot_rejects_missing_urdf(tmp_path: Path) -> None:
    provider = LocalRobotProvider()
    with pytest.raises(FileNotFoundError):
        await provider.resolve(
            RobotSpec(urdf=str(tmp_path / "no.urdf"), agents=_agents())
        )


def test_known_embodiments_covers_robosuite_robots() -> None:
    # The allowlist is the set of embodiments at least one shipped runner can
    # drive end-to-end. Robosuite is the eval runner for most arms, so every
    # Robosuite-translatable name must be in the catalog...
    from odyssey.runners.evals.robosuite import ROBOSUITE_ROBOT_NAMES

    assert set(ROBOSUITE_ROBOT_NAMES.keys()) <= KNOWN_EMBODIMENTS
    # ...and the only names beyond Robosuite are GR00T NEW_EMBODIMENT arms
    # (driven by the `gr00t` runner, never Robosuite) — today just ur10e.
    assert KNOWN_EMBODIMENTS - set(ROBOSUITE_ROBOT_NAMES.keys()) == {"ur10e"}
    # franka_panda is the alias most OpenVLA / LeRobot specs use.
    assert "franka_panda" in KNOWN_EMBODIMENTS
    # Quadrupeds and mobile bases were intentionally trimmed — no
    # runner honors them today.
    assert "unitree_go2" not in KNOWN_EMBODIMENTS
    assert "stretch3" not in KNOWN_EMBODIMENTS


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
