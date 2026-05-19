"""Tests for ProviderRegistry routing.

Uses stub implementations of the ABCs so we exercise the registry logic
without depending on any concrete provider — those land in Batch 2.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from odyssey.providers import (
    DatasetProvider,
    ModelProvider,
    ProviderNotRegisteredError,
    ProviderRegistry,
    ResolvedDataset,
    ResolvedModel,
    ResolvedRobot,
    RobotProvider,
)
from odyssey.spec.agents import AgentRole, AgentSpec
from odyssey.spec.mission import RobotSpec
from odyssey.spec.refs import (
    DatasetRef,
    DatasetSource,
    HFModelRef,
    LovellModelRef,
    ModelRef,
)


def _stub_agent() -> AgentSpec:
    return AgentSpec(
        id="pilot",
        role=AgentRole.PILOT,
        model=HFModelRef(base="openvla/openvla-7b"),
    )


def _robot(**fields: Any) -> RobotSpec:
    """RobotSpec for routing tests — supplies a placeholder loadout so
    the spec validates."""
    fields.setdefault("agents", [_stub_agent()])
    return RobotSpec(**fields)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubRobotProvider(RobotProvider):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def resolve(self, spec: RobotSpec) -> ResolvedRobot:
        return ResolvedRobot(provider=self._name, name=spec.embodiment or "x")


class _StubModelProvider(ModelProvider):
    def __init__(self, name: str, source: str):
        self._name = name
        self._source = source

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> str:
        return self._source

    async def resolve(self, ref: ModelRef) -> ResolvedModel:
        return ResolvedModel(
            provider=self._name, source=self._source, identifier="x", revision="r"
        )

    async def fetch(self, resolved: ResolvedModel, dest: Path) -> Path:
        return dest


class _StubDatasetProvider(DatasetProvider):
    def __init__(self, name: str, sources: set[DatasetSource]):
        self._name = name
        self._sources = sources

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_sources(self) -> set[DatasetSource]:
        return self._sources

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        return ResolvedDataset(
            provider=self._name, source=ref.source.value, identifier=ref.ref
        )

    async def stream_episodes(
        self,
        resolved: ResolvedDataset,
        *,
        max_episodes: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        # Empty generator — tests don't exercise the stream itself.
        if False:
            yield {}


# ---------------------------------------------------------------------------
# Robot routing
# ---------------------------------------------------------------------------

def test_robot_local_handler_for_embodiment_spec() -> None:
    reg = ProviderRegistry()
    local = _StubRobotProvider("local")
    reg.register_robot(local, handles="local")
    assert reg.for_robot_spec(_robot(embodiment="franka_panda")) is local


def test_robot_local_handler_for_urdf_spec() -> None:
    reg = ProviderRegistry()
    local = _StubRobotProvider("local")
    reg.register_robot(local, handles="local")
    assert reg.for_robot_spec(_robot(urdf="/tmp/x.urdf")) is local


def test_robot_lovell_handler_for_id_spec() -> None:
    reg = ProviderRegistry()
    lovell = _StubRobotProvider("lovell")
    reg.register_robot(lovell, handles="lovell")
    assert reg.for_robot_spec(_robot(id="r-1")) is lovell


def test_robot_missing_handler_raises() -> None:
    reg = ProviderRegistry()
    with pytest.raises(ProviderNotRegisteredError, match="kind='local'"):
        reg.for_robot_spec(_robot(embodiment="franka_panda"))
    with pytest.raises(ProviderNotRegisteredError, match="kind='lovell'"):
        reg.for_robot_spec(_robot(id="r-1"))


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("ref", "source"),
    [
        (HFModelRef(base="openvla/openvla-7b"), "huggingface"),
        (LovellModelRef(model_id="m", version="1"), "lovell"),
    ],
)
def test_model_routes_by_source(ref: ModelRef, source: str) -> None:
    reg = ProviderRegistry()
    provider = _StubModelProvider(name=source, source=source)
    reg.register_model(provider)
    assert reg.for_model_ref(ref) is provider


def test_model_missing_handler_raises() -> None:
    reg = ProviderRegistry()
    with pytest.raises(ProviderNotRegisteredError, match="source='huggingface'"):
        reg.for_model_ref(HFModelRef(base="openvla/openvla-7b"))


# ---------------------------------------------------------------------------
# Dataset routing
# ---------------------------------------------------------------------------

def test_dataset_routes_by_source() -> None:
    reg = ProviderRegistry()
    hf = _StubDatasetProvider("hf", {DatasetSource.HUGGINGFACE})
    local = _StubDatasetProvider(
        "local", {DatasetSource.LOCAL, DatasetSource.S3, DatasetSource.GCS}
    )
    reg.register_dataset(hf)
    reg.register_dataset(local)

    assert reg.for_dataset_ref(
        DatasetRef(source=DatasetSource.HUGGINGFACE, ref="lerobot/bridge_v2")
    ) is hf
    assert reg.for_dataset_ref(
        DatasetRef(source=DatasetSource.LOCAL, ref="/data/x")
    ) is local
    assert reg.for_dataset_ref(
        DatasetRef(source=DatasetSource.S3, ref="s3://bucket/path")
    ) is local


def test_dataset_missing_handler_raises() -> None:
    reg = ProviderRegistry()
    with pytest.raises(ProviderNotRegisteredError, match="source='oxe'"):
        reg.for_dataset_ref(DatasetRef(source=DatasetSource.OXE, ref="x"))


# ---------------------------------------------------------------------------
# Re-registration replaces
# ---------------------------------------------------------------------------

def test_re_registering_robot_kind_replaces_previous() -> None:
    reg = ProviderRegistry()
    first = _StubRobotProvider("first")
    second = _StubRobotProvider("second")
    reg.register_robot(first, handles="local")
    reg.register_robot(second, handles="local")
    assert reg.for_robot_spec(_robot(embodiment="franka_panda")) is second
