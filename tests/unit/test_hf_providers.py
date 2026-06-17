"""Tests for HFModelProvider + HFDatasetProvider.

The HF API is injected so tests don't need network access or the
``huggingface`` extra installed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from odyssey.providers.huggingface import HFDatasetProvider, HFModelProvider
from odyssey.spec.refs import (
    DatasetRef,
    DatasetSource,
    HFModelRef,
    LovellModelRef,
)


class _FakeHfApi:
    """Stand-in for huggingface_hub.HfApi.

    Returns a SimpleNamespace with a ``sha`` field, mirroring what HfApi's
    real ``model_info`` / ``dataset_info`` give back.
    """

    def __init__(self, *, model_sha: str = "abc123", dataset_sha: str = "def456"):
        self.model_sha = model_sha
        self.dataset_sha = dataset_sha
        self.model_info_calls: list[tuple[str, str | None]] = []
        self.dataset_info_calls: list[tuple[str, str | None]] = []

    def model_info(
        self, repo_id: str, revision: str | None = None
    ) -> SimpleNamespace:
        self.model_info_calls.append((repo_id, revision))
        return SimpleNamespace(sha=self.model_sha)

    def dataset_info(self, repo_id: str, **kwargs: Any) -> SimpleNamespace:
        self.dataset_info_calls.append((repo_id, kwargs.get("revision")))
        return SimpleNamespace(sha=self.dataset_sha)


# ---------------------------------------------------------------------------
# HFModelProvider
# ---------------------------------------------------------------------------

async def test_model_resolve_pins_sha_when_none_given() -> None:
    api = _FakeHfApi(model_sha="full-sha-here")
    provider = HFModelProvider(api=api)
    resolved = await provider.resolve(HFModelRef(base="openvla/openvla-7b"))
    assert resolved.identifier == "openvla/openvla-7b"
    assert resolved.revision == "full-sha-here"
    assert resolved.provider == "huggingface"
    assert api.model_info_calls == [("openvla/openvla-7b", None)]


async def test_model_resolve_pins_requested_revision() -> None:
    api = _FakeHfApi(model_sha="latest")
    provider = HFModelProvider(api=api)
    resolved = await provider.resolve(
        HFModelRef(base="openvla/openvla-7b", revision="v1.2.3")
    )
    # When caller pinned a revision, the api still returns the resolved
    # sha — we trust whichever HfApi gives back as canonical.
    assert resolved.revision == "latest"
    assert api.model_info_calls == [("openvla/openvla-7b", "v1.2.3")]


async def test_model_resolve_rejects_non_hf_ref() -> None:
    provider = HFModelProvider(api=_FakeHfApi())
    with pytest.raises(TypeError, match="non-HF ref"):
        await provider.resolve(LovellModelRef(model_id="m", version="1"))


async def test_model_resolve_raises_when_api_returns_no_sha() -> None:
    class _NoShaApi:
        def model_info(self, repo_id: str, revision: str | None = None) -> Any:
            return SimpleNamespace()  # no .sha attribute

    provider = HFModelProvider(api=_NoShaApi())
    with pytest.raises(RuntimeError, match="no sha"):
        await provider.resolve(HFModelRef(base="x/y"))


async def test_model_resolve_falls_back_when_hub_unreachable() -> None:
    # Offline / air-gapped (or a gated repo with no token): model_info raises,
    # and resolve falls back to a cache-resolvable revision instead of failing
    # — HEAD maps to ``main`` (no cached ``HEAD`` ref); a concrete pin is kept.
    class _OfflineApi:
        def model_info(self, repo_id: str, revision: str | None = None) -> Any:
            raise OSError("offline mode is enabled")

    provider = HFModelProvider(api=_OfflineApi())
    resolved = await provider.resolve(
        HFModelRef(base="nvidia/GR00T-N1.7-3B", revision="HEAD")
    )
    assert resolved.revision == "main"
    pinned = await provider.resolve(HFModelRef(base="x/y", revision="v9"))
    assert pinned.revision == "v9"


def test_model_provider_lazy_imports_hf() -> None:
    # Constructor without api should not raise; the import only happens
    # when a method is called and api is None.
    provider = HFModelProvider()
    assert provider.source == "huggingface"


# ---------------------------------------------------------------------------
# HFDatasetProvider
# ---------------------------------------------------------------------------

async def test_dataset_resolve_pins_sha() -> None:
    api = _FakeHfApi(dataset_sha="ds-sha-xyz")
    provider = HFDatasetProvider(api=api)
    resolved = await provider.resolve(
        DatasetRef(source=DatasetSource.HUGGINGFACE, ref="lerobot/bridge_v2")
    )
    assert resolved.identifier == "lerobot/bridge_v2"
    assert resolved.revision == "ds-sha-xyz"
    assert resolved.content_hash == "hf-sha:ds-sha-xyz"


async def test_dataset_resolve_carries_split_and_format() -> None:
    api = _FakeHfApi()
    provider = HFDatasetProvider(api=api)
    from odyssey.spec.refs import DatasetFormat
    resolved = await provider.resolve(
        DatasetRef(
            source=DatasetSource.HUGGINGFACE,
            ref="lerobot/bridge_v2",
            split="train[:1000]",
            format=DatasetFormat.LEROBOT,
        )
    )
    assert resolved.split == "train[:1000]"
    assert resolved.format == "lerobot"


def test_dataset_provider_supports_only_huggingface() -> None:
    provider = HFDatasetProvider(api=_FakeHfApi())
    assert provider.supported_sources == {DatasetSource.HUGGINGFACE}
