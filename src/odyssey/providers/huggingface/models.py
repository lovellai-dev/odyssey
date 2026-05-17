"""HFModelProvider — resolves and fetches models from the HuggingFace Hub.

``resolve`` pins ``revision`` to a concrete commit sha if the spec didn't
provide one. ``fetch`` downloads via ``snapshot_download`` into the
caller's dest directory.

The ``huggingface_hub`` import is lazy so users who don't want the
HuggingFace extra can still import the rest of odyssey. The constructor
takes an optional ``api`` so tests inject a fake without touching the
real Hub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from odyssey.providers.base import ModelProvider, ResolvedModel
from odyssey.spec.refs import HFModelRef, ModelRef


def _load_hf_api() -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise RuntimeError(
            "HuggingFace providers require the 'huggingface' extra. "
            "Install with: pip install 'lovell-odyssey[huggingface]'"
        ) from e
    return HfApi()


class HFModelProvider(ModelProvider):
    """Resolves and fetches HuggingFace Hub models."""

    def __init__(self, *, api: Any | None = None):
        self._api = api  # lazy-built on first use if None

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def source(self) -> str:
        return "huggingface"

    def _get_api(self) -> Any:
        if self._api is None:
            self._api = _load_hf_api()
        return self._api

    async def resolve(self, ref: ModelRef) -> ResolvedModel:
        if not isinstance(ref, HFModelRef):
            raise TypeError(
                f"HFModelProvider got non-HF ref of type {type(ref).__name__}"
            )

        api = self._get_api()
        # model_info accepts revision=None and returns the latest commit
        # sha, which we then pin into the resolved record.
        info = api.model_info(repo_id=ref.base, revision=ref.revision)
        sha = getattr(info, "sha", None) or ref.revision
        if not sha:
            raise RuntimeError(
                f"HF model_info returned no sha for {ref.base!r}"
            )
        return ResolvedModel(
            provider=self.name,
            source=self.source,
            identifier=ref.base,
            revision=sha,
            metadata={"hf_repo_id": ref.base},
        )

    async def fetch(self, resolved: ResolvedModel, dest: Path) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "HuggingFace fetch requires the 'huggingface' extra."
            ) from e

        dest.mkdir(parents=True, exist_ok=True)
        local_path = snapshot_download(
            repo_id=resolved.identifier,
            revision=resolved.revision,
            local_dir=str(dest),
        )
        return Path(local_path)
