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
        try:
            info = api.model_info(repo_id=ref.base, revision=ref.revision)
            sha = getattr(info, "sha", None) or ref.revision
        except Exception:
            # Hub unreachable (HF_HUB_OFFLINE / air-gapped) or a gated repo with
            # no token: fall back to a cached revision and let the downstream
            # loader resolve weights from the local HF cache. ``HEAD`` is a
            # non-cached alias for the default branch, so map it to ``main`` so
            # the cache lookup (refs/main) succeeds. Keeps training runnable
            # offline once the base model is cached.
            sha = ref.revision if (ref.revision and ref.revision != "HEAD") else "main"
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
        import os

        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "HuggingFace fetch requires the 'huggingface' extra."
            ) from e

        # Offline / air-gapped: resolve straight from the local HF cache. We
        # avoid ``local_dir`` here because it triggers a network ``repo_info``
        # call (which raises under HF_HUB_OFFLINE) plus a multi-GB copy; instead
        # we hand the cached snapshot path to the downstream loader.
        if os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes"):
            return Path(
                snapshot_download(
                    repo_id=resolved.identifier,
                    revision=resolved.revision,
                    local_files_only=True,
                )
            )

        dest.mkdir(parents=True, exist_ok=True)
        local_path = snapshot_download(
            repo_id=resolved.identifier,
            revision=resolved.revision,
            local_dir=str(dest),
        )
        return Path(local_path)
