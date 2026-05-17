"""HFDatasetProvider — resolves and streams datasets from the Hub.

``resolve`` pins the dataset revision via ``HfApi.dataset_info``.
``stream_episodes`` calls into the ``datasets`` library and yields one
dict per row, falling back to a streaming-mode load so we never pull
multi-GB shards into RAM.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from odyssey.providers.base import DatasetProvider, ResolvedDataset
from odyssey.providers.huggingface.models import _load_hf_api
from odyssey.spec.refs import DatasetRef, DatasetSource


class HFDatasetProvider(DatasetProvider):
    def __init__(self, *, api: Any | None = None):
        self._api = api

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def supported_sources(self) -> set[DatasetSource]:
        return {DatasetSource.HUGGINGFACE}

    def _get_api(self) -> Any:
        if self._api is None:
            self._api = _load_hf_api()
        return self._api

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        api = self._get_api()
        info = api.dataset_info(repo_id=ref.ref)
        sha = getattr(info, "sha", None)
        return ResolvedDataset(
            provider=self.name,
            source=ref.source.value,
            identifier=ref.ref,
            revision=sha,
            content_hash=f"hf-sha:{sha}" if sha else None,
            format=ref.format.value if ref.format else None,
            split=ref.split,
            metadata={"hf_repo_id": ref.ref},
        )

    async def stream_episodes(
        self,
        resolved: ResolvedDataset,
        *,
        max_episodes: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "HF dataset streaming requires the 'huggingface' extra "
                "(which pulls in the `datasets` library)."
            ) from e

        ds = load_dataset(
            resolved.identifier,
            revision=resolved.revision,
            split=resolved.split or "train",
            streaming=True,
        )
        for seen, row in enumerate(ds):
            if max_episodes is not None and seen >= max_episodes:
                return
            yield dict(row) if not isinstance(row, dict) else row
