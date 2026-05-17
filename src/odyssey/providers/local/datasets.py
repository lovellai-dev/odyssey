"""LocalDatasetProvider — handles datasets that live on the local filesystem.

Supports ``source: local`` refs whose ``ref`` is a path. Validates the
path exists; ``stream_episodes`` walks files matching ``*.json`` /
``*.parquet`` / ``*.rlds`` and yields one episode dict per file.

This is intentionally minimal — for Scope B users mostly use HuggingFace
datasets. Real local-file streaming with format decoders (RLDS, LeRobot)
is a v0.2.x line item.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from odyssey.providers.base import DatasetProvider, ResolvedDataset
from odyssey.spec.refs import DatasetRef, DatasetSource

_EPISODE_GLOBS = ("*.json", "*.parquet", "*.rlds", "*.tfrecord")


class LocalDatasetProvider(DatasetProvider):
    @property
    def name(self) -> str:
        return "local"

    @property
    def supported_sources(self) -> set[DatasetSource]:
        return {DatasetSource.LOCAL}

    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        path = Path(ref.ref)
        if not path.exists():
            raise FileNotFoundError(f"Local dataset path not found: {path}")

        content_hash = (
            _hash_file(path) if path.is_file() else _hash_directory_listing(path)
        )
        return ResolvedDataset(
            provider=self.name,
            source=ref.source.value,
            identifier=str(path),
            content_hash=f"sha256:{content_hash}",
            format=ref.format.value if ref.format else None,
            split=ref.split,
            metadata={"absolute_path": str(path.resolve())},
        )

    async def stream_episodes(
        self,
        resolved: ResolvedDataset,
        *,
        max_episodes: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        root = Path(resolved.identifier)
        if root.is_file():
            yield {"path": str(root), "index": 0}
            return

        seen = 0
        for pattern in _EPISODE_GLOBS:
            for entry in sorted(root.rglob(pattern)):
                if max_episodes is not None and seen >= max_episodes:
                    return
                yield {"path": str(entry), "index": seen}
                seen += 1


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _hash_directory_listing(path: Path) -> str:
    """A cheap content hash that captures the directory's file layout +
    sizes without reading every byte. Good enough as a lock-file marker
    when we'd otherwise have to scan terabytes."""
    h = hashlib.sha256()
    entries: list[tuple[str, int]] = []
    for p in sorted(path.rglob("*")):
        if p.is_file():
            entries.append((str(p.relative_to(path)), p.stat().st_size))
    for name, size in entries:
        h.update(f"{name}\0{size}\0".encode())
    return h.hexdigest()
