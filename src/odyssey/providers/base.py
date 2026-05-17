"""Provider ABCs — the swap-points between local and Lovell-hosted modes.

Three provider kinds:

  * RobotProvider — resolves a RobotSpec into a concrete robot context.
    Local mode validates URDF / embodiment names; Lovell mode talks to
    Autonomy.
  * ModelProvider — resolves a ModelRef (HF / Lovell / from_task) and
    fetches weights into a local cache.
  * DatasetProvider — resolves a DatasetRef and streams episodes
    lazily (memory-bounded).

The engine never imports concrete providers; it asks the `ProviderRegistry`
for one matching the spec. That's what makes local-mode vs. hosted-mode
a configuration choice instead of a code change.

Per design §3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from odyssey.spec.mission import RobotSpec
from odyssey.spec.refs import DatasetRef, DatasetSource, ModelRef

# ---------------------------------------------------------------------------
# Resolved-* records — what providers hand back to the engine / runners.
# ---------------------------------------------------------------------------

@dataclass
class ResolvedRobot:
    """The robot as understood by whichever provider resolved it."""

    provider: str
    name: str
    embodiment: str | None = None
    urdf_path: str | None = None
    external_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedModel:
    """A model pinned to a specific revision, optionally fetched locally."""

    provider: str
    source: str
    identifier: str
    revision: str
    local_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedDataset:
    """A dataset pinned + ready to stream."""

    provider: str
    source: str
    identifier: str
    revision: str | None = None
    content_hash: str | None = None
    format: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ABCs
# ---------------------------------------------------------------------------

class RobotProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def resolve(self, spec: RobotSpec) -> ResolvedRobot:
        """Validate the spec resolves to a real robot."""


class ModelProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def source(self) -> str:
        """The ModelRef.source discriminator this provider handles
        (``huggingface``, ``lovell``, ``from_task``)."""

    @abstractmethod
    async def resolve(self, ref: ModelRef) -> ResolvedModel:
        """Pin to a specific revision; verify access."""

    @abstractmethod
    async def fetch(self, resolved: ResolvedModel, dest: Path) -> Path:
        """Download to a local cache. Returns the local root path."""


class DatasetProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def supported_sources(self) -> set[DatasetSource]: ...

    @abstractmethod
    async def resolve(self, ref: DatasetRef) -> ResolvedDataset:
        """Pin revision, verify access, return content_hash for the lock file."""

    @abstractmethod
    def stream_episodes(
        self,
        resolved: ResolvedDataset,
        *,
        max_episodes: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream episodes lazily. Memory-bounded.

        Implementations are async generators (``async def`` with ``yield``).
        Episode shape is provider-and-format-specific; runners and providers
        must agree on the dict keys for a given dataset format.
        """
