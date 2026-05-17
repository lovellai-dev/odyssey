"""Persistence ABC — the storage seam the engine talks to.

Two implementations ship in v0.1.0-alpha: InMemoryPersistence (used by
tests and the CPU mock smoke path) and the upcoming SqlitePersistence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from odyssey.engine.records import MissionRun, TaskRun


class Persistence(ABC):
    """All engine reads and writes go through this interface."""

    @abstractmethod
    async def initialize(self) -> None: ...

    # ---- Missions ----

    @abstractmethod
    async def save_mission(self, run: MissionRun) -> MissionRun:
        """Insert or replace a mission run by id. Returns the saved record."""

    @abstractmethod
    async def get_mission(self, mission_id: str) -> MissionRun | None: ...

    @abstractmethod
    async def list_missions(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MissionRun]: ...

    @abstractmethod
    async def delete_mission(self, mission_id: str) -> bool: ...

    # ---- Tasks ----
    #
    # Tasks are owned by their mission row (cascade delete). These helpers
    # exist so the engine can update one task without rewriting the whole
    # mission record on every progress event.

    @abstractmethod
    async def update_task(
        self,
        mission_id: str,
        task_id: str,
        **fields: Any,
    ) -> TaskRun:
        """Patch one task on a mission. Returns the updated task."""
