"""In-memory `Persistence` implementation.

Backed by a single dict and an asyncio.Lock. Used by tests and by the
CPU-mock smoke path. Not durable — data evaporates when the process
exits. SQLite persistence is the durable cousin.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Any

from odyssey.engine.errors import MissionNotFoundError, TaskNotFoundError
from odyssey.engine.records import MissionRun, TaskRun
from odyssey.persistence.base import Persistence


class InMemoryPersistence(Persistence):
    def __init__(self) -> None:
        self._missions: dict[str, MissionRun] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        return

    async def save_mission(self, run: MissionRun) -> MissionRun:
        async with self._lock:
            self._missions[run.id] = run.model_copy(deep=True)
            return copy.deepcopy(run)

    async def get_mission(self, mission_id: str) -> MissionRun | None:
        async with self._lock:
            stored = self._missions.get(mission_id)
            return stored.model_copy(deep=True) if stored else None

    async def list_missions(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MissionRun]:
        async with self._lock:
            rows = list(self._missions.values())
        if status is not None:
            rows = [r for r in rows if r.status.value == status]
        rows.sort(key=lambda r: r.created_at)
        return [r.model_copy(deep=True) for r in rows[offset : offset + limit]]

    async def delete_mission(self, mission_id: str) -> bool:
        async with self._lock:
            return self._missions.pop(mission_id, None) is not None

    async def update_task(
        self,
        mission_id: str,
        task_id: str,
        **fields: Any,
    ) -> TaskRun:
        async with self._lock:
            run = self._missions.get(mission_id)
            if run is None:
                raise MissionNotFoundError(mission_id)
            for i, task in enumerate(run.tasks):
                if task.id == task_id:
                    updated = task.model_copy(update=fields)
                    run.tasks[i] = updated
                    return updated.model_copy(deep=True)
            raise TaskNotFoundError(task_id)
