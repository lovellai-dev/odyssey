"""SQLite-backed Persistence implementation.

The local-mode default, intended for ``~/.odyssey/missions.db``. Schema
is intentionally narrow: one row per mission, with the full MissionRun
serialized as JSON. The redundant scalar columns (status, created_at,
overall_grade) exist solely so ``list_missions`` can filter and sort
without re-parsing every JSON blob.

WAL mode is enabled at initialize so readers don't block the writer.
Each operation opens its own connection — SQLite is cheap for this and
it sidesteps aiosqlite's long-lived-connection footguns (matches the
pattern from lai-inference's job_table.py).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite

from odyssey.engine.errors import MissionNotFoundError, TaskNotFoundError
from odyssey.engine.records import MissionRun, TaskRun
from odyssey.persistence.base import Persistence

_SCHEMA = """
CREATE TABLE IF NOT EXISTS missions (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    status          TEXT NOT NULL,
    run_json        TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    completed_at    TEXT,
    overall_grade   REAL
);

CREATE INDEX IF NOT EXISTS idx_missions_status ON missions(status);
CREATE INDEX IF NOT EXISTS idx_missions_created ON missions(created_at);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""

_SCHEMA_VERSION = 1


class SqlitePersistence(Persistence):
    def __init__(self, db_path: str):
        self._db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    @asynccontextmanager
    async def _conn(self):  # type: ignore[no-untyped-def]
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db

    async def initialize(self) -> None:
        async with self._conn() as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(_SCHEMA)
            await db.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (_SCHEMA_VERSION,),
            )
            await db.commit()

    # ------------------------------------------------------------------
    # Missions
    # ------------------------------------------------------------------

    async def save_mission(self, run: MissionRun) -> MissionRun:
        payload = run.model_dump_json()
        async with self._conn() as db:
            await db.execute(
                """
                INSERT INTO missions (
                    id, name, status, run_json, created_at,
                    started_at, completed_at, overall_grade
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name          = excluded.name,
                    status        = excluded.status,
                    run_json      = excluded.run_json,
                    started_at    = excluded.started_at,
                    completed_at  = excluded.completed_at,
                    overall_grade = excluded.overall_grade
                """,
                (
                    run.id,
                    run.spec.metadata.name,
                    run.status.value,
                    payload,
                    run.created_at.isoformat(),
                    run.started_at.isoformat() if run.started_at else None,
                    run.completed_at.isoformat() if run.completed_at else None,
                    run.overall_grade,
                ),
            )
            await db.commit()
        return run.model_copy(deep=True)

    async def get_mission(self, mission_id: str) -> MissionRun | None:
        async with self._conn() as db:
            cur = await db.execute(
                "SELECT run_json FROM missions WHERE id = ?",
                (mission_id,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return MissionRun.model_validate_json(row["run_json"])

    async def list_missions(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MissionRun]:
        sql = "SELECT run_json FROM missions"
        params: list[Any] = []
        if status is not None:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with self._conn() as db:
            cur = await db.execute(sql, tuple(params))
            rows = await cur.fetchall()
        return [MissionRun.model_validate_json(r["run_json"]) for r in rows]

    async def delete_mission(self, mission_id: str) -> bool:
        async with self._conn() as db:
            cur = await db.execute(
                "DELETE FROM missions WHERE id = ?", (mission_id,)
            )
            await db.commit()
            return (cur.rowcount or 0) > 0

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------
    # Stored inside the parent mission's JSON blob. update_task does a
    # read-modify-write under the SQLite writer lock; the engine
    # serializes task dispatch per mission so we don't race ourselves.

    async def update_task(
        self,
        mission_id: str,
        task_id: str,
        **fields: Any,
    ) -> TaskRun:
        async with self._conn() as db:
            cur = await db.execute(
                "SELECT run_json FROM missions WHERE id = ?",
                (mission_id,),
            )
            row = await cur.fetchone()
            if row is None:
                raise MissionNotFoundError(mission_id)

            run = MissionRun.model_validate_json(row["run_json"])
            updated_task: TaskRun | None = None
            for i, task in enumerate(run.tasks):
                if task.id == task_id:
                    updated_task = task.model_copy(update=fields)
                    run.tasks[i] = updated_task
                    break
            if updated_task is None:
                raise TaskNotFoundError(task_id)

            await db.execute(
                "UPDATE missions SET run_json = ? WHERE id = ?",
                (run.model_dump_json(), mission_id),
            )
            await db.commit()
        return updated_task.model_copy(deep=True)
