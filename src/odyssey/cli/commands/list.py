"""``odyssey list`` — show recent missions from the local SQLite DB."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from odyssey.engine.lifecycle import MissionStatus
from odyssey.engine.records import MissionRun
from odyssey.persistence import SqlitePersistence
from odyssey.utils.paths import default_db_path

_STATUS_COLORS: dict[str, str] = {
    "COMPLETED": "green",
    "FAILED": "red",
    "CANCELLED": "yellow",
    "ACTIVE": "cyan",
    "QUEUED": "blue",
    "DRAFT": "white",
}


@click.command(name="list")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite missions database (defaults to ~/.odyssey/missions.db).",
)
@click.option(
    "--status",
    type=click.Choice([s.value for s in MissionStatus], case_sensitive=False),
    default=None,
    help="Filter by status.",
)
@click.option("--limit", type=int, default=20, help="Maximum rows to show.")
def list_(db: Path | None, status: str | None, limit: int) -> None:
    """List recent missions."""
    db_path = db or default_db_path()
    persistence = SqlitePersistence(str(db_path))

    missions = asyncio.run(_load(persistence, status, limit))
    if not missions:
        click.echo("(no missions)")
        return

    for m in missions:
        status_str = click.style(
            f"{m.status.value:<10}",
            fg=_STATUS_COLORS.get(m.status.value, "white"),
            bold=True,
        )
        grade = (
            f"grade={m.overall_grade:.3f}"
            if m.overall_grade is not None
            else ""
        )
        click.echo(
            f"{m.id[:12]}  {status_str}  {m.spec.metadata.name:<32}  "
            f"{m.created_at.isoformat(timespec='seconds')}  {grade}"
        )


async def _load(
    persistence: SqlitePersistence,
    status: str | None,
    limit: int,
) -> list[MissionRun]:
    await persistence.initialize()
    return await persistence.list_missions(status=status, limit=limit)
