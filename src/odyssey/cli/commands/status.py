"""``odyssey status <mission_id>`` — show one mission's detail."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from odyssey.engine.records import MissionRun
from odyssey.persistence import SqlitePersistence
from odyssey.utils.paths import default_db_path

_STATUS_COLORS: dict[str, str] = {
    "COMPLETED": "green",
    "FAILED": "red",
    "CANCELLED": "yellow",
    "IN_PROGRESS": "cyan",
    "QUEUED": "blue",
    "PENDING": "white",
}


@click.command()
@click.argument("mission_id")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite missions database (defaults to ~/.odyssey/missions.db).",
)
def status(mission_id: str, db: Path | None) -> None:
    """Show one mission's status, tasks, and any errors.

    Accepts either the full mission id or any unique prefix.
    """
    db_path = db or default_db_path()
    persistence = SqlitePersistence(str(db_path))

    mission = asyncio.run(_resolve(persistence, mission_id))
    if mission is None:
        click.echo(
            click.style("NOT FOUND", fg="red", bold=True) + f"  {mission_id}"
        )
        sys.exit(1)

    status_color = _STATUS_COLORS.get(mission.status.value, "white")
    click.echo(
        click.style(mission.status.value, fg=status_color, bold=True)
        + f"  {mission.id}"
    )
    click.echo(f"  name         : {mission.spec.metadata.name}")
    click.echo(f"  created_at   : {mission.created_at.isoformat(timespec='seconds')}")
    if mission.started_at:
        click.echo(
            f"  started_at   : {mission.started_at.isoformat(timespec='seconds')}"
        )
    if mission.completed_at:
        click.echo(
            f"  completed_at : {mission.completed_at.isoformat(timespec='seconds')}"
        )
    if mission.overall_grade is not None:
        click.echo(f"  overall_grade: {mission.overall_grade:.3f}")
    click.echo("  tasks:")
    for t in mission.tasks:
        t_color = _STATUS_COLORS.get(t.status.value, "white")
        click.echo(
            "    "
            + click.style(f"{t.status.value:<11}", fg=t_color)
            + f"  {t.spec.kind:<10}  {t.spec.name}"
        )
        if t.error_message:
            click.echo(f"      error: {t.error_message}")


async def _resolve(
    persistence: SqlitePersistence, prefix: str
) -> MissionRun | None:
    await persistence.initialize()
    # Try direct lookup first (the common case — operators paste the
    # full id from `odyssey run`'s output).
    direct = await persistence.get_mission(prefix)
    if direct is not None:
        return direct
    # Fall back to prefix match against the full list.
    rows = await persistence.list_missions(limit=10_000)
    matches = [r for r in rows if r.id.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    return None
