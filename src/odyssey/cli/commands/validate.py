"""`odyssey validate <mission.yaml>` — load + validate a mission spec.

Exit codes:
  0 — spec parsed and validated cleanly
  1 — spec failed to load or validate
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from odyssey.spec.loader import LoadError, load_mission


@click.command()
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def validate(path: Path) -> None:
    """Validate a mission YAML against the Odyssey spec."""
    try:
        mission = load_mission(path)
    except LoadError as e:
        click.echo(click.style("INVALID", fg="red", bold=True) + f"  {e.path}")
        click.echo(e.message)
        sys.exit(1)

    training = sum(1 for t in mission.tasks if t.kind == "training")
    evaluation = sum(1 for t in mission.tasks if t.kind == "evaluation")
    robot_kind = (
        "embodiment="
        + (mission.robot.embodiment or "")
        if mission.robot.embodiment
        else "urdf=" + (mission.robot.urdf or "")
        if mission.robot.urdf
        else "id=" + (mission.robot.id or "")
    )

    click.echo(click.style("OK", fg="green", bold=True) + f"  {path}")
    click.echo(f"  spec version : {mission.odysseyVersion.value}")
    click.echo(f"  mission name : {mission.metadata.name}")
    click.echo(f"  robot        : {robot_kind}")
    click.echo(f"  tasks        : {training} training, {evaluation} evaluation")
