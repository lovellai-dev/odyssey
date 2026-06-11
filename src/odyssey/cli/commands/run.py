"""``odyssey run <mission.yaml>`` — execute a mission end-to-end.

Wires up the standard local stack:

  * SqlitePersistence at ``~/.odyssey/missions.db``
  * StdoutEventPublisher for per-event JSON lines
  * RunnerRegistry with OpenVLA + Robosuite + CPU mock fallback
  * ProviderRegistry with Local + HuggingFace providers

The ``--use-mock-runner`` flag forces the CPU mock for every task,
making the command runnable on a laptop without a GPU and without HF
network access — useful for smoke-testing the plumbing.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from odyssey.engine import MissionEngine, MissionStatus
from odyssey.engine.records import MissionRun
from odyssey.persistence import SqlitePersistence
from odyssey.providers import ProviderRegistry
from odyssey.providers.huggingface import HFDatasetProvider, HFModelProvider
from odyssey.providers.local import LocalDatasetProvider, LocalRobotProvider
from odyssey.runners import (
    CPUMockRunner,
    IsaacLabRunner,
    OpenVLARunner,
    RunnerRegistry,
)
from odyssey.runners.robosuite import RobosuiteRunner
from odyssey.spec.loader import LoadError, load_mission
from odyssey.spec.mission import Mission
from odyssey.telemetry import StdoutEventPublisher
from odyssey.utils.paths import default_db_path, runs_dir


def _build_runners(use_mock: bool) -> RunnerRegistry:
    registry = RunnerRegistry()
    if use_mock:
        registry.register(CPUMockRunner())
        return registry
    # Real runners first; CPU mock as a last-resort fallback so unfamiliar
    # task types still produce *something* instead of "no runner registered."
    registry.register(OpenVLARunner())
    registry.register(RobosuiteRunner())
    registry.register(IsaacLabRunner())
    registry.register(CPUMockRunner())
    return registry


def _build_providers() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register_robot(LocalRobotProvider(), handles="local")
    registry.register_model(HFModelProvider())
    registry.register_dataset(LocalDatasetProvider())
    registry.register_dataset(HFDatasetProvider())
    return registry


@click.command()
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to the SQLite missions database (defaults to ~/.odyssey/missions.db).",
)
@click.option(
    "--working-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Per-mission artifact dir root (defaults to ~/.odyssey/runs).",
)
@click.option(
    "--use-mock-runner",
    is_flag=True,
    help="Force the CPU mock runner for every task — useful for smoke tests.",
)
def run(
    path: Path,
    db: Path | None,
    working_dir: Path | None,
    use_mock_runner: bool,
) -> None:
    """Load, validate, and execute a mission YAML."""
    try:
        spec = load_mission(path)
    except LoadError as e:
        click.echo(click.style("INVALID", fg="red", bold=True) + f"  {e.path}")
        click.echo(e.message)
        sys.exit(1)

    db_path = db or default_db_path()
    work_dir = working_dir or runs_dir()

    persistence = SqlitePersistence(str(db_path))
    runners = _build_runners(use_mock=use_mock_runner)
    providers = _build_providers()
    publisher = StdoutEventPublisher()
    engine = MissionEngine(
        persistence=persistence,
        runners=runners,
        event_publisher=publisher,
        working_dir=work_dir,
        providers=providers,
    )

    final = asyncio.run(_run_mission(engine, spec))

    click.echo("")
    if final.status == MissionStatus.COMPLETED:
        click.echo(click.style("COMPLETED", fg="green", bold=True) + f"  {final.id}")
        if final.overall_grade is not None:
            click.echo(f"  overall_grade : {final.overall_grade:.3f}")
        sys.exit(0)
    elif final.status == MissionStatus.FAILED:
        click.echo(click.style("FAILED", fg="red", bold=True) + f"  {final.id}")
        for task in final.tasks:
            if task.error_message:
                click.echo(f"  {task.spec.name}: {task.error_message}")
        sys.exit(1)
    else:
        click.echo(
            click.style(final.status.value, fg="yellow", bold=True)
            + f"  {final.id}"
        )
        sys.exit(1)


async def _run_mission(engine: MissionEngine, spec: Mission) -> MissionRun:
    await engine.initialize()
    run_record = await engine.create_mission(spec)
    return await engine.start_mission(run_record.id)
