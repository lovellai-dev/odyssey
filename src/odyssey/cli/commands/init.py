"""``odyssey init [DIRECTORY]`` — scaffold a new mission directory.

Writes a ``mission.yaml`` from a built-in template, substituting the
mission name. Flag-driven by default; prompts for required values that
weren't supplied unless ``--yes`` is passed.

Templates live under ``odyssey.cli.templates.<name>.mission.yaml`` with
``{{ name }}`` as the only substitution placeholder.
"""

from __future__ import annotations

import re
import sys
from importlib import resources
from pathlib import Path

import click

from odyssey.spec.loader import LoadError, load_mission

TEMPLATES_PACKAGE = "odyssey.cli.templates"

# Hardcoded rather than scanned via ``resources.iterdir`` so the CLI's
# --template choices are stable regardless of how the package is laid
# out on disk vs. inside a wheel.
AVAILABLE_TEMPLATES: tuple[str, ...] = ("openvla", "cpu_mock")
DEFAULT_TEMPLATE = "openvla"

# Must match ``_NAME_PATTERN`` in odyssey.spec.mission.
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")
_NAME_MAX_LEN = 64


def _slugify(text: str) -> str:
    """Reduce *text* to the mission-name character class.

    Lower-case, replace any run of non-``[a-z0-9-]`` with a single
    hyphen, strip leading/trailing hyphens. Caller still has to check
    the result against ``_NAME_PATTERN`` — slugify can produce empty or
    single-char strings that don't validate.
    """
    slug = re.sub(r"[^a-z0-9-]+", "-", text.lower())
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def _is_valid_name(name: str) -> bool:
    return bool(name) and len(name) <= _NAME_MAX_LEN and bool(_NAME_PATTERN.match(name))


def _read_template(template: str) -> str:
    return (
        resources.files(TEMPLATES_PACKAGE) / template / "mission.yaml"
    ).read_text(encoding="utf-8")


def _render(template: str, *, name: str) -> str:
    return _read_template(template).replace("{{ name }}", name)


def _resolve_template(template: str | None, no_input: bool) -> str:
    if template is not None:
        return template
    if no_input:
        return DEFAULT_TEMPLATE
    choice: str = click.prompt(
        "Template",
        type=click.Choice(AVAILABLE_TEMPLATES),
        default=DEFAULT_TEMPLATE,
        show_choices=True,
    )
    return choice


def _resolve_name(
    name: str | None,
    directory: Path,
    no_input: bool,
) -> str:
    candidate = name if name is not None else _slugify(directory.resolve().name)
    if _is_valid_name(candidate):
        return candidate

    # Slug came out empty or invalid (e.g., ``odyssey init .`` from a
    # directory whose name is "_scratch"). Either prompt or fail.
    if no_input:
        raise click.ClickException(
            f"could not derive a valid mission name from {directory!s} "
            f"(got {candidate!r}); pass --name explicitly."
        )

    while True:
        prompted: str = click.prompt(
            "Mission name", default=candidate or "my-mission"
        )
        if _is_valid_name(prompted):
            return prompted
        click.echo(
            f"  {prompted!r} is not a valid mission name. "
            "Use lowercase letters, digits, and hyphens (2-64 chars, "
            "starting and ending with a letter or digit).",
            err=True,
        )


@click.command()
@click.argument(
    "directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
)
@click.option(
    "--template",
    "-t",
    type=click.Choice(AVAILABLE_TEMPLATES),
    default=None,
    help=f"Which template to scaffold from (default: {DEFAULT_TEMPLATE}).",
)
@click.option(
    "--name",
    default=None,
    help="Mission name (slug). Defaults to the directory's basename.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite an existing mission.yaml.",
)
@click.option(
    "--yes",
    "-y",
    "no_input",
    is_flag=True,
    help="Skip prompts; use defaults and fail on missing required values.",
)
def init(
    directory: Path,
    template: str | None,
    name: str | None,
    force: bool,
    no_input: bool,
) -> None:
    """Scaffold a new mission directory with a starter mission.yaml."""
    resolved_template = _resolve_template(template, no_input)
    resolved_name = _resolve_name(name, directory, no_input)

    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "mission.yaml"
    if target.exists() and not force:
        raise click.ClickException(
            f"{target} already exists; pass --force to overwrite."
        )

    rendered = _render(resolved_template, name=resolved_name)
    target.write_text(rendered, encoding="utf-8")

    # Sanity-check the generated file. If the template + substitution
    # produced something that doesn't validate, that's a bug in the
    # template — surface it instead of leaving a broken file behind.
    try:
        load_mission(target)
    except LoadError as e:
        target.unlink(missing_ok=True)
        click.echo(
            click.style("INTERNAL ERROR", fg="red", bold=True)
            + f"  rendered template {resolved_template!r} failed to validate:",
            err=True,
        )
        click.echo(e.message, err=True)
        sys.exit(2)

    click.echo(
        click.style("Created", fg="green", bold=True) + f"  {target}"
    )
    click.echo(f"  template     : {resolved_template}")
    click.echo(f"  mission name : {resolved_name}")
    click.echo("")
    click.echo("Next:")
    click.echo(f"  odyssey validate {target}")
    if resolved_template == "cpu_mock":
        click.echo(f"  odyssey run {target} --use-mock-runner")
    else:
        click.echo(f"  odyssey run {target} --use-mock-runner   # smoke (no GPU)")
        click.echo(f"  odyssey run {target}                     # real run (24 GB GPU)")
