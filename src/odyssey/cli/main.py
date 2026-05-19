"""The `odyssey` command — Click root group.

Subcommands are registered by importing them here. Each subcommand lives
in its own module under `odyssey.cli.commands.*` so adding one is a
one-line change here plus the module.
"""

from __future__ import annotations

import click

from odyssey import __version__
from odyssey.cli.commands.init import init
from odyssey.cli.commands.list import list_
from odyssey.cli.commands.run import run
from odyssey.cli.commands.status import status
from odyssey.cli.commands.validate import validate


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="odyssey")
def cli() -> None:
    """Lovell Odyssey — robot training mission framework."""


cli.add_command(init)
cli.add_command(validate)
cli.add_command(run)
cli.add_command(list_)
cli.add_command(status)


if __name__ == "__main__":
    cli()
