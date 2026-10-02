"""Run an openpi script against a TrainConfig WITH the mission's overrides applied.

Why: ``scripts/train.py`` parses its config with
``openpi.training.config.cli()`` (``tyro.extras.overridable_config_cli``), so
the mission's tyro overrides (``--data.repo-id …``, ``--model.action-horizon …``)
reach training. ``scripts/compute_norm_stats.py`` is a plain
``tyro.cli(main)`` that only takes ``--config-name`` and loads the
*registered* config. With overrides, the two steps disagree: statistics are
computed for the config's default dataset while training reads another, and
training then fails on missing stats or normalizes with the wrong ones.

This bootstrap builds the config with the SAME parser ``train.py`` uses and
the SAME overrides, registers it under a private key, and runs the target
script against that key through ``runpy`` (as if invoked directly). The
config keeps its original ``name``, so outputs land where ``train.py`` looks
for them (``assets/<config_name>/<repo_id>/``). No openpi source is patched.

Usage (under odyssey's own interpreter, ``sys.executable``, which must have
openpi installed):

    python openpi_bootstrap.py <target_script.py> <config_name> -- <tyro overrides…>

Stdlib-only at import time: it runs as a standalone script (by path, not as
part of the odyssey package), so openpi is its only non-stdlib dependency.
"""

from __future__ import annotations

import runpy
import sys

# Private registry key; never a name a user would register.
_KEY_SUFFIX = "__odyssey_overrides"


def parse_argv(argv: list[str]) -> tuple[str, str, list[str]]:
    """Split ``[target, config_name, "--", *overrides]``."""
    if "--" not in argv:
        raise SystemExit(
            "openpi_bootstrap: usage: openpi_bootstrap.py <target.py> <config_name> "
            "-- <overrides…>"
        )
    sep = argv.index("--")
    if sep != 2:
        raise SystemExit("openpi_bootstrap: expected <target.py> <config_name> before '--'")
    return argv[0], argv[1], argv[sep + 1 :]


def registry_key(config_name: str) -> str:
    return f"{config_name}{_KEY_SUFFIX}"


def main(argv: list[str]) -> None:
    target, config_name, overrides = parse_argv(argv)

    import openpi.training.config as openpi_config  # openpi's venv only

    # Same parser and overrides as train.py, so both steps see one config.
    sys.argv = [target, config_name, *overrides]
    config = openpi_config.cli()
    key = registry_key(config_name)
    openpi_config._CONFIGS_DICT[key] = config  # keeps config.name == config_name
    print(
        f"[odyssey-bootstrap] {target}: config {config_name!r} with overrides "
        f"{overrides} registered as {key!r}",
        flush=True,
    )

    sys.argv = [target, "--config-name", key]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
