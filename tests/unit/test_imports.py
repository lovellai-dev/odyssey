"""Import-order regression tests.

Each public module must be importable as the *first* odyssey import in
a fresh interpreter. In-process tests can't check this (modules cache
after the first test imports anything), so these spawn a clean child
interpreter per module.

Regression: ``odyssey.runners.base`` imported ``odyssey.engine.records``
at runtime, which initialized the engine package, whose
``mission_engine`` imports ``runners.base`` right back — a cycle that
only bit when a runner module was the entry import.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

ENTRY_MODULES = [
    "odyssey.runners.base",
    "odyssey.runners.gr00t",
    "odyssey.runners.isaac_lab",
    "odyssey.runners.openvla",
    "odyssey.runners.robosuite",
    "odyssey.engine",
    "odyssey.spec",
    "odyssey.cli.main",
]


@pytest.mark.parametrize("module", ENTRY_MODULES)
def test_module_imports_clean_as_first_import(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"`import {module}` failed in a fresh interpreter:\n{result.stderr}"
    )
