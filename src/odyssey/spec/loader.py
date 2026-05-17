"""YAML loader for mission specs.

`load_mission(path)` reads the file, parses YAML, and validates against the
`Mission` Pydantic model. Errors are wrapped in `LoadError` with a path so
the CLI can surface a useful message.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from odyssey.spec.mission import Mission


class LoadError(Exception):
    """Raised when a mission spec fails to load.

    Wraps either YAML syntax errors or Pydantic validation errors. The
    original cause is preserved via `__cause__` for callers that want the
    structured error.
    """

    def __init__(self, path: Path, message: str):
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def load_mission(path: str | Path) -> Mission:
    """Load and validate a mission spec from a YAML file.

    Returns the parsed `Mission`. Raises `LoadError` for I/O errors, YAML
    syntax errors, or validation failures.
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise LoadError(p, f"cannot read file: {e}") from e

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise LoadError(p, f"YAML syntax error: {e}") from e

    if not isinstance(data, dict):
        raise LoadError(p, "top-level YAML document must be a mapping")

    try:
        return Mission.model_validate(data)
    except ValidationError as e:
        raise LoadError(p, f"spec validation failed:\n{e}") from e
