"""``~/.odyssey/`` path management.

All persistent local state — the missions DB, downloaded model
snapshots, run artifacts — lives under here. ``ODYSSEY_HOME`` env var
overrides for tests and unusual setups.
"""

from __future__ import annotations

import os
from pathlib import Path


def odyssey_home() -> Path:
    """Resolve the on-disk root for local Odyssey state.

    Creates it on first access so callers don't have to.
    """
    raw = os.getenv("ODYSSEY_HOME")
    root = Path(raw).expanduser() if raw else Path.home() / ".odyssey"
    root.mkdir(parents=True, exist_ok=True)
    return root


def default_db_path() -> Path:
    return odyssey_home() / "missions.db"


def runs_dir() -> Path:
    d = odyssey_home() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d
