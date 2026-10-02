"""ExecutionSpec — per-mission engine knobs."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from odyssey.spec._base import SpecModel


class ExecutionSpec(SpecModel):
    parallelism: int = Field(default=1, ge=1)
    on_task_failure: Literal["stop", "continue"] = "stop"
