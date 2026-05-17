"""ExecutionSpec — per-mission engine knobs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExecutionSpec(BaseModel):
    parallelism: int = Field(default=1, ge=1)
    on_task_failure: Literal["stop", "continue"] = "stop"
