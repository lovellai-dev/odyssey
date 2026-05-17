"""Model and dataset references used inside task specs.

Per design §2:

  * `ModelRef` is a discriminated union over three sources (huggingface,
    from_task, lovell). The runner consumes whichever variant the YAML
    declared.
  * `DatasetRef` carries a source enum + opaque ref string the provider
    interprets. Optional split / format / partial-episodes fields.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Model references
# ---------------------------------------------------------------------------

class HFModelRef(BaseModel):
    source: Literal["huggingface"] = "huggingface"
    base: str
    revision: str | None = None


class FromTaskModelRef(BaseModel):
    """A model produced by an earlier task in the same mission."""

    source: Literal["from_task"] = "from_task"
    from_task: str


class LovellModelRef(BaseModel):
    source: Literal["lovell"] = "lovell"
    model_id: str
    version: str


ModelRef = Annotated[
    HFModelRef | FromTaskModelRef | LovellModelRef,
    Field(discriminator="source"),
]


# ---------------------------------------------------------------------------
# Dataset references
# ---------------------------------------------------------------------------

class DatasetSource(str, Enum):
    HUGGINGFACE = "huggingface"
    OXE = "oxe"
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    LOVELL = "lovell"


class DatasetFormat(str, Enum):
    RLDS = "rlds"
    LEROBOT = "lerobot"
    PARQUET = "parquet"


class DatasetRef(BaseModel):
    source: DatasetSource
    ref: str
    split: str | None = None
    format: DatasetFormat | None = None
    partial: int | None = Field(default=None, ge=1)
