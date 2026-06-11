"""Model and dataset references.

Model references live on ``AgentSpec.model`` (the agent's base
checkpoint). Two sources today: HuggingFace Hub and Lovell-hosted
model management. The framework no longer carries a ``from_task``
ref — chaining is implicit through the per-agent checkpoint walk
performed by ``MissionRun.latest_checkpoint_for``.

``DatasetRef`` carries a source enum + opaque ref string the provider
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
    quantization: str | None = None


class LovellModelRef(BaseModel):
    source: Literal["lovell"] = "lovell"
    model_id: str
    version: str


ModelRef = Annotated[
    HFModelRef | LovellModelRef,
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
