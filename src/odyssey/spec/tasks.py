"""Task specs.

Two task kinds: TRAINING and EVALUATION. Each carries its own type enum
(TrainingType / EvaluationType) — matching the CC schema where
`training_type` is on the training_tasks table and `evaluation_type` is
on the evaluation_tasks table, not on the missions table.

ML hyperparameters live in `config: dict`, not as enum values. The framework
spec stays open; runners validate their own config schemas.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from odyssey.spec.refs import DatasetRef, ModelRef

_TASK_NAME_PATTERN = r"^[a-z0-9][a-z0-9-]*[a-z0-9]$"


class TaskKind(str, Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"


# ---------------------------------------------------------------------------
# Task-level enums
# ---------------------------------------------------------------------------

class TrainingType(str, Enum):
    """Training task style. Mirrors lai-trainer's models.TrainingType.

    Lower-cased here for YAML friendliness; CC's persistence layer upper-cases
    on write to match its existing schema.
    """

    NARRATION = "narration"
    DEMONSTRATION = "demonstration"
    TELEOPERATION = "teleoperation"
    EXPERIMENTATION = "experimentation"


class EvaluationType(str, Enum):
    """Evaluation task framework. Mirrors lai-trainer's models.EvaluationType."""

    ROBOSUITE = "robosuite"
    ISAAC_LAB = "isaac_lab"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Task models
# ---------------------------------------------------------------------------

class TrainingTask(BaseModel):
    name: Annotated[str, Field(pattern=_TASK_NAME_PATTERN, max_length=64)]
    kind: Literal["training"] = "training"
    description: str | None = None
    training_type: TrainingType
    depends_on: list[str] = Field(default_factory=list)
    model: ModelRef
    dataset: DatasetRef | None = None
    target_agent_id: str
    target_agent_role: str = "PILOT"
    target_model_type: str = "VLA"
    config: dict[str, Any] = Field(default_factory=dict)
    execution_order: int = 0
    timeout_seconds: int | None = Field(default=None, ge=1)
    retries: int = Field(default=0, ge=0)


class EvaluationTask(BaseModel):
    name: Annotated[str, Field(pattern=_TASK_NAME_PATTERN, max_length=64)]
    kind: Literal["evaluation"] = "evaluation"
    description: str | None = None
    evaluation_type: EvaluationType
    benchmark_name: str
    depends_on: list[str] = Field(default_factory=list)
    model: ModelRef
    target_agent_id: str
    target_agent_role: str = "PILOT"
    target_model_type: str = "VLA"
    num_episodes: int = Field(default=100, ge=1)
    config: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int | None = Field(default=None, ge=1)
    retries: int = Field(default=0, ge=0)


TaskSpec = Annotated[
    TrainingTask | EvaluationTask,
    Field(discriminator="kind"),
]
