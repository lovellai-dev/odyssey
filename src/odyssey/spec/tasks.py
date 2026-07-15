"""Task specs.

Two task kinds: TRAINING and EVALUATION.

  * A TRAINING task targets one agent on the robot (``agent_id`` refs
    ``RobotSpec.agents[].id``). The framework looks up the agent's
    base model from ``AgentSpec.model`` and chains successive training
    tasks against the same agent through the runtime's per-agent
    checkpoint walk — there is no ``model:`` field on the task.

  * An EVALUATION task runs the robot. It carries no model or agent
    reference; the runner walks the robot's agents and composes the
    current checkpoints. Today that composition reduces to one agent =
    one policy = one eval; multi-agent eval arrives with the
    multi-agent runtime.

ML hyperparameters live in ``config: dict``. The framework spec stays
open; runners validate their own config schemas.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from odyssey.spec.refs import DatasetRef

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
    LIBERO = "libero"
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
    # The agent on the robot this training task updates. The model
    # checkpoint is resolved from the agent (or from a prior training
    # task's output for the same agent) — the task does not name a
    # model directly.
    agent_id: str
    dataset: DatasetRef | None = None
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
    # No model or agent reference. The eval runs the robot — the runner
    # walks the robot's agents and composes their current checkpoints.
    # Today there is exactly one agent, so the composition is single-
    # policy; multi-agent eval arrives with the multi-agent runtime.
    num_episodes: int = Field(default=100, ge=1)
    config: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int | None = Field(default=None, ge=1)
    retries: int = Field(default=0, ge=0)


TaskSpec = Annotated[
    TrainingTask | EvaluationTask,
    Field(discriminator="kind"),
]
