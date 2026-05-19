"""Agent definitions on a robot.

In the Lovell architecture, a robot is an embodiment plus a loadout of
agents (see the robot-brain paper). Each agent has an id, a role
(PILOT or SPECIALIST), and an underlying model checkpoint. The brain
paper describes the full agent shape — persona, goals, success
criteria, materialized artifacts — that v0.0.x ``AgentSpec`` does not
yet model.

What v0.0.x ships: enough of the agent shape that training tasks can
reference an agent and the framework can look up that agent's starting
model. Today RobotSpec enforces exactly one agent; multi-agent
loadouts (a PILOT plus one or more SPECIALISTs) arrive when the
multi-agent runtime ships.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field

from odyssey.spec.refs import ModelRef

_AGENT_ID_PATTERN = r"^[a-z0-9][a-z0-9-]*[a-z0-9]$"


class AgentRole(str, Enum):
    PILOT = "PILOT"
    SPECIALIST = "SPECIALIST"


class AgentSpec(BaseModel):
    """One agent on a robot.

    ``model`` is the agent's base checkpoint — what a training task
    starts from on the first round. Subsequent training tasks on the
    same agent start from the previous task's output: the engine walks
    completed training tasks in spec order to find the latest
    checkpoint per ``agent_id``.
    """

    id: Annotated[str, Field(pattern=_AGENT_ID_PATTERN, max_length=64)]
    role: AgentRole = AgentRole.PILOT
    model: ModelRef
