"""Agent runtime protocols for multi-agent evaluation.

Two runtime protocols define the interface between eval runners (Robosuite,
Isaac Lab) and the agent models:

  * ``PilotRuntime`` — wraps a VLA model. Maps a camera image plus a
    natural-language instruction to a robot action (7-DoF ndarray).
  * ``PlannerRuntime`` — wraps a task-planner LLM. Decomposes a high-level
    task instruction into an ordered list of sub-instructions the pilot
    executes sequentially.

These are ``typing.Protocol`` classes — any object that implements the
right methods satisfies the protocol without explicit inheritance.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class PilotRuntime(Protocol):
    """Maps one observation image + instruction to one robot action."""

    def act(
        self,
        image: Any,
        instruction: str,
    ) -> NDArray[np.floating[Any]]:
        """Produce a single-step action from an RGB image and instruction.

        Parameters
        ----------
        image:
            RGB image as a PIL Image or numpy HWC uint8 array.
        instruction:
            Natural-language instruction for this step/phase.

        Returns
        -------
        7-DoF action array (end-effector delta + gripper).
        """
        ...


@runtime_checkable
class PlannerRuntime(Protocol):
    """Decomposes a task instruction into sub-instructions."""

    def plan(self, task_instruction: str) -> list[str]:
        """Break a high-level instruction into ordered sub-steps.

        Parameters
        ----------
        task_instruction:
            The top-level task description (e.g. "pick up the red cube
            and place it on the shelf").

        Returns
        -------
        Ordered list of sub-instructions the pilot should execute
        sequentially.
        """
        ...
