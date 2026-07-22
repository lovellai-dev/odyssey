"""Agent runtime protocols for multi-agent evaluation.

Three runtime protocols define the interfaces between components:

  * ``TextGenerator`` — wraps any text generation model. Maps chat
    messages to generated text. Lives at the model layer.
  * ``PilotRuntime`` — wraps a VLA model. Maps a camera image plus a
    natural-language instruction to a robot action (7-DoF ndarray).
  * ``PlannerRuntime`` — wraps a task-planner. Decomposes a high-level
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
class TextGenerator(Protocol):
    """Generates text from chat messages.

    This is the model-layer interface. Implementations live in
    ``runners/models/`` (e.g. ``GemmaVLMGenerator``). The planning
    logic in ``runners/agents/planner.py`` consumes this protocol,
    so swapping the underlying model doesn't touch the planner.

    Multimodal implementations (e.g. ``GemmaVLMGenerator``) extend
    ``generate`` with an optional ``image`` argument; ``LLMPlanner``
    forwards a scene image only when the generator accepts it, so a
    text-only generator (matching this minimal signature) keeps working
    unchanged.
    """

    def generate(self, messages: list[dict[str, Any]]) -> str:
        """Generate text from a list of chat messages.

        Parameters
        ----------
        messages:
            Chat messages, e.g. ``[{"role": "user", "content": "..."}]``.

        Returns
        -------
        Generated text string.
        """
        ...


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

    def plan(self, task_instruction: str, image: Any | None = None) -> list[str]:
        """Break a high-level instruction into ordered sub-steps.

        Parameters
        ----------
        task_instruction:
            The top-level task description (e.g. "pick up the red cube
            and place it on the shelf").
        image:
            Optional scene image (PIL Image or HWC uint8 ndarray) captured
            at the start of the episode. A multimodal planner grounds its
            plan in it; text-only planners ignore it.

        Returns
        -------
        Ordered list of sub-instructions the pilot should execute
        sequentially.
        """
        ...
