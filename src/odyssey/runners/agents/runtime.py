"""Agent runtime protocols for multi-agent evaluation.

Four runtime protocols define the interfaces between components:

  * ``TextGenerator`` — wraps any text generation model. Maps chat
    messages to generated text. Lives at the model layer.
  * ``PilotRuntime`` — wraps a VLA model. Maps a camera image plus a
    natural-language instruction to a robot action (7-DoF ndarray).
  * ``PlannerRuntime`` — wraps a task-planner. Decomposes a high-level
    task instruction into an ordered list of sub-instructions the pilot
    executes sequentially.
  * ``CompletionDetector`` — wraps a completion judge (e.g. a VLM like
    Gemma int4). Answers "is the current sub-instruction done?" so the
    orchestrator can advance the phase (closed-loop *hand-back*) instead
    of relying on a fixed step budget.

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
class CompletionDetector(Protocol):
    """Judges whether the current sub-instruction has been completed.

    This is the closed-loop counterpart to a fixed step/timeout budget:
    a completion judge (typically a VLM such as Gemma int4, or GR00T's
    ``cosmos_reason``) looks at the current observation and the active
    sub-instruction and answers whether the pilot has finished it, so the
    orchestrator can *hand back* control and advance the phase.

    Kept deliberately minimal so any fake/heuristic detector satisfies it
    structurally. The polling cadence (once per action chunk, not per step)
    is the ``ChunkCompletionGate``'s job, not the detector's — the detector
    only answers the yes/no question when asked.
    """

    def is_complete(self, observation: Any, instruction: str) -> bool:
        """Return ``True`` when ``instruction`` is judged complete.

        Parameters
        ----------
        observation:
            The current observation the judge grounds its decision in —
            typically the camera image (PIL Image or HWC uint8 ndarray).
        instruction:
            The active sub-instruction whose completion is being judged.
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
