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

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# Opt-in "who acted when" trace (PILOT / SPECIALIST / ORCHESTRATOR). Silent by
# default even under root INFO; the eval recipe bumps it to INFO on `config.trace`.
AGENT_TRACE = logging.getLogger("odyssey.agents.trace")
AGENT_TRACE.setLevel(logging.WARNING)


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


@runtime_checkable
class CompletionDetector(Protocol):
    """Answers whether a sub-instruction is satisfied in the current frame.

    Used by ``PlannedEvalRuntime``'s ``COMPLETION_GATED`` strategy to close the
    loop: instead of advancing phases on a blind step counter, it asks a VLM
    whether the active sub-instruction is done before moving on. The
    out-of-process SPECIALIST satisfies both this and ``PlannerRuntime`` from
    the same loaded model, so no second model is loaded.
    """

    def check_done(self, instruction: str, image: Any) -> bool:
        """Return True if ``instruction`` is satisfied in ``image``.

        Must be conservative and fail-safe: return False when uncertain or on
        any error, so a phase never advances on a false positive (the runtime's
        step cap still guarantees forward progress).
        """
        ...


@runtime_checkable
class GroundingProvider(Protocol):
    """Locates the scene target a delegated sub-task should act on.

    This is the delegated capability in the *delegation-driven* runtime
    (``DelegatedEvalRuntime``). Unlike ``PlannerRuntime`` — which authors the
    whole task decomposition up front — a grounder owns no sequence: the
    orchestrator invokes it on demand, once per phase, to turn a generic query
    ("the object to pick up") into a scene-grounded phrase ("the red mug on the
    left") that conditions the pilot's instruction. It is the open-weight
    analogue of GR-ER's 2D pointing / spatial grounding.

    The out-of-process SPECIALIST satisfies this alongside ``PlannerRuntime``
    and ``CompletionDetector`` from the same loaded model — no extra VRAM.
    """

    def ground(self, target_query: str, image: Any) -> str:
        """Return a scene-grounded phrase for ``target_query`` in ``image``.

        Parameters
        ----------
        target_query:
            A generic description of what to locate (e.g. "the object that must
            be picked up").
        image:
            RGB frame (PIL Image or HWC uint8 ndarray) of the current scene.

        Returns
        -------
        A short natural-language phrase naming the located target, suitable for
        splicing into a pilot instruction. Must fail safe: return the query
        unchanged (never empty) when uncertain or on any error, so the pilot
        always receives an actionable instruction.
        """
        ...


@dataclass(frozen=True)
class RouteDecision:
    """One routing decision from an ORCHESTRATOR: the next pilot sub-instruction.

    ``subtask`` is the next sub-instruction for the pilot to execute; ``done`` is
    True when the orchestrator judges the whole task already accomplished (then
    ``subtask`` is ignored). Fail-safe construction should prefer ``done=False``
    with a sensible ``subtask`` so an episode never terminates on a bad route.
    """

    subtask: str
    done: bool = False


@runtime_checkable
class OrchestratorRuntime(Protocol):
    """Decides, on demand, the next sub-instruction — the regime-D LLM router.

    Unlike ``PlannerRuntime`` (which authors the whole sequence up front) or a
    fixed ``pick -> place`` template (``DelegatedEvalRuntime``), an orchestrator
    owns the sequence **dynamically**: given the task, the current frame, and the
    sub-instructions already completed, it emits the single next sub-instruction
    (or signals done). The out-of-process SPECIALIST/ORCHESTRATOR satisfies this
    alongside ``CompletionDetector`` from the same loaded model — no extra VRAM.
    """

    def route(
        self, task_instruction: str, image: Any, history: list[str]
    ) -> RouteDecision:
        """Return the next sub-instruction (or ``done``) for ``task_instruction``.

        Parameters
        ----------
        task_instruction:
            The overall task.
        image:
            The current scene frame (RGB) to ground the decision in.
        history:
            Sub-instructions already completed this episode, so the orchestrator
            doesn't repeat them.

        Returns
        -------
        A ``RouteDecision``. Must fail safe: on any error return a decision that
        keeps the episode progressing (``done=False`` with the task as subtask),
        never a spurious ``done`` (which would end the episode early).
        """
        ...
