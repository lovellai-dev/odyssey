"""DelegatedEvalRuntime — delegation-driven multi-agent eval orchestrator.

The *delegation* counterpart to ``PlannedEvalRuntime``. Where the planner-driven
runtime has the SPECIALIST author the full task decomposition up front and the
PILOT march through that fixed list, here **no agent owns the sequence**:

  * The **orchestrator** holds a generic, task-agnostic manipulation template
    (default pick -> place — an embodiment-level skill skeleton, *not* a
    task-specific plan).
  * For each phase it **delegates grounding** to the SPECIALIST — "where is the
    object to pick up in *this* scene?" — turning a generic query into a
    scene-grounded phrase that conditions the PILOT's instruction.
  * It then **delegates execution** to the PILOT, and the phase **hands control
    back** when the SPECIALIST's completion detector confirms the sub-task is
    done (semantic hand-back), with a step cap as a safety net.

The distinction from ``PlannedEvalRuntime`` is the SPECIALIST's *role*: there it
is the sequence author (``plan``); here it is an on-demand perception tool
(``ground``) that never decomposes the task. Everything else — the model, the
PILOT, the completion detector, the environments, the metrics — is held equal,
so the two runtimes isolate exactly that variable.

Scope (v0.1 skeleton): the orchestrator's routing is a fixed template and its
hand-back is completion-gated. Capability advertising / A2A-style task lifecycle
/ an LLM router that *chooses* which capability to invoke are deferred (that is
the full delegation-driven regime). See ``PlannedEvalRuntime`` for the
planner-driven arm this is compared against.

Public surface mirrors ``PlannedEvalRuntime`` (``begin_episode`` / ``get_action``
/ ``drain_phase_events`` / ``close`` + the phase properties) so the eval runners
drive both runtimes through one code path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from odyssey.runners.agents.runtime import (
    CompletionDetector,
    GroundingProvider,
    PilotRuntime,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DelegationSkill:
    """One phase of the orchestrator's generic manipulation template.

    ``target_query`` is what the orchestrator delegates to the grounder (it may
    reference ``{task}``); the returned phrase is spliced into ``instruction``
    via ``{target}`` to produce the PILOT's grounded sub-instruction.
    """

    name: str
    target_query: str
    instruction: str


# Default template: the canonical pick-and-place skeleton. Task-agnostic — the
# scene-specific intelligence comes entirely from the delegated grounding, not
# from this list. Override via ``DelegationConfig.from_config`` (skills key) for
# other manipulation shapes.
PICK_PLACE_TEMPLATE: tuple[DelegationSkill, ...] = (
    DelegationSkill(
        name="pick",
        target_query="the object that must be picked up to accomplish the task: {task}",
        instruction="pick up {target}",
    ),
    DelegationSkill(
        name="place",
        target_query=(
            "the location or container where the held object must be placed "
            "to accomplish the task: {task}"
        ),
        instruction="place it at {target}",
    ),
)


@dataclass
class DelegationConfig:
    """Configuration for the delegation-driven orchestrator.

    Parameters
    ----------
    skills:
        The manipulation template (ordered phases). Defaults to pick -> place.
    check_every:
        Poll the completion detector every N steps for hand-back (never every
        step — a VLM round-trip costs ~1-2s).
    max_steps_per_phase:
        Safety cap: force a hand-back after this many steps so a phase never
        wedges if the detector never confirms (or is unavailable).
    """

    skills: tuple[DelegationSkill, ...] = PICK_PLACE_TEMPLATE
    check_every: int = 10
    max_steps_per_phase: int = 100

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> DelegationConfig:
        """Build a DelegationConfig from a mission ``config:`` dict (opt-in).

        Recognised keys: ``phase_check_every``, ``phase_max_steps`` (shared with
        the planner arm for parity). The template is the default pick -> place;
        a custom template is not yet mission-configurable (skeleton scope).
        """
        return cls(
            check_every=int(cfg.get("phase_check_every", 10)),
            max_steps_per_phase=int(cfg.get("phase_max_steps", 100)),
        )


@dataclass
class _DelegationState:
    """Mutable per-episode orchestrator state."""

    task: str = ""
    skill_index: int = 0
    # The current PILOT instruction (grounded). None means the next phase still
    # needs grounding — the orchestrator delegates it lazily on the next step,
    # using the frame it is about to act on.
    pilot_instruction: str | None = None
    steps_in_phase: int = 0


class DelegatedEvalRuntime:
    """Delegation-driven multi-agent eval runtime (orchestrator + PILOT + SPECIALIST).

    Parameters
    ----------
    pilot:
        A ``PilotRuntime`` (e.g. ``VLARuntime``) for action generation.
    grounder:
        A ``GroundingProvider`` (e.g. the out-of-process ``RemotePlanner``) the
        orchestrator delegates per-phase target grounding to.
    config:
        Template + hand-back cadence/cap. Defaults to pick -> place.
    detector:
        Optional ``CompletionDetector`` for semantic hand-back. If omitted, the
        grounder is auto-adopted when it exposes ``check_done`` (the SPECIALIST
        answers both from the same loaded model — zero extra VRAM). Without a
        detector, phases hand back only at the step cap.
    task_fallback:
        Instruction used if grounding yields nothing and no task is set.
    """

    def __init__(
        self,
        pilot: PilotRuntime,
        grounder: GroundingProvider,
        *,
        config: DelegationConfig | None = None,
        detector: CompletionDetector | None = None,
        task_fallback: str = "complete the task",
    ) -> None:
        self._pilot = pilot
        self._grounder = grounder
        self._config = config or DelegationConfig()
        # Reuse the grounder as the completion detector when it can answer a
        # yes/no check (the SPECIALIST does both). hasattr keeps a grounder that
        # lacks the method valid — hand-back then falls back to the step cap.
        if detector is None and hasattr(grounder, "check_done"):
            detector = grounder  # type: ignore[assignment]
        self._detector = detector
        self._fallback = task_fallback
        self._state = _DelegationState()
        # Buffered (from, to, instruction, reason, capability) records. get_action
        # is sync but telemetry is async, so the async step loop drains these.
        self._pending_events: list[dict[str, Any]] = []

    @property
    def current_phase_index(self) -> int:
        return self._state.skill_index

    @property
    def total_phases(self) -> int:
        return len(self._config.skills)

    @property
    def current_instruction(self) -> str:
        return self._state.pilot_instruction or ""

    @property
    def _is_complete(self) -> bool:
        return self._state.skill_index >= len(self._config.skills)

    def begin_episode(
        self, task_instruction: str, image: Any | None = None
    ) -> list[str]:
        """Call at the start of each episode. Returns the phase labels.

        Unlike the planner arm, there is no plan to compute here: grounding is
        delegated lazily on the first step of each phase (using the frame the
        PILOT is about to act on). The returned list is the template's phase
        names, purely for telemetry/logging parity with ``PlannedEvalRuntime``.
        ``image`` is accepted for signature parity and currently unused.
        """
        self._state = _DelegationState(task=task_instruction)
        self._pending_events = []
        labels = [s.name for s in self._config.skills]
        logger.info(
            "DelegatedEvalRuntime: episode with %d delegated phase(s): %s",
            len(labels),
            labels,
        )
        if self._detector is None:
            logger.warning(
                "No completion detector available; delegated phases will only "
                "hand back at the max_steps_per_phase cap (%d).",
                self._config.max_steps_per_phase,
            )
        return labels

    def get_action(self, image: Any) -> NDArray[np.floating[Any]]:
        """Delegate grounding (if needed), then delegate execution to the PILOT.

        Handles semantic hand-back between phases based on the completion
        detector, with the step cap as a safety net.
        """
        if self._is_complete:
            # All phases handed back: hold the last grounded instruction.
            instruction = self._state.pilot_instruction or self._state.task or self._fallback
            return self._pilot.act(image, instruction)

        if self._state.pilot_instruction is None:
            self._delegate_grounding(image)

        assert self._state.pilot_instruction is not None
        action = self._pilot.act(image, self._state.pilot_instruction)
        self._state.steps_in_phase += 1
        self._maybe_hand_back(image)
        return action

    def _delegate_grounding(self, image: Any) -> None:
        """Delegate target grounding for the current phase to the SPECIALIST."""
        skill = self._config.skills[self._state.skill_index]
        query = skill.target_query.format(task=self._state.task)
        try:
            target = self._grounder.ground(query, image)
        except Exception as e:  # grounder is fail-safe, but never let it crash the loop
            logger.warning("grounding delegation failed (%s) — using query", e)
            target = query
        target = (target or "").strip() or self._state.task or self._fallback
        self._state.pilot_instruction = skill.instruction.format(target=target)
        logger.debug(
            "Delegated grounding for phase %d (%s): %r -> %r",
            self._state.skill_index,
            skill.name,
            query,
            self._state.pilot_instruction,
        )
        self._pending_events.append(
            {
                "from": self._state.skill_index,
                "to": self._state.skill_index,
                "instruction": self._state.pilot_instruction,
                "reason": "grounding",
                "capability": "grounding",
                "skill": skill.name,
            }
        )

    def _maybe_hand_back(self, image: Any) -> None:
        cfg = self._config
        hand_back = False
        reason = ""

        n = self._state.steps_in_phase
        # Cap first: a stuck phase always hands back, even without a detector.
        if n >= cfg.max_steps_per_phase:
            hand_back, reason = True, "cap"
        elif (
            self._detector is not None
            and self._state.pilot_instruction is not None
            and n > 0
            and n % cfg.check_every == 0
            and self._detector.check_done(self._state.pilot_instruction, image)
        ):
            hand_back, reason = True, "completion"

        if not hand_back:
            return

        old_idx = self._state.skill_index
        old_instruction = self._state.pilot_instruction
        self._state.skill_index += 1
        self._state.steps_in_phase = 0
        # Next phase re-grounds lazily on its first step.
        self._state.pilot_instruction = None
        self._pending_events.append(
            {
                "from": old_idx,
                "to": self._state.skill_index,
                "instruction": old_instruction,
                "reason": reason,
                "capability": "handback",
            }
        )
        logger.debug(
            "Phase %d -> %d hand-back (%s)", old_idx, self._state.skill_index, reason
        )

    def drain_phase_events(self) -> list[dict[str, Any]]:
        """Return and clear buffered delegation/hand-back records.

        Each record is ``{"from", "to", "instruction", "reason", "capability"}``
        where reason is ``grounding`` (a new target was delegated) |
        ``completion`` (semantic hand-back) | ``cap`` (safety-cap hand-back).
        Empty (zero overhead) on non-eventful steps. Mirrors
        ``PlannedEvalRuntime.drain_phase_events`` so the async step loop drains
        both runtimes identically.
        """
        events = self._pending_events
        self._pending_events = []
        return events

    def close(self) -> None:
        """Release runtime resources. Closes the grounder if it owns any (e.g.
        an out-of-process ``RemotePlanner`` subprocess). Safe to call twice."""
        closer = getattr(self._grounder, "close", None)
        if callable(closer):
            closer()
