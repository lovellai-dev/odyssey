"""OrchestratedEvalRuntime — LLM-orchestrated multi-agent eval (regime D).

The *orchestration* counterpart to ``PlannedEvalRuntime`` (planner authors the whole
sequence up front) and ``DelegatedEvalRuntime`` (a fixed ``pick -> place`` template
owns the sequence). Here **an LLM ORCHESTRATOR owns the sequence dynamically**: at
each phase boundary it looks at the live scene + what's already been done and emits
the single next sub-instruction — or declares the task DONE. No plan up front, no
fixed template; the sequence emerges.

Roles (only two models, same as the other arms — zero extra VRAM):
  * PILOT — the chunk-emitting VLA (via ``ChunkPilotAdapter``), executes each
    sub-instruction.
  * ORCHESTRATOR — the out-of-process SPECIALIST (Gemma) answering ``route`` (next
    sub-instruction) **and** ``check_done`` (completion-gated hand-back) from one
    loaded model.

This is regime D of the coordination taxonomy: a real LLM router deciding *what
next*, versus the deterministic ``pick -> place`` skeleton of the delegation arm.

Public surface mirrors ``PlannedEvalRuntime`` / ``DelegatedEvalRuntime``
(``begin_episode`` / ``get_action`` / ``drain_phase_events`` / ``close`` + the phase
properties) so the eval recipe drives all three through one code path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from odyssey.runners.agents.runtime import (
    CompletionDetector,
    OrchestratorRuntime,
    PilotRuntime,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationConfig:
    """Configuration for the LLM-orchestrated runtime.

    Parameters
    ----------
    check_every:
        Poll the completion detector every N steps for hand-back (never every step
        — a VLM round-trip costs ~1-2s).
    max_steps_per_phase:
        Safety cap: force a hand-back after this many steps so a phase never wedges.
    max_phases:
        Hard cap on routed sub-tasks per episode — the episode's orchestration ends
        after this many phases even if the router never says DONE (guards against a
        router that re-routes forever). The env's own ``max_steps_per_episode`` and
        LIBERO's success flag still bound the episode.
    """

    check_every: int = 10
    max_steps_per_phase: int = 100
    max_phases: int = 8

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> OrchestrationConfig:
        """Build from a mission ``config:`` dict. Recognised keys:
        ``phase_check_every``, ``phase_max_steps`` (shared with the other arms) and
        ``max_phases``."""
        return cls(
            check_every=int(cfg.get("phase_check_every", 10)),
            max_steps_per_phase=int(cfg.get("phase_max_steps", 100)),
            max_phases=int(cfg.get("max_phases", 8)),
        )


@dataclass
class _OrchestrationState:
    """Mutable per-episode orchestrator state."""

    task: str = ""
    phase_index: int = 0
    # Current PILOT instruction. None means the next phase still needs routing —
    # the orchestrator emits it lazily on the next step, using the live frame.
    pilot_instruction: str | None = None
    steps_in_phase: int = 0
    complete: bool = False
    history: list[str] = field(default_factory=list)  # completed sub-instructions


class OrchestratedEvalRuntime:
    """LLM-orchestrated multi-agent eval runtime (ORCHESTRATOR + PILOT).

    Parameters
    ----------
    pilot:
        A ``PilotRuntime`` (e.g. the ``ChunkPilotAdapter`` over GR00T).
    orchestrator:
        An ``OrchestratorRuntime`` (e.g. the out-of-process ``RemotePlanner``) that
        emits the next sub-instruction per phase via ``route``.
    config:
        Cadence / caps. Defaults are conservative.
    detector:
        Optional ``CompletionDetector`` for semantic hand-back. If omitted, the
        orchestrator is auto-adopted when it exposes ``check_done`` (the SPECIALIST
        answers both). Without a detector, phases hand back only at the step cap.
    task_fallback:
        Instruction held once the task is complete / if routing yields nothing.
    """

    def __init__(
        self,
        pilot: PilotRuntime,
        orchestrator: OrchestratorRuntime,
        *,
        config: OrchestrationConfig | None = None,
        detector: CompletionDetector | None = None,
        task_fallback: str = "complete the task",
    ) -> None:
        self._pilot = pilot
        self._orchestrator = orchestrator
        self._config = config or OrchestrationConfig()
        # Reuse the orchestrator as the completion detector when it can answer a
        # yes/no check (the SPECIALIST does both). hasattr keeps an orchestrator
        # that lacks the method valid — hand-back then falls back to the step cap.
        if detector is None and hasattr(orchestrator, "check_done"):
            detector = orchestrator  # type: ignore[assignment]
        self._detector = detector
        self._fallback = task_fallback
        self._state = _OrchestrationState()
        self._pending_events: list[dict[str, Any]] = []

    @property
    def current_phase_index(self) -> int:
        return self._state.phase_index

    @property
    def total_phases(self) -> int:
        # Unknown up front (the sequence is dynamic); report phases seen so far.
        return self._state.phase_index + (0 if self._state.complete else 1)

    @property
    def current_instruction(self) -> str:
        return self._state.pilot_instruction or ""

    def begin_episode(
        self, task_instruction: str, image: Any | None = None
    ) -> list[str]:
        """Call at the start of each episode. Returns ``[]`` (no plan up front).

        The sequence is decided dynamically by the orchestrator, one sub-task at a
        time, starting on the first step. ``image`` is accepted for signature
        parity with the other runtimes and currently unused here.
        """
        self._state = _OrchestrationState(task=task_instruction)
        self._pending_events = []
        logger.info("OrchestratedEvalRuntime: episode start, task=%r", task_instruction)
        if self._detector is None:
            logger.warning(
                "No completion detector available; orchestrated phases will only "
                "hand back at the max_steps_per_phase cap (%d).",
                self._config.max_steps_per_phase,
            )
        return []

    def get_action(self, image: Any) -> NDArray[np.floating[Any]]:
        """Route the next sub-instruction (if needed), then execute it via the PILOT.

        Semantic hand-back between phases via the completion detector, with the
        step cap as a safety net; the router decides the sequence and when done.
        """
        if not self._state.complete and self._state.pilot_instruction is None:
            self._route_next(image)

        if self._state.complete:
            # Task routed as done (or max_phases hit): hold the last instruction.
            instruction = self._state.pilot_instruction or self._state.task or self._fallback
            return self._pilot.act(image, instruction)

        assert self._state.pilot_instruction is not None
        action = self._pilot.act(image, self._state.pilot_instruction)
        self._state.steps_in_phase += 1
        self._maybe_hand_back(image)
        return action

    def _route_next(self, image: Any) -> None:
        """Ask the ORCHESTRATOR for the next sub-instruction (or DONE)."""
        if self._state.phase_index >= self._config.max_phases:
            logger.info("OrchestratedEvalRuntime: max_phases reached — completing.")
            self._state.complete = True
            return
        try:
            decision = self._orchestrator.route(
                self._state.task, image, list(self._state.history)
            )
        except Exception as e:  # orchestrator is fail-safe, but never crash the loop
            logger.warning("route failed (%s) — using task", e)
            from odyssey.runners.agents.runtime import RouteDecision

            decision = RouteDecision(subtask=self._state.task, done=False)

        if decision.done:
            self._state.complete = True
            logger.info("OrchestratedEvalRuntime: orchestrator declared DONE.")
            return
        self._state.pilot_instruction = (
            (decision.subtask or "").strip() or self._state.task or self._fallback
        )
        self._pending_events.append(
            {
                "from": self._state.phase_index,
                "to": self._state.phase_index,
                "instruction": self._state.pilot_instruction,
                "reason": "route",
                "capability": "routing",
            }
        )
        logger.debug(
            "Routed phase %d: %r", self._state.phase_index, self._state.pilot_instruction
        )

    def _maybe_hand_back(self, image: Any) -> None:
        cfg = self._config
        hand_back = False
        reason = ""

        n = self._state.steps_in_phase
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

        old_idx = self._state.phase_index
        old_instruction = self._state.pilot_instruction
        if old_instruction:
            self._state.history.append(old_instruction)
        self._state.phase_index += 1
        self._state.steps_in_phase = 0
        # Next phase re-routes lazily on its first step.
        self._state.pilot_instruction = None
        self._pending_events.append(
            {
                "from": old_idx,
                "to": self._state.phase_index,
                "instruction": old_instruction,
                "reason": reason,
                "capability": "handback",
            }
        )
        logger.debug("Phase %d -> %d hand-back (%s)", old_idx, self._state.phase_index, reason)

    def drain_phase_events(self) -> list[dict[str, Any]]:
        """Return and clear buffered route/hand-back records.

        Each record is ``{"from", "to", "instruction", "reason", "capability"}``
        where reason is ``route`` (a new sub-task was routed) | ``completion``
        (semantic hand-back) | ``cap`` (safety-cap hand-back). Mirrors the other
        runtimes so the async step loop drains all three identically.
        """
        events = self._pending_events
        self._pending_events = []
        return events

    def close(self) -> None:
        """Release runtime resources. Closes the orchestrator if it owns any (e.g.
        an out-of-process ``RemotePlanner`` subprocess). Safe to call twice."""
        closer = getattr(self._orchestrator, "close", None)
        if callable(closer):
            closer()
