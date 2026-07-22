"""PlannedEvalRuntime — composes planner + pilot with phase transitions.

This is the multi-agent eval orchestrator. Before each episode:
  1. The planner decomposes the task instruction into sub-steps.
  2. The pilot executes each sub-step sequentially, with a configurable
     phase transition strategy determining when to advance.

Phase transition strategies:
  * ``fixed_steps`` (default) — advance after N steps per phase.
  * ``timeout`` — advance after T seconds per phase.

The runtime is simulator-agnostic: callers (e.g. RobosuiteRunner)
drive the step loop and call ``get_action()`` each tick. The runtime
tracks which phase is active and feeds the correct sub-instruction
to the pilot.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from odyssey.runners.agents.runtime import (
    CompletionDetector,
    PilotRuntime,
    PlannerRuntime,
)

logger = logging.getLogger(__name__)


class PhaseStrategy(str, Enum):
    FIXED_STEPS = "fixed_steps"
    TIMEOUT = "timeout"
    COMPLETION_GATED = "completion_gated"


@dataclass
class PhaseConfig:
    """Configuration for phase transitions."""

    strategy: PhaseStrategy = PhaseStrategy.FIXED_STEPS
    steps_per_phase: int = 50
    timeout_seconds: float = 10.0
    # COMPLETION_GATED only: poll the detector every ``check_every`` steps
    # (never every step — a VLM round-trip costs ~1-2s), and force-advance
    # after ``max_steps_per_phase`` as a safety cap so a phase never wedges.
    check_every: int = 10
    max_steps_per_phase: int = 100

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> PhaseConfig:
        """Build a PhaseConfig from a mission ``config:`` dict (opt-in).

        Recognised keys: ``phase_strategy`` (``fixed_steps`` | ``timeout`` |
        ``completion_gated``, default ``fixed_steps``), ``steps_per_phase``,
        ``timeout_seconds``, ``phase_check_every``, ``phase_max_steps``. An
        unknown ``phase_strategy`` raises ``ValueError`` (runners own config
        validation — the spec's ``config`` is free-form).
        """
        raw = cfg.get("phase_strategy", PhaseStrategy.FIXED_STEPS.value)
        try:
            strategy = PhaseStrategy(str(raw))
        except ValueError as exc:
            allowed = ", ".join(s.value for s in PhaseStrategy)
            raise ValueError(
                f"unknown phase_strategy {raw!r}; allowed: {allowed}"
            ) from exc
        return cls(
            strategy=strategy,
            steps_per_phase=int(cfg.get("steps_per_phase", 50)),
            timeout_seconds=float(cfg.get("timeout_seconds", 10.0)),
            check_every=int(cfg.get("phase_check_every", 10)),
            max_steps_per_phase=int(cfg.get("phase_max_steps", 100)),
        )


@dataclass
class _PhaseState:
    """Mutable state tracking the current phase within an episode."""

    sub_instructions: list[str] = field(default_factory=list)
    current_index: int = 0
    steps_in_phase: int = 0
    phase_start_time: float = 0.0

    @property
    def current_instruction(self) -> str:
        if not self.sub_instructions:
            return ""
        idx = min(self.current_index, len(self.sub_instructions) - 1)
        return self.sub_instructions[idx]

    @property
    def is_complete(self) -> bool:
        return self.current_index >= len(self.sub_instructions)

    def advance(self) -> None:
        self.current_index += 1
        self.steps_in_phase = 0
        self.phase_start_time = time.monotonic()


class PlannedEvalRuntime:
    """Multi-agent eval runtime composing planner + pilot.

    Parameters
    ----------
    pilot:
        A ``PilotRuntime`` (e.g. ``VLARuntime``) for action generation.
    planner:
        A ``PlannerRuntime`` (e.g. ``LLMPlanner``) for task decomposition.
    phase_config:
        Controls when to advance between sub-instructions.
    fallback_instruction:
        Used if the planner is None or returns no plan.
    detector:
        Optional ``CompletionDetector`` for the ``COMPLETION_GATED`` strategy.
        If omitted, the planner is auto-adopted when it exposes a ``check_done``
        method (the out-of-process SPECIALIST answers both ``plan`` and
        ``check_done`` from the same loaded model — zero extra VRAM).
    """

    def __init__(
        self,
        pilot: PilotRuntime,
        planner: PlannerRuntime | None = None,
        *,
        phase_config: PhaseConfig | None = None,
        fallback_instruction: str = "complete the task",
        detector: CompletionDetector | None = None,
    ) -> None:
        self._pilot = pilot
        self._planner = planner
        self._phase_config = phase_config or PhaseConfig()
        self._fallback = fallback_instruction
        # Reuse the planner as the completion detector when it can answer a
        # yes/no check (RemotePlanner / a multimodal LLMPlanner). hasattr keeps
        # a text-only planner that lacks the method perfectly valid.
        if detector is None and hasattr(planner, "check_done"):
            detector = planner  # type: ignore[assignment]
        self._detector = detector
        self._state = _PhaseState()
        # Buffered (from, to, instruction, reason) advance records. get_action
        # is sync but telemetry is async, so the async step loop drains these.
        self._pending_events: list[dict[str, Any]] = []

    @property
    def current_phase_index(self) -> int:
        return self._state.current_index

    @property
    def total_phases(self) -> int:
        return len(self._state.sub_instructions)

    @property
    def current_instruction(self) -> str:
        return self._state.current_instruction

    def begin_episode(
        self, task_instruction: str, image: Any | None = None
    ) -> list[str]:
        """Call at the start of each episode. Returns the plan.

        If no planner is set, returns ``[task_instruction]`` (single phase).
        ``image`` is the first observation frame; a multimodal planner grounds
        its plan in it, text-only planners ignore it.
        """
        if self._planner is not None:
            steps = self._planner.plan(task_instruction, image)
        else:
            steps = [task_instruction]

        if not steps:
            steps = [self._fallback]

        self._state = _PhaseState(
            sub_instructions=steps,
            phase_start_time=time.monotonic(),
        )
        self._pending_events = []
        logger.info(
            "PlannedEvalRuntime: episode plan with %d phases: %s",
            len(steps),
            steps,
        )
        if (
            self._phase_config.strategy == PhaseStrategy.COMPLETION_GATED
            and self._detector is None
        ):
            logger.warning(
                "COMPLETION_GATED strategy but no completion detector available; "
                "phases will only advance at the max_steps_per_phase cap (%d).",
                self._phase_config.max_steps_per_phase,
            )
        return steps

    def get_action(self, image: Any) -> NDArray[np.floating[Any]]:
        """Get the next action from the pilot using the current phase instruction.

        Also handles phase advancement based on the configured strategy.
        """
        if self._state.is_complete:
            instruction = self._state.sub_instructions[-1]
        else:
            instruction = self._state.current_instruction

        action = self._pilot.act(image, instruction)
        self._state.steps_in_phase += 1
        self._maybe_advance_phase(image)
        return action

    def _maybe_advance_phase(self, image: Any) -> None:
        if self._state.is_complete:
            return

        cfg = self._phase_config
        advance = False
        reason = ""

        if cfg.strategy == PhaseStrategy.FIXED_STEPS:
            if self._state.steps_in_phase >= cfg.steps_per_phase:
                advance, reason = True, "fixed_steps"
        elif cfg.strategy == PhaseStrategy.TIMEOUT:
            elapsed = time.monotonic() - self._state.phase_start_time
            if elapsed >= cfg.timeout_seconds:
                advance, reason = True, "timeout"
        elif cfg.strategy == PhaseStrategy.COMPLETION_GATED:
            n = self._state.steps_in_phase
            # Cap first: a stuck phase always terminates, even if the detector
            # is unavailable or never confirms.
            if n >= cfg.max_steps_per_phase:
                advance, reason = True, "cap"
            elif (
                self._detector is not None
                and n > 0
                and n % cfg.check_every == 0
                and self._detector.check_done(self._state.current_instruction, image)
            ):
                advance, reason = True, "completion"

        if advance:
            old_idx = self._state.current_index
            old_instruction = self._state.current_instruction
            self._state.advance()
            next_instruction = (
                self._state.current_instruction
                if not self._state.is_complete
                else old_instruction
            )
            self._pending_events.append(
                {
                    "from": old_idx,
                    "to": self._state.current_index,
                    "instruction": next_instruction,
                    "reason": reason,
                }
            )
            if not self._state.is_complete:
                logger.debug(
                    "Phase %d → %d (%s): %s",
                    old_idx,
                    self._state.current_index,
                    reason,
                    self._state.current_instruction,
                )

    def drain_phase_events(self) -> list[dict[str, Any]]:
        """Return and clear buffered phase-advance records.

        Each record is ``{"from", "to", "instruction", "reason"}`` where reason
        is ``fixed_steps`` | ``timeout`` | ``completion`` | ``cap``. Empty (zero
        overhead) on non-advancing steps and the single-agent path. The async
        step loop drains this after each ``get_action`` to emit telemetry that
        the sync runtime cannot emit itself.
        """
        events = self._pending_events
        self._pending_events = []
        return events

    def close(self) -> None:
        """Release runtime resources. Closes the planner if it owns any
        (e.g. an out-of-process ``RemotePlanner`` subprocess). No-op for
        in-process planners. Safe to call multiple times."""
        closer = getattr(self._planner, "close", None)
        if callable(closer):
            closer()
