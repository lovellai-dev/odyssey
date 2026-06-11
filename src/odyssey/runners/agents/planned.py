"""PlannedEvalRuntime — composes planner + pilot with phase transitions.

This is the multi-agent eval orchestrator. Before each episode:
  1. The planner decomposes the task instruction into sub-steps.
  2. The pilot executes each sub-step sequentially, with a configurable
     phase transition strategy determining when to advance.

Phase transition strategies:
  * ``fixed_steps`` (default) — advance after N steps per phase.
  * ``timeout`` — advance after T seconds per phase.

The runtime is simulator-agnostic: callers (IsaacLabRunner,
RobosuiteRunner) drive the step loop and call ``get_action()``
each tick. The runtime tracks which phase is active and feeds
the correct sub-instruction to the pilot.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from odyssey.runners.agents.runtime import PilotRuntime, PlannerRuntime

logger = logging.getLogger(__name__)


class PhaseStrategy(str, Enum):
    FIXED_STEPS = "fixed_steps"
    TIMEOUT = "timeout"


@dataclass
class PhaseConfig:
    """Configuration for phase transitions."""

    strategy: PhaseStrategy = PhaseStrategy.FIXED_STEPS
    steps_per_phase: int = 50
    timeout_seconds: float = 10.0


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
    """

    def __init__(
        self,
        pilot: PilotRuntime,
        planner: PlannerRuntime | None = None,
        *,
        phase_config: PhaseConfig | None = None,
        fallback_instruction: str = "complete the task",
    ) -> None:
        self._pilot = pilot
        self._planner = planner
        self._phase_config = phase_config or PhaseConfig()
        self._fallback = fallback_instruction
        self._state = _PhaseState()

    @property
    def current_phase_index(self) -> int:
        return self._state.current_index

    @property
    def total_phases(self) -> int:
        return len(self._state.sub_instructions)

    @property
    def current_instruction(self) -> str:
        return self._state.current_instruction

    def begin_episode(self, task_instruction: str) -> list[str]:
        """Call at the start of each episode. Returns the plan.

        If no planner is set, returns ``[task_instruction]`` (single phase).
        """
        if self._planner is not None:
            steps = self._planner.plan(task_instruction)
        else:
            steps = [task_instruction]

        if not steps:
            steps = [self._fallback]

        self._state = _PhaseState(
            sub_instructions=steps,
            phase_start_time=time.monotonic(),
        )
        logger.info(
            "PlannedEvalRuntime: episode plan with %d phases: %s",
            len(steps),
            steps,
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
        self._maybe_advance_phase()
        return action

    def _maybe_advance_phase(self) -> None:
        if self._state.is_complete:
            return

        cfg = self._phase_config
        advance = False

        if cfg.strategy == PhaseStrategy.FIXED_STEPS:
            if self._state.steps_in_phase >= cfg.steps_per_phase:
                advance = True
        elif cfg.strategy == PhaseStrategy.TIMEOUT:
            elapsed = time.monotonic() - self._state.phase_start_time
            if elapsed >= cfg.timeout_seconds:
                advance = True

        if advance:
            old_idx = self._state.current_index
            self._state.advance()
            if not self._state.is_complete:
                logger.debug(
                    "Phase %d → %d: %s",
                    old_idx,
                    self._state.current_index,
                    self._state.current_instruction,
                )
