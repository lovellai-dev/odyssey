"""Multi-agent evaluation runtimes.

Protocols:
  * ``TextGenerator`` — chat messages -> text (model-layer interface)
  * ``PilotRuntime`` — image + instruction -> action
  * ``PlannerRuntime`` — task instruction -> sub-instructions

Implementations:
  * ``LLMPlanner`` — planning logic, takes any TextGenerator
  * ``PlannedEvalRuntime`` — composes planner + pilot with phase transitions

Model loading (``VLARuntime``, ``GemmaTextGenerator``) lives in
``runners/models/``.
"""

from odyssey.runners.agents.planned import (
    PhaseConfig,
    PhaseStrategy,
    PlannedEvalRuntime,
)
from odyssey.runners.agents.planner import LLMPlanner
from odyssey.runners.agents.runtime import PilotRuntime, PlannerRuntime, TextGenerator

__all__ = [
    "LLMPlanner",
    "PhaseConfig",
    "PhaseStrategy",
    "PilotRuntime",
    "PlannedEvalRuntime",
    "PlannerRuntime",
    "TextGenerator",
]
