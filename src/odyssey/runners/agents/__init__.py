"""Multi-agent evaluation runtimes.

Protocols:
  * ``PilotRuntime`` — image + instruction → action
  * ``PlannerRuntime`` — task instruction → sub-instructions

Implementations:
  * ``VLARuntime`` — OpenVLA (requires ``openvla`` extra)
  * ``LLMPlanner`` — Gemma 4B int4 (requires ``transformers`` + ``torch``)
  * ``PlannedEvalRuntime`` — composes planner + pilot with phase transitions
"""

from odyssey.runners.agents.planned import (
    PhaseConfig,
    PhaseStrategy,
    PlannedEvalRuntime,
)
from odyssey.runners.agents.runtime import PilotRuntime, PlannerRuntime

__all__ = [
    "LLMPlanner",
    "PhaseConfig",
    "PhaseStrategy",
    "PilotRuntime",
    "PlannedEvalRuntime",
    "PlannerRuntime",
    "VLARuntime",
]


def __getattr__(name: str) -> object:
    """Lazy-import heavy implementations to avoid pulling in torch at import."""
    if name == "VLARuntime":
        from odyssey.runners.agents.vla import VLARuntime

        return VLARuntime
    if name == "LLMPlanner":
        from odyssey.runners.agents.planner import LLMPlanner

        return LLMPlanner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
