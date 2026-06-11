"""Multi-agent evaluation runtimes.

Protocols:
  * ``TextGenerator`` — chat messages → text (model-layer interface)
  * ``PilotRuntime`` — image + instruction → action
  * ``PlannerRuntime`` — task instruction → sub-instructions

Implementations:
  * ``VLARuntime`` — OpenVLA (requires ``openvla`` extra)
  * ``LLMPlanner`` — planning logic, takes any TextGenerator
  * ``PlannedEvalRuntime`` — composes planner + pilot with phase transitions

Model loading (``GemmaTextGenerator``, etc.) lives in ``runners/models/``.
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


def __getattr__(name: str) -> object:
    """Lazy-import heavy implementations to avoid pulling in torch at import."""
    if name == "VLARuntime":
        from odyssey.runners.agents.vla import VLARuntime

        return VLARuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
