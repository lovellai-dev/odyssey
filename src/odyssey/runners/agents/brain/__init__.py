"""Deterministic multi-agent brain — dispatch -> compose -> converge.

A framework-agnostic (no Google ADK) control plane that replaces the
plan-then-execute planner (``agents/planned.py``). The brain owns *who acts*
(deterministic dispatch), *how outputs compose* (advisories), and *convergence*
(a single Pilot inference over the composed context). It calls only odyssey's
existing model-layer ports — ``TextGenerator`` (specialist turn) and
``PilotRuntime`` (action turn) — so it needs no agent framework.

The pieces:
  * ``BrainContext`` / ``Advisory``  — the compose seam (``context.py``).
  * ``select_specialists`` + concept lexicon — deterministic dispatch,
    no LLM decision (``dispatch.py``).
  * ``SpecialistTurn``               — one advisory-producing cognitive turn
    over any ``TextGenerator`` (``specialist.py``).
  * ``DeterministicBrain``           — the orchestrator that satisfies the eval
    runtime surface (``runtime.py``).
  * ``SafetyBoundary``               — monitor-only action checks (``safety.py``).
"""

from __future__ import annotations

from odyssey.runners.agents.brain.context import Advisory, BrainContext
from odyssey.runners.agents.brain.dispatch import Decision, select_specialists
from odyssey.runners.agents.brain.runtime import DeterministicBrain
from odyssey.runners.agents.brain.safety import SafetyBoundary, SafetyFinding
from odyssey.runners.agents.brain.specialist import SpecialistTurn

__all__ = [
    "Advisory",
    "BrainContext",
    "Decision",
    "DeterministicBrain",
    "SafetyBoundary",
    "SafetyFinding",
    "SpecialistTurn",
    "select_specialists",
]
