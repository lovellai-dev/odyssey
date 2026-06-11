"""Unit tests for the multi-agent evaluation runtimes (Phase 4).

Tests use mock/fake implementations to avoid GPU dependencies.
Covers: protocol conformance, PlannedEvalRuntime phase transitions,
planner output parsing, fallback behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Import engine first to avoid circular import (same pattern as test_engine.py).
import odyssey.engine  # noqa: F401
from odyssey.runners.agents.planned import (
    PhaseConfig,
    PhaseStrategy,
    PlannedEvalRuntime,
    _PhaseState,
)
from odyssey.runners.agents.planner import LLMPlanner, _parse_plan
from odyssey.runners.agents.runtime import PilotRuntime, PlannerRuntime, TextGenerator

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakePilot:
    """Records every (image, instruction) call and returns a fixed action."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, str]] = []

    def act(self, image: Any, instruction: str) -> NDArray[np.floating[Any]]:
        self.calls.append((image, instruction))
        return np.zeros(7, dtype=np.float64)


class FakeTextGenerator:
    """Returns canned text for any messages."""

    def __init__(self, response: str = "1. Step one\n2. Step two") -> None:
        self._response = response
        self.call_count = 0

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.call_count += 1
        return self._response


class FakePlanner:
    """Returns a fixed plan."""

    def __init__(self, steps: list[str]) -> None:
        self._steps = steps
        self.call_count = 0

    def plan(self, task_instruction: str) -> list[str]:
        self.call_count += 1
        return list(self._steps)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_fake_pilot_satisfies_protocol() -> None:
    assert isinstance(FakePilot(), PilotRuntime)


def test_fake_planner_satisfies_protocol() -> None:
    assert isinstance(FakePlanner([]), PlannerRuntime)


# ---------------------------------------------------------------------------
# _parse_plan
# ---------------------------------------------------------------------------

def test_parse_plan_numbered_dot() -> None:
    text = "1. Pick up the cube\n2. Move to shelf\n3. Place the cube"
    assert _parse_plan(text) == [
        "Pick up the cube",
        "Move to shelf",
        "Place the cube",
    ]


def test_parse_plan_numbered_paren() -> None:
    text = "1) Grasp object\n2) Lift\n3) Release"
    assert _parse_plan(text) == ["Grasp object", "Lift", "Release"]


def test_parse_plan_with_noise() -> None:
    text = "Here is the plan:\n1. Step one\nSome noise\n2. Step two\n"
    assert _parse_plan(text) == ["Step one", "Step two"]


def test_parse_plan_empty() -> None:
    assert _parse_plan("no numbered lines here") == []


def test_parse_plan_whitespace() -> None:
    text = "  1.  Pick up   \n  2.  Place down  "
    assert _parse_plan(text) == ["Pick up", "Place down"]


# ---------------------------------------------------------------------------
# PhaseState
# ---------------------------------------------------------------------------

def test_phase_state_current_instruction() -> None:
    state = _PhaseState(sub_instructions=["a", "b", "c"])
    assert state.current_instruction == "a"
    state.advance()
    assert state.current_instruction == "b"
    state.advance()
    assert state.current_instruction == "c"


def test_phase_state_is_complete() -> None:
    state = _PhaseState(sub_instructions=["a"])
    assert not state.is_complete
    state.advance()
    assert state.is_complete


def test_phase_state_empty() -> None:
    state = _PhaseState(sub_instructions=[])
    assert state.current_instruction == ""
    assert state.is_complete


# ---------------------------------------------------------------------------
# PlannedEvalRuntime — single agent (no planner)
# ---------------------------------------------------------------------------

def test_no_planner_single_phase() -> None:
    pilot = FakePilot()
    rt = PlannedEvalRuntime(pilot, planner=None)
    plan = rt.begin_episode("pick up the cube")
    assert plan == ["pick up the cube"]
    assert rt.total_phases == 1

    img = np.zeros((256, 256, 3), dtype=np.uint8)
    rt.get_action(img)
    assert len(pilot.calls) == 1
    assert pilot.calls[0][1] == "pick up the cube"


# ---------------------------------------------------------------------------
# PlannedEvalRuntime — with planner
# ---------------------------------------------------------------------------

def test_planner_decomposes_task() -> None:
    pilot = FakePilot()
    planner = FakePlanner(["grasp cube", "lift", "move to shelf", "release"])
    rt = PlannedEvalRuntime(pilot, planner)

    plan = rt.begin_episode("pick and place the cube")
    assert plan == ["grasp cube", "lift", "move to shelf", "release"]
    assert rt.total_phases == 4
    assert planner.call_count == 1


def test_phase_advancement_fixed_steps() -> None:
    pilot = FakePilot()
    planner = FakePlanner(["phase-1", "phase-2", "phase-3"])
    cfg = PhaseConfig(strategy=PhaseStrategy.FIXED_STEPS, steps_per_phase=3)
    rt = PlannedEvalRuntime(pilot, planner, phase_config=cfg)
    rt.begin_episode("task")

    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # Phase 1: steps 1-3
    for _ in range(3):
        rt.get_action(img)
    assert rt.current_phase_index == 1  # advanced after 3 steps
    assert pilot.calls[-1][1] == "phase-1"  # step 3 still used phase-1

    # Phase 2: steps 4-6
    for _ in range(3):
        rt.get_action(img)
    assert rt.current_phase_index == 2
    # Step 4 should have used phase-2
    assert pilot.calls[3][1] == "phase-2"

    # Phase 3: steps 7-9
    for _ in range(3):
        rt.get_action(img)
    assert rt.current_phase_index == 3  # past last phase = complete

    # After all phases, keeps using last instruction
    rt.get_action(img)
    assert pilot.calls[-1][1] == "phase-3"


def test_planner_returns_empty_uses_fallback() -> None:
    pilot = FakePilot()
    planner = FakePlanner([])
    rt = PlannedEvalRuntime(
        pilot, planner, fallback_instruction="do the thing"
    )
    plan = rt.begin_episode("task")
    assert plan == ["do the thing"]


def test_begin_episode_resets_state() -> None:
    pilot = FakePilot()
    planner = FakePlanner(["a", "b"])
    cfg = PhaseConfig(steps_per_phase=2)
    rt = PlannedEvalRuntime(pilot, planner, phase_config=cfg)

    img = np.zeros((256, 256, 3), dtype=np.uint8)

    rt.begin_episode("ep1")
    for _ in range(4):
        rt.get_action(img)
    assert rt.current_phase_index == 2  # all phases done

    # Second episode resets
    rt.begin_episode("ep2")
    assert rt.current_phase_index == 0
    assert rt.total_phases == 2


def test_phase_timeout_strategy() -> None:
    pilot = FakePilot()
    planner = FakePlanner(["a", "b"])
    cfg = PhaseConfig(strategy=PhaseStrategy.TIMEOUT, timeout_seconds=0.0)
    rt = PlannedEvalRuntime(pilot, planner, phase_config=cfg)
    rt.begin_episode("task")

    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # With timeout=0.0, every step should advance
    rt.get_action(img)
    assert rt.current_phase_index == 1
    rt.get_action(img)
    assert rt.current_phase_index == 2  # complete


def test_pilot_receives_correct_image() -> None:
    pilot = FakePilot()
    rt = PlannedEvalRuntime(pilot, planner=None)
    rt.begin_episode("task")

    img = np.ones((64, 64, 3), dtype=np.uint8) * 42
    rt.get_action(img)
    np.testing.assert_array_equal(pilot.calls[0][0], img)


# ---------------------------------------------------------------------------
# PlannedEvalRuntime — action shape
# ---------------------------------------------------------------------------

def test_get_action_returns_7dof_array() -> None:
    pilot = FakePilot()
    rt = PlannedEvalRuntime(pilot, planner=None)
    rt.begin_episode("task")

    action = rt.get_action(np.zeros((256, 256, 3), dtype=np.uint8))
    assert action.shape == (7,)
    assert action.dtype == np.float64


# ---------------------------------------------------------------------------
# TextGenerator protocol + LLMPlanner with fake generator
# ---------------------------------------------------------------------------

def test_fake_text_generator_satisfies_protocol() -> None:
    assert isinstance(FakeTextGenerator(), TextGenerator)


def test_llm_planner_uses_text_generator() -> None:
    gen = FakeTextGenerator("1. Reach for cube\n2. Grasp\n3. Lift")
    planner = LLMPlanner(gen)
    steps = planner.plan("pick up the cube")
    assert steps == ["Reach for cube", "Grasp", "Lift"]
    assert gen.call_count == 1


def test_llm_planner_satisfies_planner_protocol() -> None:
    gen = FakeTextGenerator()
    assert isinstance(LLMPlanner(gen), PlannerRuntime)


def test_llm_planner_fallback_on_unparseable_output() -> None:
    gen = FakeTextGenerator("This is not a numbered list at all.")
    planner = LLMPlanner(gen)
    steps = planner.plan("do something")
    assert steps == ["do something"]


def test_llm_planner_passes_instruction_to_generator() -> None:
    calls: list[list[dict[str, str]]] = []

    class CapturingGenerator:
        def generate(self, messages: list[dict[str, str]]) -> str:
            calls.append(messages)
            return "1. Do it"

    planner = LLMPlanner(CapturingGenerator())
    planner.plan("pick up the red cube")
    assert len(calls) == 1
    assert "pick up the red cube" in calls[0][0]["content"]
