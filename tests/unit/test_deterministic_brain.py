"""Unit tests for DeterministicBrain (dispatch -> compose -> converge)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Import engine first to avoid circular import (same pattern as test_engine.py).
import odyssey.engine  # noqa: F401
from odyssey.runners.agents.brain.runtime import DeterministicBrain
from odyssey.runners.agents.brain.safety import SafetyBoundary
from odyssey.runners.agents.brain.specialist import SpecialistTurn
from odyssey.runners.agents.runtime import PilotRuntime


class FakePilot:
    """Records every (image, instruction) call and returns a fixed action."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, str]] = []

    def act(self, image: Any, instruction: str) -> NDArray[np.floating[Any]]:
        self.calls.append((image, instruction))
        return np.zeros(7, dtype=np.float64)


class FakeTextGenerator:
    """Returns canned advisory text; records how many times it ran."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.call_count = 0

    def generate(self, messages: list[dict[str, Any]]) -> str:
        self.call_count += 1
        return self._response


def _image() -> NDArray[np.uint8]:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_fake_pilot_satisfies_protocol() -> None:
    assert isinstance(FakePilot(), PilotRuntime)


def test_engaged_specialist_advisory_composes_into_pilot_instruction() -> None:
    pilot = FakePilot()
    gen = FakeTextGenerator("a red cube is on the left")
    spec = SpecialistTurn("scene-describer", gen, is_vision=True)
    brain = DeterministicBrain(pilot, [spec], safety=SafetyBoundary())

    plan = brain.begin_episode("describe the scene", _image())
    assert plan == ["describe the scene"]  # single convergent "phase"

    # Specialist ran exactly once (compose-once-per-episode).
    assert gen.call_count == 1
    composed = brain.current_instruction
    assert "[ADVISORY — scene-describer]" in composed
    assert "a red cube is on the left" in composed
    assert composed.rstrip().endswith("describe the scene")

    action = brain.get_action(_image())
    assert action.shape == (7,)
    # Pilot converged over the composed instruction, not the raw task.
    assert pilot.calls[-1][1] == composed


def test_compose_once_then_reused_across_steps() -> None:
    pilot = FakePilot()
    gen = FakeTextGenerator("advice")
    brain = DeterministicBrain(pilot, [SpecialistTurn("scene", gen, is_vision=True)])

    brain.begin_episode("do the thing", _image())
    for _ in range(5):
        brain.get_action(_image())

    assert gen.call_count == 1          # composed once for the whole episode
    assert len(pilot.calls) == 5        # pilot ran every step


def test_no_engagement_yields_bare_instruction() -> None:
    pilot = FakePilot()
    gen = FakeTextGenerator("unused")
    # No text signal, no image -> selective dispatch engages nobody.
    brain = DeterministicBrain(pilot, [SpecialistTurn("mapper", gen, is_vision=False)])

    brain.begin_episode("say hello", image=None)
    assert gen.call_count == 0
    assert brain.current_instruction == "say hello"

    brain.get_action(_image())
    assert pilot.calls[-1][1] == "say hello"


def test_close_is_safe() -> None:
    brain = DeterministicBrain(FakePilot(), [SpecialistTurn("s", FakeTextGenerator("x"), is_vision=False)])
    brain.close()  # no out-of-process resources -> no-op, must not raise
