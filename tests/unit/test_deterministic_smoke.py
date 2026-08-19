"""End-to-end smoke test for the deterministic brain — no GPU, all fakes.

Simulates the eval loop's contract (``begin_episode`` then repeated
``get_action``) against a fake env, asserting the full dispatch -> compose ->
converge -> safety path runs and produces actions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Import engine first to avoid circular import (same pattern as test_engine.py).
import odyssey.engine  # noqa: F401
from odyssey.runners.agents.brain.runtime import DeterministicBrain
from odyssey.runners.agents.brain.safety import SafetyBoundary
from odyssey.runners.agents.brain.specialist import SpecialistTurn


class _FakeEnv:
    """Yields a fresh RGB frame each step; ends after ``horizon`` steps."""

    def __init__(self, horizon: int) -> None:
        self._horizon = horizon
        self._t = 0

    def reset(self) -> NDArray[np.uint8]:
        self._t = 0
        return self._frame()

    def step(self) -> tuple[NDArray[np.uint8], bool]:
        self._t += 1
        return self._frame(), self._t >= self._horizon

    def _frame(self) -> NDArray[np.uint8]:
        return np.full((16, 16, 3), self._t % 255, dtype=np.uint8)


class _VisionGenerator:
    """A multimodal fake TextGenerator (accepts an ``image`` kwarg)."""

    def __init__(self) -> None:
        self.saw_image = False

    def generate(self, messages: list[dict[str, Any]], image: Any | None = None) -> str:
        self.saw_image = image is not None
        return "the target object is centered; approach slowly"


class _FakePilot:
    def __init__(self) -> None:
        self.instructions: list[str] = []

    def act(self, image: Any, instruction: str) -> NDArray[np.floating[Any]]:
        self.instructions.append(instruction)
        # Deliberately return an out-of-range component to exercise the safety monitor.
        return np.array([2.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)


def test_smoke_dispatch_compose_converge_safety() -> None:
    env = _FakeEnv(horizon=6)
    gen = _VisionGenerator()
    pilot = _FakePilot()
    brain = DeterministicBrain(
        pilot,
        [SpecialistTurn("scene-inspector", gen, is_vision=True)],
        safety=SafetyBoundary(expected_dim=7, mode="monitor"),
    )

    image = env.reset()
    plan = brain.begin_episode("inspect the object and lift it", image)
    assert len(plan) == 1

    # Dispatch engaged the vision specialist and forwarded the image to it.
    assert gen.saw_image is True
    assert "the target object is centered" in brain.current_instruction

    done = False
    steps = 0
    while not done:
        action = brain.get_action(image)
        assert action.shape == (7,)
        # Monitor-only: the out-of-range action is reported but NOT clamped.
        assert float(action[0]) == 2.0
        image, done = env.step()
        steps += 1

    assert steps == 6
    # Every step converged over the same composed instruction.
    assert all(instr == brain.current_instruction for instr in pilot.instructions)
    brain.close()
