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
from odyssey.runners.agents.delegated import (
    PICK_PLACE_TEMPLATE,
    DelegatedEvalRuntime,
    DelegationConfig,
)
from odyssey.runners.agents.planned import (
    PhaseConfig,
    PhaseStrategy,
    PlannedEvalRuntime,
    _PhaseState,
)
from odyssey.runners.agents.planner import (
    LLMPlanner,
    _parse_grounding,
    _parse_plan,
    _parse_yes_no,
)
from odyssey.runners.agents.runtime import (
    CompletionDetector,
    GroundingProvider,
    PilotRuntime,
    PlannerRuntime,
    TextGenerator,
)

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
    """Returns a fixed plan. Records the image it was handed (if any)."""

    def __init__(self, steps: list[str]) -> None:
        self._steps = steps
        self.call_count = 0
        self.last_image: object = None

    def plan(self, task_instruction: str, image: object = None) -> list[str]:
        self.call_count += 1
        self.last_image = image
        return list(self._steps)


class FakeDetector:
    """Completion detector stub. Returns True from the Nth call onward.

    ``done_after=None`` never confirms; records each (instruction, image) call.
    """

    def __init__(self, done_after: int | None = None) -> None:
        self._done_after = done_after
        self.calls: list[tuple[str, Any]] = []

    def check_done(self, instruction: str, image: Any) -> bool:
        self.calls.append((instruction, image))
        return self._done_after is not None and len(self.calls) >= self._done_after


class FakePlannerWithCheck(FakePlanner):
    """A planner that also satisfies CompletionDetector (like RemotePlanner)."""

    def check_done(self, instruction: str, image: Any) -> bool:
        return False


class FakeVisionGenerator:
    """Multimodal TextGenerator stub — its generate() accepts an image."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[list[dict[str, str]], Any]] = []

    def generate(self, messages: list[dict[str, str]], image: Any = None) -> str:
        self.calls.append((messages, image))
        return self._response


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


def test_begin_episode_forwards_image_to_planner() -> None:
    pilot = FakePilot()
    planner = FakePlanner(["a", "b"])
    rt = PlannedEvalRuntime(pilot, planner)

    img = np.ones((48, 48, 3), dtype=np.uint8)
    rt.begin_episode("task", img)
    np.testing.assert_array_equal(planner.last_image, img)


def test_begin_episode_image_defaults_to_none() -> None:
    pilot = FakePilot()
    planner = FakePlanner(["a"])
    rt = PlannedEvalRuntime(pilot, planner)

    rt.begin_episode("task")
    assert planner.last_image is None


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


# ---------------------------------------------------------------------------
# COMPLETION_GATED strategy
# ---------------------------------------------------------------------------

def _gated_cfg(**kw: Any) -> PhaseConfig:
    return PhaseConfig(strategy=PhaseStrategy.COMPLETION_GATED, **kw)


def test_completion_gated_advances_on_detector() -> None:
    pilot = FakePilot()
    planner = FakePlanner(["phase-1", "phase-2"])
    detector = FakeDetector(done_after=1)  # True on its first poll
    cfg = _gated_cfg(check_every=2, max_steps_per_phase=100)
    rt = PlannedEvalRuntime(pilot, planner, phase_config=cfg, detector=detector)
    rt.begin_episode("task")
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    rt.get_action(img)  # step 1: off-cadence, detector NOT polled
    assert rt.current_phase_index == 0
    assert detector.calls == []

    rt.get_action(img)  # step 2: on-cadence, detector polled -> True -> advance
    assert rt.current_phase_index == 1
    assert len(detector.calls) == 1


def test_completion_gated_advances_at_cap_when_never_done() -> None:
    pilot = FakePilot()
    planner = FakePlanner(["phase-1", "phase-2"])
    detector = FakeDetector(done_after=None)  # never confirms
    cfg = _gated_cfg(check_every=2, max_steps_per_phase=6)
    rt = PlannedEvalRuntime(pilot, planner, phase_config=cfg, detector=detector)
    rt.begin_episode("task")
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    for _ in range(5):
        rt.get_action(img)
    assert rt.current_phase_index == 0  # not yet at cap
    rt.get_action(img)  # step 6 hits the cap
    assert rt.current_phase_index == 1
    # Cap is checked before the detector, so step 6 does not poll: only steps 2 & 4.
    assert len(detector.calls) == 2
    events = rt.drain_phase_events()
    assert events == [
        {"from": 0, "to": 1, "instruction": "phase-2", "reason": "cap"}
    ]


def test_completion_gated_no_detector_falls_back_to_cap() -> None:
    pilot = FakePilot()
    planner = FakePlanner(["phase-1", "phase-2"])  # no check_done -> no detector
    cfg = _gated_cfg(check_every=2, max_steps_per_phase=3)
    rt = PlannedEvalRuntime(pilot, planner, phase_config=cfg)
    assert rt._detector is None
    rt.begin_episode("task")
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    for _ in range(2):
        rt.get_action(img)
    assert rt.current_phase_index == 0
    rt.get_action(img)  # step 3 == cap
    assert rt.current_phase_index == 1
    assert rt.drain_phase_events()[0]["reason"] == "cap"


def test_completion_gated_detector_gets_current_frame_and_instruction() -> None:
    pilot = FakePilot()
    planner = FakePlanner(["grasp the cube", "lift"])
    detector = FakeDetector(done_after=None)
    cfg = _gated_cfg(check_every=1, max_steps_per_phase=100)
    rt = PlannedEvalRuntime(pilot, planner, phase_config=cfg, detector=detector)
    rt.begin_episode("task")

    img = np.ones((8, 8, 3), dtype=np.uint8) * 7
    rt.get_action(img)
    assert detector.calls[0][0] == "grasp the cube"
    np.testing.assert_array_equal(detector.calls[0][1], img)


def test_detector_auto_discovered_from_planner() -> None:
    rt = PlannedEvalRuntime(FakePilot(), FakePlannerWithCheck(["a", "b"]))
    assert rt._detector is not None


def test_no_detector_when_planner_lacks_check_done() -> None:
    rt = PlannedEvalRuntime(FakePilot(), FakePlanner(["a", "b"]))
    assert rt._detector is None


def test_drain_phase_events_empties_the_buffer() -> None:
    pilot = FakePilot()
    planner = FakePlanner(["a", "b"])
    cfg = PhaseConfig(strategy=PhaseStrategy.FIXED_STEPS, steps_per_phase=1)
    rt = PlannedEvalRuntime(pilot, planner, phase_config=cfg)
    rt.begin_episode("task")
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    rt.get_action(img)  # advances immediately (steps_per_phase=1)
    first = rt.drain_phase_events()
    assert first and first[0]["reason"] == "fixed_steps"
    assert rt.drain_phase_events() == []  # drained


def test_fixed_steps_emits_no_events_mid_phase() -> None:
    pilot = FakePilot()
    planner = FakePlanner(["a", "b"])
    cfg = PhaseConfig(strategy=PhaseStrategy.FIXED_STEPS, steps_per_phase=5)
    rt = PlannedEvalRuntime(pilot, planner, phase_config=cfg)
    rt.begin_episode("task")
    rt.get_action(np.zeros((8, 8, 3), dtype=np.uint8))
    assert rt.drain_phase_events() == []


# ---------------------------------------------------------------------------
# _parse_yes_no + LLMPlanner.check_done
# ---------------------------------------------------------------------------

def test_parse_yes_no_true_cases() -> None:
    for text in ["YES", "yes", "Yes.", "  yes it is done", "y", "**YES**"]:
        assert _parse_yes_no(text) is True, text


def test_parse_yes_no_false_cases() -> None:
    for text in ["NO", "no", "", "maybe", "not yet", "almost yes"]:
        assert _parse_yes_no(text) is False, text


def test_llm_planner_check_done_yes() -> None:
    planner = LLMPlanner(FakeVisionGenerator("YES"))
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    assert planner.check_done("grasp the cube", img) is True


def test_llm_planner_check_done_no() -> None:
    planner = LLMPlanner(FakeVisionGenerator("NO"))
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    assert planner.check_done("grasp the cube", img) is False


def test_llm_planner_check_done_false_without_image() -> None:
    planner = LLMPlanner(FakeVisionGenerator("YES"))
    assert planner.check_done("grasp", None) is False


def test_llm_planner_check_done_false_for_text_only_generator() -> None:
    # FakeTextGenerator.generate has no image param -> can't verify -> False.
    planner = LLMPlanner(FakeTextGenerator("YES"))
    assert planner.check_done("grasp", np.zeros((8, 8, 3), dtype=np.uint8)) is False


def test_llm_planner_satisfies_completion_detector() -> None:
    assert isinstance(LLMPlanner(FakeVisionGenerator("YES")), CompletionDetector)


# ---------------------------------------------------------------------------
# Grounding — LLMPlanner.ground() + _parse_grounding
# ---------------------------------------------------------------------------

def test_parse_grounding_first_nonempty_line() -> None:
    assert _parse_grounding("the red mug on the left") == "the red mug on the left"


def test_parse_grounding_strips_markers_and_quotes() -> None:
    assert _parse_grounding('- "the top drawer".') == "the top drawer"
    assert _parse_grounding("1. **the black bowl**") == "the black bowl"


def test_parse_grounding_skips_blank_lines() -> None:
    assert _parse_grounding("\n\n  the blue plate\nextra") == "the blue plate"


def test_parse_grounding_empty() -> None:
    assert _parse_grounding("") == ""
    assert _parse_grounding("   \n  ") == ""


def test_llm_planner_ground_returns_parsed_phrase() -> None:
    gen = FakeVisionGenerator("the red cube near the plate")
    planner = LLMPlanner(gen)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    assert planner.ground("the object to pick up", img) == "the red cube near the plate"
    # The scene image is forwarded to the multimodal generator.
    assert gen.calls[-1][1] is img


def test_llm_planner_ground_echoes_query_without_image() -> None:
    planner = LLMPlanner(FakeVisionGenerator("ignored"))
    assert planner.ground("the target", None) == "the target"


def test_llm_planner_ground_echoes_query_for_text_only_generator() -> None:
    planner = LLMPlanner(FakeTextGenerator("ignored"))
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    assert planner.ground("the target", img) == "the target"


def test_llm_planner_ground_echoes_query_on_empty_output() -> None:
    planner = LLMPlanner(FakeVisionGenerator("   "))
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    assert planner.ground("the target", img) == "the target"


def test_llm_planner_satisfies_grounding_provider() -> None:
    assert isinstance(LLMPlanner(FakeVisionGenerator("x")), GroundingProvider)


# ---------------------------------------------------------------------------
# DelegatedEvalRuntime — delegation-driven orchestrator
# ---------------------------------------------------------------------------

class FakeGrounder:
    """Grounding provider stub. Records queries; returns a fixed target.

    No ``check_done`` — models a grounder that only grounds (hand-back then
    falls back to the step cap).
    """

    def __init__(self, target: str = "the red cube", *, raises: bool = False) -> None:
        self.target = target
        self.ground_calls: list[tuple[str, Any]] = []
        self._raises = raises

    def ground(self, target_query: str, image: Any) -> str:
        self.ground_calls.append((target_query, image))
        if self._raises:
            raise RuntimeError("grounding boom")
        return self.target


class FakeGrounderWithCheck(FakeGrounder):
    """Grounder that also satisfies CompletionDetector (like RemotePlanner).

    ``check_done`` returns True from the ``done_after``-th call onward.
    """

    def __init__(
        self, target: str = "the red cube", *, done_after: int | None = None
    ) -> None:
        super().__init__(target)
        self._done_after = done_after
        self.check_calls: list[tuple[str, Any]] = []

    def check_done(self, instruction: str, image: Any) -> bool:
        self.check_calls.append((instruction, image))
        return self._done_after is not None and len(self.check_calls) >= self._done_after


_IMG = np.zeros((4, 4, 3), dtype=np.uint8)


def test_fake_grounder_satisfies_protocol() -> None:
    assert isinstance(FakeGrounder(), GroundingProvider)


def test_delegated_begin_episode_returns_phase_labels() -> None:
    rt = DelegatedEvalRuntime(FakePilot(), FakeGrounder())
    assert rt.begin_episode("task") == ["pick", "place"]
    assert rt.total_phases == 2
    assert rt.current_phase_index == 0


def test_delegated_grounds_and_composes_pick_instruction() -> None:
    pilot = FakePilot()
    grounder = FakeGrounder(target="the blue mug")
    rt = DelegatedEvalRuntime(pilot, grounder)
    rt.begin_episode("put the mug on the plate")
    rt.get_action(_IMG)
    # First phase (pick) grounded once, spliced into the pilot instruction.
    assert len(grounder.ground_calls) == 1
    assert pilot.calls[0][1] == "pick up the blue mug"


def test_delegated_hand_back_on_completion_and_regrounds() -> None:
    pilot = FakePilot()
    grounder = FakeGrounderWithCheck(target="the mug", done_after=1)
    rt = DelegatedEvalRuntime(
        pilot, grounder, config=DelegationConfig(check_every=2, max_steps_per_phase=100)
    )
    rt.begin_episode("put the mug on the plate")

    rt.get_action(_IMG)  # step 1: ground pick, act; no check yet (n=1)
    assert rt.current_phase_index == 0
    rt.get_action(_IMG)  # step 2: n=2 -> check fires -> hand-back to place
    assert rt.current_phase_index == 1
    rt.get_action(_IMG)  # step 3: re-ground place, act
    assert pilot.calls[-1][1] == "place it at the mug"
    assert len(grounder.ground_calls) == 2  # exactly one grounding per phase


def test_delegated_cap_hand_back_without_detector() -> None:
    pilot = FakePilot()
    grounder = FakeGrounder(target="the cube")  # no check_done
    rt = DelegatedEvalRuntime(
        pilot, grounder, config=DelegationConfig(check_every=10, max_steps_per_phase=3)
    )
    rt.begin_episode("task")
    for _ in range(3):
        rt.get_action(_IMG)
    # Cap reached at step 3 -> hand-back even without a detector.
    assert rt.current_phase_index == 1


def test_delegated_auto_adopts_grounder_as_detector() -> None:
    # No detector passed; the grounder exposes check_done -> semantic hand-back
    # works (proves auto-adoption).
    pilot = FakePilot()
    grounder = FakeGrounderWithCheck(target="x", done_after=1)
    rt = DelegatedEvalRuntime(
        pilot, grounder, config=DelegationConfig(check_every=1, max_steps_per_phase=100)
    )
    rt.begin_episode("task")
    rt.get_action(_IMG)  # n=1 -> check -> hand-back
    assert rt.current_phase_index == 1
    assert grounder.check_calls  # detector was consulted


def test_delegated_phase_events_schema() -> None:
    pilot = FakePilot()
    grounder = FakeGrounderWithCheck(target="the cube", done_after=1)
    rt = DelegatedEvalRuntime(
        pilot, grounder, config=DelegationConfig(check_every=1, max_steps_per_phase=100)
    )
    rt.begin_episode("task")
    rt.get_action(_IMG)  # grounding event, then completion hand-back event
    events = rt.drain_phase_events()

    assert [e["reason"] for e in events] == ["grounding", "completion"]
    assert events[0]["capability"] == "grounding"
    assert events[0]["instruction"] == "pick up the cube"
    assert events[0]["skill"] == "pick"
    assert events[1]["capability"] == "handback"
    assert events[1]["from"] == 0 and events[1]["to"] == 1
    # Drain clears the buffer.
    assert rt.drain_phase_events() == []


def test_delegated_grounding_failure_falls_back_to_query() -> None:
    pilot = FakePilot()
    grounder = FakeGrounder(raises=True)
    rt = DelegatedEvalRuntime(pilot, grounder, task_fallback="do it")
    rt.begin_episode("stack the blocks")
    rt.get_action(_IMG)  # grounder raises -> falls back to the query, never crashes
    instr = pilot.calls[0][1]
    assert instr.startswith("pick up ")
    assert "stack the blocks" in instr


def test_delegated_holds_after_all_phases() -> None:
    pilot = FakePilot()
    grounder = FakeGrounderWithCheck(target="the cube", done_after=1)
    rt = DelegatedEvalRuntime(
        pilot, grounder, config=DelegationConfig(check_every=1, max_steps_per_phase=100)
    )
    rt.begin_episode("task")
    for _ in range(5):
        action = rt.get_action(_IMG)
    # Both phases handed back; runtime keeps returning valid actions.
    assert rt.current_phase_index >= rt.total_phases
    assert action.shape == (7,)


def test_delegation_config_from_config_reads_keys() -> None:
    cfg = DelegationConfig.from_config({"phase_check_every": 5, "phase_max_steps": 20})
    assert cfg.check_every == 5
    assert cfg.max_steps_per_phase == 20
    assert cfg.skills == PICK_PLACE_TEMPLATE


def test_delegation_config_defaults() -> None:
    cfg = DelegationConfig.from_config({})
    assert cfg.check_every == 10
    assert cfg.max_steps_per_phase == 100


def test_delegated_close_calls_grounder_close() -> None:
    class ClosableGrounder(FakeGrounder):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    grounder = ClosableGrounder()
    rt = DelegatedEvalRuntime(FakePilot(), grounder)
    rt.close()
    assert grounder.closed is True
