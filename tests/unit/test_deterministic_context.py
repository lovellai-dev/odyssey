"""Unit tests for BrainContext composition (the compose seam)."""

from __future__ import annotations

from odyssey.runners.agents.brain.context import Advisory, BrainContext


def test_bare_instruction_when_nothing_to_compose() -> None:
    ctx = BrainContext(instruction="pick up the cube")
    # No identity, no advisories -> byte-identical to a single-agent instruction.
    assert ctx.render_instruction() == "pick up the cube"


def test_advisories_and_identity_compose_in_order() -> None:
    ctx = BrainContext(
        instruction="pick up the cube",
        identity="I am arm-1",
        advisories=[
            Advisory(source="scene", kind="specialist", text="a red cube is left"),
        ],
    )
    out = ctx.render_instruction()
    assert out.index("[ROBOT IDENTITY]") < out.index("[ADVISORY — scene]")
    assert out.index("[ADVISORY — scene]") < out.index("[TASK]")
    assert "a red cube is left" in out
    assert out.rstrip().endswith("pick up the cube")


def test_empty_advisory_text_is_skipped() -> None:
    ctx = BrainContext(
        instruction="go",
        advisories=[Advisory(source="s", kind="specialist", text="")],
    )
    # The only advisory is empty -> nothing to compose -> bare instruction.
    assert ctx.render_instruction() == "go"
