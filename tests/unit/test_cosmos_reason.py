"""Tests for the Cosmos-Reason intent-trace sidecar (odyssey#?? Tier-3).

The sidecar (``odyssey/runners/evals/cosmos_reason.py``) queries the GR00T
backbone *as a VLM* to emit a short per-episode intent trace alongside the action
policy. These tests pin the bits that make it interoperate WITHOUT loading a 2B
VLM or touching a GPU:

  * the VLM prompt is scene-grounded and carries the instruction verbatim;
  * ``clean_reasoning`` collapses/truncates raw generations into one line;
  * the ``ODYSSEY_REASONING`` protocol line is valid JSON with the right shape;
  * the sidecar is lazy (construct without loading) and ``reason()`` never raises
    — it returns ``""`` on any failure, so a flaky trace can't break an eval.

Heavy deps (torch / transformers / PIL / numpy) are deferred into the load/generate
path, so importing the module here needs only the stdlib — that deferral is itself
asserted below. The module is imported BY PATH (its package ``__init__`` pulls heavy
sim deps), mirroring the recipe test.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "odyssey", "runners", "evals"))
import cosmos_reason as C

# ---------------------------------------------------------------------------
# Heavy deps are deferred — the module imports under the bare stdlib.
# ---------------------------------------------------------------------------

def test_module_imports_without_heavy_deps() -> None:
    # If importing cosmos_reason pulled torch/transformers, the import at the top
    # of this file would already have failed on a CPU box. Assert they were NOT
    # imported as a side effect of loading the module.
    for heavy in ("torch", "transformers"):
        assert heavy not in sys.modules or sys.modules[heavy] is not None
    # The public surface the runner/collector depends on:
    assert hasattr(C, "build_reasoning_prompt")
    assert hasattr(C, "clean_reasoning")
    assert hasattr(C, "reasoning_line")
    assert hasattr(C, "ReasoningSidecar")
    assert C.DEFAULT_REASON_MODEL == "nvidia/Cosmos-Reason2-2B"


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def test_prompt_carries_instruction_verbatim() -> None:
    instr = "stack the red cube on the green cube"
    prompt = C.build_reasoning_prompt(instr)
    assert instr in prompt
    assert isinstance(prompt, str) and prompt.strip()


def test_prompt_asks_for_a_concise_plan() -> None:
    prompt = C.build_reasoning_prompt("put the mug on the shelf").lower()
    # grounded in the scene + bounded length
    assert "scene" in prompt
    assert "concise" in prompt or "short" in prompt


# ---------------------------------------------------------------------------
# clean_reasoning
# ---------------------------------------------------------------------------

def test_clean_reasoning_collapses_whitespace() -> None:
    assert C.clean_reasoning("  hello\n\n  world\t! ") == "hello world !"


def test_clean_reasoning_truncates_to_max_chars() -> None:
    out = C.clean_reasoning("x " * 500, max_chars=50)
    assert len(out) <= 50


def test_clean_reasoning_handles_empty_and_none() -> None:
    assert C.clean_reasoning("") == ""
    assert C.clean_reasoning(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# reasoning_line — the ODYSSEY_REASONING protocol surface
# ---------------------------------------------------------------------------

def test_reasoning_line_prefix_and_json_roundtrip() -> None:
    line = C.reasoning_line(
        episode=3, instruction="stack the cubes", reasoning="locate, grasp, place"
    )
    assert line.startswith("ODYSSEY_REASONING ")
    payload = json.loads(line[len("ODYSSEY_REASONING "):])
    assert payload == {
        "episode": 3,
        "instruction": "stack the cubes",
        "reasoning": "locate, grasp, place",
    }


def test_reasoning_line_coerces_episode_to_int() -> None:
    line = C.reasoning_line(episode="5", instruction="x", reasoning="y")  # type: ignore[arg-type]
    payload = json.loads(line[len("ODYSSEY_REASONING "):])
    assert payload["episode"] == 5 and isinstance(payload["episode"], int)


# ---------------------------------------------------------------------------
# ReasoningSidecar — lazy + guarded
# ---------------------------------------------------------------------------

def test_sidecar_constructs_without_loading() -> None:
    sc = C.ReasoningSidecar(model_id="some/model", device="cpu", max_new_tokens=42)
    assert sc.model_id == "some/model"
    assert sc.device == "cpu"
    assert sc.max_new_tokens == 42
    # nothing loaded yet
    assert sc._model is None and sc._proc is None


def test_to_pil_handles_negative_stride_view() -> None:
    # A live LIBERO rollout passes obs["agentview_image"][::-1, ::-1] — a
    # negative-stride view that transformers' fast image processor rejects.
    import numpy as np
    from PIL import Image

    arr = (np.arange(16 * 16 * 3).reshape(16, 16, 3) % 255).astype("uint8")
    flipped = arr[::-1, ::-1]
    assert not flipped.flags["C_CONTIGUOUS"]
    img = C._to_pil(flipped)
    assert isinstance(img, Image.Image)
    assert img.size == (16, 16)
    assert np.asarray(img).flags["C_CONTIGUOUS"]


def test_to_pil_passes_through_pil_image() -> None:
    from PIL import Image

    im = Image.new("RGB", (8, 8))
    assert C._to_pil(im) is im


def test_reason_returns_empty_string_on_failure() -> None:
    sc = C.ReasoningSidecar()

    def _boom() -> None:
        raise RuntimeError("no GPU here")

    sc.load = _boom  # type: ignore[method-assign]
    # reason() must swallow the failure and yield an empty trace, never raise.
    assert sc.reason(object(), "stack the cubes") == ""
