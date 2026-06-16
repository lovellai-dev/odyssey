"""Standalone smoke test for the multi-agent SPECIALIST (Gemma planner).

Loads the Gemma task planner and decomposes a high-level instruction into
sub-steps — WITHOUT any training, checkpoint, OpenVLA, or simulator. This
isolates the genuinely-new multi-agent piece (planner-driven task
decomposition) so it can be validated even while the training/eval
end-to-end path is blocked.

Footprint: ~1.5 GB VRAM (Gemma 2B int4). No distributed setup needed.

Usage (from the odyssey repo root, venv active, HF auth done):
    python scripts/smoke_planner.py
    python scripts/smoke_planner.py "stack the blue block on the red block"
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main() -> None:
    instruction = sys.argv[1] if len(sys.argv) > 1 else "pick up the red cube"

    # Imported here so --help / import errors surface clearly.
    from odyssey.runners.agents.planner import LLMPlanner
    from odyssey.runners.models.gemma import GemmaTextGenerator

    print("\n=== Loading SPECIALIST: google/gemma-2b-it (int4) ===", flush=True)
    generator = GemmaTextGenerator("google/gemma-2b-it", quantization="int4")
    planner = LLMPlanner(generator)

    print(f"\n=== Decomposing: {instruction!r} ===", flush=True)
    steps = planner.plan(instruction)

    print(f"\n--- Plan: {len(steps)} sub-instruction(s) ---")
    for i, step in enumerate(steps):
        print(f"  phase {i}: {step}")
    print()

    if len(steps) == 1:
        print(
            "NOTE: only one step — either the planner fell back (no parseable "
            "numbered list) or the instruction was already atomic. Check the "
            "INFO log above for the raw Gemma output."
        )


if __name__ == "__main__":
    main()
