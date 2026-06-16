"""Standalone smoke test for the MULTIMODAL SPECIALIST (Gemma 4 vision-language).

Loads the vision-language Gemma 4 planner, feeds it a synthetic scene image
plus a high-level instruction, and decomposes it into sub-steps — WITHOUT any
training, checkpoint, OpenVLA, or simulator. This isolates the genuinely-new
multimodal piece (image-grounded task decomposition) and reports peak VRAM so
you can confirm the budget on a real 24 GB card.

Footprint: ~9.3 GB VRAM (Gemma 4 E4B-it int4).

REQUIRES the specialist venv (modern transformers + torchvision):
    python -m venv ~/specialist-venv
    ~/specialist-venv/bin/pip install -e ".[specialist]" -c constraints/specialist-known-good.txt

Usage (run with the SPECIALIST venv's python, HF auth done):
    ~/specialist-venv/bin/python tests/manual/smoke_multimodal_planner.py
    ~/specialist-venv/bin/python tests/manual/smoke_multimodal_planner.py "stack the blocks"
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def _demo_image() -> object:
    """A small synthetic RGB scene (a red square on grey) as a PIL Image."""
    import numpy as np
    from PIL import Image

    arr = np.full((256, 256, 3), 128, dtype=np.uint8)
    arr[150:210, 90:150] = (200, 30, 30)  # a red "cube"
    return Image.fromarray(arr)


def main() -> None:
    instruction = sys.argv[1] if len(sys.argv) > 1 else "pick up the red cube"

    # Import engine first to break the engine<->runners circular import (see
    # smoke_planner.py for the full explanation).
    import odyssey.engine  # noqa: F401
    from odyssey.runners.agents.planner import LLMPlanner
    from odyssey.runners.models.gemma_vlm import GemmaVLMGenerator

    print("\n=== Loading SPECIALIST: google/gemma-4-E4B-it (int4, multimodal) ===", flush=True)
    generator = GemmaVLMGenerator("google/gemma-4-E4B-it", quantization="int4")
    planner = LLMPlanner(generator)

    image = _demo_image()
    print(f"\n=== Decomposing (with scene image): {instruction!r} ===", flush=True)
    steps = planner.plan(instruction, image=image)

    print(f"\n--- Plan: {len(steps)} sub-instruction(s) ---")
    for i, step in enumerate(steps):
        print(f"  phase {i}: {step}")

    try:
        import torch

        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"\n--- Peak VRAM (SPECIALIST only): {peak_gb:.2f} GB ---")
            print("    Budget on 24 GB shared with the ~14 GB bf16 pilot: keep total <~20 GB.")
    except Exception:  # pragma: no cover - diagnostics only
        pass
    print()

    if len(steps) == 1:
        print(
            "NOTE: only one step — either the planner fell back (no parseable "
            "numbered list) or the instruction was already atomic. Check the "
            "INFO log above for the raw Gemma output."
        )


if __name__ == "__main__":
    main()
