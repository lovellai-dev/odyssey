"""Diagnostic: dump the RAW Gemma 3 multimodal output (no plan parsing).

When the multimodal smoke test falls back to a single step, it means the
model generated text that ``LLMPlanner._parse_plan`` couldn't read as a
numbered list. This script bypasses the planner entirely and prints the
*literal* model output (``repr``) so we can see exactly what Gemma 3
produced — to decide whether it's a prompt issue, a markdown/format issue
(``**1.**``, bullets), or an empty/EOS decode problem.

REQUIRES the specialist venv (modern transformers >=4.50):
    ~/specialist-venv/bin/python tests/manual/debug_multimodal_raw.py
    ~/specialist-venv/bin/python tests/manual/debug_multimodal_raw.py "stack the blocks"

Footprint: ~5 GB VRAM (Gemma 3 4B int4 incl. SigLIP vision tower).
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

    # Same import order as the smoke (breaks the engine<->runners cycle).
    import odyssey.engine  # noqa: F401
    from odyssey.runners.models.gemma_vlm import GemmaVLMGenerator

    print("\n=== Loading google/gemma-3-4b-it (int4, multimodal) ===", flush=True)
    gen = GemmaVLMGenerator("google/gemma-3-4b-it", quantization="int4")

    image = _demo_image()
    messages = [
        {
            "role": "user",
            "content": (
                "You are a robot task planner. Given the scene image and a task, "
                "decompose it into a numbered list of simple sequential "
                "sub-instructions (1., 2., 3. ...). Output ONLY the numbered "
                f"list.\n\nTask: {instruction}"
            ),
        }
    ]

    print(f"\n=== Generating (with scene image): {instruction!r} ===", flush=True)
    raw = gen.generate(messages, image=image)

    print("\n=== RAW OUTPUT START ===")
    print(repr(raw))
    print("=== RAW OUTPUT END ===")
    print(f"\nlen={len(raw)}  stripped_len={len(raw.strip())}")
    print("first 5 lines:")
    for i, line in enumerate(raw.strip().splitlines()[:5]):
        print(f"  [{i}] {line!r}")


if __name__ == "__main__":
    main()
