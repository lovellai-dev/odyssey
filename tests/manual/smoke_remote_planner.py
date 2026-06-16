"""Manual smoke test for the out-of-process SPECIALIST planner.

Validates that ``RemotePlanner`` launches the planner server in the *specialist*
venv, loads an advanced Gemma there, and returns a decomposition — WITHOUT any
training, OpenVLA, or simulator. Exercises the real cross-venv/cross-process
path that the unit tests fake.

Requirements:
  * A specialist venv with the [specialist] extra installed (modern transformers
    + Gemma deps), and odyssey importable in it:
        python -m venv ~/specialist-venv
        ~/specialist-venv/bin/pip install -e ".[specialist]" -c constraints/specialist-known-good.txt
  * export ODYSSEY_SPECIALIST_PYTHON=~/specialist-venv/bin/python
  * HF auth (Gemma 4 is Apache-2.0, no gating).

Usage (from the odyssey repo root, MAIN venv active):
    python tests/manual/smoke_remote_planner.py
    python tests/manual/smoke_remote_planner.py --model google/gemma-4-E4B-it "stack the blocks"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction", nargs="?", default="pick up the red cube")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--quantization", default="int4")
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Launch the vision-language server (GemmaVLMGenerator). Required "
        "for Gemma 4 — the text path uses AutoModelForCausalLM, which doesn't "
        "fit the multimodal E-models. (This smoke sends no image, so the VLM "
        "just runs text-only here; the mission exercises the image path.)",
    )
    args = parser.parse_args()

    specialist_python = os.getenv("ODYSSEY_SPECIALIST_PYTHON")
    if not specialist_python:
        print(
            "ERROR: set ODYSSEY_SPECIALIST_PYTHON to the specialist venv's python, e.g.\n"
            "  export ODYSSEY_SPECIALIST_PYTHON=~/specialist-venv/bin/python",
            file=sys.stderr,
        )
        sys.exit(2)

    # Import engine first to break the pre-existing engine<->runners circular
    # import (importing anything under odyssey.runners triggers the cycle).
    import odyssey.engine  # noqa: F401
    from odyssey.runners.agents.remote_planner import RemotePlanner

    print(f"\n=== Launching SPECIALIST out-of-process: {args.model} "
          f"(multimodal={args.multimodal}) via {specialist_python} ===", flush=True)
    planner = RemotePlanner(
        args.model,
        args.quantization,
        python_path=specialist_python,
        multimodal=args.multimodal,
    )
    try:
        print(f"\n=== Decomposing: {args.instruction!r} ===", flush=True)
        steps = planner.plan(args.instruction)
        print(f"\n--- Plan: {len(steps)} sub-instruction(s) ---")
        for i, step in enumerate(steps):
            print(f"  phase {i}: {step}")
        print()
        if steps == [args.instruction]:
            print("NOTE: single-step fallback — the server failed or returned no plan. "
                  "Check the server's stderr above (model load / auth / transformers version).")
    finally:
        planner.close()


if __name__ == "__main__":
    main()
