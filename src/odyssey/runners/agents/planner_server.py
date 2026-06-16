"""Out-of-process SPECIALIST planner server.

Runs in the *specialist* venv — a modern ``transformers`` that can host an
advanced Gemma (2/3), free of the OpenVLA-pinned ``transformers==4.40.1`` that
constrains the main venv. Loads the planner once, then answers planning
requests over a JSON-lines stdin/stdout protocol:

    <- {"ready": true}                       (once, after the model loads)
    -> {"instruction": "pick up the cube"}   (one request per line, on stdin)
    <- {"plan": ["...", "..."]}              (one response per line, on stdout)
    <- {"error": "..."}                      (on failure; the client falls back)
    -> {"shutdown": true}                    (client asks the server to exit)

**stdout carries ONLY protocol JSON.** Model-loading / log noise is forced to
stderr so it never corrupts the channel; the client (``RemotePlanner``) also
skips any non-JSON stdout line defensively.

The planner logic is reused verbatim from the in-process path:
``LLMPlanner`` (``agents/planner.py``) + ``GemmaTextGenerator``
(``models/gemma.py``). Heavy imports are deferred into ``main()`` so this
module imports cheaply for unit-testing ``serve()``.

Usage (normally launched by ``RemotePlanner``, not by hand):
    python -m odyssey.runners.agents.planner_server \
        --model google/gemma-3-4b-it --quantization int4
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from odyssey.runners.agents.runtime import PlannerRuntime


def _emit(stream: TextIO, obj: dict[str, Any]) -> None:
    """Write one protocol JSON line and flush."""
    stream.write(json.dumps(obj) + "\n")
    stream.flush()


def serve(
    planner: PlannerRuntime,
    instream: TextIO,
    outstream: TextIO,
) -> None:
    """Emit ``{"ready": true}``, then answer one request per stdin line.

    Pure I/O loop over the streams — no model loading here — so it can be
    unit-tested with ``io.StringIO`` and a fake planner.
    """
    _emit(outstream, {"ready": True})
    for raw in instream:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _emit(outstream, {"error": "invalid JSON request"})
            continue
        if not isinstance(req, dict):
            _emit(outstream, {"error": "request must be a JSON object"})
            continue
        if req.get("shutdown"):
            break
        instruction = req.get("instruction")
        if not isinstance(instruction, str):
            _emit(outstream, {"error": "missing 'instruction' string"})
            continue
        try:
            plan = planner.plan(instruction)
            _emit(outstream, {"plan": list(plan)})
        except BaseException as e:
            _emit(outstream, {"error": f"plan failed: {type(e).__name__}: {e}"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Odyssey out-of-process planner server")
    parser.add_argument("--model", required=True, help="HF id of the SPECIALIST model")
    parser.add_argument("--quantization", default=None, help="e.g. int4 (or omit)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    # Keep stdout clean for the protocol: route any model-loading prints to
    # stderr while we import + load the model, then restore.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from odyssey.runners.agents.planner import LLMPlanner
        from odyssey.runners.models.gemma import GemmaTextGenerator

        generator = GemmaTextGenerator(
            args.model,
            quantization=args.quantization,
            max_new_tokens=args.max_new_tokens,
        )
        planner: PlannerRuntime = LLMPlanner(generator)
    except BaseException as e:
        sys.stdout = real_stdout
        _emit(sys.stdout, {"error": f"planner load failed: {type(e).__name__}: {e}"})
        return
    finally:
        sys.stdout = real_stdout

    serve(planner, sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
