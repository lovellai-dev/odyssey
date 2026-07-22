"""Out-of-process SPECIALIST planner server.

Runs in the *specialist* venv — a modern ``transformers`` + ``torchvision``
that can host the multimodal Gemma 4 planner, free of the OpenVLA-pinned
``transformers==4.40.1`` that constrains the main venv. Loads the planner once,
then answers planning requests over a JSON-lines stdin/stdout protocol:

    <- {"ready": true}                       (once, after the model loads)
    -> {"instruction": "pick up the cube"}   (one request per line, on stdin)
    -> {"instruction": "...", "image": "<base64 PNG>"}   (multimodal request)
    <- {"plan": ["...", "..."]}              (one response per line, on stdout)
    -> {"check": {"instruction": "...", "image": "<base64 PNG>"}}  (completion check)
    <- {"done": true|false}                  (is the sub-instruction satisfied?)
    -> {"ground": {"query": "...", "image": "<base64 PNG>"}}  (target grounding)
    <- {"target": "..."}                     (scene-grounded phrase for the query)
    -> {"route": {"task": "...", "history": [...], "image": "<base64 PNG>"}}  (next sub-task)
    <- {"subtask": "...", "done": false}     (the ORCHESTRATOR's next sub-instruction)
    <- {"error": "..."}                      (on failure; the client falls back)
    -> {"shutdown": true}                    (client asks the server to exit)

**stdout carries ONLY protocol JSON.** Model-loading / log noise is forced to
stderr so it never corrupts the channel; the client (``RemotePlanner``) also
skips any non-JSON stdout line defensively.

The planner logic reuses ``LLMPlanner`` (``agents/planner.py``) on top of the
multimodal ``GemmaVLMGenerator`` (``models/gemma_vlm.py``). Heavy imports are
deferred into ``main()`` so this module imports cheaply for unit-testing
``serve()``.

Usage (normally launched by ``RemotePlanner``, not by hand):
    python -m odyssey.runners.agents.planner_server \
        --model google/gemma-4-E2B-it --quantization int4
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


def _decode_image(data: str) -> Any:
    """Decode a base64 PNG string into a PIL Image (deferred PIL import)."""
    import base64
    import io

    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


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
        check = req.get("check")
        if isinstance(check, dict):
            try:
                check_instruction = check.get("instruction")
                if not isinstance(check_instruction, str):
                    _emit(outstream, {"error": "check missing 'instruction' string"})
                    continue
                checker = getattr(planner, "check_done", None)
                if not callable(checker):
                    _emit(outstream, {"error": "check unsupported"})
                    continue
                check_image = None
                raw_check_image = check.get("image")
                if isinstance(raw_check_image, str):
                    check_image = _decode_image(raw_check_image)
                done = bool(checker(check_instruction, check_image))
                _emit(outstream, {"done": done, "raw": getattr(planner, "_last_check_raw", None)})
            except BaseException as e:
                _emit(outstream, {"error": f"check failed: {type(e).__name__}: {e}"})
            continue
        grounding = req.get("ground")
        if isinstance(grounding, dict):
            try:
                query = grounding.get("query")
                if not isinstance(query, str):
                    _emit(outstream, {"error": "ground missing 'query' string"})
                    continue
                grounder = getattr(planner, "ground", None)
                if not callable(grounder):
                    _emit(outstream, {"error": "ground unsupported"})
                    continue
                ground_image = None
                raw_ground_image = grounding.get("image")
                if isinstance(raw_ground_image, str):
                    ground_image = _decode_image(raw_ground_image)
                _emit(outstream, {"target": str(grounder(query, ground_image))})
            except BaseException as e:
                _emit(outstream, {"error": f"ground failed: {type(e).__name__}: {e}"})
            continue
        routing = req.get("route")
        if isinstance(routing, dict):
            try:
                task = routing.get("task")
                if not isinstance(task, str):
                    _emit(outstream, {"error": "route missing 'task' string"})
                    continue
                router = getattr(planner, "route", None)
                if not callable(router):
                    _emit(outstream, {"error": "route unsupported"})
                    continue
                route_image = None
                raw_route_image = routing.get("image")
                if isinstance(raw_route_image, str):
                    route_image = _decode_image(raw_route_image)
                hist = routing.get("history")
                history = [str(h) for h in hist] if isinstance(hist, list) else []
                decision = router(task, route_image, history)
                _emit(outstream, {"subtask": str(decision.subtask), "done": bool(decision.done)})
            except BaseException as e:
                _emit(outstream, {"error": f"route failed: {type(e).__name__}: {e}"})
            continue
        instruction = req.get("instruction")
        if not isinstance(instruction, str):
            _emit(outstream, {"error": "missing 'instruction' string"})
            continue
        try:
            image = None
            raw_image = req.get("image")
            if isinstance(raw_image, str):
                image = _decode_image(raw_image)
            plan = planner.plan(instruction, image)
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
        from odyssey.runners.models.gemma_vlm import GemmaVLMGenerator

        generator = GemmaVLMGenerator(
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
