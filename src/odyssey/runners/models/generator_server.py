"""Out-of-process, prompt-agnostic text-generation server.

Runs in the *specialist* venv — a modern ``transformers`` + ``torchvision`` that
can host the multimodal Gemma 4 generator, free of the OpenVLA-pinned
``transformers==4.40.1`` that constrains the main venv. This is INFRASTRUCTURE,
not brain: it hosts a ``TextGenerator`` and knows nothing about tasks, plans, or
advisories — the caller sends fully-composed messages, the server returns text.

JSON-lines stdin/stdout protocol:

    <- {"ready": true}                              (once, after the model loads)
    -> {"messages": [...], "image": "<base64 PNG>"} (one request per line)
    <- {"text": "<generated text>"}                 (one response per line)
    <- {"error": "..."}                             (on failure; client falls back)
    -> {"shutdown": true}                           (client asks the server to exit)

**stdout carries ONLY protocol JSON.** Model-loading / log noise is forced to
stderr. Heavy imports are deferred into ``main()`` so this module imports
cheaply for unit-testing ``serve()`` with ``io.StringIO`` and a fake generator.

Usage (normally launched by ``RemoteGenerator``, not by hand):
    python -m odyssey.runners.models.generator_server \
        --model google/gemma-4-E2B-it --quantization int4
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from odyssey.runners.agents.runtime import TextGenerator


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
    generator: TextGenerator,
    instream: TextIO,
    outstream: TextIO,
) -> None:
    """Emit ``{"ready": true}``, then answer one request per stdin line.

    Pure I/O loop over the streams — no model loading here — so it can be
    unit-tested with ``io.StringIO`` and a fake generator.
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
        messages = req.get("messages")
        if not isinstance(messages, list):
            _emit(outstream, {"error": "missing 'messages' list"})
            continue
        try:
            raw_image = req.get("image")
            if isinstance(raw_image, str):
                text = generator.generate(messages, image=_decode_image(raw_image))  # type: ignore[call-arg]
            else:
                text = generator.generate(messages)
            _emit(outstream, {"text": str(text)})
        except BaseException as e:
            _emit(outstream, {"error": f"generate failed: {type(e).__name__}: {e}"})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Odyssey out-of-process text-generation server"
    )
    parser.add_argument("--model", required=True, help="HF id of the SPECIALIST model")
    parser.add_argument("--quantization", default=None, help="e.g. int4 (or omit)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    # Keep stdout clean for the protocol: route any model-loading prints to
    # stderr while we import + load the model, then restore.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from odyssey.runners.models.gemma_vlm import GemmaVLMGenerator

        generator: TextGenerator = GemmaVLMGenerator(
            args.model,
            quantization=args.quantization,
            max_new_tokens=args.max_new_tokens,
        )
    except BaseException as e:
        sys.stdout = real_stdout
        _emit(sys.stdout, {"error": f"generator load failed: {type(e).__name__}: {e}"})
        return
    finally:
        sys.stdout = real_stdout

    serve(generator, sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
