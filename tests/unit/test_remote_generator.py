"""Tests for the generic out-of-process TextGenerator.

No GPU / no real model: ``serve()`` is driven with a fake generator over
in-memory streams, and ``RemoteGenerator`` is driven against a tiny fake server
script run as a real subprocess (exercises the real Popen + JSON protocol +
lifecycle). Mirrors ``test_remote_planner.py``.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import odyssey.engine  # noqa: F401 — import engine first to break the runners<->engine cycle
from odyssey.runners.models.generator_server import serve
from odyssey.runners.models.remote_generator import RemoteGenerator


class _FakeGen:
    """TextGenerator stub: returns fixed text; records the image it was handed."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.last_image: object = "unset"

    def generate(self, messages: list[dict[str, Any]], image: object = None) -> str:
        self.last_image = image
        return self._text


def _tiny_png_b64() -> str:
    import base64

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# serve() — the server-side request/response loop
# --------------------------------------------------------------------------- #


def _run_serve(gen: _FakeGen, requests: list[dict[str, object]]) -> list[dict[str, Any]]:
    instream = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    outstream = io.StringIO()
    serve(gen, instream, outstream)
    return [json.loads(line) for line in outstream.getvalue().splitlines() if line.strip()]


def test_serve_emits_ready_then_text() -> None:
    out = _run_serve(
        _FakeGen("the cube is centered"),
        [{"messages": [{"role": "user", "content": "hi"}]}, {"shutdown": True}],
    )
    assert out[0] == {"ready": True}
    assert out[1] == {"text": "the cube is centered"}


def test_serve_rejects_invalid_json() -> None:
    instream = io.StringIO('not-json\n{"shutdown": true}\n')
    outstream = io.StringIO()
    serve(_FakeGen("x"), instream, outstream)
    msgs = [json.loads(x) for x in outstream.getvalue().splitlines() if x.strip()]
    assert msgs[0] == {"ready": True}
    assert any("error" in m for m in msgs)


def test_serve_missing_messages() -> None:
    out = _run_serve(_FakeGen("x"), [{"foo": "bar"}, {"shutdown": True}])
    assert any("error" in m for m in out)


def test_serve_decodes_image_and_passes_to_generator() -> None:
    from PIL import Image

    gen = _FakeGen("ok")
    out = _run_serve(
        gen,
        [{"messages": [{"role": "user", "content": "x"}], "image": _tiny_png_b64()},
         {"shutdown": True}],
    )
    assert out[1] == {"text": "ok"}
    assert isinstance(gen.last_image, Image.Image)
    assert gen.last_image.size == (2, 2)


# --------------------------------------------------------------------------- #
# RemoteGenerator — the client, against a fake server subprocess
# --------------------------------------------------------------------------- #

# Emits stderr + non-JSON stdout noise (client must skip), then the protocol.
_FAKE_SERVER_OK = """
import argparse, json, sys
p = argparse.ArgumentParser()
p.add_argument("--model"); p.add_argument("--quantization"); p.add_argument("--max-new-tokens")
p.parse_args()
sys.stderr.write("specialist: loading model noise\\n"); sys.stderr.flush()
print("non-json noise on stdout that the client must skip")
sys.stdout.write(json.dumps({"ready": True}) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("shutdown"):
        break
    has_img = isinstance(req.get("image"), str) and len(req["image"]) > 0
    n = len(req.get("messages", []))
    sys.stdout.write(json.dumps({"text": "msgs=%d image=%s" % (n, has_img)}) + "\\n")
    sys.stdout.flush()
"""

# Never prints {"ready": true} — exits immediately.
_FAKE_SERVER_DIES = "import sys; sys.exit(1)\n"

# Ready, but returns a non-string text -> client should fall back to "".
_FAKE_SERVER_BAD = """
import json, sys
sys.stdout.write(json.dumps({"ready": True}) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    if line.strip():
        sys.stdout.write(json.dumps({"text": None}) + "\\n"); sys.stdout.flush()
"""


def _generator_for(script_body: str, tmp_path: Path) -> RemoteGenerator:
    script = tmp_path / "fake_server.py"
    script.write_text(script_body)
    return RemoteGenerator(
        "fake/model",
        "int4",
        python_path=sys.executable,
        launch_args=(str(script),),
        startup_timeout=30.0,
        request_timeout=30.0,
    )


def test_remote_generator_roundtrip(tmp_path: Path) -> None:
    gen = _generator_for(_FAKE_SERVER_OK, tmp_path)
    try:
        msgs = [{"role": "user", "content": "advise me"}]
        assert gen.generate(msgs) == "msgs=1 image=False"
        # Persistent: a second call reuses the same process.
        assert gen.generate(msgs) == "msgs=1 image=False"
    finally:
        gen.close()
        gen.close()  # idempotent


def test_remote_generator_encodes_image_into_request(tmp_path: Path) -> None:
    import numpy as np

    gen = _generator_for(_FAKE_SERVER_OK, tmp_path)
    try:
        msgs = [{"role": "user", "content": "x"}]
        assert gen.generate(msgs) == "msgs=1 image=False"
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        assert gen.generate(msgs, image=img) == "msgs=1 image=True"
    finally:
        gen.close()


def test_remote_generator_falls_back_when_server_dies(tmp_path: Path) -> None:
    gen = _generator_for(_FAKE_SERVER_DIES, tmp_path)
    try:
        assert gen.generate([{"role": "user", "content": "x"}]) == ""
    finally:
        gen.close()


def test_remote_generator_falls_back_on_bad_response(tmp_path: Path) -> None:
    gen = _generator_for(_FAKE_SERVER_BAD, tmp_path)
    try:
        assert gen.generate([{"role": "user", "content": "x"}]) == ""
    finally:
        gen.close()


def test_remote_generator_satisfies_textgenerator_protocol(tmp_path: Path) -> None:
    from odyssey.runners.agents.runtime import TextGenerator

    gen = _generator_for(_FAKE_SERVER_OK, tmp_path)
    try:
        assert isinstance(gen, TextGenerator)
    finally:
        gen.close()
