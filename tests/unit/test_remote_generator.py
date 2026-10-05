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


def test_serve_echoes_request_id() -> None:
    out = _run_serve(
        _FakeGen("ok"),
        [{"id": 7, "messages": []}, {"id": 8, "foo": "bar"}, {"shutdown": True}],
    )
    assert out[1] == {"id": 7, "text": "ok"}
    assert out[2]["id"] == 8 and "error" in out[2]


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
    sys.stdout.write(json.dumps({"id": req["id"], "text": "msgs=%d image=%s" % (n, has_img)}) + "\\n")
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
        req = json.loads(line)
        sys.stdout.write(json.dumps({"id": req["id"], "text": None}) + "\\n"); sys.stdout.flush()
"""


# Answers with the scene named in the last message; the FIRST reply is late.
_FAKE_SERVER_SLOW_FIRST = """
import json, sys, time
sys.stdout.write(json.dumps({"ready": True}) + "\\n"); sys.stdout.flush()
first = True
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("shutdown"):
        break
    if first:
        time.sleep(0.3)
        first = False
    scene = req["messages"][-1]["content"]
    sys.stdout.write(json.dumps({"id": req["id"], "text": scene}) + "\\n"); sys.stdout.flush()
"""

# First launch reports a model-load error and exits; later launches are healthy.
_FAKE_SERVER_FAILS_ONCE = """
import json, os, sys
marker = {marker!r}
if not os.path.exists(marker):
    open(marker, "w").close()
    sys.stdout.write(json.dumps({{"error": "generator load failed: transient"}}) + "\\n")
    sys.stdout.flush()
    sys.exit(1)
sys.stdout.write(json.dumps({{"ready": True}}) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("shutdown"):
        break
    sys.stdout.write(json.dumps({{"id": req["id"], "text": "healthy"}}) + "\\n")
    sys.stdout.flush()
"""


# Answers the first request, then crashes (exits without replying) on the
# second. Each launch appends to a counter file so tests can count restarts.
_FAKE_SERVER_CRASHES = """
import json, sys
open({counter!r}, "a").write("x")
sys.stdout.write(json.dumps({{"ready": True}}) + "\\n"); sys.stdout.flush()
served = 0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("shutdown"):
        break
    if served == 1:
        sys.exit(1)
    served += 1
    sys.stdout.write(json.dumps({{"id": req["id"], "text": "alive"}}) + "\\n")
    sys.stdout.flush()
"""

# Never becomes ready; counts its launches.
_FAKE_SERVER_ALWAYS_FAILS = """
import json, sys
open({counter!r}, "a").write("x")
sys.stdout.write(json.dumps({{"error": "generator load failed: bad model"}}) + "\\n")
sys.stdout.flush()
sys.exit(1)
"""


# First launch: reads one request, then stops reading stdin (stuck in
# "inference") so the pipe fills. Later launches are healthy.
_FAKE_SERVER_STOPS_READING = """
import json, os, sys, time
marker = {marker!r}
first = not os.path.exists(marker)
open(marker, "a").close()
sys.stdout.write(json.dumps({{"ready": True}}) + "\\n"); sys.stdout.flush()
if first:
    sys.stdin.readline()
    time.sleep(10)
    sys.exit(0)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("shutdown"):
        break
    sys.stdout.write(json.dumps({{"id": req["id"], "text": "healthy"}}) + "\\n")
    sys.stdout.flush()
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


def test_remote_generator_discards_late_reply_after_timeout(tmp_path: Path) -> None:
    gen = _generator_for(_FAKE_SERVER_SLOW_FIRST, tmp_path)
    try:
        gen._request_timeout = 0.05
        assert gen.generate([{"role": "user", "content": "scene A"}]) == ""
        # The late "scene A" reply must not be read as the answer for scene B.
        gen._request_timeout = 30.0
        assert gen.generate([{"role": "user", "content": "scene B"}]) == "scene B"
        assert gen.generate([{"role": "user", "content": "scene C"}]) == "scene C"
    finally:
        gen.close()


def test_remote_generator_restarts_after_startup_failure(tmp_path: Path) -> None:
    script = _FAKE_SERVER_FAILS_ONCE.format(marker=str(tmp_path / "failed_once"))
    gen = _generator_for(script, tmp_path)
    try:
        msgs = [{"role": "user", "content": "x"}]
        assert gen.generate(msgs) == ""
        # The dead process's EOF must not close the healthy replacement.
        assert gen.generate(msgs) == "healthy"
        assert gen.generate(msgs) == "healthy"
    finally:
        gen.close()


def test_remote_generator_restarts_after_server_dies_mid_run(tmp_path: Path) -> None:
    counter = tmp_path / "launches"
    gen = _generator_for(_FAKE_SERVER_CRASHES.format(counter=str(counter)), tmp_path)
    try:
        msgs = [{"role": "user", "content": "x"}]
        assert gen.generate(msgs) == "alive"
        assert gen.generate(msgs) == ""  # server crashes on this request
        # The dead server is replaced, not reused for the rest of the run.
        assert gen.generate(msgs) == "alive"
        assert counter.read_text() == "xx"
    finally:
        gen.close()


def test_remote_generator_stops_relaunching_after_budget(tmp_path: Path) -> None:
    counter = tmp_path / "launches"
    gen = _generator_for(_FAKE_SERVER_ALWAYS_FAILS.format(counter=str(counter)), tmp_path)
    try:
        msgs = [{"role": "user", "content": "x"}]
        for _ in range(5):
            assert gen.generate(msgs) == ""
        # Default budget is 3 launches: episodes 4 and 5 don't reload the model.
        assert counter.read_text() == "xxx"
    finally:
        gen.close()


def test_remote_generator_bounds_send_to_a_server_that_stopped_reading(
    tmp_path: Path,
) -> None:
    import time

    import numpy as np

    script = _FAKE_SERVER_STOPS_READING.format(marker=str(tmp_path / "stuck_once"))
    gen = _generator_for(script, tmp_path)
    try:
        gen._request_timeout = 0.5
        msgs = [{"role": "user", "content": "x"}]
        # The server reads this request, then hangs without answering.
        assert gen.generate(msgs) == ""
        # Random noise defeats PNG compression: ~260 KB of base64, far beyond
        # a pipe buffer, so the write blocks on a server that stopped reading.
        image = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
        start = time.monotonic()
        assert gen.generate(msgs, image=image) == ""
        # Bounded: send timeout + bounded cleanup, not the server's 10 s hang.
        assert time.monotonic() - start < 5.0
        # The stuck server was retired; the next call gets a healthy one.
        gen._request_timeout = 30.0
        assert gen.generate(msgs, image=image) == "healthy"
    finally:
        start = time.monotonic()
        gen.close()
        assert time.monotonic() - start < 5.0


def test_remote_generator_satisfies_textgenerator_protocol(tmp_path: Path) -> None:
    from odyssey.runners.agents.runtime import TextGenerator

    gen = _generator_for(_FAKE_SERVER_OK, tmp_path)
    try:
        assert isinstance(gen, TextGenerator)
    finally:
        gen.close()
