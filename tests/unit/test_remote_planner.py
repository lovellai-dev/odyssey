"""Tests for the out-of-process SPECIALIST planner.

No GPU / no real model: ``serve()`` is driven with a fake TextGenerator over
in-memory streams, and ``RemotePlanner`` is driven against a tiny fake server
script run as a real subprocess (exercises the real Popen + select + JSON
protocol + lifecycle).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import odyssey.engine  # noqa: F401 — import engine first to break the runners<->engine cycle
from odyssey.runners.agents.planner import LLMPlanner
from odyssey.runners.agents.planner_server import serve
from odyssey.runners.agents.remote_planner import RemotePlanner


class _FakeGen:
    """TextGenerator stub: returns a fixed numbered list."""

    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, messages: list[dict[str, str]]) -> str:
        return self._text


class _FakeVLMGen:
    """Multimodal generator stub: records the image it was handed."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.last_image: object = "unset"

    def generate(self, messages: list[dict[str, object]], image: object = None) -> str:
        self.last_image = image
        return self._text


def _tiny_png_b64() -> str:
    """A 2x2 PNG as base64 (built with PIL, the base image dep)."""
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_llmplanner_forwards_image_when_generator_is_multimodal() -> None:
    gen = _FakeVLMGen("1. look\n2. grasp\n")
    planner = LLMPlanner(gen)
    sentinel = object()
    assert planner.plan("pick up the cube", image=sentinel) == ["look", "grasp"]
    assert gen.last_image is sentinel


def test_llmplanner_ignores_image_when_generator_is_text_only() -> None:
    # _FakeGen.generate has no `image` param → image must not be forwarded.
    planner = LLMPlanner(_FakeGen("1. a\n2. b\n"))
    assert planner.plan("task", image=object()) == ["a", "b"]


# --------------------------------------------------------------------------- #
# serve() — the server-side request/response loop
# --------------------------------------------------------------------------- #


def _run_serve(planner: LLMPlanner, requests: list[dict[str, object]]) -> list[dict]:
    instream = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    outstream = io.StringIO()
    serve(planner, instream, outstream)
    return [json.loads(line) for line in outstream.getvalue().splitlines() if line.strip()]


def test_serve_emits_ready_then_plan() -> None:
    planner = LLMPlanner(_FakeGen("1. move above cube\n2. grasp\n"))
    out = _run_serve(planner, [{"instruction": "pick up the cube"}, {"shutdown": True}])
    assert out[0] == {"ready": True}
    assert out[1] == {"plan": ["move above cube", "grasp"]}


def test_serve_rejects_invalid_json() -> None:
    instream = io.StringIO('not-json\n{"shutdown": true}\n')
    outstream = io.StringIO()
    serve(LLMPlanner(_FakeGen("1. a\n")), instream, outstream)
    msgs = [json.loads(x) for x in outstream.getvalue().splitlines() if x.strip()]
    assert msgs[0] == {"ready": True}
    assert any("error" in m for m in msgs)


def test_serve_missing_instruction() -> None:
    out = _run_serve(LLMPlanner(_FakeGen("1. a\n")), [{"foo": "bar"}, {"shutdown": True}])
    assert any("error" in m for m in out)


def test_serve_decodes_image_and_passes_to_planner() -> None:
    """serve() base64-decodes 'image' into a PIL Image for the planner."""

    class _RecordingPlanner:
        def __init__(self) -> None:
            self.last_image: object = "unset"

        def plan(self, instruction: str, image: object = None) -> list[str]:
            self.last_image = image
            return ["ok"]

    from PIL import Image

    planner = _RecordingPlanner()
    out = _run_serve(  # type: ignore[arg-type]
        planner,
        [{"instruction": "pick up", "image": _tiny_png_b64()}, {"shutdown": True}],
    )
    assert out[1] == {"plan": ["ok"]}
    assert isinstance(planner.last_image, Image.Image)
    assert planner.last_image.size == (2, 2)


# --------------------------------------------------------------------------- #
# serve() — completion-check requests (additive; legacy plan path untouched)
# --------------------------------------------------------------------------- #


def test_serve_completion_check_returns_done() -> None:
    # Multimodal generator -> check_done runs; "YES" -> done True.
    planner = LLMPlanner(_FakeVLMGen("YES"))
    out = _run_serve(
        planner,
        [
            {"check": {"instruction": "grasp the cube", "image": _tiny_png_b64()}},
            {"shutdown": True},
        ],
    )
    assert out[0] == {"ready": True}
    assert out[1] == {"done": True}


def test_serve_completion_check_returns_not_done() -> None:
    planner = LLMPlanner(_FakeVLMGen("NO"))
    out = _run_serve(
        planner,
        [
            {"check": {"instruction": "grasp", "image": _tiny_png_b64()}},
            {"shutdown": True},
        ],
    )
    assert out[1] == {"done": False}


def test_serve_check_unsupported_when_planner_lacks_check_done() -> None:
    class _PlanOnly:
        def plan(self, instruction: str, image: object = None) -> list[str]:
            return ["a"]

    out = _run_serve(  # type: ignore[arg-type]
        _PlanOnly(),
        [
            {"check": {"instruction": "x", "image": _tiny_png_b64()}},
            {"shutdown": True},
        ],
    )
    assert any(m.get("error") == "check unsupported" for m in out)


def test_serve_legacy_plan_still_works() -> None:
    # Back-compat: an instruction request is unaffected by the new check branch.
    planner = LLMPlanner(_FakeGen("1. a\n2. b\n"))
    out = _run_serve(planner, [{"instruction": "task"}, {"shutdown": True}])
    assert out[1] == {"plan": ["a", "b"]}


# --------------------------------------------------------------------------- #
# RemotePlanner — the client, against a fake server subprocess
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
    instr = req.get("instruction", "")
    sys.stdout.write(json.dumps({"plan": ["phase 0 for " + instr, "phase 1"]}) + "\\n")
    sys.stdout.flush()
"""

# Never prints {"ready": true} — exits immediately.
_FAKE_SERVER_DIES = "import sys; sys.exit(1)\n"

# Ready, but returns an empty plan -> client should fall back.
_FAKE_SERVER_EMPTY = """
import json, sys
sys.stdout.write(json.dumps({"ready": True}) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    if line.strip():
        sys.stdout.write(json.dumps({"plan": []}) + "\\n"); sys.stdout.flush()
"""

# Answers completion checks: done iff the instruction contains "done".
_FAKE_SERVER_CHECK = """
import json, sys
sys.stdout.write(json.dumps({"ready": True}) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("shutdown"):
        break
    if isinstance(req.get("check"), dict):
        instr = req["check"].get("instruction", "")
        sys.stdout.write(json.dumps({"done": "done" in instr}) + "\\n"); sys.stdout.flush()
        continue
    sys.stdout.write(json.dumps({"plan": ["p0", "p1"]}) + "\\n"); sys.stdout.flush()
"""

# Echoes whether the request carried an "image" field (a base64 string).
_FAKE_SERVER_ECHO_IMG = """
import json, sys
sys.stdout.write(json.dumps({"ready": True}) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    if req.get("shutdown"):
        break
    has_img = isinstance(req.get("image"), str) and len(req["image"]) > 0
    sys.stdout.write(json.dumps({"plan": ["image=" + str(has_img)]}) + "\\n")
    sys.stdout.flush()
"""


def _planner_for(script_body: str, tmp_path: Path, **kw: object) -> RemotePlanner:
    script = tmp_path / "fake_server.py"
    script.write_text(script_body)
    return RemotePlanner(
        "fake/model",
        "int4",
        python_path=sys.executable,
        launch_args=(str(script),),
        startup_timeout=30.0,
        request_timeout=30.0,
        **kw,  # type: ignore[arg-type]
    )


def test_remote_planner_roundtrip(tmp_path: Path) -> None:
    planner = _planner_for(_FAKE_SERVER_OK, tmp_path)
    try:
        assert planner.plan("pick up the red cube") == [
            "phase 0 for pick up the red cube",
            "phase 1",
        ]
        # Persistent: a second call reuses the same process.
        assert planner.plan("stack the blocks") == ["phase 0 for stack the blocks", "phase 1"]
    finally:
        planner.close()
        planner.close()  # idempotent


def test_remote_planner_close_falls_back_without_posix_process_groups(monkeypatch) -> None:
    import odyssey.runners.agents.remote_planner as remote_planner

    monkeypatch.delattr(remote_planner.os, "killpg", raising=False)
    monkeypatch.delattr(remote_planner.os, "getpgid", raising=False)

    class _FakeStdin:
        closed = False

        def write(self, _text: str) -> None:
            pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    class _FakeProc:
        stdin = _FakeStdin()
        pid = 123
        terminated = False
        killed = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            return 0

    planner = RemotePlanner("fake/model", python_path=sys.executable)
    proc = _FakeProc()
    planner._proc = proc  # type: ignore[assignment]

    planner.close()

    assert planner._proc is None
    assert proc.terminated is True
    assert proc.killed is False


def test_remote_planner_encodes_image_into_request(tmp_path: Path) -> None:
    import numpy as np

    planner = _planner_for(_FAKE_SERVER_ECHO_IMG, tmp_path)
    try:
        # No image -> request has no "image" field.
        assert planner.plan("do the task") == ["image=False"]
        # With an image (HWC uint8 ndarray) -> base64-encoded into the request.
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        assert planner.plan("do the task", image=img) == ["image=True"]
    finally:
        planner.close()


def test_remote_planner_falls_back_when_server_dies(tmp_path: Path) -> None:
    planner = _planner_for(_FAKE_SERVER_DIES, tmp_path)
    try:
        # No ready signal -> startup fails -> plan() falls back to single-step.
        assert planner.plan("do the task") == ["do the task"]
    finally:
        planner.close()


def test_remote_planner_falls_back_on_empty_plan(tmp_path: Path) -> None:
    planner = _planner_for(_FAKE_SERVER_EMPTY, tmp_path)
    try:
        assert planner.plan("do the task") == ["do the task"]
    finally:
        planner.close()


def test_remote_planner_satisfies_protocol(tmp_path: Path) -> None:
    from odyssey.runners.agents.runtime import PlannerRuntime

    planner = _planner_for(_FAKE_SERVER_OK, tmp_path)
    try:
        assert isinstance(planner, PlannerRuntime)
    finally:
        planner.close()


def test_remote_planner_satisfies_completion_detector(tmp_path: Path) -> None:
    from odyssey.runners.agents.runtime import CompletionDetector

    planner = _planner_for(_FAKE_SERVER_OK, tmp_path)
    try:
        assert isinstance(planner, CompletionDetector)
    finally:
        planner.close()


def test_remote_planner_check_done_roundtrip(tmp_path: Path) -> None:
    import numpy as np

    planner = _planner_for(_FAKE_SERVER_CHECK, tmp_path)
    try:
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        assert planner.check_done("the task is done now", img) is True
        assert planner.check_done("still working", img) is False
    finally:
        planner.close()


def test_remote_planner_check_done_false_without_image(tmp_path: Path) -> None:
    # image=None short-circuits to False without starting the subprocess.
    planner = _planner_for(_FAKE_SERVER_CHECK, tmp_path)
    try:
        assert planner.check_done("done", None) is False
        assert planner._proc is None  # never launched
    finally:
        planner.close()


def test_remote_planner_check_done_false_against_old_server(tmp_path: Path) -> None:
    # An old server that doesn't understand {"check": ...} replies with a plan;
    # the client fails safe to False (no "done" key).
    import numpy as np

    planner = _planner_for(_FAKE_SERVER_OK, tmp_path)
    try:
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        assert planner.check_done("anything", img) is False
    finally:
        planner.close()
