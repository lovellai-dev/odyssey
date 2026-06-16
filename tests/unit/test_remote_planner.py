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
