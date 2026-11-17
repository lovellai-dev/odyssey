"""Tests for the MolmoAct2 chunk-aware pilot and its wire transforms.

The MolmoAct2 pilot reuses the pilot-agnostic ``ChunkPilotAdapter`` (chunk replay
+ flush-on-instruction-change already pinned by ``test_pi05_pilot_*``), so these
tests cover only what is MolmoAct2-specific — WITHOUT a GPU, a served
checkpoint, transformers or json-numpy installed:

  * the wire transforms (``runners/evals/molmoact2_transforms.py``): the
    json_numpy codec (byte-compatible with Ai2's ``json_numpy`` format), the
    8-D LIBERO state, the ``POST /act`` body, and the 7-D action decoder;
  * ``make_molmoact2_pilot`` (``runners/models/molmoact2.py``): end-to-end
    wiring with an injected fake client, adopting ``action_horizon`` from
    ``GET /act``, chunk replay + re-query, short-chunk guard;
  * the stdlib HTTP client against a real local ``http.server`` speaking the
    ``/act`` protocol (round-trip + error surfacing);
  * the model module imports under the bare stdlib.

All tests are named ``test_molmoact2_pilot_*`` (the ``-k molmoact2_pilot`` gate).
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from odyssey.runners.agents.runtime import PilotRuntime
from odyssey.runners.models.molmoact2 import MolmoAct2HttpClient, make_molmoact2_pilot

np = pytest.importorskip("numpy")


def _raw_obs() -> dict[str, Any]:
    return {
        "image": np.zeros((8, 8, 3), np.uint8),
        "wrist_image": np.full((8, 8, 3), 255, np.uint8),
        "eef_pos": np.array([0.1, 0.2, 0.3]),
        # 90° about z, xyzw
        "eef_quat_xyzw": np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
        "gripper_qpos": np.array([0.04, -0.04]),
    }


def _chunk(horizon: int = 10) -> np.ndarray:
    c = np.zeros((horizon, 7), np.float32)
    c[:, 0] = np.arange(horizon)  # dx encodes the step index
    c[:, 3:6] = 0.05
    c[:, 6] = 1.0  # close
    return c


class _FakeClient:
    def __init__(self, response: Any, info: dict[str, Any] | None = None) -> None:
        self.response = response
        self._info = info
        self.requests: list[dict[str, Any]] = []

    def info(self) -> dict[str, Any]:
        if self._info is None:
            raise RuntimeError("no /act health endpoint")
        return self._info

    def infer(self, request: dict[str, Any]) -> Any:
        self.requests.append(request)
        return self.response


# ---------------------------------------------------------------------------
# json_numpy codec — must match Ai2's json_numpy wire format exactly.
# ---------------------------------------------------------------------------

def test_molmoact2_pilot_json_numpy_matches_reference_format() -> None:
    from odyssey.runners.evals import molmoact2_transforms as t

    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    doc = json.loads(t.json_numpy_dumps({"a": arr}))["a"]
    # json_numpy.default: {"__numpy__": b64(raw bytes), "dtype": descr, "shape": shape}
    assert doc == {
        "__numpy__": base64.b64encode(arr.tobytes()).decode(),
        "dtype": "<f4",
        "shape": [2, 3],
    }


def test_molmoact2_pilot_json_numpy_round_trips() -> None:
    from odyssey.runners.evals import molmoact2_transforms as t

    payload = {
        "img": np.random.default_rng(0).integers(0, 255, (4, 4, 3), dtype=np.uint8),
        "state": np.linspace(0, 1, 8, dtype=np.float32),
        "scalar": np.float32(2.5),
        "text": "pick",
        "nested": {"x": [1, 2]},
    }
    out = t.json_numpy_loads(t.json_numpy_dumps(payload))
    np.testing.assert_array_equal(out["img"], payload["img"])
    np.testing.assert_array_equal(out["state"], payload["state"])
    assert out["scalar"] == pytest.approx(2.5)
    assert out["text"] == "pick" and out["nested"] == {"x": [1, 2]}
    assert out["img"].flags.writeable


# ---------------------------------------------------------------------------
# Request packing / response decoding.
# ---------------------------------------------------------------------------

def test_molmoact2_pilot_request_matches_server_contract() -> None:
    from odyssey.runners.evals import molmoact2_transforms as t

    req = t.build_molmoact2_libero_request(instruction="pick the bowl", num_steps=7,
                                           **_raw_obs())
    assert set(req) == {"image", "wrist_image", "state", "instruction", "num_steps"}
    assert req["image"].dtype == np.uint8 and req["image"].shape == (8, 8, 3)
    assert (req["wrist_image"] == 255).all()
    assert req["num_steps"] == 7
    state = req["state"]
    assert state.dtype == np.float32 and state.shape == (8,)
    np.testing.assert_allclose(state[:3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(state[3:6], [0.0, 0.0, np.pi / 2], atol=1e-5)  # axis-angle
    np.testing.assert_allclose(state[6:], [0.04, -0.04], atol=1e-6)


def test_molmoact2_pilot_chunk_from_response_shapes_and_errors() -> None:
    from odyssey.runners.evals import molmoact2_transforms as t

    for resp in ({"actions": _chunk()}, {"actions": _chunk()[None]}, _chunk()):
        assert t.molmoact2_chunk_from_response(resp).shape == (10, 7)
    with pytest.raises(RuntimeError, match="boom"):
        t.molmoact2_chunk_from_response({"error": "boom"})
    with pytest.raises(KeyError, match="actions"):
        t.molmoact2_chunk_from_response({"dt_ms": 1.0})


def test_molmoact2_pilot_action_decoder_passthrough_and_knobs() -> None:
    from odyssey.runners.evals import molmoact2_transforms as t

    action = t.molmoact2_action_to_libero(_chunk(), 4)
    assert action.dtype == np.float32
    np.testing.assert_allclose(action, [4, 0, 0, 0.05, 0.05, 0.05, 1.0], atol=1e-6)
    assert t.molmoact2_action_to_libero(_chunk(), 0, invert_gripper=True)[6] == -1.0
    tr = t.molmoact2_action_to_libero(_chunk(), 0, translation_only=True)
    assert (tr[3:6] == 0).all() and tr[6] == -1.0
    with pytest.raises(ValueError, match="LIBERO needs"):
        t.molmoact2_action_to_libero(np.zeros((3, 5)), 0)


# ---------------------------------------------------------------------------
# make_molmoact2_pilot — ChunkPilotAdapter wiring with a fake client.
# ---------------------------------------------------------------------------

def test_molmoact2_pilot_adopts_server_action_horizon() -> None:
    client = _FakeClient({"actions": _chunk()}, info={"action_horizon": 6})
    pilot = make_molmoact2_pilot(client=client)
    assert isinstance(pilot, PilotRuntime)
    assert pilot.n_action_steps == 6


def test_molmoact2_pilot_falls_back_to_default_horizon() -> None:
    pilot = make_molmoact2_pilot(client=_FakeClient({"actions": _chunk()}))
    assert pilot.n_action_steps == 10


def test_molmoact2_pilot_replays_chunk_then_requeries() -> None:
    client = _FakeClient({"actions": _chunk()})
    pilot = make_molmoact2_pilot(n_action_steps=4, num_steps=5, client=client)

    xs = [float(pilot.act(_raw_obs(), "pick")[0]) for _ in range(6)]
    assert xs == [0, 1, 2, 3, 0, 1]  # 4 replayed, then a fresh chunk
    assert len(client.requests) == 2
    assert client.requests[0]["num_steps"] == 5
    assert client.requests[0]["instruction"] == "pick"


def test_molmoact2_pilot_rejects_chunk_shorter_than_n_action_steps() -> None:
    pilot = make_molmoact2_pilot(
        n_action_steps=10, client=_FakeClient({"actions": _chunk(horizon=4)})
    )
    with pytest.raises(ValueError, match="n_action_steps"):
        pilot.act(_raw_obs(), "pick")


# ---------------------------------------------------------------------------
# Stdlib HTTP client against a real local /act server.
# ---------------------------------------------------------------------------

@pytest.fixture
def act_server() -> Iterator[tuple[str, int, list[dict[str, Any]]]]:
    from odyssey.runners.evals import molmoact2_transforms as t

    seen: list[dict[str, Any]] = []

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # silence test output
            return

        def _reply(self, code: int, body: dict[str, Any]) -> None:
            data = t.json_numpy_dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._reply(200, {"status": "ok", "action_horizon": 10})

        def do_POST(self) -> None:
            body = t.json_numpy_loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if body.get("instruction") == "explode":
                self._reply(500, {"error": "inference failed: boom"})
                return
            self._reply(200, {"actions": _chunk(), "dt_ms": 1.0})

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "127.0.0.1", server.server_address[1], seen
    finally:
        server.shutdown()
        server.server_close()


def test_molmoact2_pilot_http_client_round_trip(act_server) -> None:
    host, port, seen = act_server
    pilot = make_molmoact2_pilot(host=host, port=port)
    assert pilot.n_action_steps == 10  # from GET /act

    action = pilot.act(_raw_obs(), "pick")
    np.testing.assert_allclose(action, [0, 0, 0, 0.05, 0.05, 0.05, 1.0], atol=1e-6)
    assert seen[0]["state"].shape == (8,) and seen[0]["image"].dtype == np.uint8


def test_molmoact2_pilot_http_client_surfaces_server_error(act_server) -> None:
    host, port, _ = act_server
    client = MolmoAct2HttpClient(host=host, port=port)
    with pytest.raises(RuntimeError, match="boom"):
        client.infer({"instruction": "explode"})


def test_molmoact2_pilot_http_client_unreachable_has_hint() -> None:
    client = MolmoAct2HttpClient(host="127.0.0.1", port=1, timeout_seconds=2)
    with pytest.raises(RuntimeError, match=r"host_server_libero\.py"):
        client.info()


def test_molmoact2_pilot_module_imports_under_bare_stdlib() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(src)!r})\n"
        "import odyssey.runners.models.molmoact2\n"
        "leaked = [m for m in ('numpy', 'torch', 'transformers') if m in sys.modules]\n"
        "assert not leaked, leaked\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
