"""Tests for the WALL-OSS chunk-aware pilot and its wire transforms.

The WALL-OSS pilot reuses the pilot-agnostic ``ChunkPilotAdapter`` (chunk replay
+ flush-on-instruction-change already pinned by ``test_pi05_pilot_*``), so these
tests cover only what is WALL-OSS-specific — WITHOUT a GPU, a served checkpoint,
wall-x, websockets or msgpack installed:

  * the wire transforms (``runners/evals/wall_oss_transforms.py``): the
    ``follow2_pos`` state (axis-angle), base64-JPEG views, the msgpack-numpy
    ext-dict decoder, and the 26-wide -> 7-D LIBERO action decoder + gripper knobs;
  * ``make_wall_oss_pilot`` (``runners/models/wall_oss.py``): end-to-end wiring
    with an injected fake client, chunk replay + re-query, short-chunk guard;
  * the model module imports under the bare stdlib.

All tests are named ``test_wall_oss_pilot_*`` (the ``-k wall_oss_pilot`` gate).
"""

from __future__ import annotations

import base64
import io
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from odyssey.runners.agents.runtime import PilotRuntime
from odyssey.runners.models.wall_oss import WallOSSWebsocketClient, make_wall_oss_pilot

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


def _chunk(horizon: int = 10, width: int = 26) -> np.ndarray:
    c = np.zeros((horizon, width), np.float32)
    c[:, 0] = np.arange(horizon)  # dx encodes the step index
    c[:, 3:6] = 0.5
    c[:, 6] = 0.7
    c[:, 7:] = 99.0  # action_padding — must never reach the env
    return c


class _FakeClient:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def infer(self, request: dict[str, Any]) -> Any:
        self.requests.append(request)
        return self.response


# ---------------------------------------------------------------------------
# Wire transforms — request packing.
# ---------------------------------------------------------------------------

def test_wall_oss_pilot_request_matches_server_contract() -> None:
    from odyssey.runners.evals import wall_oss_transforms as t

    req = t.build_wall_oss_libero_request(instruction="pick the bowl", **_raw_obs())

    assert set(req) == {"state", "views", "instruction"}
    assert req["instruction"] == "pick the bowl"
    assert set(req["views"]) == {"face_view", "right_wrist_view"}
    state = req["state"]["follow2_pos"]
    assert isinstance(state, list) and len(state) == 8
    assert all(isinstance(v, float) for v in state)  # msgpack-safe, no numpy
    np.testing.assert_allclose(state[:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(state[3:6], [0.0, 0.0, np.pi / 2], atol=1e-5)  # axis-angle
    np.testing.assert_allclose(state[6:], [0.04, -0.04])


def test_wall_oss_pilot_views_are_decodable_jpeg() -> None:
    Image = pytest.importorskip("PIL.Image")
    from odyssey.runners.evals import wall_oss_transforms as t

    b64 = t.encode_image_b64_jpeg(np.full((8, 8, 3), 200, np.uint8))
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))
    assert decoded.shape == (8, 8, 3)
    assert abs(int(decoded.mean()) - 200) <= 2  # lossy but close


# ---------------------------------------------------------------------------
# Wire transforms — response decoding.
# ---------------------------------------------------------------------------

def _msgpack_numpy_ext(arr: np.ndarray, *, byte_keys: bool = True) -> dict[Any, Any]:
    """What msgpack_numpy.encode produces for an ndarray (pre-msgpack)."""
    d = {"nd": True, "type": arr.dtype.str, "kind": b"", "shape": list(arr.shape),
         "data": arr.tobytes()}
    return {k.encode(): v for k, v in d.items()} if byte_keys else d


@pytest.mark.parametrize("byte_keys", [True, False])
def test_wall_oss_pilot_decodes_msgpack_numpy_arrays(byte_keys: bool) -> None:
    from odyssey.runners.evals import wall_oss_transforms as t

    arr = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    out = t.decode_msgpack_numpy(_msgpack_numpy_ext(arr, byte_keys=byte_keys))
    np.testing.assert_array_equal(out, arr)
    assert out.flags.writeable  # copied off the read-only frame buffer
    plain = {"server_timing": {"infer_ms": 1.0}}
    assert t.decode_msgpack_numpy(plain) is plain


def test_wall_oss_pilot_chunk_from_response_squeezes_batch() -> None:
    from odyssey.runners.evals import wall_oss_transforms as t

    batched = _chunk()[None]
    for resp in ({"predict_action": batched}, {b"predict_action": batched}, batched):
        assert t.wall_oss_chunk_from_response(resp).shape == (10, 26)


def test_wall_oss_pilot_chunk_from_response_raises_on_error_frame() -> None:
    from odyssey.runners.evals import wall_oss_transforms as t

    with pytest.raises(RuntimeError, match="Traceback"):
        t.wall_oss_chunk_from_response("Traceback (most recent call last): ...")
    with pytest.raises(KeyError, match="predict_action"):
        t.wall_oss_chunk_from_response({"server_timing": {}})


def test_wall_oss_pilot_action_decoder_slices_padding_off() -> None:
    from odyssey.runners.evals import wall_oss_transforms as t

    action = t.wall_oss_action_to_libero(_chunk(), 3)
    assert action.shape == (7,) and action.dtype == np.float32
    np.testing.assert_allclose(action, [3, 0, 0, 0.5, 0.5, 0.5, 0.7])


def test_wall_oss_pilot_action_decoder_gripper_knobs() -> None:
    from odyssey.runners.evals import wall_oss_transforms as t

    c = _chunk()
    assert t.wall_oss_action_to_libero(c, 0, binarize_gripper=True)[6] == 1.0
    assert t.wall_oss_action_to_libero(c, 0, invert_gripper=True)[6] == pytest.approx(-0.7)
    c[0, 6] = 0.0
    assert t.wall_oss_action_to_libero(c, 0, binarize_gripper=True)[6] == -1.0
    tr = t.wall_oss_action_to_libero(_chunk(), 0, translation_only=True)
    assert (tr[3:6] == 0).all() and tr[6] == -1.0


def test_wall_oss_pilot_action_decoder_rejects_narrow_rows() -> None:
    from odyssey.runners.evals import wall_oss_transforms as t

    with pytest.raises(ValueError, match="LIBERO needs"):
        t.wall_oss_action_to_libero(np.zeros((4, 5)), 0)


# ---------------------------------------------------------------------------
# make_wall_oss_pilot — ChunkPilotAdapter wiring with a fake client.
# ---------------------------------------------------------------------------

def test_wall_oss_pilot_replays_chunk_then_requeries() -> None:
    client = _FakeClient({"predict_action": _chunk()[None]})
    pilot = make_wall_oss_pilot(n_action_steps=4, client=client)
    assert isinstance(pilot, PilotRuntime)

    xs = [float(pilot.act(_raw_obs(), "pick")[0]) for _ in range(6)]
    assert xs == [0, 1, 2, 3, 0, 1]  # 4 replayed, then a fresh chunk
    assert len(client.requests) == 2
    assert client.requests[0]["state"]["follow2_pos"][:3] == pytest.approx([0.1, 0.2, 0.3])


def test_wall_oss_pilot_rejects_chunk_shorter_than_n_action_steps() -> None:
    client = _FakeClient({"predict_action": _chunk(horizon=4)})
    pilot = make_wall_oss_pilot(n_action_steps=10, client=client)
    with pytest.raises(ValueError, match="ACTION_HORIZON"):
        pilot.act(_raw_obs(), "pick")


def test_wall_oss_pilot_missing_client_deps_raise_actionable_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "msgpack", None)  # simulate not installed
    client = WallOSSWebsocketClient(host="127.0.0.1", port=1)
    with pytest.raises(NotImplementedError, match=r"\[wall-oss\]"):
        client.infer({})


def test_wall_oss_pilot_module_imports_under_bare_stdlib() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(src)!r})\n"
        "import odyssey.runners.models.wall_oss\n"
        "leaked = [m for m in ('numpy', 'torch', 'websockets', 'msgpack') if m in sys.modules]\n"
        "assert not leaked, leaked\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_wall_oss_pilot_client_msgpack_round_trip() -> None:
    """Real msgpack frames through the client (metadata frame, then an action frame)."""
    msgpack = pytest.importorskip("msgpack")

    chunk = _chunk()[None]
    frames = [
        msgpack.packb({"env": "libero"}),
        msgpack.packb({"predict_action": _msgpack_numpy_ext(chunk),
                       "server_timing": {"infer_ms": 12.5}}),
    ]

    class _FakeWS:
        def __init__(self) -> None:
            self.sent: list[bytes] = []

        def recv(self) -> bytes:
            return frames.pop(0)

        def send(self, data: bytes) -> None:
            self.sent.append(data)

    client = WallOSSWebsocketClient()
    ws = _FakeWS()
    client._ws = ws
    client._metadata = client._unpack(ws.recv())  # what connect() does after opening
    assert client._metadata == {"env": "libero"}

    pilot = make_wall_oss_pilot(n_action_steps=10, client=client)
    action = pilot.act(_raw_obs(), "pick")
    np.testing.assert_allclose(action, [0, 0, 0, 0.5, 0.5, 0.5, 0.7])
    sent = msgpack.unpackb(ws.sent[0])
    assert sent["instruction"] == "pick" and len(sent["state"]["follow2_pos"]) == 8


def test_wall_oss_pilot_client_surfaces_server_traceback() -> None:
    pytest.importorskip("msgpack")
    client = WallOSSWebsocketClient()
    with pytest.raises(RuntimeError, match="boom"):
        client._unpack("Traceback ... boom")
