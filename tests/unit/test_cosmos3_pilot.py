"""Tests for the Cosmos 3 (WAM) chunk-aware pilot and its wire transforms.

The Cosmos3 pilot reuses the pilot-agnostic ``ChunkPilotAdapter`` (chunk replay
+ flush-on-instruction-change already pinned by ``test_pi05_pilot_*``), so these
tests cover only what is Cosmos3-specific — WITHOUT a GPU, a served checkpoint,
or cosmos-framework installed:

  * the wire transforms (``runners/evals/cosmos3_transforms.py``): the
    concat_view packer, the base64-PNG ``POST /predict`` request, and the
    rot6d -> axis-angle chunk decoder (reusing GR00T's shared kinematics);
  * ``make_cosmos3_pilot`` (``runners/models/cosmos3.py``): end-to-end wiring
    with an injected fake client, including adopting the server's
    ``action_chunk_size`` from ``GET /info`` when ``n_action_steps`` is omitted;
  * the model module imports under the bare stdlib (the HTTP client is urllib).

All tests are named ``test_cosmos3_pilot_*`` (the ``-k cosmos3_pilot`` gate).
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from odyssey.runners.agents.runtime import PilotRuntime
from odyssey.runners.models.cosmos3 import make_cosmos3_pilot

# ---------------------------------------------------------------------------
# Wire transforms — concat_view + base64 PNG request packing.
# ---------------------------------------------------------------------------

def test_cosmos3_pilot_concat_view_stitches_views_side_by_side() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import cosmos3_transforms as t

    image = np.zeros((4, 6, 3), np.uint8)
    wrist = np.full((4, 2, 3), 255, np.uint8)
    concat = t.build_cosmos3_concat_view(image, wrist)

    assert concat.shape == (4, 8, 3)  # widths add, height preserved
    assert (concat[:, :6] == 0).all() and (concat[:, 6:] == 255).all()


def test_cosmos3_pilot_concat_view_rejects_height_mismatch() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import cosmos3_transforms as t

    with pytest.raises(ValueError, match="equal heights"):
        t.build_cosmos3_concat_view(
            np.zeros((4, 4, 3), np.uint8), np.zeros((5, 4, 3), np.uint8)
        )


def test_cosmos3_pilot_png_encoding_round_trips() -> None:
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from odyssey.runners.evals import cosmos3_transforms as t

    rng = np.random.default_rng(7)
    img = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
    b64 = t.encode_image_b64_png(img)

    import io
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))
    np.testing.assert_array_equal(decoded, img)  # PNG is lossless


def test_cosmos3_pilot_predict_request_matches_server_contract() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import cosmos3_transforms as t

    req = t.build_cosmos3_predict_request(
        image=np.zeros((4, 4, 3), np.uint8),
        wrist_image=np.zeros((4, 4, 3), np.uint8),
        instruction="put the banana in the bowl",
        domain_name="libero",
        image_size=256,
    )
    # Exactly the action_policy_server_libero /predict fields.
    assert set(req) == {"image", "prompt", "domain_name", "image_size"}
    assert req["prompt"] == "put the banana in the bowl"
    assert req["domain_name"] == "libero"
    assert req["image_size"] == 256
    assert isinstance(req["image"], str) and base64.b64decode(req["image"])


def test_cosmos3_pilot_predict_request_single_view_skips_concat() -> None:
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from odyssey.runners.evals import cosmos3_transforms as t

    req = t.build_cosmos3_predict_request(
        image=np.zeros((4, 6, 3), np.uint8), instruction="x",
    )
    import io
    decoded = Image.open(io.BytesIO(base64.b64decode(req["image"])))
    assert decoded.size == (6, 4)  # (W, H): no wrist view concatenated


# ---------------------------------------------------------------------------
# Wire transforms — chunk coercion + rot6d action decoding.
# ---------------------------------------------------------------------------

def test_cosmos3_pilot_chunk_accepts_server_response_shapes() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import cosmos3_transforms as t

    rows = [[1.0] * 10, [2.0] * 10]
    # Single-request shape: {"action": [[...], ...], "video": [...]} — video dropped.
    single = t.cosmos3_chunk_from_response({"action": rows, "video": ["png1", "png2"]})
    assert single.shape == (2, 10)
    # Batch shape: {"actions": [chunk, ...]} — first chunk taken.
    batch = t.cosmos3_chunk_from_response({"actions": [rows]})
    np.testing.assert_array_equal(batch, single)
    # Bare arrays, incl. a flat single action -> a chunk of one.
    assert t.cosmos3_chunk_from_response(np.asarray(rows)).shape == (2, 10)
    assert t.cosmos3_chunk_from_response([1.0] * 10).shape == (1, 10)


def test_cosmos3_pilot_action_decodes_rot6d_identity_to_zero_rotation() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import cosmos3_transforms as t

    # LIBERO SFT row: [dpos(3), rot6d(6), gripper(1)]; identity rot6d.
    row = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0]
    action = t.cosmos3_action_to_libero({"action": [row]}, 0)

    assert action.shape == (7,)
    np.testing.assert_allclose(action[:3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(action[3:6], [0.0, 0.0, 0.0], atol=1e-6)
    assert action[6] == pytest.approx(-1.0)  # gripper passthrough (no fix-up)


def test_cosmos3_pilot_action_decodes_rot6d_via_shared_kinematics() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import cosmos3_transforms as t
    from odyssey.runners.evals.gr00t_transforms import rot6d_to_axis_angle

    theta = 0.4  # rotation about z; rot6d = first two columns of R
    c, s = np.cos(theta), np.sin(theta)
    rot6d = [c, s, 0.0, -s, c, 0.0]
    row = [0.0, 0.0, 0.0, *rot6d, 0.5]
    action = t.cosmos3_action_to_libero({"action": [row]}, 0)

    # Byte-for-byte the shared GR00T rot6d kinematics (no duplicate math).
    np.testing.assert_allclose(action[3:6], rot6d_to_axis_angle(rot6d), atol=1e-6)
    np.testing.assert_allclose(action[3:6], [0.0, 0.0, theta], atol=1e-5)


def test_cosmos3_pilot_action_passes_env_native_rows_through() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import cosmos3_transforms as t

    # A 7-D row (a family member already emitting env-native actions).
    chunk = np.arange(3 * 7, dtype=float).reshape(3, 7)
    action = t.cosmos3_action_to_libero(chunk, 1)
    np.testing.assert_allclose(action, np.arange(7, 14))


def test_cosmos3_pilot_action_translation_only_zeros_rotation() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import cosmos3_transforms as t

    row = [0.1, 0.1, 0.1, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0]
    action = t.cosmos3_action_to_libero({"action": [row]}, 0, translation_only=True)
    np.testing.assert_allclose(action[3:6], [0.0, 0.0, 0.0])
    assert action[6] == pytest.approx(1.0)  # gripper forced open


# ---------------------------------------------------------------------------
# make_cosmos3_pilot — end-to-end wiring with an injected fake client.
# ---------------------------------------------------------------------------

class _FakeClient:
    """Fake policy server client: records requests, returns a 10-D chunk."""

    def __init__(self, *, chunk_len: int = 4, info: dict[str, Any] | None = None,
                 info_raises: bool = False) -> None:
        self.calls: list[Any] = []
        self.info_calls = 0
        self._chunk_len = chunk_len
        self._info = info or {}
        self._info_raises = info_raises

    def info(self) -> dict[str, Any]:
        self.info_calls += 1
        if self._info_raises:
            raise RuntimeError("server not up yet")
        return self._info

    def infer(self, request: Any) -> Any:
        self.calls.append(request)
        row = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0]
        return {"action": [row] * self._chunk_len, "video": ["<b64png>"]}


def _raw_obs(np: Any) -> dict[str, Any]:
    return {
        "image": np.zeros((4, 4, 3), np.uint8),
        "wrist_image": np.zeros((4, 4, 3), np.uint8),
    }


def test_cosmos3_pilot_make_wires_injected_client_end_to_end() -> None:
    np = pytest.importorskip("numpy")

    client = _FakeClient(chunk_len=4)
    pilot = make_cosmos3_pilot(client=client, n_action_steps=4,
                               domain_name="libero", image_size=256)
    assert isinstance(pilot, PilotRuntime)

    action = pilot.act(_raw_obs(np), "put the banana in the bowl")

    # One query; the wire request is the /predict contract.
    assert len(client.calls) == 1
    req = client.calls[0]
    assert set(req) == {"image", "prompt", "domain_name", "image_size"}
    assert req["prompt"] == "put the banana in the bowl"
    assert req["domain_name"] == "libero"
    # The decoded action is a clean 7-DoF vector (rot6d block collapsed).
    assert action.shape == (7,)
    np.testing.assert_allclose(action[:3], [0.1, 0.2, 0.3], atol=1e-6)

    # Chunk replay: 3 more acts drain the buffer without re-querying...
    for _ in range(3):
        pilot.act(_raw_obs(np), "put the banana in the bowl")
    assert len(client.calls) == 1
    # ...and the 5th act re-queries.
    pilot.act(_raw_obs(np), "put the banana in the bowl")
    assert len(client.calls) == 2


def test_cosmos3_pilot_make_adopts_server_chunk_size_from_info() -> None:
    np = pytest.importorskip("numpy")

    client = _FakeClient(chunk_len=6, info={"action_chunk_size": 6})
    pilot = make_cosmos3_pilot(client=client)  # n_action_steps omitted

    assert client.info_calls == 1
    assert pilot.n_action_steps == 6
    for _ in range(6):
        pilot.act(_raw_obs(np), "instr")
    assert len(client.calls) == 1  # the whole server-sized chunk replayed


def test_cosmos3_pilot_make_falls_back_when_info_unavailable() -> None:
    client = _FakeClient(info_raises=True)
    pilot = make_cosmos3_pilot(client=client)
    from odyssey.runners.evals.cosmos3_transforms import COSMOS3_DEFAULT_CHUNK_SIZE
    assert pilot.n_action_steps == COSMOS3_DEFAULT_CHUNK_SIZE


def test_cosmos3_pilot_make_explicit_n_action_steps_skips_info() -> None:
    client = _FakeClient(info={"action_chunk_size": 6})
    pilot = make_cosmos3_pilot(client=client, n_action_steps=2)
    assert client.info_calls == 0  # explicit value wins; no /info round-trip
    assert pilot.n_action_steps == 2


# ---------------------------------------------------------------------------
# The model module is stdlib-only at import (urllib client, lazy transforms).
# ---------------------------------------------------------------------------

def test_cosmos3_pilot_model_module_imports_without_heavy_deps() -> None:
    heavy = ("numpy", "torch", "PIL")
    # Point the fresh interpreter at THIS worktree's src (it doesn't inherit
    # pytest's `pythonpath`, and an editable-install .pth may redirect elsewhere).
    src_dir = Path(__file__).resolve().parents[2] / "src"
    script = (
        "import importlib, json, sys\n"
        f"sys.path.insert(0, {str(src_dir)!r})\n"
        "importlib.import_module('odyssey.runners.models.cosmos3')\n"
        f"print(json.dumps([m for m in {heavy!r} if m in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"cosmos3 model module failed to import cleanly:\n{result.stderr}"
    )
    leaked = json.loads(result.stdout.strip().splitlines()[-1])
    assert leaked == [], f"cosmos3 model module imported heavy deps at load: {leaked}"
