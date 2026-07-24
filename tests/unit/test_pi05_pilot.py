"""Tests for the π0.5 chunk-aware pilot adapter (issue #74).

The π0.5 pilot reuses the pilot-agnostic ``ChunkPilotAdapter`` (chunk replay +
flush-on-instruction-change gating) and thin π0.5 wire transforms, so the whole
adapter is exercisable WITHOUT a GPU, a served checkpoint, or downloaded weights:

  * the gating engine (``runners/agents/chunk_pilot.py``) carries no heavy deps —
    tested with plain fake ``predict_chunk``/``action_decoder`` callables;
  * the wire transforms (``runners/evals/pi05_transforms.py``) are numpy-only and
    reuse GR00T's Franka Panda kinematics — tested against tiny arrays;
  * ``make_pi05_pilot`` (``runners/models/pi05.py``) defers the openpi client
    import, so we can wire it end-to-end with an injected fake client and pin the
    missing-dependency contract without openpi installed.

All tests are named ``test_pi05_pilot_*`` (the ``-k pi05_pilot`` gate).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from odyssey.runners.agents.chunk_pilot import ChunkPilotAdapter
from odyssey.runners.agents.runtime import PilotRuntime
from odyssey.runners.models.pi05 import make_pi05_pilot

# ---------------------------------------------------------------------------
# Fakes — a chunk-emitting policy + a decoder, both dependency-free.
# ---------------------------------------------------------------------------

class _RecordingPolicy:
    """Fake chunk-emitting policy: records every query, returns a tagged chunk."""

    def __init__(self, result: Any = None) -> None:
        self.calls: list[Any] = []
        self._result = result

    def infer(self, wire_obs: Any) -> Any:
        self.calls.append(wire_obs)
        # Default chunk tags each step so the decoder can prove which step it got.
        if self._result is not None:
            return self._result
        return {"query": len(self.calls)}


def _index_decoder(chunk: Any, k: int) -> tuple[Any, int]:
    """Decode = surface (chunk, k) so tests can assert the replay cursor."""
    return (chunk, k)


def _make_adapter(policy: _RecordingPolicy, *, n_action_steps: int = 3,
                  observation_builder=None) -> ChunkPilotAdapter:
    return ChunkPilotAdapter(
        predict_chunk=policy.infer,
        action_decoder=_index_decoder,
        n_action_steps=n_action_steps,
        observation_builder=observation_builder,
    )


# ---------------------------------------------------------------------------
# ChunkPilotAdapter — chunk buffering + replay.
# ---------------------------------------------------------------------------

def test_pi05_pilot_replays_chunk_without_requerying() -> None:
    policy = _RecordingPolicy()
    adapter = _make_adapter(policy, n_action_steps=3)

    outs = [adapter.act("obs", "pick up the cube") for _ in range(3)]

    # One query filled the chunk; all 3 steps replayed from it.
    assert len(policy.calls) == 1
    # Cursor advanced 0,1,2 over the same (single) chunk.
    assert [k for _chunk, k in outs] == [0, 1, 2]
    assert all(chunk == {"query": 1} for chunk, _k in outs)


def test_pi05_pilot_requeries_when_chunk_drains() -> None:
    policy = _RecordingPolicy()
    adapter = _make_adapter(policy, n_action_steps=2)

    for _ in range(4):  # two full chunks of 2 steps
        adapter.act("obs", "same instruction")

    assert len(policy.calls) == 2  # re-queried once the first chunk was spent


def test_pi05_pilot_decoder_cursor_resets_each_chunk() -> None:
    policy = _RecordingPolicy()
    adapter = _make_adapter(policy, n_action_steps=2)

    ks = [adapter.act("obs", "instr")[1] for _ in range(4)]

    assert ks == [0, 1, 0, 1]  # cursor resets to 0 on each re-query


# ---------------------------------------------------------------------------
# ChunkPilotAdapter — flush-on-instruction-change gating.
# ---------------------------------------------------------------------------

def test_pi05_pilot_flushes_on_instruction_change() -> None:
    policy = _RecordingPolicy()
    adapter = _make_adapter(policy, n_action_steps=5)

    adapter.act("obs", "reach for the bowl")   # query 1
    adapter.act("obs", "reach for the bowl")   # replay (no query)
    assert len(policy.calls) == 1

    # Instruction changes mid-chunk -> immediate re-query, stale chunk discarded.
    _chunk, k = adapter.act("obs", "now grasp the bowl")
    assert len(policy.calls) == 2
    assert k == 0  # fresh chunk starts at step 0


def test_pi05_pilot_same_instruction_does_not_flush() -> None:
    policy = _RecordingPolicy()
    adapter = _make_adapter(policy, n_action_steps=10)

    for _ in range(5):
        adapter.act("obs", "identical")

    assert len(policy.calls) == 1  # never re-queried within one chunk


def test_pi05_pilot_reset_forces_requery() -> None:
    policy = _RecordingPolicy()
    adapter = _make_adapter(policy, n_action_steps=5)

    adapter.act("obs", "instr")
    assert adapter.steps_remaining == 4
    adapter.reset()  # per-episode flush
    assert adapter.steps_remaining == 0

    adapter.act("obs", "instr")
    assert len(policy.calls) == 2  # reset dropped the buffer -> new query


# ---------------------------------------------------------------------------
# ChunkPilotAdapter — plumbing details.
# ---------------------------------------------------------------------------

def test_pi05_pilot_unwraps_tuple_policy_result() -> None:
    # A policy may return (chunk, extra); the adapter uses the leading element.
    policy = _RecordingPolicy(result=("the-chunk", {"latency_ms": 12}))
    adapter = _make_adapter(policy, n_action_steps=1)

    chunk, k = adapter.act("obs", "instr")
    assert chunk == "the-chunk"
    assert k == 0


def test_pi05_pilot_observation_builder_shapes_the_wire_obs() -> None:
    seen: list[tuple[Any, str]] = []

    def builder(raw_obs: Any, instruction: str) -> Any:
        seen.append((raw_obs, instruction))
        return {"wire": raw_obs, "prompt": instruction}

    policy = _RecordingPolicy()
    adapter = _make_adapter(policy, n_action_steps=1, observation_builder=builder)

    adapter.act({"eef": 1}, "do the thing")

    assert seen == [({"eef": 1}, "do the thing")]
    # The built wire obs (not the raw obs) is what reaches the policy.
    assert policy.calls == [{"wire": {"eef": 1}, "prompt": "do the thing"}]


def test_pi05_pilot_passes_observation_through_without_builder() -> None:
    policy = _RecordingPolicy()
    adapter = _make_adapter(policy, n_action_steps=1, observation_builder=None)

    adapter.act({"already": "wire"}, "instr")
    assert policy.calls == [{"already": "wire"}]


def test_pi05_pilot_introspection_tracks_buffer() -> None:
    policy = _RecordingPolicy()
    adapter = _make_adapter(policy, n_action_steps=3)

    assert adapter.n_action_steps == 3
    assert adapter.steps_remaining == 0  # nothing buffered yet
    adapter.act("obs", "instr")
    assert adapter.steps_remaining == 2
    adapter.act("obs", "instr")
    assert adapter.steps_remaining == 1


def test_pi05_pilot_rejects_non_positive_n_action_steps() -> None:
    with pytest.raises(ValueError, match="n_action_steps"):
        ChunkPilotAdapter(
            predict_chunk=lambda o: o, action_decoder=_index_decoder, n_action_steps=0,
        )


def test_pi05_pilot_satisfies_pilot_runtime_protocol() -> None:
    adapter = _make_adapter(_RecordingPolicy(), n_action_steps=1)
    # runtime_checkable Protocol: structural match on act(observation, instruction).
    assert isinstance(adapter, PilotRuntime)


# ---------------------------------------------------------------------------
# π0.5 wire transforms — numpy-only, reuse GR00T's Franka Panda kinematics.
# ---------------------------------------------------------------------------

def test_pi05_pilot_obs_uses_openpi_keys() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import pi05_transforms as t

    obs = t.build_pi05_libero_obs(
        image=np.zeros((4, 4, 3), np.uint8),
        wrist_image=np.zeros((4, 4, 3), np.uint8),
        eef_pos=[0.1, 0.2, 0.3],
        eef_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        gripper_qpos=[0.01, -0.01],
        instruction="pick up the cube",
    )
    assert set(obs) == {
        "observation/image", "observation/wrist_image",
        "observation/state", "prompt",
    }
    assert obs["prompt"] == "pick up the cube"
    assert obs["observation/image"].dtype == np.uint8


def test_pi05_pilot_state_is_eight_dim_franka_panda() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import pi05_transforms as t

    state = t.build_pi05_libero_state(
        eef_pos=[0.1, 0.2, 0.3],
        eef_quat_xyzw=[0.0, 0.0, 0.0, 1.0],  # identity -> zero axis-angle
        gripper_qpos=[0.04, -0.04],
    )
    assert state.shape == (8,)
    # eef_pos(3) + axis-angle(3) + gripper qpos(2)
    np.testing.assert_allclose(state[:3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(state[3:6], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(state[6:8], [0.04, -0.04], atol=1e-6)


def test_pi05_pilot_state_reuses_gr00t_axis_angle() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import pi05_transforms as t
    from odyssey.runners.evals.gr00t_transforms import quat_xyzw_to_axis_angle

    quat = [0.0, 0.3826834, 0.0, 0.9238795]  # 45° about y
    state = t.build_pi05_libero_state(
        eef_pos=[0, 0, 0], eef_quat_xyzw=quat, gripper_qpos=[0, 0],
    )
    # The axis-angle block is byte-for-byte the shared GR00T kinematics (no dup).
    np.testing.assert_allclose(state[3:6], quat_xyzw_to_axis_angle(quat), atol=1e-6)


def test_pi05_pilot_action_slices_chunk_to_seven_dof() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import pi05_transforms as t

    # openpi wire result: a (horizon, 32) chunk under "actions" (pre-slice).
    chunk = {"actions": np.arange(10 * 32, dtype=float).reshape(10, 32)}
    action = t.pi05_action_to_libero(chunk, 4)
    assert action.shape == (7,)
    np.testing.assert_allclose(action, np.arange(4 * 32, 4 * 32 + 7))


def test_pi05_pilot_action_has_no_gripper_fixup() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import pi05_transforms as t

    # A gripper value in [0,1]: π0.5 passes it through verbatim (NO
    # normalize/binarize/invert), unlike gr00t_action_to_libero.
    chunk = {"actions": np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]])}
    action = t.pi05_action_to_libero(chunk, 0)
    assert action[6] == pytest.approx(0.2)  # unchanged: no fix-up


def test_pi05_pilot_action_accepts_bare_array_chunk() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import pi05_transforms as t

    chunk = np.arange(5 * 7, dtype=float).reshape(5, 7)  # no dict wrapper
    action = t.pi05_action_to_libero(chunk, 1)
    np.testing.assert_allclose(action, np.arange(7, 14))


def test_pi05_pilot_action_translation_only_zeros_rotation() -> None:
    np = pytest.importorskip("numpy")
    from odyssey.runners.evals import pi05_transforms as t

    chunk = {"actions": np.ones((1, 7))}
    action = t.pi05_action_to_libero(chunk, 0, translation_only=True)
    np.testing.assert_allclose(action[3:6], [0.0, 0.0, 0.0])  # rotation zeroed
    assert action[6] == pytest.approx(1.0)  # gripper forced open


# ---------------------------------------------------------------------------
# make_pi05_pilot — end-to-end wiring with an injected fake client; and the
# missing-openpi contract. No GPU, no served checkpoint, no weights.
# ---------------------------------------------------------------------------

def test_pi05_pilot_make_wires_injected_client_end_to_end() -> None:
    np = pytest.importorskip("numpy")

    # Fake openpi client: returns a (horizon, 32) chunk, records the wire obs.
    class _FakeClient:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def infer(self, obs: Any) -> Any:
            self.calls.append(obs)
            return {"actions": np.arange(10 * 32, dtype=float).reshape(10, 32)}

    client = _FakeClient()
    pilot = make_pi05_pilot(client=client, n_action_steps=10)
    assert isinstance(pilot, PilotRuntime)

    raw_obs = {
        "image": np.zeros((4, 4, 3), np.uint8),
        "wrist_image": np.zeros((4, 4, 3), np.uint8),
        "eef_pos": [0.1, 0.2, 0.3],
        "eef_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "gripper_qpos": [0.01, -0.01],
    }
    action = pilot.act(raw_obs, "pick up the cube")

    # One query; the wire obs handed to the client has openpi keys + the prompt.
    assert len(client.calls) == 1
    wire = client.calls[0]
    assert set(wire) == {
        "observation/image", "observation/wrist_image", "observation/state", "prompt",
    }
    assert wire["prompt"] == "pick up the cube"
    # The decoded action is a clean 7-DoF vector.
    assert action.shape == (7,)


def test_pi05_pilot_make_raises_without_openpi_client() -> None:
    if importlib.util.find_spec("openpi_client") is not None:
        pytest.skip("openpi_client installed; the missing-dep path is not exercised")
    with pytest.raises(NotImplementedError, match="openpi"):
        make_pi05_pilot(host="127.0.0.1", port=8000)  # no injected client


# ---------------------------------------------------------------------------
# The model module defers heavy deps (openpi/numpy) — imports under bare stdlib.
# ---------------------------------------------------------------------------

def test_pi05_pilot_model_module_imports_without_heavy_deps() -> None:
    heavy = ("numpy", "openpi_client", "torch", "jax")
    # Point the fresh interpreter at THIS worktree's src (it doesn't inherit
    # pytest's `pythonpath`, and an editable-install .pth may redirect elsewhere).
    src_dir = Path(__file__).resolve().parents[2] / "src"
    script = (
        "import importlib, json, sys\n"
        f"sys.path.insert(0, {str(src_dir)!r})\n"
        "importlib.import_module('odyssey.runners.models.pi05')\n"
        f"print(json.dumps([m for m in {heavy!r} if m in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"pi05 model module failed to import cleanly:\n{result.stderr}"
    )
    leaked = json.loads(result.stdout.strip().splitlines()[-1])
    assert leaked == [], f"pi05 model module imported heavy deps at load: {leaked}"
