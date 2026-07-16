"""TDD for the FlowDAgger Stage-B deploy seam (pure logic, numpy-only).

Covers the two pieces the honest A/B rides on:

* :class:`PhaseInference` (scripts/serve_observer_conditioning.py) — the
  deploy-time 4-bucket phase state machine that replaces the GT-sidecar phase
  labels the steering net was trained with (A3 ``PHASE_TO_BUCKET`` semantics).
  Exercised with a stubbed FK so no mujoco is needed.
* the ``init_noise`` wire encoding — sidecar ``encode_init_noise`` -> JSON ->
  bridge ``decode_init_noise`` must round-trip the (40,132) float32 tensor
  bit-exactly (the A1 patch then guarantees byte-identical decode on the GR00T
  side for identical noise).

Runs locally::

    env -u PYTHONPATH .venv-ur5e/bin/python -m pytest tests/unit/test_stageb_steering.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import serve_groot_http_bridge as bridge  # noqa: E402
import serve_observer_conditioning as cond  # noqa: E402

HOME_Q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])


def _linear_fk(q6):
    """Stub FK: pinch xy = (q0, q1), z = 0.3 — lets tests move the pinch directly."""
    q = np.asarray(q6, dtype=np.float64)
    return np.array([q[0], q[1], 0.3])


def _state(x, y, grip):
    """proprio7 whose stub-FK pinch is at (x, y)."""
    return [x, y, 1.5708, -1.5708, -1.5708, 0.0, grip]


def test_reach_before_any_close():
    pi = cond.PhaseInference(_linear_fk, HOME_Q)
    assert pi.infer("s", _state(0.4, 0.1, 0.0)) == pi.REACH
    assert pi.infer("s", _state(0.45, 0.12, 0.1)) == pi.REACH  # descend, still open


def test_grasp_lock_then_transport_then_place():
    pi = cond.PhaseInference(_linear_fk, HOME_Q)
    pi.infer("s", _state(0.45, 0.15, 0.0))                       # reach
    assert pi.infer("s", _state(0.45, 0.15, 0.9)) == pi.GRASP    # close: lock xy
    # straight-up lift: xy unchanged -> still GRASP
    assert pi.infer("s", _state(0.46, 0.16, 0.9)) == pi.GRASP
    # carry departs the grasp column (> 10 cm xy) -> TRANSPORT
    assert pi.infer("s", _state(0.45, 0.50, 0.9)) == pi.TRANSPORT
    # release after a close -> PLACE (and stays PLACE while open, away from home)
    assert pi.infer("s", _state(0.45, 0.50, 0.05)) == pi.PLACE
    assert pi.infer("s", _state(0.40, 0.45, 0.05)) == pi.PLACE


def test_home_reset_clears_has_closed():
    pi = cond.PhaseInference(_linear_fk, HOME_Q)
    pi.infer("s", _state(0.45, 0.15, 0.9))                       # grasp
    assert pi.infer("s", _state(0.45, 0.50, 0.05)) == pi.PLACE   # release
    # next episode starts at home, gripper open -> reset -> REACH
    home = list(HOME_Q) + [0.0]
    assert pi.infer("s", home) == pi.REACH
    # and a later non-home open state is REACH again (has_closed cleared)
    assert pi.infer("s", _state(0.4, 0.1, 0.0)) == pi.REACH


def test_reclose_relocks_grasp_column():
    pi = cond.PhaseInference(_linear_fk, HOME_Q)
    pi.infer("s", _state(0.45, 0.15, 0.9))                       # first close at (0.45,0.15)
    pi.infer("s", _state(0.45, 0.50, 0.05))                      # open (PLACE)
    # re-close at a NEW column: must re-lock there, not compare to the old lock
    assert pi.infer("s", _state(0.45, 0.50, 0.9)) == pi.GRASP


def test_sessions_are_independent():
    pi = cond.PhaseInference(_linear_fk, HOME_Q)
    assert pi.infer("a", _state(0.45, 0.15, 0.9)) == pi.GRASP
    assert pi.infer("b", _state(0.45, 0.15, 0.0)) == pi.REACH    # b never closed


def test_init_noise_b64_roundtrip_bit_exact():
    rng = np.random.default_rng(7)
    noise = rng.standard_normal((cond.ACTION_HORIZON, cond.ACTION_DIM)).astype(np.float32)
    b64, shape = cond.encode_init_noise(noise)
    assert shape == [cond.ACTION_HORIZON, cond.ACTION_DIM]
    out = bridge.decode_init_noise(b64, shape)
    assert out.shape == (1, cond.ACTION_HORIZON, cond.ACTION_DIM)
    assert out.dtype == np.float32
    assert np.array_equal(out[0], noise)                          # bit-exact


def test_decode_init_noise_defaults_and_rejects_bad_size():
    import pytest

    rng = np.random.default_rng(0)
    noise = rng.standard_normal((cond.ACTION_HORIZON, cond.ACTION_DIM)).astype(np.float32)
    b64, _ = cond.encode_init_noise(noise)
    out = bridge.decode_init_noise(b64, None)                     # default shape
    assert out.shape == (1, cond.ACTION_HORIZON, cond.ACTION_DIM)
    with pytest.raises(ValueError):
        bridge.decode_init_noise(b64, [16, 7])                    # size mismatch
