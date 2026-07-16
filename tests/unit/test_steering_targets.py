"""TDD for the FlowDAgger steering-net data pipeline (A4, numpy-only).

Exercises the pure helpers in ``scripts/train_steering_ur5e.py`` — target
assembly (X/Y shapes, phase one-hot), the EPISODE-level train/val split (no
episode may appear in both sides), fixed-seed deploy pads, and init-noise
assembly. No torch / GR00T / GPU needed (the local .venv-ur5e has numpy only),
so this runs locally with::

    env -u PYTHONPATH .venv-ur5e/bin/python -m pytest tests/unit/test_steering_targets.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

import train_steering_ur5e as ts  # noqa: E402


def _fake_arrays(n=200, n_eps=10, seed=0):
    rng = np.random.default_rng(seed)
    episodes = rng.integers(0, n_eps, size=n).astype(np.int64)
    return {
        "proprio7": rng.standard_normal((n, 7)).astype(np.float32),
        "grasp_target3": rng.standard_normal((n, 3)).astype(np.float32),
        "phase_label": rng.integers(0, ts.N_PHASES, size=n).astype(np.int64),
        "w_star_real": rng.standard_normal((n, ts.REAL_STEPS, ts.REAL_DIMS)).astype(np.float32),
        "recon_mse": rng.random(n).astype(np.float32) * 1e-4,
        "episode": episodes,
        "frame_idx": rng.integers(0, 900, size=n).astype(np.int64),
    }


def test_phase_onehot_is_correct():
    pl = np.array([0, 1, 2, 3, 3, 0], dtype=np.int64)
    oh = ts.phase_onehot(pl)
    assert oh.shape == (6, ts.N_PHASES)
    assert oh.dtype == np.float32
    # each row is a single 1.0 at the label position
    assert np.array_equal(oh.argmax(axis=1), pl)
    assert np.allclose(oh.sum(axis=1), 1.0)


def test_build_xy_real_design_shapes():
    arr = _fake_arrays()
    X, Y, eps = ts.build_xy(arr, "real112")
    assert X.shape == (200, 14)      # 7 + 3 + 4
    assert Y.shape == (200, ts.REAL_OUT)   # 112
    assert X.dtype == np.float32 and Y.dtype == np.float32
    assert eps.shape == (200,)
    # X first 7 cols == proprio, next 3 == grasp_target, last 4 == onehot
    assert np.allclose(X[:, :7], arr["proprio7"])
    assert np.allclose(X[:, 7:10], arr["grasp_target3"])
    assert np.array_equal(X[:, 10:].argmax(axis=1), arr["phase_label"])


def test_build_xy_full_design_requires_full_and_shapes():
    arr = _fake_arrays()
    with pytest.raises(KeyError):
        ts.build_xy(arr, "full5280")   # no w_star_full present
    arr["w_star_full"] = np.random.default_rng(1).standard_normal(
        (200, ts.ACTION_HORIZON, ts.ACTION_DIM)).astype(np.float32)
    _X, Y, _eps = ts.build_xy(arr, "full5280")
    assert Y.shape == (200, ts.FULL_OUT)   # 5280


def test_episode_split_is_disjoint_by_episode():
    arr = _fake_arrays(n=500, n_eps=20, seed=3)
    tr, va = ts.episode_split(arr["episode"], val_frac=0.1, seed=7)
    # partition: every index appears exactly once
    assert len(tr) + len(va) == 500
    assert set(tr.tolist()).isdisjoint(va.tolist())
    tr_eps = set(arr["episode"][tr].tolist())
    va_eps = set(arr["episode"][va].tolist())
    # NO episode may appear on both sides
    assert tr_eps.isdisjoint(va_eps)
    # ~10% of the 20 episodes -> 2 val episodes
    assert len(va_eps) == 2


def test_episode_split_is_deterministic():
    eps = _fake_arrays(seed=5)["episode"]
    a1 = ts.episode_split(eps, seed=1)
    a2 = ts.episode_split(eps, seed=1)
    assert np.array_equal(a1[0], a2[0]) and np.array_equal(a1[1], a2[1])


def test_deploy_pads_fixed_seed_reproducible():
    p1 = ts.deploy_pads(5)
    p2 = ts.deploy_pads(5)
    assert p1.shape == (5, ts.ACTION_HORIZON, ts.ACTION_DIM)
    assert np.array_equal(p1, p2)   # fixed PAD_SEED -> identical at train and deploy


def test_assemble_init_noise_places_real_dims():
    real = np.random.default_rng(2).standard_normal((3, ts.REAL_STEPS, ts.REAL_DIMS)).astype(np.float32)
    full = ts.assemble_init_noise(real)
    assert full.shape == (3, ts.ACTION_HORIZON, ts.ACTION_DIM)
    # real block matches exactly
    assert np.allclose(full[:, :ts.REAL_STEPS, :ts.REAL_DIMS], real)
    # pad block matches the fixed-seed deploy pads
    pads = ts.deploy_pads(3)
    assert np.allclose(full[:, ts.REAL_STEPS:, :], pads[:, ts.REAL_STEPS:, :])
    assert np.allclose(full[:, :ts.REAL_STEPS, ts.REAL_DIMS:], pads[:, :ts.REAL_STEPS, ts.REAL_DIMS:])


def test_load_shards_roundtrip(tmp_path):
    arr = _fake_arrays(n=50)
    # write two shards
    for si, sl in enumerate((slice(0, 30), slice(30, 50))):
        np.savez(tmp_path / f"shard_{si:04d}.npz", **{k: v[sl] for k, v in arr.items()})
    loaded = ts.load_shards(tmp_path)
    assert loaded["proprio7"].shape == (50, 7)
    assert loaded["w_star_real"].shape == (50, ts.REAL_STEPS, ts.REAL_DIMS)
    assert np.allclose(loaded["proprio7"], arr["proprio7"])
