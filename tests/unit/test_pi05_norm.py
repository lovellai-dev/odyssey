"""Tests for π0.5's action-space normalization — the ``unnorm_key`` analog (issue #74).

π0.5 has NO mission-config ``unnorm_key``: its de-normalization is baked into the
checkpoint (``norm_stats``/``asset_id``) and openpi applies it server-side. This
module (``runners/evals/pi05_norm.py``) mirrors that math in numpy plus the pad→32 /
slice→7 bridge and the asset-id resolution, so the mapping is exercisable WITHOUT a
GPU, a served checkpoint, or downloaded weights — every case here uses tiny synthetic
vectors and hand-built ``NormStats``.

All tests are named ``test_pi05_norm_*`` (the ``-k pi05_norm`` gate). See
``docs/pi05-scoping.md`` → "Mapeo de espacio de acción".
"""

from __future__ import annotations

import numpy as np
import pytest

from odyssey.runners.evals import pi05_norm as n
from odyssey.runners.evals.pi05_norm import NormStats


# ---------------------------------------------------------------------------
# mean/std normalization (openpi default) — matches (x-mean)/(std+eps).
# ---------------------------------------------------------------------------

def test_pi05_norm_mean_std_matches_openpi_formula() -> None:
    stats = NormStats.from_mean_std(mean=[1.0, -2.0, 0.5], std=[2.0, 4.0, 0.25])
    x = np.array([3.0, 2.0, 1.0])

    got = n.normalize(x, stats)
    expected = (x - np.array([1.0, -2.0, 0.5])) / (np.array([2.0, 4.0, 0.25]) + n.NORM_EPS)
    np.testing.assert_allclose(got, expected, rtol=1e-9)


def test_pi05_norm_mean_std_round_trips() -> None:
    stats = NormStats.from_mean_std(mean=[0.3, -1.1, 5.0, 0.0], std=[1.5, 0.2, 3.0, 1.0])
    x = np.array([0.7, -0.4, 2.2, 9.9])

    round_tripped = n.unnormalize(n.normalize(x, stats), stats)
    np.testing.assert_allclose(round_tripped, x, rtol=1e-9, atol=1e-9)


def test_pi05_norm_unnormalize_recovers_physical_units() -> None:
    # A zero in normalized space maps back to the mean (physical units).
    stats = NormStats.from_mean_std(mean=[10.0, -3.0], std=[2.0, 5.0])
    np.testing.assert_allclose(
        n.unnormalize(np.zeros(2), stats), [10.0, -3.0], atol=1e-4
    )


# ---------------------------------------------------------------------------
# quantile normalization (openpi use_quantiles) — maps [q01,q99] -> [-1,1].
# ---------------------------------------------------------------------------

def test_pi05_norm_quantile_maps_endpoints_to_pm_one() -> None:
    stats = NormStats.from_quantiles(q01=[-1.0, 0.0], q99=[1.0, 10.0])
    # q01 -> -1, q99 -> +1 (up to eps).
    np.testing.assert_allclose(n.normalize([-1.0, 0.0], stats, use_quantiles=True), [-1.0, -1.0], atol=1e-5)
    np.testing.assert_allclose(n.normalize([1.0, 10.0], stats, use_quantiles=True), [1.0, 1.0], atol=1e-5)


def test_pi05_norm_quantile_round_trips() -> None:
    stats = NormStats.from_quantiles(q01=[-2.0, 1.0, 0.0], q99=[2.0, 5.0, 100.0])
    x = np.array([0.5, 3.0, 42.0])

    round_tripped = n.unnormalize(
        n.normalize(x, stats, use_quantiles=True), stats, use_quantiles=True
    )
    np.testing.assert_allclose(round_tripped, x, rtol=1e-6, atol=1e-6)


def test_pi05_norm_missing_stats_for_mode_raises() -> None:
    quant_only = NormStats.from_quantiles(q01=[0.0], q99=[1.0])
    with pytest.raises(ValueError, match="mean_std"):
        n.normalize([0.5], quant_only)  # default mode needs mean/std

    mean_only = NormStats.from_mean_std(mean=[0.0], std=[1.0])
    with pytest.raises(ValueError, match="quantile"):
        n.normalize([0.5], mean_only, use_quantiles=True)


# ---------------------------------------------------------------------------
# pad-to-32 / slice-to-7 bridge (openpi pad_to_dim + LiberoOutputs[..., :7]).
# ---------------------------------------------------------------------------

def test_pi05_norm_pad_state_to_model_dim() -> None:
    state8 = np.arange(8, dtype=float)
    padded = n.pad_to_dim(state8)  # 8 -> 32
    assert padded.shape == (n.PI0_ACTION_DIM,)
    np.testing.assert_allclose(padded[:8], state8)
    np.testing.assert_allclose(padded[8:], 0.0)  # tail is zero padding


def test_pi05_norm_pad_is_noop_when_already_wide_enough() -> None:
    x = np.arange(32, dtype=float)
    np.testing.assert_array_equal(n.pad_to_dim(x), x)


def test_pi05_norm_slice_action_to_seven_dof() -> None:
    chunk32 = np.arange(3 * 32, dtype=float).reshape(3, 32)  # (horizon, 32)
    sliced = n.slice_to_libero(chunk32)
    assert sliced.shape == (3, n.LIBERO_ACTION_DIM)
    np.testing.assert_allclose(sliced[0], np.arange(7))


# ---------------------------------------------------------------------------
# End-to-end state/action mapping to the simulator format.
# ---------------------------------------------------------------------------

def test_pi05_norm_state_to_model_pads_then_normalizes() -> None:
    # 8-D state, model dim 32 -> normalized 32-vector. Use unit std / zero mean so
    # the normalized values equal the padded input (easy to assert).
    stats = NormStats.from_mean_std(mean=np.zeros(32), std=np.ones(32))
    state8 = np.arange(1, 9, dtype=float)

    model_in = n.pi05_state_to_model(state8, stats)
    assert model_in.shape == (32,)
    np.testing.assert_allclose(model_in[:8], state8, atol=1e-4)
    np.testing.assert_allclose(model_in[8:], 0.0, atol=1e-4)


def test_pi05_norm_action_to_sim_unnormalizes_then_slices() -> None:
    # Model emits a normalized 32-D action; unnormalize with per-dim stats, take 7-D.
    mean = np.arange(32, dtype=float)          # physical offset per dim
    stats = NormStats.from_mean_std(mean=mean, std=np.ones(32))
    normalized_action = np.zeros(32)           # zero -> recovers the mean

    sim_action = n.pi05_action_to_sim(normalized_action, stats)
    assert sim_action.shape == (n.LIBERO_ACTION_DIM,)
    # Un-normalized value at each dim is its mean; sliced to the first 7.
    np.testing.assert_allclose(sim_action, mean[:7], atol=1e-4)


def test_pi05_norm_action_to_sim_no_gripper_fixup() -> None:
    # Identity stats: a gripper in [0,1] passes through un-normalized and un-touched
    # (NO binarize/invert, unlike GR00T). Here dim 6 (gripper) stays 0.2.
    stats = NormStats.from_mean_std(mean=np.zeros(7), std=np.ones(7))
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
    sim = n.pi05_action_to_sim(action, stats)
    assert sim[6] == pytest.approx(0.2, abs=1e-4)


# ---------------------------------------------------------------------------
# The ``unnorm_key`` analog: asset_id resolution + the no-unnorm_key guard.
# ---------------------------------------------------------------------------

def test_pi05_norm_resolves_libero_asset_id() -> None:
    # The analog of OpenVLA's unnorm_key: which baked norm_stats asset the checkpoint uses.
    assert n.resolve_pi05_asset_id("lerobot/pi05_libero") == "physical-intelligence/libero"
    assert (
        n.resolve_pi05_asset_id("gs://openpi-assets/checkpoints/pi05_libero")
        == "physical-intelligence/libero"
    )


def test_pi05_norm_resolve_asset_id_override_wins() -> None:
    assert (
        n.resolve_pi05_asset_id("lerobot/pi05_libero", override="my-org/custom")
        == "my-org/custom"
    )


def test_pi05_norm_resolve_unknown_checkpoint_raises() -> None:
    with pytest.raises(KeyError, match="pi05"):
        n.resolve_pi05_asset_id("some/unknown_checkpoint")


def test_pi05_norm_rejects_stray_unnorm_key_in_config() -> None:
    # π0.5 mission configs must not carry unnorm_key (inert/misleading vs OpenVLA).
    with pytest.raises(ValueError, match="unnorm_key"):
        n.assert_no_unnorm_key({"pilot": "pi05", "unnorm_key": "libero_object"})


def test_pi05_norm_accepts_config_without_unnorm_key() -> None:
    # A clean π0.5 config passes the guard silently.
    n.assert_no_unnorm_key({"pilot": "pi05", "checkpoint": "lerobot/pi05_libero"})
    n.assert_no_unnorm_key(None)  # non-dict is a no-op
