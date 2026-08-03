"""TDD for the open-loop GT eval metrics (scripts/eval_openloop_gt.py).

Pure-numpy metric functions: per-joint arm MAE and gripper agreement between a
predicted action chunk and the recorded expert chunk (the training ground truth).
No server / dataset / heavy deps touched (those are lazy inside the CLI ``main``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import eval_openloop_gt as ol  # noqa: E402


def test_arm_mae_zero_when_identical() -> None:
    chunk = np.random.default_rng(0).uniform(-1, 1, size=(16, 6))
    mae = ol.arm_chunk_mae(chunk, chunk)
    assert mae.shape == (6,)
    assert np.allclose(mae, 0.0)


def test_arm_mae_matches_known_offset() -> None:
    gt = np.zeros((16, 6))
    pred = gt.copy()
    pred[:, 2] += 0.1  # a constant 0.1 rad error on joint 2 only
    mae = ol.arm_chunk_mae(pred, gt)
    expected = np.array([0, 0, 0.1, 0, 0, 0])
    assert np.allclose(mae, expected)


def test_arm_mae_uses_overlapping_horizon() -> None:
    # Predicted horizon shorter than the expert tail near the episode end.
    pred = np.ones((3, 6))
    gt = np.ones((16, 6))
    mae = ol.arm_chunk_mae(pred, gt)  # compares only the 3 overlapping steps
    assert np.allclose(mae, 0.0)


def test_arm_mae_empty_overlap_is_nan() -> None:
    mae = ol.arm_chunk_mae(np.empty((0, 6)), np.ones((5, 6)))
    assert mae.shape == (6,) and np.isnan(mae).all()


def test_gripper_agreement_perfect_and_inverted() -> None:
    gt = np.array([0.0, 0.0, 1.0, 1.0])
    assert ol.gripper_agreement(gt, gt) == 1.0
    assert ol.gripper_agreement(1.0 - gt, gt) == 0.0


def test_gripper_agreement_thresholds_at_half() -> None:
    pred = np.array([0.49, 0.51, 0.9, 0.1])   # -> open, closed, closed, open
    gt = np.array([0.0, 1.0, 1.0, 0.0])       # -> open, closed, closed, open
    assert ol.gripper_agreement(pred, gt) == 1.0


def test_gripper_agreement_empty_is_nan() -> None:
    assert np.isnan(ol.gripper_agreement(np.empty(0), np.empty(0)))


def test_aggregate_means_over_ticks() -> None:
    maes = [np.array([0.0, 0.2, 0, 0, 0, 0]), np.array([0.0, 0.4, 0, 0, 0, 0])]
    accs = [1.0, 0.5]
    out = ol.aggregate(maes, accs)
    assert out["n_ticks"] == 2
    assert np.isclose(out["arm_mae_per_joint_rad"][1], 0.3)
    assert np.isclose(out["gripper_agreement"], 0.75)


def test_aggregate_empty_is_nan_safe() -> None:
    out = ol.aggregate([], [])
    assert out["n_ticks"] == 0
    assert np.isnan(out["arm_mae_overall_rad"])
    assert np.isnan(out["gripper_agreement"])
