"""Unit tests for the pure metric/transform helpers of the Cosmos3 open-loop eval.

The script lives beside its mission (examples/drugsort-cosmos3-train/) rather than
on the package path, so it is loaded by file path. Only the numpy-only helpers are
exercised here — the CLI's server/FK/dataset IO is integration-tested on the box.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "drugsort-cosmos3-train"
    / "eval_openloop_gt_cosmos3.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("eval_openloop_gt_cosmos3", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def test_rot6d_to_matrix_identity_is_orthonormal():
    # rot6d for identity: first two columns of I3.
    r = mod.rot6d_to_matrix([1, 0, 0, 0, 1, 0])
    assert np.allclose(r, np.eye(3), atol=1e-6)
    # columns orthonormal, det +1
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(r), 1.0, atol=1e-6)


def test_rot6d_to_matrix_gram_schmidt_orthogonalises():
    # non-orthogonal input still yields an orthonormal matrix
    r = mod.rot6d_to_matrix([2, 0, 0, 1, 1, 0])
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-6)


def test_geodesic_deg_mat_zero_for_equal():
    r = mod.rot6d_to_matrix([1, 0, 0, 0, 1, 0])
    assert mod.geodesic_deg_mat(r, r) == pytest.approx(0.0, abs=1e-2)


def test_geodesic_deg_mat_ninety_degrees():
    # identity vs 90deg about z: R = [[0,-1,0],[1,0,0],[0,0,1]] -> cols [0,1,0],[-1,0,0]
    ident = mod.rot6d_to_matrix([1, 0, 0, 0, 1, 0])
    rot_z90 = mod.rot6d_to_matrix([0, 1, 0, -1, 0, 0])
    assert mod.geodesic_deg_mat(ident, rot_z90) == pytest.approx(90.0, abs=1e-3)


def test_axis_angle_to_matrix_zero_is_identity():
    assert np.allclose(mod.axis_angle_to_matrix([0, 0, 0]), np.eye(3), atol=1e-9)


def test_axis_angle_to_matrix_ninety_about_z():
    import math
    r = mod.axis_angle_to_matrix([0, 0, math.pi / 2])
    assert np.allclose(r, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)


def test_axis_angle_and_rot6d_agree():
    # a rot6d identity and an axis-angle zero rotation are the same R -> 0 geodesic
    import math
    r_rot6d = mod.rot6d_to_matrix([0, 1, 0, -1, 0, 0])           # 90deg about z
    r_aa = mod.axis_angle_to_matrix([0, 0, math.pi / 2])          # 90deg about z
    assert mod.geodesic_deg_mat(r_rot6d, r_aa) == pytest.approx(0.0, abs=1e-2)


def test_cartesian_metrics_mixed_widths():
    # pred is the 7-D server row [dpos(3), axis-angle(3), gripper(1)];
    # gt is the 10-D FK'd row [dpos(3), rot6d(6), gripper(1)] — same pose -> ~0 error.
    pred = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])       # identity rot, closed
    gt = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]])  # identity rot6d, closed
    tm, rm, gm = mod.cartesian_chunk_metrics(pred, gt)
    assert tm == pytest.approx(0.0)
    assert rm == pytest.approx(0.0, abs=1e-2)
    assert gm == pytest.approx(1.0)


def test_cartesian_chunk_metrics_zero_when_equal():
    chunk = np.zeros((4, 10), dtype=np.float64)
    chunk[:, 3:9] = [1, 0, 0, 0, 1, 0]  # identity rot6d
    chunk[:, 9] = 1.0
    tm, rm, gm = mod.cartesian_chunk_metrics(chunk, chunk)
    assert tm == pytest.approx(0.0)
    assert rm == pytest.approx(0.0, abs=1e-2)
    assert gm == pytest.approx(1.0)


def test_cartesian_chunk_metrics_translation_and_gripper():
    gt = np.zeros((3, 10))
    gt[:, 3:9] = [1, 0, 0, 0, 1, 0]
    gt[:, 9] = 1.0  # closed
    pred = gt.copy()
    pred[:, 0] = 0.1  # 0.1 m off in x each step
    pred[:, 9] = 0.0  # gripper disagrees on all steps
    tm, _rm, gm = mod.cartesian_chunk_metrics(pred, gt)
    assert tm == pytest.approx(0.1 / 3, abs=1e-6)  # mean over dpos block (x=0.1, y=z=0)
    assert gm == pytest.approx(0.0)


def test_cartesian_chunk_metrics_overlap_horizon():
    gt = np.zeros((2, 10))
    gt[:, 3:9] = [1, 0, 0, 0, 1, 0]
    pred = np.zeros((5, 10))
    pred[:, 3:9] = [1, 0, 0, 0, 1, 0]
    tm, _rm, _gm = mod.cartesian_chunk_metrics(pred, gt)  # min horizon = 2
    assert not np.isnan(tm)


def test_cartesian_chunk_metrics_empty_is_nan():
    tm, rm, gm = mod.cartesian_chunk_metrics(np.empty((0, 10)), np.empty((0, 10)))
    assert np.isnan(tm) and np.isnan(rm) and np.isnan(gm)


def test_aggregate_nan_safe():
    out = mod.aggregate([0.1, float("nan"), 0.3], [1.0, 2.0, float("nan")], [1.0, 0.0, 1.0])
    assert out["n_ticks"] == 3
    assert out["trans_mae_m"] == pytest.approx(0.2)  # nanmean(0.1,0.3)
    assert out["gripper_agreement"] == pytest.approx(2 / 3)


def test_aggregate_empty():
    out = mod.aggregate([], [], [])
    assert out["n_ticks"] == 0
    assert np.isnan(out["trans_mae_m"])
