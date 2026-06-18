"""Unit tests for the GR00T obs/action transforms (odyssey#17 eval recipe).

The transforms back the blessed GR00T Isaac Lab eval script
(``scripts/eval/gr00t_isaac_eval.py``). They are pure-numpy and live beside
the script (run under Isaac Sim's bundled python), so the test imports the
module by path. numpy is optional in the core env — skip when absent, the
same convention as the Isaac-GR00T conformance tests.
"""
from __future__ import annotations

import os
import sys

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "eval"))
import gr00t_transforms as T  # noqa: E402


def _Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def test_identity_quat_to_rot6d():
    assert np.allclose(T.quat_wxyz_to_rot6d([1, 0, 0, 0]), [1, 0, 0, 0, 1, 0], atol=1e-6)


def test_identity_rot6d_to_axis_angle_zero():
    assert np.allclose(T.rot6d_to_axis_angle([1, 0, 0, 0, 1, 0]), [0, 0, 0], atol=1e-6)


def test_rz90_rot6d_and_axis_angle():
    R = _Rz(np.pi / 2)
    assert np.allclose(T.matrix_to_rot6d(R), [0, 1, 0, -1, 0, 0], atol=1e-6)
    assert np.allclose(T.matrix_to_axis_angle(R), [0, 0, np.pi / 2], atol=1e-5)


def test_rot6d_matrix_roundtrip_proper_rotation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = T.quat_wxyz_to_matrix(q)
        R2 = T.rot6d_to_matrix(T.matrix_to_rot6d(R))
        assert np.allclose(R, R2, atol=1e-5)
        assert np.isclose(np.linalg.det(R2), 1.0, atol=1e-5)


def test_build_gr00t_obs_nested_shapes():
    ext = np.zeros((2, 16, 16, 3), np.uint8)
    obs = T.build_gr00t_obs(exterior_seq=ext, wrist_seq=ext, eef_pos=np.zeros(3),
                            eef_quat_wxyz=[1, 0, 0, 0], gripper=0.0,
                            arm_joints=np.zeros(7), instruction="stack")
    assert obs["video"]["exterior_image_1_left"].shape == (1, 2, 16, 16, 3)
    assert obs["video"]["exterior_image_1_left"].dtype == np.uint8
    assert obs["state"]["eef_9d"].shape == (1, 1, 9) and obs["state"]["eef_9d"].dtype == np.float32
    assert obs["state"]["gripper_position"].shape == (1, 1, 1)
    assert obs["state"]["joint_position"].shape == (1, 1, 7)
    assert obs["language"]["annotation.language.language_instruction"] == [["stack"]]
    assert np.allclose(obs["state"]["eef_9d"].reshape(9)[3:], [1, 0, 0, 0, 1, 0], atol=1e-6)


def test_gr00t_action_to_isaac_shape_scale_gripper():
    eef = np.array([0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0], np.float32)  # identity rot6d
    chunk = {
        "eef_9d": np.tile(eef, (T.ACTION_HORIZON, 1))[None],
        "gripper_position": np.zeros((1, T.ACTION_HORIZON, 1), np.float32),
        "joint_position": np.zeros((1, T.ACTION_HORIZON, 7), np.float32),
    }
    a = T.gr00t_action_to_isaac(chunk, 0, pos_scale=2.0, rot_scale=1.0, gripper_threshold=0.5)
    assert a.shape == (7,)
    assert np.allclose(a[0:3], [0.2, 0.4, 0.6], atol=1e-6)   # pos_scale applied
    assert np.allclose(a[3:6], [0, 0, 0], atol=1e-6)         # identity rot -> 0 axis-angle
    assert a[6] == 1.0                                       # grip 0 < 0.5 -> open
