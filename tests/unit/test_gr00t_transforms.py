"""Unit tests for the GR00T obs/action transforms (odyssey#17 eval recipe).

The transforms back the blessed GR00T Isaac Lab eval script
(``odyssey/runners/evals/gr00t_isaac_eval.py``). They are pure-numpy and live
beside the script (run under Isaac Sim's bundled python), so the test imports
the module by path. numpy is optional in the core env — skip when absent, the
same convention as the Isaac-GR00T conformance tests.
"""
from __future__ import annotations

import os
import sys

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "odyssey", "runners", "evals"))
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


def test_quat_xyzw_to_wxyz_rolls_scalar_to_front():
    # robosuite exposes eef_quat as xyzw; GR00T wants wxyz.
    assert np.allclose(T.quat_xyzw_to_wxyz([0, 0, 0, 1]), [1, 0, 0, 0])       # identity
    assert np.allclose(T.quat_xyzw_to_wxyz([0.1, 0.2, 0.3, 0.4]), [0.4, 0.1, 0.2, 0.3])


# ---------------------------------------------------------------------------
# LIBERO (embodiment LIBERO_PANDA). The checkpoint emits the action ALREADY in
# LIBERO's 7-DoF space; per-axis chunk keys, no eef-delta / rot6d.
# ---------------------------------------------------------------------------

_H = 16  # a representative action horizon


def _libero_chunk(*, grip: float, pose=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
    chunk = {a: np.full((_H, 1), pose[i], np.float32)
             for i, a in enumerate(T.LIBERO_POSE_AXES)}
    chunk["gripper"] = np.full((_H, 1), grip, np.float32)
    return chunk


def test_quat_xyzw_to_axis_angle_identity_is_zero():
    assert np.allclose(T.quat_xyzw_to_axis_angle([0, 0, 0, 1]), [0, 0, 0], atol=1e-6)


def test_quat_xyzw_to_axis_angle_z90():
    # 90° about +z: xyzw = [0, 0, sin(45°), cos(45°)] -> axis-angle [0,0,pi/2].
    s = np.sqrt(0.5)
    assert np.allclose(T.quat_xyzw_to_axis_angle([0, 0, s, s]), [0, 0, np.pi / 2], atol=1e-5)


def test_build_gr00t_libero_obs_flat_keys_and_shapes():
    img = np.zeros((256, 256, 3), np.uint8)
    obs = T.build_gr00t_libero_obs(
        image=img, wrist_image=img, eef_pos=[0.1, 0.2, 0.3],
        eef_quat_xyzw=[0, 0, 0, 1], gripper_qpos=[0.02, -0.02], instruction="pick up the can",
    )
    assert obs["video.image"].shape == (1, 1, 256, 256, 3)
    assert obs["video.image"].dtype == np.uint8
    assert obs["state.x"].shape == (1, 1, 1) and obs["state.gripper"].shape == (1, 1, 2)
    assert np.allclose(obs["state.x"].reshape(-1), [0.1])
    assert np.allclose(obs["state.roll"].reshape(-1), [0.0])   # identity quat -> 0 axis-angle
    assert obs["annotation.human.action.task_description"] == [["pick up the can"]]


def test_gr00t_action_to_libero_passthrough_pose_and_gripper_open():
    a = T.gr00t_action_to_libero(_libero_chunk(grip=0.0), 0)
    assert a.shape == (7,)
    assert np.allclose(a[0:6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-6)  # absolute passthrough
    # grip 0.0 -> normalize -1 -> invert +1
    assert a[6] == 1.0


def test_gr00t_action_to_libero_gripper_close():
    # grip ~1.0 -> normalize +1 -> invert -1
    assert T.gr00t_action_to_libero(_libero_chunk(grip=0.9), 0)[6] == -1.0


def test_gr00t_action_to_libero_accepts_composed_action_array():
    chunk = {"action": np.tile(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0], np.float32), (_H, 1))}
    a = T.gr00t_action_to_libero(chunk, 0)
    assert np.allclose(a[0:6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-6)
    assert a[6] == 1.0


def test_gr00t_action_to_libero_translation_only_zeros_rot_and_opens_gripper():
    a = T.gr00t_action_to_libero(_libero_chunk(grip=0.9), 0, translation_only=True)
    assert np.allclose(a[3:6], [0, 0, 0], atol=1e-6)   # rotation zeroed
    assert a[6] == 1.0                                 # gripper forced open despite grip=0.9


def test_gr00t_action_to_libero_swallows_legacy_kwargs():
    # Old configs/argv passed pos_scale/rot_scale/gripper_* — must not raise now.
    a = T.gr00t_action_to_libero(
        _libero_chunk(grip=0.0), 0, pos_scale=2.0, rot_scale=1.0,
        gripper_threshold=0.5, gripper_open=1.0, gripper_close=-1.0,
    )
    assert np.allclose(a[0:6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], atol=1e-6)
