"""Pure GR00T obs/action transforms for the Isaac Lab eval recipe (odyssey#17).

NO heavy deps (numpy only) so it imports and unit-tests without gr00t / isaaclab /
torch. This is the design-agnostic core of the GR00T<->Isaac adapter — it carried
over unchanged when odyssey#17 moved from an in-process runner onto Jeanine's
subprocess ``IsaacLabRunner`` contract; only the surrounding harness changed.

Embodiment: ``oxe_droid_relative_eef_relative_joint`` (GR00T-N1.7 DROID-family).
The GR00T policy server (``run_gr00t_server.py``) expects a NESTED obs dict and
returns ``(action, info)`` with action keys ``eef_9d`` / ``gripper_position`` /
``joint_position``. rot6d = first two columns of R (Zhou et al. 2019;
identity rotation -> ``[1,0,0, 0,1,0]``).
"""
from __future__ import annotations

import numpy as np

DROID_VIDEO_KEYS = ["exterior_image_1_left", "wrist_image_left"]
T_VIDEO = 2          # video delta_indices = [-15, 0]
ACTION_HORIZON = 40  # action delta_indices = range(40)


def quat_wxyz_to_matrix(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def matrix_to_rot6d(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)  # first two columns


def rot6d_to_matrix(r6) -> np.ndarray:
    r6 = np.asarray(r6, dtype=np.float64)
    a1, a2 = r6[0:3], r6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2p = a2 - np.dot(b1, a2) * b1
    b2 = a2p / (np.linalg.norm(a2p) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def matrix_to_axis_angle(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-6:
        return np.zeros(3, dtype=np.float32)
    if abs(angle - np.pi) < 1e-6:
        axis = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None))
        s = np.sign([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis * np.where(s == 0, 1.0, s)
        return (axis * angle).astype(np.float32)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(angle))
    return (axis * angle).astype(np.float32)


def quat_wxyz_to_rot6d(q) -> np.ndarray:
    return matrix_to_rot6d(quat_wxyz_to_matrix(q))


def rot6d_to_axis_angle(r6) -> np.ndarray:
    return matrix_to_axis_angle(rot6d_to_matrix(r6))


def build_gr00t_obs(*, exterior_seq, wrist_seq, eef_pos, eef_quat_wxyz,
                    gripper, arm_joints, instruction) -> dict:
    """Nested GR00T obs dict (video/state/language) the server validates.
    exterior_seq/wrist_seq: (T_VIDEO,H,W,3) uint8; language is list[list[str]] (B,T)."""
    eef_9d = np.concatenate([
        np.asarray(eef_pos, dtype=np.float32).reshape(3),
        quat_wxyz_to_rot6d(eef_quat_wxyz),
    ]).astype(np.float32)
    return {
        "video": {
            "exterior_image_1_left": np.asarray(exterior_seq, dtype=np.uint8)[None],
            "wrist_image_left": np.asarray(wrist_seq, dtype=np.uint8)[None],
        },
        "state": {
            "eef_9d": eef_9d.reshape(1, 1, 9),
            "gripper_position": np.asarray(gripper, dtype=np.float32).reshape(1, 1, 1),
            "joint_position": np.asarray(arm_joints, dtype=np.float32).reshape(1, 1, 7),
        },
        "language": {
            "annotation.language.language_instruction": [[str(instruction)]],
        },
    }


def gr00t_action_to_isaac(chunk, k, *, pos_scale=1.0, rot_scale=1.0,
                          gripper_threshold=0.5, gripper_open=1.0,
                          gripper_close=-1.0) -> np.ndarray:
    """Map step k of GR00T's action chunk to the env's 7-D IK-rel + gripper action.
    Uses eef_9d (relative xyz + rot6d) + gripper; ignores joint_position (env is EEF/IK)."""
    eef = np.asarray(chunk["eef_9d"]).reshape(-1, 9)[k]
    grip = float(np.asarray(chunk["gripper_position"]).reshape(-1, 1)[k, 0])
    dpos = eef[0:3].astype(np.float64) * pos_scale
    drot = rot6d_to_axis_angle(eef[3:9]).astype(np.float64) * rot_scale
    # Gripper polarity (justified + configurable): GR00T emits `gripper_position`
    # in [0, 1]; this recipe assumes a LOW value means "open", so
    # grip < gripper_threshold ⇒ command the env's binary gripper OPEN
    # (gripper_open = +1.0), else CLOSE (-1.0) — matching Isaac's IK-rel
    # visuomotor convention (+1 open / -1 close). If a checkpoint's gripper
    # convention is inverted, flip it with no code change via the gripper_open /
    # gripper_close kwargs. NOTE: confirm the direction against the GR00T server
    # on the first live Cosmos rollout (silent-failure footgun if reversed).
    g = gripper_open if grip < gripper_threshold else gripper_close
    return np.concatenate([dpos, drot, [g]]).astype(np.float32)
