"""Pure GR00T obs/action transforms for the Isaac Lab + LIBERO eval recipes.

NO heavy deps (numpy only) so it imports and unit-tests without gr00t / isaaclab /
torch. This is the design-agnostic core of the GR00T<->env adapters — it carried
over unchanged when odyssey#17 moved from an in-process runner onto Jeanine's
subprocess ``IsaacLabRunner`` contract; only the surrounding harness changed.

TWO embodiments live here, side by side:

* **Isaac / DROID** (``oxe_droid_relative_eef_relative_joint``) — the original
  odyssey#17 path. NESTED obs dict; the server returns action keys ``eef_9d`` /
  ``gripper_position`` / ``joint_position`` as RELATIVE eef deltas (rot6d = first
  two columns of R, Zhou et al. 2019; identity rotation -> ``[1,0,0, 0,1,0]``).
  Everything above ``# --- LIBERO`` is this path and is UNCHANGED.

* **LIBERO** (``LIBERO_PANDA``) — the GR00T-N1.7-LIBERO checkpoint. Its real
  contract (``examples/LIBERO/modality.json`` in NVIDIA/Isaac-GR00T) is different:
  FLAT dotted-key obs (``video.image`` / ``state.x`` / ...), single video frame,
  state = eef pos + eef quat→axis-angle + 2 gripper finger qpos, and the action is
  ALREADY in LIBERO's 7-DoF OSC_POSE space (ABSOLUTE) — so NO rot6d / eef-delta
  step; only a gripper fix-up. Mirrors ``gr00t/eval/sim/LIBERO/libero_env.py``.
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


def quat_xyzw_to_wxyz(q) -> np.ndarray:
    """robosuite/LIBERO quaternion order (xyzw) -> GR00T's wxyz.

    robosuite exposes ``robot0_eef_quat`` as xyzw; ``build_gr00t_obs`` /
    ``quat_wxyz_to_rot6d`` expect wxyz. Roll the scalar term to the front.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


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


def _gr00t_eef_delta(chunk, k, *, pos_scale, rot_scale):
    """Step k of a GR00T chunk -> (dpos[3], drot_axis_angle[3]) relative EEF delta.

    Uses eef_9d (relative xyz + rot6d); ignores joint_position (envs are EEF/IK).
    Shared by the Isaac and LIBERO action maps — both consume the same 6-D delta.
    """
    eef = np.asarray(chunk["eef_9d"]).reshape(-1, 9)[k]
    dpos = eef[0:3].astype(np.float64) * pos_scale
    drot = rot6d_to_axis_angle(eef[3:9]).astype(np.float64) * rot_scale
    return dpos, drot


def _gr00t_gripper_bit(chunk, k, *, threshold, open_val, close_val):
    """GR00T ``gripper_position`` in [0,1] -> a binary open/close command.

    Polarity (justified + configurable): a LOW value means "open", so
    grip < threshold ⇒ open_val, else close_val. If a checkpoint's gripper
    convention is inverted, flip it with no code change via the kwargs. NOTE:
    confirm the direction against the GR00T server on the first live rollout
    (silent-failure footgun if reversed — the arm moves but never grasps).
    """
    grip = float(np.asarray(chunk["gripper_position"]).reshape(-1, 1)[k, 0])
    return open_val if grip < threshold else close_val


def gr00t_action_to_isaac(chunk, k, *, pos_scale=1.0, rot_scale=1.0,
                          gripper_threshold=0.5, gripper_open=1.0,
                          gripper_close=-1.0) -> np.ndarray:
    """Map step k of GR00T's action chunk to the env's 7-D IK-rel + gripper action.

    Isaac IK-rel visuomotor convention: +1 open / -1 close.
    """
    dpos, drot = _gr00t_eef_delta(chunk, k, pos_scale=pos_scale, rot_scale=rot_scale)
    g = _gr00t_gripper_bit(
        chunk, k, threshold=gripper_threshold,
        open_val=gripper_open, close_val=gripper_close,
    )
    return np.concatenate([dpos, drot, [g]]).astype(np.float32)


# ---------------------------------------------------------------------------
# --- LIBERO (embodiment LIBERO_PANDA) --------------------------------------
# The GR00T-N1.7-LIBERO checkpoint's schema (examples/LIBERO/modality.json):
#   state  : x,y,z (eef pos) + roll,pitch,yaw (eef quat -> axis-angle) + gripper(2 finger qpos)
#   action : x,y,z,roll,pitch,yaw,gripper  (ABSOLUTE, LIBERO's own 7-DoF space)
#   video  : image, wrist_image (single frame, T=1)
#   lang   : annotation.human.action.task_description
# No rot6d / eef-delta plumbing: the server returns the LIBERO action directly; we
# only fix the gripper channel. Mirrors NVIDIA's gr00t/eval/sim/LIBERO/libero_env.py.
# ---------------------------------------------------------------------------

# The 6 pose axes of the LIBERO state/action (each a scalar); gripper is separate.
LIBERO_POSE_AXES = ("x", "y", "z", "roll", "pitch", "yaw")


def quat_xyzw_to_axis_angle(q) -> np.ndarray:
    """robosuite eef quat (xyzw) -> axis-angle (3,), matching robosuite ``quat2axisangle``.

    NVIDIA's ``libero_env`` builds state roll/pitch/yaw via ``quat2axisangle(eef_quat)``;
    replicated here so the wire state matches what the checkpoint was trained on.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q = q / (np.linalg.norm(q) + 1e-12)
    w = np.clip(q[3], -1.0, 1.0)                 # xyzw -> scalar is last
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den < 1e-8:                               # ~identity rotation
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * np.arccos(w) / den).astype(np.float32)


def build_gr00t_libero_obs(*, image, wrist_image, eef_pos, eef_quat_xyzw,
                           gripper_qpos, instruction) -> dict:
    """FLAT dotted-key GR00T obs for the LIBERO_PANDA embodiment (modality.json).

    Mirrors ``libero_env._process_observation``: images are already 180°-flipped by
    the caller; state = eef_pos(3) + quat2axisangle(eef_quat)(3) + gripper_qpos(2).
    Array shapes follow the working Isaac ``PolicyClient`` convention — a leading
    ``(batch=1, time=1)`` pair on every tensor. ⚠ VERIFY the (batch,time) rank on
    the first VM smoke: if the sim-policy-wrapped server rejects it, drop a dim.
    """
    img = np.asarray(image, dtype=np.uint8)
    wrist = np.asarray(wrist_image, dtype=np.uint8)
    pose = np.concatenate([
        np.asarray(eef_pos, dtype=np.float32).reshape(3),
        quat_xyzw_to_axis_angle(eef_quat_xyzw),
    ]).astype(np.float32)                        # x,y,z,roll,pitch,yaw
    grip = np.asarray(gripper_qpos, dtype=np.float32).reshape(-1)[:2]
    obs = {
        "video.image": img[None, None],          # (1,1,H,W,3) uint8
        "video.wrist_image": wrist[None, None],  # (1,1,H,W,3) uint8
        # sim-policy-wrapper validates a BATCH of strings — one level, not [[str]]:
        # each batch item must be a str (a nested [str] fails "Language batch item
        # must be a string. Got <class 'list'>").
        "annotation.human.action.task_description": [str(instruction)],
        "state.gripper": grip.reshape(1, 1, 2),  # both finger joints
    }
    for i, axis in enumerate(LIBERO_POSE_AXES):
        obs[f"state.{axis}"] = pose[i].reshape(1, 1, 1)
    return obs


def normalize_gripper_action(g, *, binarize: bool = True) -> float:
    """GR00T gripper in [0,1] -> [-1,1] (NVIDIA ``normalize_gripper_action``)."""
    v = 2.0 * float(g) - 1.0
    if binarize:
        v = 1.0 if v >= 0.0 else -1.0
    return v


def invert_gripper_action(g) -> float:
    """Flip gripper sign (NVIDIA ``invert_gripper_action``) for LIBERO's convention."""
    return -float(g)


def _libero_action_vec(chunk, k) -> np.ndarray:
    """Step k of a GR00T LIBERO action chunk as a raw 7-vec [x,y,z,roll,pitch,yaw,gripper].

    Accepts either per-axis keys (mirrors the Isaac path's bare modality keys) or a
    single composed ``action`` array. ⚠ VERIFY which the sim-policy-wrapped server
    returns on the first smoke and drop the branch that doesn't apply.
    """
    if "action" in chunk:                        # composed (H,7) fallback
        return np.asarray(chunk["action"], np.float64).reshape(-1, 7)[k]
    axes = (*LIBERO_POSE_AXES, "gripper")
    return np.array(
        [float(np.asarray(chunk[a]).reshape(-1)[k]) for a in axes], np.float64
    )


def gr00t_action_to_libero(chunk, k, *, translation_only=False,
                           binarize_gripper=True, **_legacy) -> np.ndarray:
    """Map step k of a GR00T LIBERO action chunk to LIBERO's 7-DoF OSC_POSE action.

    The GR00T-N1.7-LIBERO checkpoint emits the action ALREADY in LIBERO's action
    space ``[x,y,z,roll,pitch,yaw,gripper]`` (ABSOLUTE) — so there is NO eef-delta /
    rot6d step (that path is the Isaac ``gr00t_action_to_isaac`` above). We only fix
    the gripper channel to LIBERO's convention: normalize [0,1]->[-1,1] then invert
    sign, exactly like NVIDIA's ``libero_env`` (``normalize_gripper_action`` +
    ``invert_gripper_action``). ``translation_only`` zeroes rotation and forces the
    gripper open (a de-risking knob). ``**_legacy`` swallows the retired
    pos_scale/rot_scale/gripper_* kwargs so old configs/argv stay accepted.
    ⚠ Validate the gripper polarity on the first live rollout (silent-failure footgun:
    a reversed sign makes the arm move but never grasp).
    """
    vec = _libero_action_vec(chunk, k)
    pose = vec[:6].astype(np.float64)
    if translation_only:
        pose[3:6] = 0.0
        return np.concatenate([pose, [1.0]]).astype(np.float32)  # gripper forced open
    g = invert_gripper_action(normalize_gripper_action(vec[6], binarize=binarize_gripper))
    return np.concatenate([pose, [g]]).astype(np.float32)
