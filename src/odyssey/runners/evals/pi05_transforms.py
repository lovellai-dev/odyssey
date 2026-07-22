"""Pure π0.5 (openpi) obs/action transforms for the LIBERO eval recipe.

π0.5-LIBERO shares the **same Franka Panda convention** as GR00T-N1.7-LIBERO and
OpenVLA (see ``docs/pi05-scoping.md`` → "Mapeo de espacio de acción"): the 8-D
proprio state (eef pos + eef quat→axis-angle + 2 gripper finger qpos) and the
7-DoF OSC_POSE action ``[dx,dy,dz,droll,dpitch,dyaw,gripper]``. So this module
does NOT re-derive the kinematics — it **reuses** ``quat_xyzw_to_axis_angle``
from ``gr00t_transforms`` and only adds the two things π0.5 does differently:

  * **wire format** — openpi's ``LiberoInputs`` expects flat ``observation/*``
    keys (``observation/image``, ``observation/wrist_image``, ``observation/state``,
    ``prompt``) and a single 8-D state vector, NOT GR00T's dotted ``state.x`` /
    ``video.image`` fan-out;
  * **NO gripper fix-up** — π0.5's checkpoint was trained on data already in
    LIBERO's gripper convention and its ``norm_stats`` fix the scale, so the
    action is sliced to 7-D and passed to ``env.step`` verbatim (no
    normalize/binarize/invert, unlike GR00T's ``gr00t_action_to_libero``).
    ⚠ Verify gripper polarity on the first GPU rollout (silent-failure footgun).

numpy-only (via ``gr00t_transforms``) so it imports and unit-tests without
openpi / torch / a GPU.
"""

from __future__ import annotations

import numpy as np

# Reuse the shared Franka Panda kinematics rather than duplicating it. This is
# the exact same state convention NVIDIA's LIBERO eval + openpi's LiberoInputs
# build: quat (xyzw) -> axis-angle.
from odyssey.runners.evals.gr00t_transforms import quat_xyzw_to_axis_angle

# openpi LiberoInputs wire keys (examples/libero/main.py in Physical-Intelligence/openpi).
PI05_IMAGE_KEY = "observation/image"
PI05_WRIST_IMAGE_KEY = "observation/wrist_image"
PI05_STATE_KEY = "observation/state"
PI05_PROMPT_KEY = "prompt"

# LIBERO's native action width applied to env.step (dx,dy,dz,droll,dpitch,dyaw,gripper).
LIBERO_ACTION_DIM = 7


def build_pi05_libero_state(*, eef_pos, eef_quat_xyzw, gripper_qpos) -> np.ndarray:
    """8-D Franka Panda proprio state openpi's ``LiberoInputs`` consumes.

    ``[eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)]`` — identical to
    the vector GR00T's ``build_gr00t_libero_obs`` builds; only the packaging into
    the wire dict differs (one flat vector here vs. per-axis ``state.*`` there).
    """
    return np.concatenate([
        np.asarray(eef_pos, dtype=np.float32).reshape(-1)[:3],
        quat_xyzw_to_axis_angle(eef_quat_xyzw),
        np.asarray(gripper_qpos, dtype=np.float32).reshape(-1)[:2],
    ]).astype(np.float32)


def build_pi05_libero_obs(*, image, wrist_image, eef_pos, eef_quat_xyzw,
                          gripper_qpos, instruction) -> dict:
    """FLAT openpi ``observation/*`` obs dict for π0.5-LIBERO.

    Images are ``uint8 (H, W, C)`` (already 180°-flipped by the caller, matching
    the GR00T-LIBERO recipe). openpi's server-side ``LiberoInputs`` handles the
    third (right-wrist) camera mask, the pad-to-32 of the state, and normalization
    with the checkpoint's ``norm_stats`` — so this stays a thin, single-arm packer.
    """
    return {
        PI05_IMAGE_KEY: np.asarray(image, dtype=np.uint8),
        PI05_WRIST_IMAGE_KEY: np.asarray(wrist_image, dtype=np.uint8),
        PI05_STATE_KEY: build_pi05_libero_state(
            eef_pos=eef_pos, eef_quat_xyzw=eef_quat_xyzw, gripper_qpos=gripper_qpos,
        ),
        PI05_PROMPT_KEY: str(instruction),
    }


def _pi05_action_chunk(chunk) -> np.ndarray:
    """Coerce an openpi inference result into a ``(horizon, >=7)`` float array.

    The websocket server returns ``{"actions": ndarray}`` (openpi ``LiberoOutputs``
    already slices to 7-D); accept either that dict or a bare array, and normalize
    to a 2-D ``(horizon, width)`` so step indexing is uniform.
    """
    arr = chunk["actions"] if isinstance(chunk, dict) and "actions" in chunk else chunk
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:  # a single flat action -> a horizon-1 chunk
        arr = arr[None, :]
    return arr.reshape(arr.shape[0], -1)


def pi05_action_to_libero(chunk, k, *, translation_only=False, **_legacy) -> np.ndarray:
    """Map step ``k`` of a π0.5 action chunk to LIBERO's 7-DoF OSC_POSE action.

    Slice the (already de-normalized) action to its first 7 dims and pass it
    through **without** any gripper fix-up — π0.5's gripper is baked to LIBERO's
    convention in the checkpoint (contrast ``gr00t_action_to_libero``, which
    normalizes+inverts because GR00T emits the gripper in ``[0,1]``).
    ``translation_only`` zeroes rotation and forces the gripper open (de-risk knob);
    ``**_legacy`` swallows retired scale/gripper kwargs so old configs stay accepted.
    """
    vec = _pi05_action_chunk(chunk)[int(k)]
    action = vec[:LIBERO_ACTION_DIM].astype(np.float32)
    if action.shape[0] < LIBERO_ACTION_DIM:  # pad a short row (shouldn't happen)
        action = np.concatenate([action, np.zeros(LIBERO_ACTION_DIM - action.shape[0], np.float32)])
    if translation_only:
        action[3:6] = 0.0
        action[6] = 1.0  # gripper forced open
    return action.astype(np.float32)
