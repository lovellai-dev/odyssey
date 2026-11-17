"""Pure WALL-OSS (X Square Robot) obs/action transforms for the LIBERO eval recipe.

WALL-OSS is served out-of-process by the official ``wall-x`` repo's WebSocket
policy server (``scripts/run_serving.sh --env LIBERO``,
``wall_x/_vendor/harrix/serving``). The wire is openpi-*shaped* (msgpack frames,
a metadata frame on connect) but NOT openpi-compatible: the observation schema
differs and numpy arrays travel in the ``msgpack-numpy`` library's ext-dict
format, so ``openpi_client`` can't be reused. This module is the thin, numpy +
stdlib packer/decoder for that wire.

Wire contract (``WallXPolicy.infer`` with a single-arm LIBERO train config)::

    request  {"state": {"follow2_pos": [x, y, z, r1, r2, r3, g1, g2]},
              "views": {"face_view": "<b64 jpeg>", "right_wrist_view": "<b64 jpeg>"},
              "instruction": "<task>"}
    response {"predict_action": ndarray (T, D) | (1, T, D), "server_timing": {...}}

* **state** — ``follow2_pos`` is the right arm: eef pos(3) + rotation(3) +
  gripper qpos(2). The rotation slot carries robosuite **axis-angle**, exactly
  what wall-x's own LIBERO adapter feeds the model
  (``harrix/envs/libero_common.encode_raw_obs`` -> ``eef_axisangle``), even
  though the server names that slot "euler".
* **views** — LIBERO agentview / eye-in-hand, 180°-rotated (the caller flips,
  same as every other LIBERO pilot), JPEG-encoded and base64'd (the server's
  default ``IMAGE_PASSING_MODE=base64``). Pure str/float payload -> plain
  ``msgpack`` is enough client-side.
* **action** — single-arm LIBERO configs get the RAW chunk back (no robot
  serialization). Its row layout follows the checkpoint's ``dof_config``; the
  LIBERO template (``workspace/example/libero.yml``) is ``[dpos(3),
  drot(3, ZYX euler), gripper(1), action_padding(19)]`` = 26 wide. The first 7
  columns are LIBERO's OSC_POSE delta; padding is dropped. Actions arrive
  de-normalized (the server's model wrapper owns the ``norm_key`` stats).
  ⚠ Verify gripper polarity on the first GPU rollout — ``invert_gripper`` /
  ``binarize_gripper`` mirror wall-x's ``WALLX_LIBERO_*_GRIPPER`` env knobs.

numpy + PIL only (PIL lazily, for the JPEG wire encoding) so it imports and
unit-tests without wall-x / torch / a GPU.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

# Reuse the shared Franka Panda kinematics (robosuite quat2axisangle) rather than
# duplicating it — the same state convention GR00T / π0.5 / wall-x LIBERO use.
from odyssey.runners.evals.gr00t_transforms import quat_xyzw_to_axis_angle

# Wire keys (wall_x/_vendor/harrix/serving/policy/wall_x_policy.py + robot.py).
WALL_OSS_STATE_KEY = "state"
WALL_OSS_VIEWS_KEY = "views"
WALL_OSS_INSTRUCTION_KEY = "instruction"
WALL_OSS_RIGHT_ARM_KEY = "follow2_pos"  # follow1 = left arm, follow2 = right arm
WALL_OSS_FACE_VIEW = "face_view"
WALL_OSS_WRIST_VIEW = "right_wrist_view"
WALL_OSS_ACTION_KEY = "predict_action"

# Server defaults for --env LIBERO (launch_serving.DEFAULT_CONFIGS[LIBERO]).
WALL_OSS_DEFAULT_PORT = 32195
WALL_OSS_DEFAULT_CHUNK_SIZE = 10

# LIBERO's native action width applied to env.step (dx,dy,dz,droll,dpitch,dyaw,gripper).
LIBERO_ACTION_DIM = 7


# ---------------------------------------------------------------------------
# Request packing
# ---------------------------------------------------------------------------

def encode_image_b64_jpeg(image: Any, *, quality: int = 95) -> str:
    """Encode an ``(H, W, 3) uint8`` RGB array as the base64 JPEG the server decodes.

    The server runs ``cv2.imdecode`` + BGR->RGB, so a standard RGB JPEG round-trips.
    """
    from PIL import Image  # lazy: pillow is a dev/eval dep, not a core one

    arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_wall_oss_libero_state(*, eef_pos: Any, eef_quat_xyzw: Any,
                                gripper_qpos: Any) -> list[float]:
    """8-D ``follow2_pos``: eef pos(3) + axis-angle(3) + gripper finger qpos(2)."""
    vec = np.concatenate([
        np.asarray(eef_pos, dtype=np.float64).reshape(-1)[:3],
        quat_xyzw_to_axis_angle(eef_quat_xyzw).astype(np.float64),
        np.asarray(gripper_qpos, dtype=np.float64).reshape(-1)[:2],
    ])
    return [float(v) for v in vec]


def build_wall_oss_libero_request(
    *,
    image: Any,
    wrist_image: Any,
    eef_pos: Any,
    eef_quat_xyzw: Any,
    gripper_qpos: Any,
    instruction: str,
    jpeg_quality: int = 95,
) -> dict[str, Any]:
    """One msgpack-safe WALL-OSS policy request for a single LIBERO step.

    Images are ``uint8 (H, W, C)`` already 180°-flipped by the caller.
    """
    return {
        WALL_OSS_STATE_KEY: {
            WALL_OSS_RIGHT_ARM_KEY: build_wall_oss_libero_state(
                eef_pos=eef_pos, eef_quat_xyzw=eef_quat_xyzw, gripper_qpos=gripper_qpos,
            ),
        },
        WALL_OSS_VIEWS_KEY: {
            WALL_OSS_FACE_VIEW: encode_image_b64_jpeg(image, quality=jpeg_quality),
            WALL_OSS_WRIST_VIEW: encode_image_b64_jpeg(wrist_image, quality=jpeg_quality),
        },
        WALL_OSS_INSTRUCTION_KEY: str(instruction),
    }


# ---------------------------------------------------------------------------
# Response decoding
# ---------------------------------------------------------------------------

def _get(d: dict[Any, Any], key: str) -> Any:
    """Fetch ``key`` from a msgpack dict whose keys may be ``bytes`` or ``str``."""
    if key in d:
        return d[key]
    return d.get(key.encode("ascii"))


def _as_str(value: Any) -> str:
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


def decode_msgpack_numpy(obj: dict[Any, Any]) -> Any:
    """``msgpack.unpackb`` ``object_hook`` for the msgpack-numpy library's ext dicts.

    The wall-x server packs arrays via ``msgpack_numpy.patch()``:
    ``{b"nd": True, b"type": dtype.str, b"kind": ..., b"shape": [...], b"data": bytes}``
    (``nd: False`` for a numpy scalar). Decoding that here avoids a hard
    ``msgpack-numpy`` dependency. Non-array dicts are returned unchanged.
    """
    nd = _get(obj, "nd")
    if nd is None:
        return obj
    dtype = np.dtype(_as_str(_get(obj, "type")))
    data = _get(obj, "data")
    if nd:
        shape = tuple(int(s) for s in _get(obj, "shape"))
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    return np.frombuffer(data, dtype=dtype)[0]


def wall_oss_chunk_from_response(resp: Any) -> np.ndarray:
    """Coerce a WALL-OSS response into a ``(horizon, width)`` float array.

    Accepts the server dict (``{"predict_action": ...}``, str or bytes key) or a
    bare array; a leading batch dim of 1 is squeezed. A ``str`` response is the
    server's error frame (it sends the traceback before closing) -> raise.
    """
    if isinstance(resp, (str, bytes)):
        raise RuntimeError(f"WALL-OSS policy server error:\n{_as_str(resp)}")
    if isinstance(resp, dict):
        action = _get(resp, WALL_OSS_ACTION_KEY)
        if action is None:
            raise KeyError(
                f"WALL-OSS response has no {WALL_OSS_ACTION_KEY!r} "
                f"(keys={sorted(_as_str(k) for k in resp)})"
            )
        resp = action
    arr = np.asarray(resp, dtype=np.float64)
    if arr.ndim == 3:  # (batch=1, T, D)
        arr = arr[0]
    if arr.ndim == 1:  # a single flat action -> a chunk of one
        arr = arr[None, :]
    return arr.reshape(arr.shape[0], -1)


def wall_oss_action_to_libero(
    chunk: Any,
    k: int,
    *,
    binarize_gripper: bool = False,
    invert_gripper: bool = False,
    translation_only: bool = False,
) -> np.ndarray:
    """Map step ``k`` of a WALL-OSS chunk to LIBERO's 7-DoF OSC_POSE action.

    Slices ``[dpos(3), drot(3), gripper(1)]`` off the front of the row (the
    LIBERO ``dof_config`` order; ``action_padding`` is the tail). Gripper
    handling mirrors wall-x's ``gripper_model_to_libero_osc``: passthrough by
    default, optional sign-binarize (``0 -> -1``) and invert.
    ``translation_only`` zeroes rotation and forces the gripper open (LIBERO:
    -1 = open) — the same de-risk knob the other pilots carry.
    """
    vec = wall_oss_chunk_from_response(chunk)[int(k)]
    if vec.shape[0] < LIBERO_ACTION_DIM:
        raise ValueError(
            f"WALL-OSS action row has {vec.shape[0]} dims; LIBERO needs >= "
            f"{LIBERO_ACTION_DIM} ([dpos(3), drot(3), gripper(1)])"
        )
    action = vec[:LIBERO_ACTION_DIM].astype(np.float32)
    grip = float(action[6])
    if binarize_gripper:
        grip = -1.0 if abs(grip) < 1e-6 else float(np.sign(grip))
    if invert_gripper:
        grip = -grip
    action[6] = grip
    if translation_only:
        action[3:6] = 0.0
        action[6] = -1.0
    return action
