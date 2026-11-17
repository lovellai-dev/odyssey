"""Pure MolmoAct2 (Ai2) obs/action transforms for the LIBERO eval recipe.

MolmoAct2 (Molmo2-ER reasoning backbone + flow-matching action expert,
Apache-2.0) is served out-of-process by a FastAPI ``/act`` server — the same
wire protocol Ai2 ships for DROID / YAM (``allenai/molmoact2`` →
``examples/droid/host_server_droid.py``), with a LIBERO schema. Ai2 ships no
LIBERO server, so ours lives next to the example
(``examples/quickstart-molmoact2/host_server_libero.py``).

Wire contract (json_numpy-encoded bodies)::

    GET  /act -> {"status": "ok", "repo_id", "norm_tag", "action_horizon", ...}
    POST /act
      {"image": ndarray(H,W,3) uint8, "wrist_image": ndarray(H,W,3) uint8,
       "state": ndarray(8,) float32, "instruction": str, "num_steps": int}
    -> {"actions": ndarray(N, 7) float32, "dt_ms": float}  |  {"error": str}

``MolmoAct2-LIBERO`` (``norm_tag="libero"``) shares the LIBERO convention every
other pilot here uses — confirmed from the checkpoint's ``norm_stats.json`` and
LeRobot's ``LiberoProcessorStep`` (Ai2's LIBERO eval stack):

  * **images** — agentview then wrist (order matters), 180°-rotated (the
    caller flips);
  * **state** — 8-D ``[eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)]``;
  * **action** — 7-D delta EEF ``[dx,dy,dz,droll,dpitch,dyaw,gripper]``,
    de-normalized server-side (q01/q99) and returned "in robot scale"; gripper
    already in LIBERO's ``[-1, 1]`` convention (not normalized in training).
    ``action_horizon`` = 10.

json_numpy is re-implemented here (stdlib ``json`` + numpy) so the client needs
no ``json-numpy`` dependency; the format is
``{"__numpy__": <b64 raw bytes>, "dtype": <descr>, "shape": [...]}``.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

# Shared Franka Panda kinematics (robosuite quat2axisangle) — the same state
# convention GR00T / π0.5 / LeRobot's LiberoProcessorStep build.
from odyssey.runners.evals.gr00t_transforms import quat_xyzw_to_axis_angle

# POST /act wire keys (examples/quickstart-molmoact2/host_server_libero.py).
MOLMOACT2_IMAGE_KEY = "image"
MOLMOACT2_WRIST_IMAGE_KEY = "wrist_image"
MOLMOACT2_STATE_KEY = "state"
MOLMOACT2_INSTRUCTION_KEY = "instruction"
MOLMOACT2_NUM_STEPS_KEY = "num_steps"
MOLMOACT2_ACTIONS_KEY = "actions"

MOLMOACT2_DEFAULT_PORT = 8000
MOLMOACT2_DEFAULT_CHUNK_SIZE = 10  # norm_stats.json: libero action_horizon
MOLMOACT2_DEFAULT_FLOW_STEPS = 10  # flow_matching_num_steps

# LIBERO's native action width applied to env.step (dx,dy,dz,droll,dpitch,dyaw,gripper).
LIBERO_ACTION_DIM = 7


# ---------------------------------------------------------------------------
# json_numpy wire codec (stdlib + numpy)
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)):
        arr = np.ascontiguousarray(obj)
        return {
            "__numpy__": base64.b64encode(arr.tobytes()).decode("ascii"),
            "dtype": np.lib.format.dtype_to_descr(arr.dtype),
            "shape": list(arr.shape),
        }
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _json_object_hook(dct: dict[str, Any]) -> Any:
    if "__numpy__" in dct:
        dtype = np.lib.format.descr_to_dtype(dct["dtype"])
        arr = np.frombuffer(base64.b64decode(dct["__numpy__"]), dtype=dtype)
        shape = dct["shape"]
        return arr.reshape(shape).copy() if shape else arr[0]
    return dct


def json_numpy_dumps(obj: Any) -> str:
    """``json.dumps`` with numpy arrays encoded the json_numpy way."""
    return json.dumps(obj, default=_json_default)


def json_numpy_loads(text: str | bytes) -> Any:
    """``json.loads`` decoding json_numpy array dicts back to ``np.ndarray``."""
    return json.loads(text, object_hook=_json_object_hook)


# ---------------------------------------------------------------------------
# Request packing
# ---------------------------------------------------------------------------

def build_molmoact2_libero_state(*, eef_pos: Any, eef_quat_xyzw: Any,
                                 gripper_qpos: Any) -> np.ndarray:
    """8-D ``[eef_pos(3), axis-angle(3), gripper_qpos(2)]`` float32 state."""
    return np.concatenate([
        np.asarray(eef_pos, dtype=np.float32).reshape(-1)[:3],
        quat_xyzw_to_axis_angle(eef_quat_xyzw),
        np.asarray(gripper_qpos, dtype=np.float32).reshape(-1)[:2],
    ]).astype(np.float32)


def build_molmoact2_libero_request(
    *,
    image: Any,
    wrist_image: Any,
    eef_pos: Any,
    eef_quat_xyzw: Any,
    gripper_qpos: Any,
    instruction: str,
    num_steps: int = MOLMOACT2_DEFAULT_FLOW_STEPS,
) -> dict[str, Any]:
    """One ``POST /act`` body (pre-json_numpy) for a single LIBERO step.

    Images are ``uint8 (H, W, 3)`` already 180°-flipped by the caller.
    """
    return {
        MOLMOACT2_IMAGE_KEY: np.ascontiguousarray(np.asarray(image, dtype=np.uint8)),
        MOLMOACT2_WRIST_IMAGE_KEY: np.ascontiguousarray(
            np.asarray(wrist_image, dtype=np.uint8)
        ),
        MOLMOACT2_STATE_KEY: build_molmoact2_libero_state(
            eef_pos=eef_pos, eef_quat_xyzw=eef_quat_xyzw, gripper_qpos=gripper_qpos,
        ),
        MOLMOACT2_INSTRUCTION_KEY: str(instruction),
        MOLMOACT2_NUM_STEPS_KEY: int(num_steps),
    }


# ---------------------------------------------------------------------------
# Response decoding
# ---------------------------------------------------------------------------

def molmoact2_chunk_from_response(resp: Any) -> np.ndarray:
    """Coerce a ``/act`` response into a ``(horizon, width)`` float array.

    Accepts the server dict (``{"actions": ...}``) or a bare array; a leading
    batch dim of 1 is squeezed. ``{"error": ...}`` raises.
    """
    if isinstance(resp, dict):
        if "error" in resp:
            raise RuntimeError(f"MolmoAct2 policy server error: {resp['error']}")
        if MOLMOACT2_ACTIONS_KEY not in resp:
            raise KeyError(
                f"MolmoAct2 response has no {MOLMOACT2_ACTIONS_KEY!r} (keys={sorted(resp)})"
            )
        resp = resp[MOLMOACT2_ACTIONS_KEY]
    arr = np.asarray(resp, dtype=np.float64)
    if arr.ndim == 3:  # (batch=1, T, D)
        arr = arr[0]
    if arr.ndim == 1:  # a single flat action -> a chunk of one
        arr = arr[None, :]
    return arr.reshape(arr.shape[0], -1)


def molmoact2_action_to_libero(
    chunk: Any,
    k: int,
    *,
    invert_gripper: bool = False,
    translation_only: bool = False,
) -> np.ndarray:
    """Map step ``k`` of a MolmoAct2 chunk to LIBERO's 7-DoF OSC_POSE action.

    The checkpoint already emits the LIBERO action layout in robot scale, so
    this is a slice + passthrough (⚠ verify gripper polarity on the first GPU
    rollout; ``invert_gripper`` is the escape hatch). ``translation_only``
    zeroes rotation and forces the gripper open (LIBERO: -1 = open).
    """
    vec = molmoact2_chunk_from_response(chunk)[int(k)]
    if vec.shape[0] < LIBERO_ACTION_DIM:
        raise ValueError(
            f"MolmoAct2 action row has {vec.shape[0]} dims; LIBERO needs "
            f"{LIBERO_ACTION_DIM} ([dpos(3), drot(3), gripper(1)])"
        )
    action = vec[:LIBERO_ACTION_DIM].astype(np.float32)
    if invert_gripper:
        action[6] = -action[6]
    if translation_only:
        action[3:6] = 0.0
        action[6] = -1.0
    return action
