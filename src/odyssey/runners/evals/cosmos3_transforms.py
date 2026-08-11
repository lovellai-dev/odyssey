"""Pure Cosmos 3 (WAM) obs/action transforms for the LIBERO eval recipe.

Cosmos 3 "policy" mode is a **World Action Model**: one query returns the
predicted future observation *video* AND an action chunk. The chunk is what the
eval replays; the video rides along in the response and is ignored here (the
sim is the world — we don't need the model's imagination of it).

Wire contract (cosmos-framework ``action_policy_server_libero``, HTTP)::

    POST /predict
      {"image": "<base64_png>", "prompt": "<task>", "domain_name": "<name>",
       "image_size": <int>}
    -> {"action": [[a0_0, ...], ...], "video": ["<base64_png>", ...]}

The server owns everything model-side: prompt augmentation (duration/FPS /
resolution metadata or the JSON prompt formatter), reflection padding, the
diffusion sampling, and **action de-normalization** (it loads the checkpoint's
``action_stats`` and inverts meanstd/minmax/quantile/quantile_rot). So this
module stays a thin packer/decoder — the family-wide knobs (``domain_name``,
``image_size``, chunk length) are *parameters*, never per-checkpoint code:

  * **obs** — the LIBERO recipes train on ``concat_view`` (third-person +
    wrist concatenated side-by-side); the client builds that concat and ships
    it as one base64 PNG. No proprioceptive state on the wire — the WAM is
    video-conditioned (contrast π0.5/GR00T's 8-D state vector).
  * **action** — the LIBERO SFT recipe emits 10-D ``frame_wise_relative``
    rows ``[dpos(3), rot6d(6), gripper(1)]`` (``quantile_rot``-normalized in
    training, de-normalized server-side). Decode rot6d -> axis-angle via the
    shared helpers in ``gr00t_transforms`` to LIBERO's 7-DoF OSC_POSE action.
    A >=7-but-<10-wide row is passed through verbatim (a family member already
    emitting env-native actions).
    ⚠ Verify gripper polarity and rot6d axis order on the first GPU rollout —
    the same silent-failure footgun π0.5 documented.

numpy + PIL only (PIL lazily, for the PNG wire encoding) so it imports and
unit-tests without cosmos-framework / torch / a GPU.
"""

from __future__ import annotations

import base64
import io

import numpy as np

# Reuse the shared rotation kinematics (Gram-Schmidt rot6d -> R -> axis-angle)
# rather than duplicating it; GR00T's LIBERO recipe established these helpers.
from odyssey.runners.evals.gr00t_transforms import rot6d_to_axis_angle

# POST /predict wire keys (action_policy_server_libero._prep_policy_item).
COSMOS3_IMAGE_KEY = "image"
COSMOS3_PROMPT_KEY = "prompt"
COSMOS3_DOMAIN_KEY = "domain_name"
COSMOS3_IMAGE_SIZE_KEY = "image_size"

# Family defaults — overridable per mission, resolved against GET /info.
COSMOS3_DEFAULT_DOMAIN = "libero"
COSMOS3_DEFAULT_IMAGE_SIZE = 256
COSMOS3_DEFAULT_CHUNK_SIZE = 16  # server _DEFAULT_ACTION_CHUNK_SIZE

# LIBERO's native action width applied to env.step (dx,dy,dz,droll,dpitch,dyaw,gripper).
LIBERO_ACTION_DIM = 7
# The LIBERO SFT recipe's action row: [dpos(3), rot6d(6), gripper(1)].
_ROT6D_ROW_DIM = 10


def encode_image_b64_png(image) -> str:
    """Encode an ``(H, W, 3) uint8`` array as the base64 PNG the server decodes."""
    from PIL import Image  # lazy: pillow is a dev/eval dep, not a core one

    arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_cosmos3_concat_view(image, wrist_image) -> np.ndarray:
    """Side-by-side ``concat_view`` (third-person | wrist) the recipes train on.

    Heights must match (both cameras render at the same ``camera_height`` in the
    LIBERO env); the server scales non-square multi-view images proportionally
    by height, so the concat is shipped at native resolution.
    """
    img = np.asarray(image, dtype=np.uint8)
    wrist = np.asarray(wrist_image, dtype=np.uint8)
    if img.shape[0] != wrist.shape[0]:
        raise ValueError(
            f"concat_view needs equal heights, got {img.shape[0]} vs {wrist.shape[0]}"
        )
    return np.ascontiguousarray(np.concatenate([img, wrist], axis=1))


def build_cosmos3_predict_request(
    *,
    image,
    wrist_image=None,
    instruction: str,
    domain_name: str = COSMOS3_DEFAULT_DOMAIN,
    image_size: int = COSMOS3_DEFAULT_IMAGE_SIZE,
) -> dict:
    """The ``POST /predict`` JSON body for one policy query.

    With ``wrist_image`` present the two views are concatenated (``concat_view``,
    what the LIBERO recipes train on); a single-view family member just passes
    ``image`` alone.
    """
    frame = (
        build_cosmos3_concat_view(image, wrist_image)
        if wrist_image is not None
        else np.asarray(image, dtype=np.uint8)
    )
    return {
        COSMOS3_IMAGE_KEY: encode_image_b64_png(frame),
        COSMOS3_PROMPT_KEY: str(instruction),
        COSMOS3_DOMAIN_KEY: str(domain_name),
        COSMOS3_IMAGE_SIZE_KEY: int(image_size),
    }


def cosmos3_chunk_from_response(resp) -> np.ndarray:
    """Coerce a ``/predict`` response into a ``(chunk, width)`` float array.

    Accepts the single-request shape (``{"action": [[...], ...]}``), the batch
    shape (``{"actions": [chunk, ...]}`` — first chunk taken), or a bare array.
    The predicted ``video`` frames are deliberately dropped: the WAM's imagined
    rollout is a training/debug artifact, not an eval input.
    """
    if isinstance(resp, dict):
        if "action" in resp:
            resp = resp["action"]
        elif "actions" in resp:
            batch = resp["actions"]
            resp = batch[0] if isinstance(batch, list) and batch else batch
    arr = np.asarray(resp, dtype=np.float64)
    if arr.ndim == 1:  # a single flat action -> a chunk of one
        arr = arr[None, :]
    return arr.reshape(arr.shape[0], -1)


def cosmos3_action_to_libero(chunk, k, *, translation_only: bool = False) -> np.ndarray:
    """Map step ``k`` of a Cosmos 3 action chunk to LIBERO's 7-DoF OSC_POSE action.

    A 10-D+ row is the LIBERO SFT layout ``[dpos(3), rot6d(6), gripper(1)]`` —
    decode the rot6d block to axis-angle. A narrower (>=7-D) row is passed
    through verbatim (already env-native). Actions arrive de-normalized (the
    server inverts ``quantile_rot``), and the recipe is ``frame_wise_relative``,
    i.e. per-step deltas — exactly what OSC_POSE consumes; no gripper fix-up
    (⚠ verify polarity on the first GPU rollout).

    ``translation_only`` zeroes rotation and forces the gripper open (the same
    de-risk knob the GR00T/π0.5 recipes carry).
    """
    vec = cosmos3_chunk_from_response(chunk)[int(k)]
    if vec.shape[0] >= _ROT6D_ROW_DIM:
        action = np.concatenate([
            vec[:3],
            rot6d_to_axis_angle(vec[3:9]),
            vec[9:10],
        ]).astype(np.float32)
    else:
        action = vec[:LIBERO_ACTION_DIM].astype(np.float32)
        if action.shape[0] < LIBERO_ACTION_DIM:  # pad a short row (shouldn't happen)
            action = np.concatenate(
                [action, np.zeros(LIBERO_ACTION_DIM - action.shape[0], np.float32)]
            )
    if translation_only:
        action[3:6] = 0.0
        action[6] = -1.0  # gripper forced open (LIBERO: -1 = open, +1 = close)
    return action.astype(np.float32)
