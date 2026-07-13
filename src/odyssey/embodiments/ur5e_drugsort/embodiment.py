"""UR5e / Robotiq 2F-85 drug-sorting embodiment definition (GR00T).

Single source of truth for the observation/action layout of the AseptiPack
fill-finish cell, shared by the demonstration-data generator (writes the
LeRobot dataset) and by tests (assert the schema). It maps 1:1 onto the
GR00T-flavoured LeRobot ``meta/modality.json`` + ``meta/info.json`` format
that ``gr00t.experiment.launch_finetune`` consumes for a ``new_embodiment``
fine-tune.

Layout
------
* **state** (proprioception, 7-D ``observation.state``): the 6 UR5e arm joint
  positions (rad) + normalised gripper closure in ``[0, 1]`` (0 open, 1 closed).
* **action** (7-D ``action``): the 6 arm joint-position targets (rad) + the
  commanded gripper closure in ``[0, 1]``.
* **video**: a single ``exterior`` view — the fixed workcell overview
  (MJCF ``room`` camera), stored as ``observation.images.exterior``.
* **language**: a single task-description annotation resolved from
  ``task_index``.

The camera contract is chosen to match the **Lovell AI Robot Playground** deploy
target, not Isaac. The Playground's ``TrueSensorRenderer.capturePrimary()`` is
the single image authority the closed-loop inference client sends each tick; it
returns the ``exterior`` primary mount view at the mount's ``render_resolution``
(default 256x256). Training on any view the inference loop does not publish
(e.g. a wrist camera the browser bridge doesn't emit as primary) would break
transfer, so we record exactly the one primary ``exterior`` view. Likewise the
action space mirrors the Playground ``setTargets(q, grip)`` bridge: 6 arm joint
targets (rad) + gripper closure in ``[0, 1]``.

The state/action share feature names (joint ``.pos`` + ``gripper.pos``), the
same convention the upstream SO-100 demo set uses.
"""

from __future__ import annotations

from typing import Any

# --- Identity ---------------------------------------------------------------
ROBOT_TYPE = "ur5e_robotiq_2f85"
EMBODIMENT_TAG = "new_embodiment"
DEFAULT_INSTRUCTION = "pick up the vial and place it in the rack"

# --- Joints / dims ----------------------------------------------------------
ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
ARM_DIM = len(ARM_JOINT_NAMES)  # 6
GRIPPER_DIM = 1
STATE_DIM = ARM_DIM + GRIPPER_DIM  # 7
ACTION_DIM = ARM_DIM + GRIPPER_DIM  # 7

# Per-dimension feature names, shared by observation.state and action.
FEATURE_NAMES: tuple[str, ...] = (*(f"{j}.pos" for j in ARM_JOINT_NAMES), "gripper.pos")

# --- Cameras / video keys ---------------------------------------------------
# LeRobot video key  ->  MJCF camera name. A single ``exterior`` view maps to the
# fixed workcell-overview ``room`` camera, matching the Playground's primary
# (``exterior``) TrueSensorRenderer mount — the one image the inference loop
# publishes each tick. Add a second entry here ONLY if the Playground embodiment
# also publishes that mount as an observation the inference client sends.
VIDEO_KEY_TO_CAMERA: dict[str, str] = {
    "exterior": "room",
}
# LeRobot video key  ->  parquet/video "original_key" used in modality.json.
VIDEO_ORIGINAL_KEYS: dict[str, str] = {
    key: f"observation.images.{key}" for key in VIDEO_KEY_TO_CAMERA
}

# Default render resolution for the offscreen cameras (even dims for h264).
# 256x256 matches the Playground camera-mount default render_resolution.
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256
DEFAULT_FPS = 20

CODEBASE_VERSION = "v2.1"
CHUNKS_SIZE = 1000


def modality_json() -> dict[str, Any]:
    """The GR00T ``meta/modality.json`` for this embodiment.

    ``state``/``action`` slice the flat 7-D vectors into named groups; ``video``
    maps friendly keys onto the stored ``observation.images.*`` columns;
    ``annotation`` resolves the language instruction from ``task_index``.
    """
    slices = {
        "single_arm": {"start": 0, "end": ARM_DIM},
        "gripper": {"start": ARM_DIM, "end": STATE_DIM},
    }
    return {
        "state": {k: dict(v) for k, v in slices.items()},
        "action": {k: dict(v) for k, v in slices.items()},
        "video": {
            key: {"original_key": original}
            for key, original in VIDEO_ORIGINAL_KEYS.items()
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def _video_feature(width: int, height: int, fps: int) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def features(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
) -> dict[str, Any]:
    """The ``features`` block of ``meta/info.json``."""
    vec_feature = {
        "dtype": "float32",
        "shape": [STATE_DIM],
        "names": list(FEATURE_NAMES),
    }
    feats: dict[str, Any] = {
        "action": dict(vec_feature),
        "observation.state": dict(vec_feature),
    }
    for key in VIDEO_KEY_TO_CAMERA:
        feats[f"observation.images.{key}"] = _video_feature(width, height, fps)
    feats.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return feats


def info_json(
    *,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    fps: int = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> dict[str, Any]:
    """The full ``meta/info.json`` for a written dataset."""
    total_videos = total_episodes * len(VIDEO_KEY_TO_CAMERA)
    total_chunks = (total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE if total_episodes else 0
    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": ROBOT_TYPE,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features(width, height, fps),
    }
