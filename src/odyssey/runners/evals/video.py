"""Rollout video capture — shared across evaluation runners.

The camera-enabled eval env already renders an RGB frame every step (the
policy consumes it), so capturing a rollout video costs only a per-step list
append plus one encode per episode — negligible next to per-step policy
inference. Video writing is **best-effort**: a missing/failed encoder logs a
warning and returns ``None`` rather than failing the evaluation.

Format follows the output path suffix: ``.mp4`` (needs imageio's ffmpeg
plugin — the ``robosuite`` extra pulls in ``imageio[ffmpeg]``) or ``.gif``
(pillow plugin, always available). Used today by the Robosuite multi-agent
eval; any future eval backend that can produce RGB frames reuses it as-is.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def to_uint8_frame(image: Any) -> Any:
    """Normalize one rendered frame to a contiguous HWC uint8 RGB ndarray.

    Accepts an ndarray (uint8, or float in [0,1] / [0,255]) or anything
    ``numpy.asarray`` can ingest (e.g. a PIL Image). Returns ``None`` when the
    input isn't image-shaped — e.g. the non-camera obs fallback ``_extract_image``
    yields when no frame is present — so callers can skip non-frames without
    special-casing.
    """
    import numpy as np

    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        return None
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr * 255.0 if float(arr.max()) <= 1.0 else arr
        arr = arr.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def save_rollout_video(frames: list[Any], path: Path, fps: int = 24) -> Path | None:
    """Encode accumulated RGB frames to ``path``. Best-effort.

    Returns the path on success, ``None`` when there are no frames or the
    encoder is unavailable / errors (logged as a warning — a missing video must
    never abort an evaluation).
    """
    if not frames:
        return None
    try:
        import imageio.v2 as imageio

        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(path), frames, fps=fps)
        logger.info("wrote rollout video -> %s (%d frames)", path, len(frames))
        return path
    except Exception as e:
        logger.warning("could not write rollout video %s: %s", path, e)
        return None
