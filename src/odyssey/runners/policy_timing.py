"""Policy timing contract: ``control_hz`` and ``action_horizon``.

A training task can declare the rate its actions are executed at
(``control_hz``) and the number of actions the policy predicts per call
(``action_horizon``). Both are fixed by training and inherited by the
checkpoint. These helpers are framework-agnostic:

  * ``check_control_hz`` compares the declared rate with the dataset's
    recorded rate (LeRobot ``meta/info.json`` ``fps``) and fails before any
    GPU work when they disagree.
  * ``check_action_horizon`` compares the declared chunk length with the one
    the training framework will actually use (each runner resolves that).
  * ``eval_timing_defaults`` hands the declared values to an evaluation, so
    the eval runs the policy at the rate and chunk length it was trained for.

A runner that cannot read a value (a hub dataset not on disk, a framework it
cannot introspect) skips that check with a log line; it never guesses.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

from odyssey.spec.mission import Mission
from odyssey.spec.refs import DatasetRef, DatasetSource
from odyssey.spec.tasks import TrainingTask

logger = logging.getLogger(__name__)


class PolicyTimingError(RuntimeError):
    """A declared timing value disagrees with the data or the framework."""


def lerobot_dataset_fps(dataset: DatasetRef | None) -> float | None:
    """Recorded rate of a local LeRobot dataset, or None when it can't be read.

    Only an absolute ``source: local`` path is read; hub datasets are not
    downloaded just to check a number.
    """
    if dataset is None or dataset.source != DatasetSource.LOCAL:
        return None
    if not os.path.isabs(dataset.ref):
        return None
    info = Path(dataset.ref) / "meta" / "info.json"
    if not info.is_file():
        return None
    try:
        fps = json.loads(info.read_text(encoding="utf-8")).get("fps")
    except (OSError, ValueError):
        return None
    return float(fps) if isinstance(fps, (int, float)) else None


def check_control_hz(task: TrainingTask) -> None:
    """Fail when ``task.control_hz`` disagrees with the dataset's fps.

    No-op when ``control_hz`` is not declared or the fps can't be read.
    """
    if task.control_hz is None:
        return
    fps = lerobot_dataset_fps(task.dataset)
    if fps is None:
        logger.info(
            "task %s: control_hz=%s declared but the dataset fps could not be "
            "read; skipping the check",
            task.name,
            task.control_hz,
        )
        return
    if not math.isclose(fps, task.control_hz, rel_tol=0, abs_tol=1e-6):
        raise PolicyTimingError(
            f"task {task.name!r}: control_hz={task.control_hz} but the dataset "
            f"was recorded at {fps} fps ({task.dataset.ref if task.dataset else ''}"
            "/meta/info.json). A policy trained on this data executes its "
            "actions at the recorded rate; fix control_hz or resample the "
            "dataset."
        )


def check_action_horizon(task: TrainingTask, effective: int, source: str) -> None:
    """Fail when ``task.action_horizon`` disagrees with the framework's value.

    ``effective`` is the chunk length training will actually use; ``source``
    names where it came from, for the error message.
    """
    if task.action_horizon is None:
        return
    if effective != task.action_horizon:
        raise PolicyTimingError(
            f"task {task.name!r}: action_horizon={task.action_horizon} but "
            f"training would use {effective} ({source}). Fix action_horizon or "
            "the training config so they agree."
        )


def declared_timing(task: TrainingTask) -> dict[str, float | int]:
    """The timing values a training task declares, for its result record."""
    out: dict[str, float | int] = {}
    if task.control_hz is not None:
        out["control_hz"] = task.control_hz
    if task.action_horizon is not None:
        out["action_horizon"] = task.action_horizon
    return out


def eval_timing_defaults(mission: Mission, agent_id: str | None) -> dict[str, float | int]:
    """Timing values an evaluation inherits from training.

    Training tasks run in spec order and each one updates its agent, so the
    LAST training task that targets ``agent_id`` produced the checkpoint under
    evaluation. Values come from that task only; nothing is merged across
    tasks. Empty for an eval-only mission or when nothing was declared.
    """
    last: TrainingTask | None = None
    for task in mission.tasks:
        if isinstance(task, TrainingTask) and task.agent_id == agent_id:
            last = task
    return declared_timing(last) if last is not None else {}
