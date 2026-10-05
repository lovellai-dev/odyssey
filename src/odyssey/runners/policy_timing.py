"""Policy timing contract: ``control_hz`` and ``action_horizon``.

A training task can declare the rate its actions are executed at
(``control_hz``) and the number of actions the policy predicts per call
(``action_horizon``). Both are fixed by training and inherited by the
checkpoint. These helpers are framework-agnostic:

  * ``check_control_hz`` compares the declared rate with the recorded rate
    (LeRobot ``meta/info.json`` ``fps``) of the dataset training actually
    loads — which the runner resolves, since a config override can select
    another directory than the declared ``dataset`` — and fails before any
    GPU work when they disagree.
  * ``check_action_horizon`` compares the declared chunk length with the one
    the training framework will actually use (each runner resolves that).
  * ``eval_timing_defaults`` hands the values declared by the training task
    that produced the evaluated checkpoint to the evaluation, so the eval
    runs the policy at the rate and chunk length it was trained for.

A runner that cannot read a value (a hub dataset not on disk, a framework it
cannot introspect) skips that check with a log line; it never guesses.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from odyssey.engine.lifecycle import TaskStatus
from odyssey.engine.records import MissionRun
from odyssey.spec.tasks import TrainingTask

logger = logging.getLogger(__name__)


class PolicyTimingError(RuntimeError):
    """A declared timing value disagrees with the data or the framework."""


def lerobot_dataset_fps(dataset_dir: str | None) -> float | None:
    """Recorded rate of the LeRobot dataset in ``dataset_dir``, or None.

    None when there is no directory or it holds no readable fps (e.g. a hub
    dataset not downloaded yet; it is not fetched just to check a number).
    """
    if dataset_dir is None:
        return None
    info = Path(dataset_dir) / "meta" / "info.json"
    if not info.is_file():
        return None
    try:
        fps = json.loads(info.read_text(encoding="utf-8")).get("fps")
    except (OSError, ValueError):
        return None
    return float(fps) if isinstance(fps, (int, float)) else None


def check_control_hz(task: TrainingTask, dataset_dir: str | None) -> None:
    """Fail when ``task.control_hz`` disagrees with the loaded dataset's fps.

    ``dataset_dir`` is the directory training will load, as the runner
    resolved it. No-op when ``control_hz`` is not declared or the fps can't
    be read.
    """
    if task.control_hz is None:
        return
    fps = lerobot_dataset_fps(dataset_dir)
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
            f"training loads was recorded at {fps} fps ({dataset_dir}"
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


def eval_timing_defaults(mission: MissionRun, checkpoint: str) -> dict[str, float | int]:
    """Timing values an evaluation inherits from the checkpoint it evaluates.

    Values come from the COMPLETED training task whose recorded
    ``checkpoint_path`` is ``checkpoint`` — the task that actually produced
    it — never from the last training task declared in the spec, which may
    have failed or targeted another checkpoint. Nothing is merged across
    tasks. Empty for a checkpoint no task in this mission produced (an
    explicit ``config.checkpoint``, e.g. a published one: its timing is
    unknown, so the eval must set it itself) or when nothing was declared.
    """
    # Compare in Path form: eval runners hand the checkpoint over as a Path,
    # which collapses "//" (``mock://t/final`` → ``mock:/t/final``), while the
    # record keeps the runner's raw string.
    wanted = str(Path(checkpoint))
    for task in reversed(mission.tasks):
        recorded = task.result_summary.get("checkpoint_path")
        if (
            isinstance(task.spec, TrainingTask)
            and task.status == TaskStatus.COMPLETED
            and recorded
            and str(Path(str(recorded))) == wanted
        ):
            return declared_timing(task.spec)
    return {}
