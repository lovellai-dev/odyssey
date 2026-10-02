"""Which dataset openpi training will actually load.

A mission names its data twice: ``dataset`` (the declared ref) and, for
openpi, whatever the training config ends up with — the registered config's
``data.repo_id`` unless the mission overrides it (``config.data.repo_id`` →
``--data.repo-id``). openpi's LeRobot loader then reads
``HF_LEROBOT_HOME / repo_id``. The two can name different directories, so
any check about "the dataset" (its recorded fps, a pinned revision) must be
made against the one training loads, not the one the spec declares.

``probe_openpi_training_config`` answers that by building the config the way
``scripts/train.py`` does — openpi's own CLI parser, the mission's overrides,
under the same interpreter and env — and resolving the directory with
lerobot's own ``HF_LEROBOT_HOME``. ``check_declared_dataset`` then rejects a
declared dataset that contradicts it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from odyssey.spec.refs import DatasetSource
from odyssey.spec.tasks import TrainingTask


class DatasetMismatchError(RuntimeError):
    """The declared dataset is not the one training will load."""


class OpenpiProbeError(RuntimeError):
    """The effective openpi training config could not be read."""


@dataclass(frozen=True)
class OpenpiTrainingConfig:
    """What ``train.py`` will use, with the mission's overrides applied."""

    action_horizon: int
    repo_id: str | None
    # HF_LEROBOT_HOME / repo_id as lerobot resolves it in training's env;
    # None when the config has no LeRobot repo_id.
    dataset_dir: str | None


# Builds the TrainConfig with the parser and overrides train.py uses (see
# openpi_bootstrap.py) and prints its timing + data source as one JSON line.
# argv: <config_name> <exp_name> <tyro overrides…>
_TRAINING_CONFIG_PROBE = """\
import json, os, sys
from pathlib import Path
from openpi.training import config as c
name, exp_name, overrides = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.argv = ["train.py", name, *overrides, "--exp-name", exp_name]
cfg = c.cli()
repo_id = getattr(cfg.data, "repo_id", None)
repo_id = repo_id if isinstance(repo_id, str) and repo_id else None
home = None
for mod in ("lerobot.common.constants", "lerobot.constants", "lerobot.utils.constants"):
    try:
        home = __import__(mod, fromlist=["HF_LEROBOT_HOME"]).HF_LEROBOT_HOME
        break
    except (ImportError, AttributeError):
        pass
if home is None:
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    home = os.environ.get("HF_LEROBOT_HOME", str(Path(hf_home) / "lerobot"))
print(json.dumps({
    "action_horizon": int(cfg.model.action_horizon),
    "repo_id": repo_id,
    "dataset_dir": str(Path(home) / repo_id) if repo_id else None,
}))
"""
_PROBE_TIMEOUT_S = 300.0


async def probe_openpi_training_config(
    config_name: str,
    exp_name: str,
    overrides: list[str],
    env: dict[str, str],
    cancel_event: asyncio.Event | None = None,
) -> OpenpiTrainingConfig | None:
    """The effective openpi training config, as ``train.py`` will build it.

    Runs under odyssey's interpreter (``sys.executable``, the one that runs
    train.py) with training's ``env``. Raises ``OpenpiProbeError`` when the
    config can't be read. Returns None when ``cancel_event`` (the mission's
    cancellation) trips first. The child is killed and reaped on EVERY
    interrupted path — mission cancellation, timeout, or this coroutine
    being cancelled — so no probe outlives the task.
    """
    child_env = {**os.environ, **env}
    child_env.pop("PYTHONPATH", None)
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", _TRAINING_CONFIG_PROBE, config_name, exp_name, *overrides,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=child_env,
    )
    communicate = asyncio.ensure_future(proc.communicate())
    waiters: set[asyncio.Future[Any]] = {communicate}
    if cancel_event is not None:
        waiters.add(asyncio.ensure_future(cancel_event.wait()))
    try:
        done, _ = await asyncio.wait(
            waiters, timeout=_PROBE_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED
        )
        if communicate not in done:
            if done:  # the cancel_event waiter
                return None
            raise OpenpiProbeError(
                f"could not read openpi config {config_name!r}: "
                f"probe timed out after {_PROBE_TIMEOUT_S:.0f}s"
            )
        out, err = communicate.result()
    finally:
        for waiter in waiters:
            waiter.cancel()
        if proc.returncode is None:
            with suppress(ProcessLookupError):
                proc.kill()
            # Shield the reap so a cancellation arriving now can't abort it
            # (the killed child still gets reaped), but let that
            # CancelledError propagate: it is BaseException, not Exception.
            with suppress(Exception):
                await asyncio.shield(proc.wait())
    lines = out.decode().strip().splitlines()
    try:
        if proc.returncode != 0 or not lines:
            raise ValueError
        data = json.loads(lines[-1])
        return OpenpiTrainingConfig(
            action_horizon=int(data["action_horizon"]),
            repo_id=data.get("repo_id") or None,
            dataset_dir=data.get("dataset_dir") or None,
        )
    except (ValueError, KeyError, TypeError):
        tail = err.decode().strip().splitlines()[-3:]
        raise OpenpiProbeError(
            f"could not read openpi config {config_name!r} "
            f"(exit {proc.returncode}): {' | '.join(tail) or 'no output'}"
        ) from None


def check_declared_dataset(task: TrainingTask, effective: OpenpiTrainingConfig) -> None:
    """Fail when ``task.dataset`` names a different dataset than training loads.

    A local dataset must be the directory training reads; a Hugging Face
    dataset must be the ``repo_id`` it loads. No-op when nothing is declared,
    for other sources, or when training's source is unknown (the config has
    no LeRobot ``repo_id``).
    """
    declared = task.dataset
    if declared is None or effective.repo_id is None:
        return
    if declared.source == DatasetSource.LOCAL:
        if not os.path.isabs(declared.ref) or effective.dataset_dir is None:
            return
        same = os.path.realpath(declared.ref) == os.path.realpath(effective.dataset_dir)
        loaded = effective.dataset_dir
    elif declared.source == DatasetSource.HUGGINGFACE:
        same = declared.ref == effective.repo_id
        loaded = effective.repo_id
    else:
        return  # S3/GCS/OXE/…: no LeRobot repo_id to compare against
    if not same:
        raise DatasetMismatchError(
            f"task {task.name!r}: dataset {declared.ref!r} is declared, but "
            f"training loads {loaded!r} (data.repo_id={effective.repo_id!r}). "
            "Checks on the declared dataset would not describe the data "
            "trained on; make dataset and config.data.repo_id agree."
        )
