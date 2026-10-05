"""π0.5 (openpi) training runner.

Fine-tunes a Physical Intelligence π0.5 checkpoint by shelling out to the
upstream **openpi** training entry points — the openpi package must be
installed and its repo checked out on disk (``pip install -e`` of
https://github.com/Physical-Intelligence/openpi), pointed at via
``$OPENPI_REPO_PATH`` (default ``/srv/openpi``).

Why this looks different from the GR00T / OpenVLA runners
--------------------------------------------------------
GR00T and OpenVLA take a flat bag of ``--flag value`` overrides. openpi is
**config-name-driven**: training is selected by a registered ``TrainConfig``
name (e.g. ``pi05_libero``) that bundles the data pipeline, weight loader and
optimizer, and CLI ``--overrides`` (parsed by *tyro*) tweak individual fields.
So the mission's ``config:`` here carries:

  * ``config_name`` — REQUIRED, the registered openpi ``TrainConfig``. It reaches
    the two scripts differently (see the argv builders): ``train.py`` takes it
    **positionally** (``overridable_config_cli`` makes it a subcommand), while
    ``compute_norm_stats.py`` takes it as a ``--config-name`` flag (plain
    ``tyro.cli``). Fine-tuning a *new* embodiment/dataset means adding a
    config entry to openpi's ``src/openpi/training/config.py`` (the π0.5 analogue
    of GR00T's ``modality_config_path`` file) whose ``data.repo_id`` points at the
    LeRobot dataset — see ``examples/quickstart-pi05-train/README.md``.
  * everything else — passed through as tyro overrides (``num_train_steps`` →
    ``--num-train-steps``, ``data.repo_id`` → ``--data.repo-id``). Booleans become
    tyro flags (``--overwrite`` / ``--no-overwrite``), NOT ``--flag True``.

Two subprocesses, in order (openpi requires norm stats before training):

  1. ``scripts/compute_norm_stats.py --config-name <config_name>`` — computes
     dataset normalization statistics into ``assets/`` (skippable via
     ``config: {compute_norm_stats: false}`` when they already exist). Results are
     cached across runs per ``<config_name>/<repo_id>``; set
     ``config: {norm_stats_cache: false}`` to force a recompute when a dataset's
     content changed under an unchanged ``repo_id``.
  2. ``scripts/train.py <config_name> --exp-name <name> [--overwrite] [overrides]``.

Both run with ``cwd = <task output_dir>`` so openpi's cwd-relative ``./assets``
and ``./checkpoints`` land under the Odyssey task dir (no guessing an upstream
``--checkpoint-base-dir`` flag name). openpi is import-resolved from the installed
package, so the cwd only steers where artifacts are written.

Routing: registered behind OpenVLA's wildcard training slot, reached via the
task-level ``config: {runner: pi05}`` override — same mechanism as ``gr00t``.

Licensing note: π0.5 weights are distributed under Physical Intelligence's terms.
This runner contains no openpi code — it shells out to the user's own checkout.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from odyssey.runners.base import (
    WILDCARD_TYPE,
    Runner,
    TaskContext,
)
from odyssey.runners.dataset_revision import (
    DatasetRevisionError,
    check_local_dataset_revision,
    pins_commit,
)
from odyssey.runners.models.openpi_dataset import (
    OpenpiProbeError,
    OpenpiTrainingConfig,
    check_declared_dataset,
    probe_openpi_training_config,
)

# Shared flatten helper (dotted keys for nested dicts).
from odyssey.runners.models.openvla_train import _flatten_config
from odyssey.runners.policy_timing import (
    PolicyTimingError,
    check_action_horizon,
    check_control_hz,
    declared_timing,
)
from odyssey.runners.subprocess import (
    TrainingProcessSpec,
    output_path,
    run_training_subprocess,
)
from odyssey.spec.refs import DatasetSource
from odyssey.spec.tasks import TaskKind, TrainingTask, TrainingType

logger = logging.getLogger(__name__)


_DEFAULT_REPO_PATH = "/srv/openpi"
_TRAIN_SCRIPT_REL = "scripts/train.py"
_NORM_STATS_SCRIPT_REL = "scripts/compute_norm_stats.py"

# Config keys the runner consumes directly — never forwarded as tyro overrides.
_CONTROL_KEYS = frozenset(
    {
        "config_name",
        "runner",
        "exp_name",
        "overwrite",
        "resume",
        "compute_norm_stats",
        "norm_stats_cache",
    }
)

# openpi's train loop logs metrics dicts and a tqdm bar, e.g.:
#   "step=1000 loss=0.234 grad_norm=1.2"
#   "Step 1000: {'loss': 0.234, 'learning_rate': 1e-05}"
#   " 10%|█  | 100/1000 [00:42<06:18,  2.38it/s]"
#   "Saving checkpoint to ./checkpoints/pi05_libero/exp/1000"
_PI05_LOSS_RE = re.compile(r"(?:'loss':|\bloss=)\s*([\d.eE+-]+)")
_PI05_STEP_RE = re.compile(r"(?:'step':|\bstep=|\bStep\s+)\s*(\d+)", re.IGNORECASE)
_PI05_TQDM_RE = re.compile(r"\b(\d+)/(\d+)\s*\[")
_PI05_SAVE_RE = re.compile(r"(?i)\b(saving|saved|wrote)\b.*\bcheckpoint\b")
_PI05_NORM_RE = re.compile(r"(?i)\b(computing|writing)\b.*\bnorm(alization)?\s*stat")
_PI05_DATASET_RE = re.compile(r"(?i)\b(loading|building)\b.*\b(dataset|lerobot)\b")


def parse_pi05_train_line(line: str) -> dict[str, Any] | None:
    """Extract a progress-event dict from an openpi train/norm-stats stdout line.

    Public for tests and for users embedding openpi stdout parsing in a custom
    runner. Precedence mirrors the GR00T parser: a metric line (loss / step)
    wins over the coarse phase markers.
    """
    loss_match = _PI05_LOSS_RE.search(line)
    step_match = _PI05_STEP_RE.search(line)
    if loss_match or step_match:
        payload: dict[str, Any] = {"stage": "executing", "step": "training_step"}
        if step_match:
            payload["step_index"] = int(step_match.group(1))
        if loss_match:
            payload["step_label"] = f"loss={loss_match.group(1)}"
        return payload
    tqdm_match = _PI05_TQDM_RE.search(line)
    if tqdm_match:
        return {
            "stage": "executing",
            "step": "training_step",
            "step_index": int(tqdm_match.group(1)),
            "step_total": int(tqdm_match.group(2)),
        }
    if _PI05_SAVE_RE.search(line):
        return {"stage": "checkpoint_saving"}
    if _PI05_NORM_RE.search(line):
        return {"stage": "dataset_loading", "step": "compute_norm_stats"}
    if _PI05_DATASET_RE.search(line):
        return {"stage": "dataset_loading"}
    return None


def _resolve_openpi_script(rel: str) -> str:
    """Absolute path to an openpi script under ``$OPENPI_REPO_PATH``.

    Raises a runner-actionable error (not a bare FileNotFoundError) when the
    checkout is missing, matching the OpenVLA runner's contract.
    """
    repo_path = os.getenv("OPENPI_REPO_PATH", _DEFAULT_REPO_PATH)
    script_path = os.path.join(repo_path, rel)
    if not os.path.isfile(script_path):
        raise RuntimeError(
            f"openpi script not found at {script_path!r}; clone "
            "https://github.com/Physical-Intelligence/openpi and set "
            "OPENPI_REPO_PATH (or place it at /srv/openpi)."
        )
    return script_path


def _tyro_overrides(config: dict[str, Any]) -> list[str]:
    """Flatten ``config`` into tyro CLI overrides, skipping control keys.

    Nested dicts become dotted flags (``data.repo_id`` → ``--data.repo-id``);
    booleans become tyro toggle flags (``--overwrite`` / ``--no-overwrite``)
    rather than ``--flag True`` (which tyro rejects for bool fields).
    """
    argv: list[str] = []
    for key, value in _flatten_config(config):
        # _flatten_config yields the dotted path; only the LEAF may be a control
        # key (nested overrides like data.repo_id are always forwarded).
        if "." not in key and key in _CONTROL_KEYS:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            argv.append(flag if value else f"--no-{key.replace('_', '-')}")
        else:
            argv += [flag, str(value)]
    return argv


def _pins_local_commit(task: TrainingTask) -> bool:
    """True when the task's local dataset pins a commit sha to verify."""
    return (
        task.dataset is not None
        and task.dataset.source == DatasetSource.LOCAL
        and pins_commit(task.dataset)
    )


async def check_pi05_policy_timing(
    task: TrainingTask,
    env: dict[str, str],
    exp_name: str,
    cancel_event: asyncio.Event | None = None,
) -> OpenpiTrainingConfig | None:
    """Pre-flight: declared control_hz / action_horizon must match reality.

    Runs before norm stats and training, so a mismatch costs seconds, not a
    GPU run. Both values are checked against the config ``train.py`` will
    build — the registered openpi config WITH the mission's overrides — so
    ``control_hz`` is compared with the fps of the dataset training actually
    loads (``HF_LEROBOT_HOME / data.repo_id``), and that dataset must be the
    declared one. Returns early, unchecked, when ``cancel_event`` trips during
    the probe; the caller checks ``context.cancelled()`` right after.

    The same probe serves a pinned dataset revision: when the local dataset
    pins a commit sha, the config is probed (even with no timing declared),
    the declared dataset must be the loaded one, and the probed config is
    returned so the caller verifies the revision against the directory
    training loads. None when nothing needed the probe, or on cancellation.
    """
    pinned = _pins_local_commit(task)
    if task.control_hz is None and task.action_horizon is None and not pinned:
        return None
    config = task.config or {}
    config_name = str(config.get("config_name") or "")
    if not config_name:
        if pinned:  # the revision check runs before the argv builders would raise
            raise RuntimeError(
                "π0.5 runner: config['config_name'] is required to resolve the "
                "dataset training loads, and so to verify its pinned revision."
            )
        return None  # the argv builders raise the actionable "config_name required"
    try:
        effective = await probe_openpi_training_config(
            config_name, exp_name, _tyro_overrides(config), env, cancel_event
        )
    except OpenpiProbeError as e:
        # A declared value can't be verified, so running would defeat the check.
        if task.control_hz is None and task.action_horizon is None:
            raise DatasetRevisionError(f"task {task.name!r}: {e}") from e
        raise PolicyTimingError(f"task {task.name!r}: {e}") from e
    if effective is None:
        return None  # cancelled mid-probe
    if task.control_hz is not None or pinned:
        check_declared_dataset(task, effective)
    check_control_hz(task, effective.dataset_dir)
    check_action_horizon(
        task, effective.action_horizon, f"openpi config {config_name!r} with overrides"
    )
    return effective


def build_pi05_train_argv(*, task: TrainingTask, exp_name: str) -> list[str]:
    """Build the openpi ``scripts/train.py`` argv.

    Shape: ``<config_name> --exp-name <name> [--overwrite|--resume] [overrides…]``.
    ``config_name`` is REQUIRED — it selects the registered openpi ``TrainConfig``.
    ``--overwrite`` is emitted by default (re-running a mission clobbers the prior
    exp dir); set ``config: {overwrite: false}`` to keep it, or ``resume: true`` to
    continue from the latest checkpoint instead.
    """
    config = task.config or {}
    config_name = config.get("config_name")
    if not config_name:
        raise RuntimeError(
            "π0.5 runner: config['config_name'] is required — it names the "
            "registered openpi TrainConfig (e.g. 'pi05_libero'). Add a config "
            "entry to openpi's src/openpi/training/config.py for a custom dataset."
        )

    argv: list[str] = [str(config_name), "--exp-name", exp_name]
    if config.get("resume"):
        argv.append("--resume")
    elif config.get("overwrite", True):
        argv.append("--overwrite")
    argv += _tyro_overrides(config)
    return argv


def build_pi05_norm_stats_argv(*, task: TrainingTask) -> list[str]:
    """Build the openpi ``scripts/compute_norm_stats.py`` argv.

    Unlike ``train.py`` (positional config name via ``overridable_config_cli``),
    ``compute_norm_stats.py`` is a plain ``tyro.cli(main)`` whose ``config_name``
    parameter is exposed as a REQUIRED ``--config-name`` flag. The dataset it reads
    is fixed by the config's ``data`` factory.
    """
    config = task.config or {}
    config_name = config.get("config_name")
    if not config_name:
        raise RuntimeError(
            "π0.5 runner: config['config_name'] is required for norm-stats."
        )
    return ["--config-name", str(config_name)]


# Runs compute_norm_stats.py against the config train.py will build (see the
# module docstring of openpi_bootstrap.py for why the direct call can't).
_OPENPI_BOOTSTRAP = str(Path(__file__).with_name("openpi_bootstrap.py"))


def build_pi05_norm_stats_launch(
    *, task: TrainingTask, exp_name: str, norm_stats_script: str
) -> tuple[str, list[str]]:
    """``(script_path, argv)`` for the norm-stats step.

    Without tyro overrides the registered config IS what train.py uses, so
    ``compute_norm_stats.py`` runs directly (unchanged behaviour). With
    overrides (``data.repo_id``, ``model.action_horizon``, …) it runs through
    ``openpi_bootstrap.py``, which applies the SAME overrides with the SAME
    parser as train.py. Both steps then see one config and one dataset.
    """
    config = task.config or {}
    overrides = _tyro_overrides(config)
    if not overrides:
        return norm_stats_script, build_pi05_norm_stats_argv(task=task)
    config_name = config.get("config_name")
    if not config_name:
        raise RuntimeError(
            "π0.5 runner: config['config_name'] is required for norm-stats."
        )
    # openpi's CLI requires --exp-name (TrainConfig.exp_name is MISSING); pass
    # train.py's so the parsed config is identical. It doesn't affect stats.
    return _OPENPI_BOOTSTRAP, [
        norm_stats_script,
        str(config_name),
        "--",
        *overrides,
        "--exp-name",
        exp_name,
    ]


def _lerobot_env_for_dataset(task: TrainingTask) -> dict[str, str]:
    """Env overlay so openpi's LeRobot loader finds a LOCAL dataset.

    openpi resolves ``data.repo_id`` against ``HF_LEROBOT_HOME`` (default
    ``~/.cache/huggingface/lerobot``). For an absolute local dataset dir we point
    that at the PARENT so ``repo_id`` == the dataset folder name resolves. HF/hub
    refs and relative paths are left to the config to interpret.
    """
    if task.dataset is None:
        return {}
    ref = task.dataset.ref
    if task.dataset.source == DatasetSource.LOCAL and os.path.isabs(ref):
        parent = os.path.dirname(os.path.normpath(ref))
        # Only HF_LEROBOT_HOME — do NOT also set the legacy LEROBOT_HOME. Recent
        # lerobot HARD-FAILS at import (raises ValueError) if the deprecated
        # LEROBOT_HOME var is present, which killed compute_norm_stats.
        return {"HF_LEROBOT_HOME": parent}
    return {}


def _resolve_output_checkpoint(output_dir: Path) -> Path | None:
    """Locate the trained checkpoint under ``<output_dir>/checkpoints``.

    openpi writes ``checkpoints/<config_name>/<exp_name>/<step>/`` (step is an
    integer dir holding ``params/``). Return the highest-numbered step dir; None
    if training wrote nothing.
    """
    checkpoints_root = output_dir / "checkpoints"
    if not checkpoints_root.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for entry in checkpoints_root.rglob("*"):
        if entry.is_dir() and entry.name.isdigit():
            step = int(entry.name)
            if best is None or step > best[0]:
                best = (step, entry)
    return best[1] if best is not None else None


def _dataset_repo_id(task: TrainingTask) -> str:
    """The LeRobot ``repo_id`` openpi keys norm stats by.

    openpi writes ``assets/<config_name>/<repo_id>/norm_stats.json``. ``repo_id``
    is the mission's ``data.repo_id`` tyro override when present, else the local
    dataset folder name (same source as ``_lerobot_env_for_dataset``). Empty when
    neither is set — the registered config's *default* repo_id is then not known
    to the runner, so the cache cannot be pinned and must recompute.
    """
    config = task.config or {}
    data = config.get("data")
    if isinstance(data, dict) and data.get("repo_id"):
        return str(data["repo_id"])
    if task.dataset is not None and task.dataset.ref:
        return os.path.basename(os.path.normpath(task.dataset.ref))
    return ""


def _norm_stats_cache_root(config_name: str, revision: str | None) -> Path:
    """Stable cache dir for one config, partitioned by dataset revision.

    openpi's own layout below it (``<config_name>/<repo_id>/norm_stats.json``)
    is unchanged. With a verified revision the root is per revision, so a new
    version of a dataset under the SAME repo_id misses the cache and its
    statistics are recomputed instead of reusing the previous version's.
    """
    root = Path.home() / ".odyssey" / "pi05_assets"
    if not revision:
        return root / config_name
    return root / f"{config_name}@{revision.replace('/', '_')}"


def _link_norm_stats_cache(
    output_dir: Path, config_name: str, repo_id: str, revision: str | None = None
) -> tuple[Path | None, bool]:
    """Point openpi's cwd-relative ``./assets`` at a STABLE per-config cache.

    openpi writes norm stats into ``./assets`` relative to cwd (= the per-run
    output_dir), so a fresh output_dir every run recomputes the *deterministic*
    stats from scratch — a multi-minute full-dataset scan each time. Symlink
    ``output_dir/assets`` at a stable cache keyed by ``config_name`` so the stats
    persist and are reused across runs; checkpoints still land in the per-run
    ``output_dir/checkpoints`` (only ``./assets`` is redirected).

    The cache-hit test is keyed on the EXACT ``<config_name>/<repo_id>`` path
    openpi writes, not a recursive glob: one registered config can hold stats for
    several datasets (iterating datasets against one config is the intended
    workflow, and what the shipped example does via ``data.repo_id``), so a glob
    would report a hit for *any* dataset under the config, skip the step, and then
    ``train.py`` dies looking for the ``repo_id`` it actually needs. An empty
    ``repo_id`` (config default, unknown here) recomputes rather than risk a wrong
    hit.

    The cache is also keyed by ``revision`` when the dataset's pinned commit
    was verified on disk (see ``_norm_stats_cache_root``). Without a revision two versions of a dataset
    published under one repo_id are indistinguishable and share a cache entry.

    Returns ``(cache_dir, already_has_norm_stats)``; ``(None, False)`` when there
    is no config_name to key the cache on.
    """
    if not config_name:
        return None, False
    cache = _norm_stats_cache_root(config_name, revision)
    cache.mkdir(parents=True, exist_ok=True)
    link = output_dir / "assets"
    if not link.is_symlink() and not link.exists():
        link.symlink_to(cache, target_is_directory=True)
    cached = bool(repo_id) and (
        cache / config_name / repo_id / "norm_stats.json"
    ).is_file()
    return cache, cached


class Pi05Runner(Runner):
    """Fine-tune a π0.5 model. Subprocess-based — actual training happens in
    openpi's ``scripts/compute_norm_stats.py`` then ``scripts/train.py``."""

    @property
    def name(self) -> str:
        return "pi05"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.TRAINING}

    @property
    def supported_types(self) -> set[str]:
        # Registered behind OpenVLA's wildcard; reached via the task-level
        # ``config: {runner: pi05}`` override (same pattern as GR00T).
        return {WILDCARD_TYPE}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        spec = context.task.spec
        if not isinstance(spec, TrainingTask):
            raise TypeError(
                f"Pi05Runner expects TrainingTask, got {type(spec).__name__}"
            )
        if context.agent is None:
            raise RuntimeError(
                "Pi05Runner: TaskContext.agent is None — training tasks must be "
                "invoked through the engine, which resolves the agent from "
                "spec.robot.agents[task.agent_id]."
            )

        output_dir = output_path(context)
        output_dir.mkdir(parents=True, exist_ok=True)
        config = dict(spec.config)
        config_name = str(config.get("config_name") or "")
        exp_name = str(config.get("exp_name") or spec.name)
        timeout = getattr(spec, "timeout_seconds", None)
        child_env = _lerobot_env_for_dataset(spec)

        # Step 0: the declared timing contract must match the data and the
        # openpi config before anything heavy starts.
        effective = await check_pi05_policy_timing(
            spec, child_env, exp_name, context.cancel_event
        )
        if context.cancelled():
            logger.info("π0.5 task %s cancelled during the timing pre-flight", context.task.id)
            return {"cancelled": True}

        # The pinned dataset revision must be the data training will read: it is
        # verified, file by file, in the directory the pre-flight probe resolved.
        revision = spec.dataset.revision if spec.dataset is not None else None
        if (
            revision is not None
            and spec.dataset is not None
            and spec.dataset.source != DatasetSource.LOCAL
        ):
            raise DatasetRevisionError(
                f"π0.5 task {spec.name!r}: dataset revision {revision} is pinned, "
                "but openpi loads a hub dataset by repo_id and cannot be pinned to a "
                "revision. Download that revision (hf download <repo> --repo-type "
                "dataset --revision <sha> --local-dir <dir>) and use source: local."
            )
        loaded_dir = effective.dataset_dir if effective else None
        # The full, lowercase sha the files record (None if unverified): an
        # abbreviated or upper-case pin keys the cache and the record the same.
        verified_sha = check_local_dataset_revision(spec.name, spec.dataset, loaded_dir)
        revision_verified = verified_sha is not None

        # Redirect openpi's ./assets at a stable per-config cache so norm stats are
        # computed once and reused, not recomputed on every fresh output_dir. The
        # hit is keyed on the exact dataset (config_name + repo_id) and, when the
        # dataset's pinned commit was VERIFIED on disk, that revision (a branch or
        # tag moves, so it never keys the cache). Otherwise a dataset recaptured
        # under the SAME repo_id would silently reuse stale stats: pin the
        # revision, or set ``config: {norm_stats_cache: false}`` to force a fresh
        # recompute into the per-run dir (no shared cache).
        if config.get("norm_stats_cache", True):
            repo_id = _dataset_repo_id(spec)
            assets_cache, norm_cached = _link_norm_stats_cache(
                output_dir, config_name, repo_id, verified_sha
            )
        else:
            logger.warning(
                "π0.5 task %s: norm-stats cache disabled (norm_stats_cache=false)"
                " — recomputing statistics into the per-run assets dir.",
                context.task.id,
            )
            assets_cache, norm_cached = None, False

        # Step 1: normalization statistics (openpi requires them before training).
        # Skip when a prior run already cached them for this config.
        if config.get("compute_norm_stats", True) and not norm_cached:
            await context.emit_progress(
                "dataset_loading", step="compute_norm_stats", step_label=exp_name
            )
            norm_script, norm_argv = build_pi05_norm_stats_launch(
                task=spec,
                exp_name=exp_name,
                norm_stats_script=_resolve_openpi_script(_NORM_STATS_SCRIPT_REL),
            )
            norm_spec = TrainingProcessSpec(
                timeout_seconds=timeout,
                script_path=norm_script,
                argv_extra=norm_argv,
                env=child_env,
                cwd=str(output_dir),
                line_parser=parse_pi05_train_line,
            )
            rc = await run_training_subprocess(context, norm_spec)
            if context.cancelled():
                logger.info("π0.5 task %s cancelled during norm-stats", context.task.id)
                return {"cancelled": True}
            if rc != 0:
                raise RuntimeError(
                    f"openpi compute_norm_stats exited with code {rc}"
                )
        elif norm_cached:
            logger.info(
                "π0.5 task %s: reusing cached norm stats at %s",
                context.task.id,
                assets_cache,
            )
            await context.emit_progress(
                "dataset_loading", step="norm_stats_cached", step_label=exp_name
            )

        # Step 2: fine-tune. Re-verify the pin first: norm stats can take a
        # while, and "verified" must describe what train.py reads now, not
        # what was on disk at step 0. Costs one stat + one small read per file.
        if verified_sha is not None:
            again = check_local_dataset_revision(spec.name, spec.dataset, loaded_dir)
            if again != verified_sha:
                raise DatasetRevisionError(
                    f"π0.5 task {spec.name!r}: {loaded_dir} changed from commit "
                    f"{verified_sha} to {again} during norm stats; not training on it."
                )
        train_spec = TrainingProcessSpec(
            timeout_seconds=timeout,
            script_path=_resolve_openpi_script(_TRAIN_SCRIPT_REL),
            argv_extra=build_pi05_train_argv(task=spec, exp_name=exp_name),
            env=child_env,
            cwd=str(output_dir),
            line_parser=parse_pi05_train_line,
        )
        rc = await run_training_subprocess(context, train_spec)
        if context.cancelled():
            logger.info("π0.5 task %s cancelled by user", context.task.id)
            return {"cancelled": True}
        if rc != 0:
            raise RuntimeError(f"openpi train.py exited with code {rc}")

        checkpoint = _resolve_output_checkpoint(output_dir)
        if checkpoint is None:
            raise RuntimeError(
                f"openpi train.py finished but no checkpoint found under "
                f"{output_dir / 'checkpoints'!r}"
            )
        return {
            "checkpoint_path": str(checkpoint),
            "agent_id": context.agent.id,
            "exp_name": exp_name,
            "config_name": config.get("config_name"),
            "training_config": spec.config,
            **declared_timing(spec),
            "dataset_revision": verified_sha or revision,
            "dataset_revision_verified": revision_verified,
            "training_type": (
                spec.training_type.value
                if isinstance(spec.training_type, TrainingType)
                else spec.training_type
            ),
        }
