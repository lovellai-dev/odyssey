"""Cosmos 3 (WAM) SFT training runner.

Fine-tunes an NVIDIA Cosmos 3 action policy by shelling out to the upstream
**cosmos-framework** training entry points — the ``cosmos_framework`` package
must be installed and its repo checked out on disk (clone of
https://github.com/NVIDIA/cosmos-framework), pointed at via
``$COSMOS_FRAMEWORK_REPO_PATH`` (default ``/srv/cosmos-framework``). The
framework's SFT recipe TOMLs live under ``examples/toml/sft_config/`` there.

Why this looks different from the GR00T / OpenVLA runners
--------------------------------------------------------
Like π0.5, Cosmos is **config-name-driven**: training is selected by a
registered SFT recipe TOML (e.g. ``action_policy_libero_nano``,
``action_policy_droid_nano``) that bundles the data pipeline, weight loader and
optimizer. Individual fields are tweaked by trailing Hydra-style dotlist
positionals applied after the TOML (``trainer.max_iter=10`` — NOT
``--trainer.max-iter``; cosmos ``train.py`` reads them via argparse REMAINDER).
Paths inside the
TOML use OmegaConf env interpolation (``${oc.env:BASE_CHECKPOINT_PATH}`` …), so
the runner steers data / checkpoint / output locations through **environment
variables**, not flags — matching Odyssey's "no hardcoded env in runners; the
runner only bridges the resolved refs" convention.

Three subprocesses, in order (cosmos-framework needs a DCP base checkpoint
before training, and an exported checkpoint after):

  1. ``python -m cosmos_framework.scripts.convert_model_to_dcp
     --checkpoint-path <base> -o $BASE_CHECKPOINT_PATH`` — converts the chosen
     base model (Cosmos3-Nano / Cosmos3-Edge) into a distributed-checkpoint dir.
     Skipped when ``$BASE_CHECKPOINT_PATH`` already exists or
     ``config: {convert_dcp: false}``.
  2. ``torchrun --nproc_per_node=<N> -m cosmos_framework.scripts.train
     --sft-toml=<recipe.toml> [dotted overrides]`` — the SFT run itself. Writes
     into ``$IMAGINAIRE_OUTPUT_ROOT`` (pointed at the task output dir).
  3. ``python -m cosmos_framework.scripts.export_model
     --checkpoint-path <trained-dcp> --config-file <run>/config.yaml
     -o <run>/model`` — exports HF safetensors that the ``pilot: cosmos3``
     policy server (``action_policy_server_libero``) loads directly. Skipped via
     ``config: {export: false}``.

Venv note (option A): cosmos-framework usually lives in its OWN heavy venv.
``run_training_subprocess`` launches via ``sys.executable``, so run ``odyssey``
*from inside* that venv (the UR10e remote-venv pattern) — then every subprocess,
torchrun workers included, inherits the correct interpreter with zero core
changes.

Dataset: **LeRobot v3.0**, consumed natively by cosmos-framework's loader — no
Odyssey-side conversion. ``$DATASET_PATH`` must point at the PARENT of the
``success/`` folder, and the loader infers the schema from the dataset DIR NAME
(e.g. ``droid_plus_lerobot_640x360_20260412``), so keep the vendor naming.

Routing: registered behind OpenVLA's wildcard training slot, reached via the
task-level ``config: {runner: cosmos3}`` override — same mechanism as ``gr00t``
and ``pi05``.

Licensing note: Cosmos 3 weights are distributed under NVIDIA's terms. This
runner contains no cosmos-framework code — it shells out to the user's own
checkout.
"""

from __future__ import annotations

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

# Shared flatten helper (dotted keys for nested dicts).
from odyssey.runners.models.openvla import _flatten_config
from odyssey.runners.subprocess import (
    TrainingProcessSpec,
    output_path,
    run_training_subprocess,
)
from odyssey.spec.refs import DatasetSource, HFModelRef
from odyssey.spec.tasks import TaskKind, TrainingTask, TrainingType

logger = logging.getLogger(__name__)


_DEFAULT_REPO_PATH = "/srv/cosmos-framework"
_RECIPE_REL = "examples/toml/sft_config"

_DCP_MODULE = "cosmos_framework.scripts.convert_model_to_dcp"
_TRAIN_MODULE = "cosmos_framework.scripts.train"
_EXPORT_MODULE = "cosmos_framework.scripts.export_model"

_DEFAULT_NPROC = 8
_DEFAULT_BASE_MODEL = "Cosmos3-Nano"

# Config keys the runner consumes directly — never forwarded as tyro overrides
# on the train step.
_CONTROL_KEYS = frozenset(
    {
        "config_name",
        "runner",
        "base_model",
        "convert_dcp",
        "export",
        "nproc_per_node",
        "base_checkpoint_path",
        "wan_vae_path",
        "filter_dir",
        "dataset_path",
        "dataset_env",
    }
)

# Which env var carries the dataset path into the recipe TOML's ``${oc.env:...}``
# interpolation — RECIPE-SPECIFIC (verified on cosmos-framework): the DROID recipe
# reads ``DATASET_PATH``; the LIBERO recipe reads ``LIBERO_ROOT`` (via
# LIBEROLeRobotDataset). Override per mission with ``config: {dataset_env: ...}``.
_DEFAULT_DATASET_ENV = "DATASET_PATH"


# cosmos-framework's train loop logs an imaginaire-style tqdm bar plus periodic
# metric dicts, e.g.:
#   "iter=1000 loss=0.234 ..."
#   " 10%|█  | 100/1000 [00:42<06:18,  2.38it/s]"
#   "Saving checkpoint to outputs/train/.../iter_000001000"
#   "Loading LeRobot dataset ..."
_COSMOS_LOSS_RE = re.compile(r"(?:'loss':|\bloss=)\s*([\d.eE+-]+)")
_COSMOS_ITER_RE = re.compile(r"(?:'iter':|\biter=|\bStep\s+)\s*(\d+)", re.IGNORECASE)
_COSMOS_TQDM_RE = re.compile(r"\b(\d+)/(\d+)\s*\[")
_COSMOS_SAVE_RE = re.compile(r"(?i)\b(saving|saved|wrote)\b.*\bcheckpoint\b")
_COSMOS_DCP_RE = re.compile(r"(?i)\b(convert\w*|writing|loading)\b.*\bdcp\b")
_COSMOS_DATASET_RE = re.compile(r"(?i)\b(loading|building)\b.*\b(dataset|lerobot)\b")


def parse_cosmos3_train_line(line: str) -> dict[str, Any] | None:
    """Extract a progress-event dict from a cosmos-framework stdout line.

    Public for tests and for users embedding cosmos stdout parsing in a custom
    runner. Precedence mirrors the GR00T / π0.5 parsers: a metric line
    (loss / iter) wins over the coarse phase markers.
    """
    loss_match = _COSMOS_LOSS_RE.search(line)
    iter_match = _COSMOS_ITER_RE.search(line)
    if loss_match or iter_match:
        payload: dict[str, Any] = {"stage": "executing", "step": "training_step"}
        if iter_match:
            payload["step_index"] = int(iter_match.group(1))
        if loss_match:
            payload["step_label"] = f"loss={loss_match.group(1)}"
        return payload
    tqdm_match = _COSMOS_TQDM_RE.search(line)
    if tqdm_match:
        return {
            "stage": "executing",
            "step": "training_step",
            "step_index": int(tqdm_match.group(1)),
            "step_total": int(tqdm_match.group(2)),
        }
    if _COSMOS_SAVE_RE.search(line):
        return {"stage": "checkpoint_saving"}
    if _COSMOS_DCP_RE.search(line):
        return {"stage": "dataset_loading", "step": "convert_dcp"}
    if _COSMOS_DATASET_RE.search(line):
        return {"stage": "dataset_loading"}
    return None


def _resolve_repo_path() -> str:
    """The cosmos-framework checkout root (``$COSMOS_FRAMEWORK_REPO_PATH``)."""
    return os.getenv("COSMOS_FRAMEWORK_REPO_PATH", _DEFAULT_REPO_PATH)


def _resolve_cosmos_recipe(config_name: str) -> str:
    """Absolute path to an SFT recipe TOML under ``$COSMOS_FRAMEWORK_REPO_PATH``.

    Accepts either a bare recipe name (``action_policy_libero_nano``) resolved
    against ``examples/toml/sft_config/`` or an already-absolute ``.toml`` path.
    Raises a runner-actionable error (not a bare FileNotFoundError) when the
    checkout / recipe is missing, matching the OpenVLA / π0.5 runner contract.
    """
    if os.path.isabs(config_name):
        recipe = config_name
    else:
        name = config_name if config_name.endswith(".toml") else f"{config_name}.toml"
        recipe = os.path.join(_resolve_repo_path(), _RECIPE_REL, name)
    if not os.path.isfile(recipe):
        raise RuntimeError(
            f"cosmos-framework SFT recipe not found at {recipe!r}; clone "
            "https://github.com/NVIDIA/cosmos-framework and set "
            "COSMOS_FRAMEWORK_REPO_PATH (or place it at /srv/cosmos-framework), "
            "then set config['config_name'] to a recipe under "
            "examples/toml/sft_config/ (e.g. 'action_policy_libero_nano')."
        )
    return recipe


def _dotlist_overrides(config: dict[str, Any]) -> list[str]:
    """Flatten ``config`` into Hydra-style ``key.path=value`` positionals.

    cosmos-framework's ``train.py`` consumes overrides as trailing
    ``key.path=value`` positionals (argparse ``REMAINDER``, applied AFTER the
    TOML via OmegaConf) — NOT ``--flag value``. So a nested ``trainer.max_iter``
    becomes the positional ``trainer.max_iter=10`` and a bool becomes
    ``key=true`` / ``key=false`` (OmegaConf parses those to real bools). Control
    keys (consumed by the runner) are skipped.
    """
    argv: list[str] = []
    for key, value in _flatten_config(config):
        # _flatten_config yields the dotted path; only the LEAF may be a control
        # key (nested overrides like trainer.max_iter are always forwarded).
        if "." not in key and key in _CONTROL_KEYS:
            continue
        if isinstance(value, bool):
            argv.append(f"{key}={'true' if value else 'false'}")
        else:
            argv.append(f"{key}={value}")
    return argv


def _resolve_base_model(task: TrainingTask, agent_model_base: str | None) -> str:
    """The base model handed to ``convert_model_to_dcp``.

    Precedence: explicit ``config['base_model']`` → the agent's HF model base →
    the ``Cosmos3-Nano`` default. cosmos-framework resolves a bare id like
    ``Cosmos3-Nano`` against its own checkpoint cache.
    """
    config = task.config or {}
    base = config.get("base_model") or agent_model_base or _DEFAULT_BASE_MODEL
    return str(base)


def build_cosmos3_dcp_argv(
    *, base_model: str, base_checkpoint_path: str
) -> list[str]:
    """Build the ``convert_model_to_dcp`` argv (step 1)."""
    return ["--checkpoint-path", base_model, "-o", base_checkpoint_path]


def build_cosmos3_train_argv(*, task: TrainingTask) -> list[str]:
    """Build the ``cosmos_framework.scripts.train`` argv (step 2).

    Shape: ``--sft-toml=<recipe> [key.path=value ...]``. ``config_name`` is
    REQUIRED — it names the registered SFT recipe TOML. Non-control ``config``
    keys become trailing Hydra-style dotlist positionals (applied AFTER the
    TOML), matching cosmos-framework's ``train.py`` REMAINDER override contract.
    """
    config = task.config or {}
    config_name = config.get("config_name")
    if not config_name:
        raise RuntimeError(
            "Cosmos 3 runner: config['config_name'] is required — it names the "
            "cosmos-framework SFT recipe TOML under examples/toml/sft_config/ "
            "(e.g. 'action_policy_libero_nano'). Add a recipe there for a custom "
            "dataset/embodiment."
        )
    recipe = _resolve_cosmos_recipe(str(config_name))
    return [f"--sft-toml={recipe}", *_dotlist_overrides(config)]


def build_cosmos3_export_argv(
    *, checkpoint_path: str, config_file: str, out_dir: str
) -> list[str]:
    """Build the ``export_model`` argv (step 3)."""
    return [
        "--checkpoint-path",
        checkpoint_path,
        "--config-file",
        config_file,
        "-o",
        out_dir,
    ]


def _dataset_env(task: TrainingTask) -> dict[str, str]:
    """Env overlay so cosmos-framework's dataset loader finds the local dataset.

    The env VAR NAME is recipe-specific (``config['dataset_env']``, default
    ``DATASET_PATH``): the DROID recipe reads ``DATASET_PATH``, the LIBERO recipe
    reads ``LIBERO_ROOT``. The VALUE is a LOCAL absolute path — explicit
    ``config['dataset_path']`` wins, else ``dataset.ref``.

    FOOTGUN (DROID): point at the PARENT of the ``success/`` folder (not
    ``success/`` itself), keeping the vendor DIR NAME (e.g.
    ``droid_plus_lerobot_640x360_20260412``) — the loader infers its schema from
    that name. FOOTGUN (LIBERO): ``LIBERO_ROOT`` is the suite dir holding
    ``meta/info.json`` (e.g. ``<dir>/libero_10``). HF/hub refs are left for the
    recipe TOML to resolve.
    """
    config = task.config or {}
    env_name = str(config.get("dataset_env") or _DEFAULT_DATASET_ENV)
    if config.get("dataset_path"):
        return {env_name: str(config["dataset_path"])}
    if task.dataset is None:
        return {}
    ref = task.dataset.ref
    if task.dataset.source == DatasetSource.LOCAL and os.path.isabs(ref):
        return {env_name: os.path.normpath(ref)}
    return {}


def _path_env(task: TrainingTask, output_dir: Path, base_ckpt_dcp: str) -> dict[str, str]:
    """Bridge the resolved paths into the env the recipe TOML interpolates.

    Only sets what the runner resolves; anything omitted falls through from the
    user's shell (``WAN_VAE_PATH``, ``HF_TOKEN`` are the operator's to provide).
    """
    config = task.config or {}
    env: dict[str, str] = {
        "BASE_CHECKPOINT_PATH": base_ckpt_dcp,
        "IMAGINAIRE_OUTPUT_ROOT": str(output_dir),
        "NPROC_PER_NODE": str(config.get("nproc_per_node", _DEFAULT_NPROC)),
    }
    if config.get("wan_vae_path"):
        env["WAN_VAE_PATH"] = str(config["wan_vae_path"])
    if config.get("filter_dir"):
        env["FILTER_DIR"] = str(config["filter_dir"])
    return env


def _resolve_run_dir(output_dir: Path) -> Path | None:
    """The cosmos-framework run dir under ``$IMAGINAIRE_OUTPUT_ROOT`` holding
    ``config.yaml``.

    cosmos-framework writes ``<output_root>/<experiment>/config.yaml`` alongside
    a ``checkpoints/`` subtree. Return the run dir whose ``config.yaml`` is
    newest; None if training wrote nothing. Heuristic (pinned by unit tests,
    verified at GPU smoke) — the exact layout is the vendor's.
    """
    candidates = sorted(
        output_dir.rglob("config.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].parent if candidates else None


def _resolve_trained_checkpoint(run_dir: Path) -> Path | None:
    """Locate the latest trained DCP checkpoint under a run dir.

    cosmos-framework saves ``.../checkpoints/iter_<NNN>`` (imaginaire style);
    return the highest-numbered one, or the run dir's ``checkpoints`` root if no
    numbered dir is present. None when nothing was written.
    """
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for entry in ckpt_root.iterdir():
        if not entry.is_dir():
            continue
        digits = re.findall(r"\d+", entry.name)
        if digits:
            step = int(digits[-1])
            if best is None or step > best[0]:
                best = (step, entry)
    if best is not None:
        return best[1]
    return ckpt_root


class Cosmos3Runner(Runner):
    """Fine-tune a Cosmos 3 action policy. Subprocess-based — actual training
    happens in cosmos-framework's ``convert_model_to_dcp`` → ``train`` →
    ``export_model`` scripts."""

    @property
    def name(self) -> str:
        return "cosmos3"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.TRAINING}

    @property
    def supported_types(self) -> set[str]:
        # Registered behind OpenVLA's wildcard; reached via the task-level
        # ``config: {runner: cosmos3}`` override (same pattern as GR00T / π0.5).
        return {WILDCARD_TYPE}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        spec = context.task.spec
        if not isinstance(spec, TrainingTask):
            raise TypeError(
                f"Cosmos3Runner expects TrainingTask, got {type(spec).__name__}"
            )
        if context.agent is None:
            raise RuntimeError(
                "Cosmos3Runner: TaskContext.agent is None — training tasks must "
                "be invoked through the engine, which resolves the agent from "
                "spec.robot.agents[task.agent_id]."
            )

        output_dir = output_path(context)
        output_dir.mkdir(parents=True, exist_ok=True)
        config = dict(spec.config)
        exp_name = str(config.get("exp_name") or spec.name)
        timeout = getattr(spec, "timeout_seconds", None)
        repo_path = _resolve_repo_path()

        agent_model_base = (
            context.agent.model.base
            if isinstance(context.agent.model, HFModelRef)
            else None
        )
        base_model = _resolve_base_model(spec, agent_model_base)

        # BASE_CHECKPOINT_PATH: explicit config wins, else a per-run DCP dir.
        base_ckpt_dcp = str(
            config.get("base_checkpoint_path") or output_dir / "base_checkpoint_dcp"
        )

        child_env = {
            **_dataset_env(spec),
            **_path_env(spec, output_dir, base_ckpt_dcp),
        }

        # Step 1: convert base model → DCP (cosmos needs it before training).
        # Skip when a prior run already produced it, or explicitly disabled.
        if config.get("convert_dcp", True) and not os.path.isdir(base_ckpt_dcp):
            await context.emit_progress(
                "dataset_loading", step="convert_dcp", step_label=base_model
            )
            dcp_spec = TrainingProcessSpec(
                timeout_seconds=timeout,
                entry_module=_DCP_MODULE,
                argv_extra=build_cosmos3_dcp_argv(
                    base_model=base_model, base_checkpoint_path=base_ckpt_dcp
                ),
                env=child_env,
                cwd=repo_path,
                line_parser=parse_cosmos3_train_line,
            )
            rc = await run_training_subprocess(context, dcp_spec)
            if context.cancelled():
                logger.info("Cosmos 3 task %s cancelled during DCP convert", context.task.id)
                return {"cancelled": True}
            if rc != 0:
                raise RuntimeError(f"cosmos-framework convert_model_to_dcp exited with code {rc}")
        elif os.path.isdir(base_ckpt_dcp):
            logger.info(
                "Cosmos 3 task %s: reusing existing DCP base checkpoint at %s",
                context.task.id,
                base_ckpt_dcp,
            )

        # Step 2: SFT fine-tune (torchrun).
        train_spec = TrainingProcessSpec(
            timeout_seconds=timeout,
            entry_module=_TRAIN_MODULE,
            argv_extra=build_cosmos3_train_argv(task=spec),
            env=child_env,
            cwd=repo_path,
            line_parser=parse_cosmos3_train_line,
            use_torchrun=True,
            torchrun_nproc=int(config.get("nproc_per_node", _DEFAULT_NPROC)),
        )
        rc = await run_training_subprocess(context, train_spec)
        if context.cancelled():
            logger.info("Cosmos 3 task %s cancelled by user", context.task.id)
            return {"cancelled": True}
        if rc != 0:
            raise RuntimeError(f"cosmos-framework train exited with code {rc}")

        run_dir = _resolve_run_dir(output_dir)
        if run_dir is None:
            raise RuntimeError(
                f"cosmos-framework train finished but no run dir (config.yaml) "
                f"found under {output_dir!r}"
            )
        trained_ckpt = _resolve_trained_checkpoint(run_dir)
        if trained_ckpt is None:
            raise RuntimeError(
                f"cosmos-framework train finished but no checkpoint found under "
                f"{run_dir / 'checkpoints'!r}"
            )

        # Step 3: export DCP → HF safetensors the policy server loads directly.
        export_dir = str(run_dir / "model")
        if config.get("export", True):
            await context.emit_progress(
                "checkpoint_saving", step="export_model", step_label=exp_name
            )
            export_spec = TrainingProcessSpec(
                timeout_seconds=timeout,
                entry_module=_EXPORT_MODULE,
                argv_extra=build_cosmos3_export_argv(
                    checkpoint_path=str(trained_ckpt),
                    config_file=str(run_dir / "config.yaml"),
                    out_dir=export_dir,
                ),
                env=child_env,
                cwd=repo_path,
                line_parser=parse_cosmos3_train_line,
            )
            rc = await run_training_subprocess(context, export_spec)
            if context.cancelled():
                logger.info("Cosmos 3 task %s cancelled during export", context.task.id)
                return {"cancelled": True}
            if rc != 0:
                raise RuntimeError(f"cosmos-framework export_model exited with code {rc}")
            checkpoint_path = export_dir
        else:
            # No export requested — hand back the trained DCP dir instead.
            checkpoint_path = str(trained_ckpt)

        return {
            "checkpoint_path": checkpoint_path,
            "agent_id": context.agent.id,
            "exp_name": exp_name,
            "config_name": config.get("config_name"),
            "base_model": base_model,
            "training_config": spec.config,
            "training_type": (
                spec.training_type.value
                if isinstance(spec.training_type, TrainingType)
                else spec.training_type
            ),
        }
