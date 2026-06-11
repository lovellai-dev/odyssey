"""Conformance tests against a real Isaac-GR00T checkout.

The GR00T runner's CLI surface was derived from NVIDIA's docs; these
tests pin it against the actual upstream source so drift shows up as a
red test instead of a failed training run on a rented GPU.

Env-gated: set ``ISAAC_GR00T_REPO_PATH`` to an Isaac-GR00T checkout to
enable (skipped otherwise, so CI without the checkout stays green).
The upstream config is inspected via ``ast`` — no ``gr00t`` install or
GPU is needed.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from odyssey.runners.gr00t import _ENTRY_MODULE, build_gr00t_argv
from odyssey.spec import TrainingTask
from odyssey.spec.loader import load_mission

REPO_ROOT = Path(__file__).resolve().parents[2]
QUICKSTART = REPO_ROOT / "examples" / "quickstart-gr00t" / "mission.yaml"

ISAAC_GR00T_REPO = os.getenv("ISAAC_GR00T_REPO_PATH")

pytestmark = pytest.mark.skipif(
    not ISAAC_GR00T_REPO,
    reason="ISAAC_GR00T_REPO_PATH not set — Isaac-GR00T conformance skipped",
)


def _repo() -> Path:
    assert ISAAC_GR00T_REPO is not None  # guarded by pytestmark
    return Path(ISAAC_GR00T_REPO)


def _upstream_finetune_fields() -> set[str]:
    """Field names of upstream FinetuneConfig, discovered via ast.

    tyro turns each dataclass field into a kebab-case CLI flag, so
    these are the complete legal flag surface of launch_finetune.
    """
    config_path = _repo() / "gr00t" / "configs" / "finetune_config.py"
    assert config_path.is_file(), (
        f"Upstream layout changed: {config_path} not found — the runner's "
        "entry assumptions need re-validation."
    )
    tree = ast.parse(config_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FinetuneConfig":
            return {
                stmt.target.id
                for stmt in node.body
                if isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
            }
    raise AssertionError(
        "FinetuneConfig class not found in upstream finetune_config.py"
    )


def _quickstart_training_task() -> TrainingTask:
    mission = load_mission(QUICKSTART)
    task = mission.tasks[0]
    assert isinstance(task, TrainingTask)
    return task


def test_entry_module_exists_upstream() -> None:
    module_path = _repo().joinpath(*_ENTRY_MODULE.split(".")).with_suffix(".py")
    assert module_path.is_file(), (
        f"Runner entry module {_ENTRY_MODULE!r} has no source file at "
        f"{module_path} — upstream moved the finetune entrypoint."
    )


def test_quickstart_demo_dataset_ships_in_repo() -> None:
    task = _quickstart_training_task()
    assert task.dataset is not None
    dataset_dir = _repo() / task.dataset.ref
    assert dataset_dir.is_dir(), (
        f"Quickstart dataset ref {task.dataset.ref!r} not found in the "
        "Isaac-GR00T checkout — pick a demo set that ships upstream."
    )


def test_quickstart_argv_flags_all_exist_upstream() -> None:
    """Every flag the runner emits for the shipped quickstart must be a
    real FinetuneConfig field — tyro rejects unknown flags."""
    fields = _upstream_finetune_fields()
    argv = build_gr00t_argv(
        task=_quickstart_training_task(),
        agent_model_base="nvidia/GR00T-N1.7-3B",
        output_dir=Path("/tmp/out"),
    )
    flags = [arg for arg in argv if arg.startswith("--")]
    unknown = [
        flag for flag in flags
        if flag.removeprefix("--").replace("-", "_") not in fields
    ]
    assert not unknown, (
        f"Runner emits flags with no upstream FinetuneConfig field: "
        f"{unknown}. Upstream fields: {sorted(fields)}"
    )


def test_core_contract_fields_exist_upstream() -> None:
    """The fields the runner injects itself (not user passthrough)."""
    fields = _upstream_finetune_fields()
    required = {"base_model_path", "dataset_path", "output_dir"}
    missing = required - fields
    assert not missing, (
        f"Upstream FinetuneConfig lost core fields the runner relies on: "
        f"{sorted(missing)}"
    )
