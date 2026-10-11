"""Worker-owned entry points, without importing Cosmos or starting training."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from odyssey.engine.records import MissionRun
from odyssey.runners.base import TaskContext
from odyssey.runners.models.cosmos3_train import (
    Cosmos3Runner,
    build_cosmos3_train_argv,
)
from odyssey.runners.subprocess import TrainingProcessSpec
from odyssey.spec import (
    AgentSpec,
    EvaluationTask,
    EvaluationType,
    HFModelRef,
    Mission,
    MissionMetadata,
    RobotSpec,
    TrainingTask,
    TrainingType,
)
from odyssey.telemetry import EventPublisher

STOCK_ENTRY = "cosmos_framework.scripts.train"
WORKER_ENTRY = "app.jobs.training._cosmos3_bootstrap"
RUNNER_MODULE = "odyssey.runners.models.cosmos3_train"


class CapturingPublisher(EventPublisher):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))


@pytest.fixture
def context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TaskContext:
    recipe = tmp_path / "framework/examples/toml/sft_config/ur10e.toml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("# Unit fixture: no Cosmos configuration is composed.\n")
    monkeypatch.setenv("COSMOS_FRAMEWORK_REPO_PATH", str(tmp_path / "framework"))
    agent = AgentSpec(id="pilot", model=HFModelRef(base="nvidia/Cosmos3-Nano"))
    mission = MissionRun.from_spec(
        Mission(
            metadata=MissionMetadata(name="ur10e-entry-test"),
            objective="Check subprocess selection, without loading a model.",
            acceptance_criteria="Only the worker can choose the training module.",
            robot=RobotSpec(embodiment="ur10e", agents=[agent]),
            tasks=[
                TrainingTask(
                    name="train-ur10e",
                    training_type=TrainingType.DEMONSTRATION,
                    agent_id="pilot",
                    timeout_seconds=75,
                    config={
                        "config_name": "ur10e",
                        "convert_dcp": False,
                        "export": False,
                        "nproc_per_node": 1,
                        "trainer": {"max_iter": 1},
                    },
                ),
                EvaluationTask(
                    name="evaluate-ur10e",
                    evaluation_type=EvaluationType.ROBOSUITE,
                    benchmark_name="Lift",
                ),
            ],
        )
    )
    return TaskContext(
        task=mission.tasks[0],
        mission=mission,
        publisher=CapturingPublisher(),
        agent=agent,
        output_dir=tmp_path / "output",
    )


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    async def simulate_process(context: TaskContext, spec: TrainingProcessSpec) -> int:
        # Only the external process is replaced. This is not checkpoint or
        # training qualification: minimal files exercise the runner's locator.
        if spec.use_torchrun:
            assert context.output_dir is not None
            run_dir = context.output_dir / "ur10e-run"
            (run_dir / "checkpoints/iter_000000001").mkdir(parents=True, exist_ok=True)
            (run_dir / "config.yaml").write_text("unit_fixture: true\n")
        return 0

    process = AsyncMock(side_effect=simulate_process)
    monkeypatch.setattr(f"{RUNNER_MODULE}.run_training_subprocess", process)
    return process


@pytest.mark.parametrize(
    "entry", [None, STOCK_ENTRY, WORKER_ENTRY], ids=["default", "stock", "worker"]
)
async def test_training_entry_uses_only_constructor_selection(
    context: TaskContext, launched: AsyncMock, entry: str | None
) -> None:
    runner = Cosmos3Runner() if entry is None else Cosmos3Runner(training_entry_module=entry)

    result = await runner.run(context)

    launched.assert_awaited_once()
    called_context, spec = launched.await_args.args
    assert called_context is context
    assert spec.entry_module == (entry or STOCK_ENTRY)
    assert spec.script_path is None
    assert spec.use_torchrun is True
    assert spec.torchrun_nproc == 1
    assert spec.timeout_seconds == 75
    assert spec.argv_extra == build_cosmos3_train_argv(task=context.task.spec)
    assert spec.env["IMAGINAIRE_OUTPUT_ROOT"] == str(context.output_dir)
    assert spec.env["NPROC_PER_NODE"] == "1"
    assert result["checkpoint_path"].endswith("checkpoints/iter_000000001")


async def test_worker_hook_changes_training_only_not_conversion_or_export(
    context: TaskContext, launched: AsyncMock
) -> None:
    context.task.spec.config.update(convert_dcp=True, export=True)

    result = await Cosmos3Runner(training_entry_module=WORKER_ENTRY).run(context)

    assert launched.await_count == 3
    specs = [call.args[1] for call in launched.await_args_list]
    assert [spec.entry_module for spec in specs] == [
        "cosmos_framework.scripts.convert_model_to_dcp",
        WORKER_ENTRY,
        "cosmos_framework.scripts.export_model",
    ]
    assert [spec.use_torchrun for spec in specs] == [False, True, False]
    assert all(spec.timeout_seconds == 75 for spec in specs)
    assert result["checkpoint_path"].endswith("ur10e-run/model")


async def test_worker_hook_is_per_instance_not_a_global_override(
    context: TaskContext, launched: AsyncMock
) -> None:
    await Cosmos3Runner(training_entry_module=WORKER_ENTRY).run(context)
    await Cosmos3Runner().run(context)

    assert launched.await_count == 2
    assert [call.args[1].entry_module for call in launched.await_args_list] == [
        WORKER_ENTRY,
        STOCK_ENTRY,
    ]


@pytest.mark.parametrize("entry", [STOCK_ENTRY, WORKER_ENTRY], ids=["stock", "worker"])
async def test_mission_cannot_override_entry_before_any_subprocess(
    context: TaskContext, launched: AsyncMock, entry: str
) -> None:
    context.task.spec.config.update(
        convert_dcp=True, training_entry_module="untrusted_module.execute"
    )

    with pytest.raises(ValueError, match=r"training_entry_module.*constructor"):
        await Cosmos3Runner(training_entry_module=entry).run(context)

    launched.assert_not_awaited()
    assert not context.output_dir.exists()


def test_mission_entry_is_not_forwarded_as_a_hydra_override(context: TaskContext) -> None:
    context.task.spec.config["training_entry_module"] = "untrusted_module.execute"

    with pytest.raises(ValueError, match=r"training_entry_module.*constructor"):
        build_cosmos3_train_argv(task=context.task.spec)


@pytest.mark.parametrize(
    "entry",
    [
        "",
        " module",
        "module ",
        "module\n",
        "-m",
        "../bootstrap",
        "bootstrap.py/",
        "a..b",
        ".a",
        "a.",
        "a;print(1)",
        "a.b --flag",
        "a\x00b",
        42,
        None,
        True,
    ],
    ids=[
        "empty",
        "leading-space",
        "trailing-space",
        "newline",
        "option",
        "relative-path",
        "script-path",
        "empty-part",
        "relative-import",
        "trailing-dot",
        "shell",
        "arguments",
        "nul",
        "integer",
        "none",
        "boolean",
    ],
)
def test_worker_entry_requires_an_absolute_python_module_name(entry: Any) -> None:
    with pytest.raises(ValueError, match="training_entry_module"):
        Cosmos3Runner(training_entry_module=entry)


def test_worker_entry_is_not_imported_in_orchestrator() -> None:
    # Deliberately unavailable in this light environment. Importing it here
    # would load GPU dependencies in the job orchestrator rather than the child.
    Cosmos3Runner(training_entry_module="uninstalled_worker._training_bootstrap")


async def test_worker_entry_failure_does_not_start_export(
    context: TaskContext, launched: AsyncMock
) -> None:
    context.task.spec.config["export"] = True
    launched.side_effect = None
    launched.return_value = 7

    with pytest.raises(RuntimeError, match="train exited with code 7"):
        await Cosmos3Runner(training_entry_module=WORKER_ENTRY).run(context)

    launched.assert_awaited_once()
    assert launched.await_args.args[1].entry_module == WORKER_ENTRY


async def test_worker_entry_cancellation_does_not_start_export(
    context: TaskContext, launched: AsyncMock
) -> None:
    context.task.spec.config["export"] = True

    async def cancel_at_train(ctx: TaskContext, spec: TrainingProcessSpec) -> int:
        assert spec.entry_module == WORKER_ENTRY
        ctx.request_cancel()
        return 0

    launched.side_effect = cancel_at_train

    result = await Cosmos3Runner(training_entry_module=WORKER_ENTRY).run(context)

    launched.assert_awaited_once()
    assert result == {"cancelled": True}
