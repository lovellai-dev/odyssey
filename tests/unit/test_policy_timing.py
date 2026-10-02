"""Tests for the policy timing contract (``control_hz`` / ``action_horizon``).

Covers the framework-agnostic checks, the π0.5 runner's pre-flight (with the
openpi probe faked: no openpi or GPU needed) and the values a custom eval
inherits from the training task that produced its checkpoint.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

import odyssey.runners.models.pi05_train as pi05
from odyssey.engine.lifecycle import TaskStatus
from odyssey.engine.records import MissionRun
from odyssey.runners.base import TaskContext
from odyssey.runners.evals.custom import eval_config_with_timing
from odyssey.runners.policy_timing import (
    PolicyTimingError,
    check_action_horizon,
    check_control_hz,
    declared_timing,
    eval_timing_defaults,
    lerobot_dataset_fps,
)
from odyssey.spec import (
    AgentRole,
    AgentSpec,
    DatasetRef,
    DatasetSource,
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


def _dataset(tmp_path: Path, fps: Any = 10) -> DatasetRef:
    root = tmp_path / "my_dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": fps}), encoding="utf-8")
    return DatasetRef(source=DatasetSource.LOCAL, ref=str(root))


def _task(name: str = "finetune", **overrides: Any) -> TrainingTask:
    fields: dict[str, Any] = {
        "name": name,
        "training_type": TrainingType.DEMONSTRATION,
        "agent_id": "pilot",
    }
    fields.update(overrides)
    return TrainingTask(**fields)


# ---------------------------------------------------------------------------
# Dataset fps
# ---------------------------------------------------------------------------

def test_fps_read_from_local_lerobot_dataset(tmp_path: Path) -> None:
    assert lerobot_dataset_fps(_dataset(tmp_path, fps=15)) == 15.0


@pytest.mark.parametrize(
    "dataset",
    [
        None,
        DatasetRef(source=DatasetSource.HUGGINGFACE, ref="org/name"),
        DatasetRef(source=DatasetSource.LOCAL, ref="relative/path"),
        DatasetRef(source=DatasetSource.LOCAL, ref="/does/not/exist"),
    ],
)
def test_fps_unreadable_is_none(dataset: DatasetRef | None) -> None:
    assert lerobot_dataset_fps(dataset) is None


def test_fps_non_numeric_is_none(tmp_path: Path) -> None:
    assert lerobot_dataset_fps(_dataset(tmp_path, fps="ten")) is None


# ---------------------------------------------------------------------------
# control_hz
# ---------------------------------------------------------------------------

def test_control_hz_matching_dataset_passes(tmp_path: Path) -> None:
    check_control_hz(_task(control_hz=10, dataset=_dataset(tmp_path, fps=10)))


def test_control_hz_mismatch_raises(tmp_path: Path) -> None:
    task = _task(control_hz=15, dataset=_dataset(tmp_path, fps=10))
    with pytest.raises(PolicyTimingError, match=r"control_hz=15.*recorded at 10"):
        check_control_hz(task)


def test_control_hz_undeclared_skips(tmp_path: Path) -> None:
    check_control_hz(_task(dataset=_dataset(tmp_path, fps=10)))


def test_control_hz_unreadable_dataset_skips() -> None:
    check_control_hz(
        _task(control_hz=10, dataset=DatasetRef(source=DatasetSource.HUGGINGFACE, ref="o/n"))
    )


# ---------------------------------------------------------------------------
# action_horizon
# ---------------------------------------------------------------------------

def test_action_horizon_match_passes() -> None:
    check_action_horizon(_task(action_horizon=16), 16, "cfg")


def test_action_horizon_mismatch_raises() -> None:
    with pytest.raises(PolicyTimingError, match=r"action_horizon=16.*use 10"):
        check_action_horizon(_task(action_horizon=16), 10, "cfg")


def test_action_horizon_undeclared_skips() -> None:
    check_action_horizon(_task(), 10, "cfg")


def test_declared_timing_only_reports_declared_values() -> None:
    assert declared_timing(_task()) == {}
    assert declared_timing(_task(control_hz=10, action_horizon=16)) == {
        "control_hz": 10,
        "action_horizon": 16,
    }


# ---------------------------------------------------------------------------
# π0.5 runner pre-flight
# ---------------------------------------------------------------------------

def test_pi05_configured_action_horizon_from_nested_override() -> None:
    assert pi05.configured_action_horizon({"model": {"action_horizon": 8}}) == 8
    assert pi05.configured_action_horizon({"config_name": "x"}) is None


def test_pi05_preflight_uses_config_override_without_probing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _never(*_: Any) -> int:
        raise AssertionError("probe must not run when config overrides the horizon")

    monkeypatch.setattr(pi05, "probe_openpi_action_horizon", _never)
    task = _task(
        action_horizon=8,
        config={"config_name": "cfg", "model": {"action_horizon": 8}},
    )
    asyncio.run(pi05.check_pi05_policy_timing(task, {}))


def test_pi05_preflight_probes_registered_config(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    async def _probe(config_name: str, env: dict[str, str], cancel_event: Any) -> int:
        seen.append(config_name)
        return 10

    monkeypatch.setattr(pi05, "probe_openpi_action_horizon", _probe)
    task = _task(action_horizon=16, config={"config_name": "my_config"})
    with pytest.raises(PolicyTimingError, match="openpi config 'my_config'"):
        asyncio.run(pi05.check_pi05_policy_timing(task, {}))
    assert seen == ["my_config"]


def test_pi05_preflight_control_hz_mismatch_raises(tmp_path: Path) -> None:
    task = _task(
        control_hz=30,
        dataset=_dataset(tmp_path, fps=10),
        config={"config_name": "cfg"},
    )
    with pytest.raises(PolicyTimingError, match="control_hz=30"):
        asyncio.run(pi05.check_pi05_policy_timing(task, {}))


def test_pi05_preflight_noop_when_nothing_declared(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _never(*_: Any) -> int:
        raise AssertionError("probe must not run when action_horizon is undeclared")

    monkeypatch.setattr(pi05, "probe_openpi_action_horizon", _never)
    asyncio.run(pi05.check_pi05_policy_timing(_task(config={"config_name": "c"}), {}))


def test_pi05_probe_reports_unreadable_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real subprocess: a probe that fails surfaces as PolicyTimingError."""
    monkeypatch.setattr(pi05, "_ACTION_HORIZON_PROBE", "import sys; sys.exit(3)")
    with pytest.raises(PolicyTimingError, match="exit 3"):
        asyncio.run(pi05.probe_openpi_action_horizon("cfg", {}))


def test_pi05_probe_parses_last_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pi05, "_ACTION_HORIZON_PROBE", "print('loading...'); print(16)"
    )
    assert asyncio.run(pi05.probe_openpi_action_horizon("cfg", {})) == 16


# ---------------------------------------------------------------------------
# Evaluation inherits the trained values
# ---------------------------------------------------------------------------

def _mission(train_tasks: list[TrainingTask], eval_config: dict[str, Any]) -> Mission:
    return Mission(
        metadata=MissionMetadata(name="msn"),
        objective="o",
        acceptance_criteria="a",
        robot=RobotSpec(
            embodiment="franka_panda",
            agents=[
                AgentSpec(id="pilot", role=AgentRole.PILOT, model=HFModelRef(base="m/m"))
            ],
        ),
        tasks=[
            *train_tasks,
            EvaluationTask(
                name="bench",
                evaluation_type=EvaluationType.CUSTOM,
                benchmark_name="b",
                config=eval_config,
            ),
        ],
    )


class _NullPublisher(EventPublisher):
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        pass


def _eval_context(mission: Mission) -> TaskContext:
    run = MissionRun.from_spec(mission)
    return TaskContext(
        task=run.tasks[-1],
        mission=run,
        publisher=_NullPublisher(),
        agents=list(mission.robot.agents),
    )


def _run(mission: Mission, outcomes: list[str | None]) -> MissionRun:
    """A MissionRun whose training records ended as ``outcomes``.

    One entry per training task, in spec order: a checkpoint path marks the
    task COMPLETED with that ``checkpoint_path``; None marks it FAILED.
    """
    run = MissionRun.from_spec(mission)
    for record, checkpoint in zip(run.tasks[: len(outcomes)], outcomes, strict=True):
        if checkpoint is None:
            record.status = TaskStatus.FAILED
        else:
            record.status = TaskStatus.COMPLETED
            record.result_summary = {"checkpoint_path": checkpoint}
    return run


def _eval_context(run: MissionRun) -> TaskContext:
    return TaskContext(
        task=run.tasks[-1],
        mission=run,
        publisher=_NullPublisher(),
        agents=list(run.spec.robot.agents),
    )


def test_eval_defaults_come_from_task_that_produced_checkpoint() -> None:
    run = _run(
        _mission(
            [
                _task("first", control_hz=5, action_horizon=4),
                _task("second", control_hz=10, action_horizon=16),
            ],
            {},
        ),
        ["/ckpt/a", "/ckpt/b"],
    )
    assert eval_timing_defaults(run, "/ckpt/b") == {"control_hz": 10, "action_horizon": 16}
    assert eval_timing_defaults(run, "/ckpt/a") == {"control_hz": 5, "action_horizon": 4}


def test_eval_defaults_skip_failed_last_training() -> None:
    """Regression: a FAILED later training must not lend its timing.

    With ``on_task_failure: continue`` the eval runs the first task's
    checkpoint, so it must get 10 Hz / 16 rows, not the failed 30 Hz / 50.
    """
    run = _run(
        _mission(
            [
                _task("first", control_hz=10, action_horizon=16),
                _task("second", control_hz=30, action_horizon=50),
            ],
            {},
        ),
        ["/ckpt/a", None],
    )
    assert run.latest_checkpoint_for("pilot") == "/ckpt/a"
    assert eval_timing_defaults(run, "/ckpt/a") == {"control_hz": 10, "action_horizon": 16}


def test_eval_defaults_empty_for_explicit_checkpoint() -> None:
    """Regression: a checkpoint no task produced inherits nothing."""
    run = _run(
        _mission([_task(control_hz=30, action_horizon=50)], {}),
        ["/ckpt/trained"],
    )
    assert eval_timing_defaults(run, "org/published-model") == {}


def test_eval_defaults_match_uri_checkpoint_through_path() -> None:
    """Regression: eval runners pass a Path, which collapses ``//`` in URIs."""
    run = _run(_mission([_task(control_hz=10, action_horizon=16)], {}), ["mock://t/final"])
    assert eval_timing_defaults(run, str(Path("mock://t/final"))) == {
        "control_hz": 10,
        "action_horizon": 16,
    }


def test_eval_defaults_not_merged_across_tasks() -> None:
    run = _run(
        _mission([_task("first", control_hz=5), _task("second")], {}),
        ["/ckpt/a", "/ckpt/b"],
    )
    assert eval_timing_defaults(run, "/ckpt/b") == {}


def test_eval_defaults_empty_for_eval_only_mission() -> None:
    assert eval_timing_defaults(_run(_mission([], {}), []), "org/m") == {}


def test_custom_eval_config_gets_trained_timing() -> None:
    run = _run(_mission([_task(control_hz=10, action_horizon=16)], {"x": 1}), ["/ckpt/a"])
    assert eval_config_with_timing(_eval_context(run), Path("/ckpt/a"), {"x": 1}) == {
        "control_hz": 10,
        "action_horizon": 16,
        "x": 1,
    }


def test_custom_eval_explicit_config_wins() -> None:
    eval_config = {"control_hz": 5}
    run = _run(_mission([_task(control_hz=10, action_horizon=16)], eval_config), ["/ckpt/a"])
    assert eval_config_with_timing(_eval_context(run), Path("/ckpt/a"), eval_config) == {
        "control_hz": 5,
        "action_horizon": 16,
    }


_TIMING_EVAL_SCRIPT = (
    "import argparse, json\n"
    "p = argparse.ArgumentParser()\n"
    "p.add_argument('--checkpoint'); p.add_argument('--out-json')\n"
    "p.add_argument('--control_hz', type=float)\n"
    "p.add_argument('--action_horizon', type=int)\n"
    "a = p.parse_args()\n"
    "json.dump({'metrics': {'ckpt': a.checkpoint, 'hz': a.control_hz,"
    " 'h': a.action_horizon}}, open(a.out_json, 'w'))\n"
)


def _run_custom_eval(
    tmp_path: Path, train_tasks: list[TrainingTask], outcomes: list[str | None],
    eval_config: dict[str, Any],
) -> dict[str, Any]:
    from odyssey.runners.evals.custom import CustomEvalRunner

    script = tmp_path / "eval.py"
    script.write_text(_TIMING_EVAL_SCRIPT, encoding="utf-8")
    config = {"eval_script": str(script), "eval_python": sys.executable, **eval_config}
    ctx = _eval_context(_run(_mission(train_tasks, config), outcomes))
    ctx.output_dir = tmp_path / "out"
    metrics: dict[str, Any] = asyncio.run(CustomEvalRunner().run(ctx))["metrics"]
    return {k: metrics[k] for k in ("ckpt", "hz", "h")}  # what the script saw


def test_custom_eval_script_receives_timing_flags(tmp_path: Path) -> None:
    """End to end: the eval script is launched with --control_hz/--action_horizon."""
    metrics = _run_custom_eval(
        tmp_path, [_task(control_hz=10, action_horizon=16)], ["/ckpt/a"], {}
    )
    assert metrics == {"ckpt": "/ckpt/a", "hz": 10.0, "h": 16}


def test_custom_eval_after_failed_last_training_gets_matching_timing(
    tmp_path: Path,
) -> None:
    """End to end: checkpoint and timing both come from the completed task."""
    metrics = _run_custom_eval(
        tmp_path,
        [
            _task("first", control_hz=10, action_horizon=16),
            _task("second", control_hz=30, action_horizon=50),
        ],
        ["/ckpt/a", None],
        {},
    )
    assert metrics == {"ckpt": "/ckpt/a", "hz": 10.0, "h": 16}


def test_custom_eval_explicit_checkpoint_inherits_no_timing(tmp_path: Path) -> None:
    """End to end: an explicit external checkpoint runs on eval config only."""
    metrics = _run_custom_eval(
        tmp_path,
        [_task(control_hz=30, action_horizon=50)],
        ["/ckpt/trained"],
        {"checkpoint": "org/published-model", "control_hz": 5},
    )
    assert metrics == {"ckpt": "org/published-model", "hz": 5.0, "h": None}


def test_pi05_run_fails_before_any_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timing mismatch stops the task before norm stats or training launch."""

    async def _no_subprocess(*_: Any, **__: Any) -> int:
        raise AssertionError("no openpi subprocess may start on a timing mismatch")

    monkeypatch.setattr(pi05, "run_training_subprocess", _no_subprocess)
    monkeypatch.setattr(pi05.Path, "home", lambda: tmp_path)
    train = _task(
        control_hz=30,
        dataset=_dataset(tmp_path, fps=10),
        config={"runner": "pi05", "config_name": "cfg"},
    )
    mission = _mission([train], {})
    run = MissionRun.from_spec(mission)
    ctx = TaskContext(
        task=run.tasks[0],
        mission=run,
        publisher=_NullPublisher(),
        output_dir=tmp_path / "out",
        agent=mission.robot.agents[0],
    )
    with pytest.raises(PolicyTimingError, match="control_hz=30"):
        asyncio.run(pi05.Pi05Runner().run(ctx))


# ---------------------------------------------------------------------------
# The probe child never outlives an interrupted pre-flight
# ---------------------------------------------------------------------------

_SLEEPING_PROBE = "import time; time.sleep(60); print(16)"


def _capture_probe_proc(monkeypatch: pytest.MonkeyPatch) -> list[asyncio.subprocess.Process]:
    """Run the real probe subprocess, but sleeping, and record its handle."""
    monkeypatch.setattr(pi05, "_ACTION_HORIZON_PROBE", _SLEEPING_PROBE)
    procs: list[asyncio.subprocess.Process] = []
    real_exec = asyncio.create_subprocess_exec

    async def _exec(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await real_exec(*args, **kwargs)
        procs.append(proc)
        return proc

    monkeypatch.setattr(pi05.asyncio, "create_subprocess_exec", _exec)
    return procs


async def _until_started(procs: list[asyncio.subprocess.Process]) -> None:
    while not procs:
        await asyncio.sleep(0.01)


def test_pi05_probe_reaped_when_coroutine_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    procs = _capture_probe_proc(monkeypatch)

    async def _scenario() -> None:
        probe = asyncio.create_task(pi05.probe_openpi_action_horizon("cfg", {}))
        await _until_started(procs)
        probe.cancel()
        with pytest.raises(asyncio.CancelledError):
            await probe

    asyncio.run(_scenario())
    assert procs[0].returncode is not None


def test_pi05_probe_stops_on_mission_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mission's cancel_event ends the probe at once, not after the timeout."""
    procs = _capture_probe_proc(monkeypatch)

    async def _scenario() -> int | None:
        cancel_event = asyncio.Event()
        probe = asyncio.create_task(
            pi05.probe_openpi_action_horizon("cfg", {}, cancel_event)
        )
        await _until_started(procs)
        cancel_event.set()
        return await asyncio.wait_for(probe, timeout=10)

    assert asyncio.run(_scenario()) is None
    assert procs[0].returncode is not None


def test_pi05_probe_reaped_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    procs = _capture_probe_proc(monkeypatch)
    monkeypatch.setattr(pi05, "_PROBE_TIMEOUT_S", 0.5)
    with pytest.raises(PolicyTimingError, match="timed out"):
        asyncio.run(pi05.probe_openpi_action_horizon("cfg", {}))
    assert procs[0].returncode is not None


def test_pi05_run_cancelled_during_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mission cancellation during the probe ends the task as cancelled."""
    procs = _capture_probe_proc(monkeypatch)

    async def _no_subprocess(*_: Any, **__: Any) -> int:
        raise AssertionError("no openpi subprocess may start after cancellation")

    monkeypatch.setattr(pi05, "run_training_subprocess", _no_subprocess)
    monkeypatch.setattr(pi05.Path, "home", lambda: tmp_path)
    mission = _mission(
        [_task(action_horizon=16, config={"runner": "pi05", "config_name": "cfg"})], {}
    )
    run = MissionRun.from_spec(mission)
    ctx = TaskContext(
        task=run.tasks[0],
        mission=run,
        publisher=_NullPublisher(),
        output_dir=tmp_path / "out",
        agent=mission.robot.agents[0],
    )

    async def _scenario() -> dict[str, Any]:
        task = asyncio.create_task(pi05.Pi05Runner().run(ctx))
        await _until_started(procs)
        ctx.request_cancel()
        return await asyncio.wait_for(task, timeout=10)

    assert asyncio.run(_scenario()) == {"cancelled": True}
    assert procs[0].returncode is not None
