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

import odyssey.runners.models.openpi_dataset as openpi_dataset
import odyssey.runners.models.pi05_train as pi05
from odyssey.engine.lifecycle import TaskStatus
from odyssey.engine.records import MissionRun
from odyssey.runners.base import TaskContext
from odyssey.runners.evals.custom import eval_config_with_timing
from odyssey.runners.models.openpi_dataset import (
    DatasetMismatchError,
    OpenpiProbeError,
    OpenpiTrainingConfig,
    check_declared_dataset,
)
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
    assert lerobot_dataset_fps(_dataset(tmp_path, fps=15).ref) == 15.0


@pytest.mark.parametrize("dataset_dir", [None, "/does/not/exist"])
def test_fps_unreadable_is_none(dataset_dir: str | None) -> None:
    assert lerobot_dataset_fps(dataset_dir) is None


def test_fps_non_numeric_is_none(tmp_path: Path) -> None:
    assert lerobot_dataset_fps(_dataset(tmp_path, fps="ten").ref) is None


# ---------------------------------------------------------------------------
# control_hz
# ---------------------------------------------------------------------------

def test_control_hz_matching_dataset_passes(tmp_path: Path) -> None:
    check_control_hz(_task(control_hz=10), _dataset(tmp_path, fps=10).ref)


def test_control_hz_mismatch_raises(tmp_path: Path) -> None:
    with pytest.raises(PolicyTimingError, match=r"control_hz=15.*recorded at 10"):
        check_control_hz(_task(control_hz=15), _dataset(tmp_path, fps=10).ref)


def test_control_hz_undeclared_skips(tmp_path: Path) -> None:
    check_control_hz(_task(), _dataset(tmp_path, fps=10).ref)


def test_control_hz_unreadable_dataset_skips() -> None:
    check_control_hz(_task(control_hz=10), None)


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

def _effective(
    action_horizon: int = 16, repo_id: str | None = None, dataset_dir: str | None = None
) -> OpenpiTrainingConfig:
    return OpenpiTrainingConfig(action_horizon, repo_id, dataset_dir)


def _fake_probe(
    monkeypatch: pytest.MonkeyPatch, effective: OpenpiTrainingConfig
) -> list[tuple[Any, ...]]:
    seen: list[tuple[Any, ...]] = []

    async def _probe(*args: Any) -> OpenpiTrainingConfig:
        seen.append(args)
        return effective

    monkeypatch.setattr(pi05, "probe_openpi_training_config", _probe)
    return seen


def test_pi05_preflight_noop_when_nothing_declared(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _never(*_: Any) -> None:
        raise AssertionError("probe must not run when no timing is declared")

    monkeypatch.setattr(pi05, "probe_openpi_training_config", _never)
    asyncio.run(pi05.check_pi05_policy_timing(_task(config={"config_name": "c"}), {}, "e"))


def test_pi05_preflight_probes_config_with_mission_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe builds the config train.py will: name, exp name and overrides."""
    seen = _fake_probe(monkeypatch, _effective(action_horizon=8))
    task = _task(
        action_horizon=8,
        config={"config_name": "my_config", "model": {"action_horizon": 8}},
    )
    asyncio.run(pi05.check_pi05_policy_timing(task, {"HF_LEROBOT_HOME": "/h"}, "exp"))
    config_name, exp_name, overrides, env, _ = seen[0]
    assert (config_name, exp_name, env) == ("my_config", "exp", {"HF_LEROBOT_HOME": "/h"})
    assert overrides == ["--model.action-horizon", "8"]


def test_pi05_preflight_action_horizon_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_probe(monkeypatch, _effective(action_horizon=10))
    task = _task(action_horizon=16, config={"config_name": "my_config"})
    with pytest.raises(PolicyTimingError, match="openpi config 'my_config'"):
        asyncio.run(pi05.check_pi05_policy_timing(task, {}, "e"))


def test_pi05_preflight_control_hz_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset(tmp_path, fps=10)
    _fake_probe(monkeypatch, _effective(repo_id="my_dataset", dataset_dir=dataset.ref))
    task = _task(control_hz=30, dataset=dataset, config={"config_name": "cfg"})
    with pytest.raises(PolicyTimingError, match="control_hz=30"):
        asyncio.run(pi05.check_pi05_policy_timing(task, {}, "e"))


def test_pi05_preflight_unreadable_config_is_a_timing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fails(*_: Any) -> None:
        raise OpenpiProbeError("could not read openpi config 'cfg' (exit 1)")

    monkeypatch.setattr(pi05, "probe_openpi_training_config", _fails)
    task = _task(action_horizon=16, config={"config_name": "cfg"})
    with pytest.raises(PolicyTimingError, match="could not read"):
        asyncio.run(pi05.check_pi05_policy_timing(task, {}, "e"))


# ---------------------------------------------------------------------------
# The dataset training loads vs the declared one
# ---------------------------------------------------------------------------

def _lerobot_dataset(root: Path, fps: int) -> DatasetRef:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": fps}), encoding="utf-8")
    return DatasetRef(source=DatasetSource.LOCAL, ref=str(root))


def test_declared_dataset_matching_loaded_dir_passes(tmp_path: Path) -> None:
    ds = _lerobot_dataset(tmp_path / "ds", 10)
    check_declared_dataset(_task(dataset=ds), _effective(repo_id="ds", dataset_dir=ds.ref))


def test_declared_dataset_other_dir_raises(tmp_path: Path) -> None:
    declared = _lerobot_dataset(tmp_path / "declared", 10)
    actual = _lerobot_dataset(tmp_path / "actual", 30)
    with pytest.raises(DatasetMismatchError, match="training loads"):
        check_declared_dataset(
            _task(dataset=declared), _effective(repo_id="actual", dataset_dir=actual.ref)
        )


def test_declared_hub_dataset_must_be_loaded_repo_id() -> None:
    hub = DatasetRef(source=DatasetSource.HUGGINGFACE, ref="org/a")
    check_declared_dataset(_task(dataset=hub), _effective(repo_id="org/a"))
    with pytest.raises(DatasetMismatchError, match="'org/b'"):
        check_declared_dataset(_task(dataset=hub), _effective(repo_id="org/b"))


@pytest.mark.parametrize(
    "effective",
    [_effective(repo_id=None), _effective(repo_id="x", dataset_dir="/elsewhere/x")],
)
def test_declared_dataset_unknown_or_other_source_skips(
    effective: OpenpiTrainingConfig,
) -> None:
    oxe = DatasetRef(source=DatasetSource.OXE, ref="bridge_orig")
    check_declared_dataset(_task(dataset=oxe), effective)
    check_declared_dataset(_task(), effective)


# A stand-in openpi + lerobot, importable by the REAL probe subprocess: the
# probe runs ``python -c`` in the caller's cwd, so a package dir as cwd is on
# its sys.path. ``cli()`` mimics openpi's: <config_name> + tyro overrides.
_FAKE_OPENPI_CONFIG = """
import sys
from types import SimpleNamespace

def cli():
    args = sys.argv[1:]
    flags = dict(zip(args[1::2], args[2::2]))
    return SimpleNamespace(
        data=SimpleNamespace(repo_id=flags.get("--data.repo-id", "declared")),
        model=SimpleNamespace(action_horizon=int(flags.get("--model.action-horizon", 10))),
    )
"""
_FAKE_LEROBOT_CONSTANTS = """
import os
from pathlib import Path
HF_LEROBOT_HOME = Path(os.environ["HF_LEROBOT_HOME"])
"""


def _fake_openpi_on_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkgs = tmp_path / "pkgs"
    for mod, body in [
        ("openpi/__init__.py", ""),
        ("openpi/training/__init__.py", ""),
        ("openpi/training/config.py", _FAKE_OPENPI_CONFIG),
        ("lerobot/__init__.py", ""),
        ("lerobot/common/__init__.py", ""),
        ("lerobot/common/constants.py", _FAKE_LEROBOT_CONSTANTS),
    ]:
        (pkgs / mod).parent.mkdir(parents=True, exist_ok=True)
        (pkgs / mod).write_text(body, encoding="utf-8")
    monkeypatch.chdir(pkgs)


def _sibling_datasets(tmp_path: Path) -> DatasetRef:
    """``declared`` at 10 fps (the spec's dataset) next to ``actual`` at 30 fps."""
    declared = _lerobot_dataset(tmp_path / "data" / "declared", 10)
    _lerobot_dataset(tmp_path / "data" / "actual", 30)
    return declared


def test_pi05_preflight_rejects_repo_id_selecting_another_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: dataset.ref → declared (10 fps), data.repo_id → actual (30 fps).

    control_hz=10 matches the DECLARED dataset, but training loads the other
    one. Real env builder, real overrides, real probe subprocess.
    """
    _fake_openpi_on_cwd(tmp_path, monkeypatch)
    task = _task(
        control_hz=10,
        dataset=_sibling_datasets(tmp_path),
        config={"config_name": "cfg", "data": {"repo_id": "actual"}},
    )
    env = pi05._lerobot_env_for_dataset(task)
    with pytest.raises(DatasetMismatchError, match=r"training loads .*actual"):
        asyncio.run(pi05.check_pi05_policy_timing(task, env, "e"))


def test_pi05_preflight_checks_fps_of_the_loaded_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same path end to end when they agree: the loaded dataset's fps is checked."""
    _fake_openpi_on_cwd(tmp_path, monkeypatch)
    declared = _sibling_datasets(tmp_path)
    ok = _task(control_hz=10, dataset=declared, config={"config_name": "cfg"})
    asyncio.run(pi05.check_pi05_policy_timing(ok, pi05._lerobot_env_for_dataset(ok), "e"))
    wrong = _task(control_hz=30, dataset=declared, config={"config_name": "cfg"})
    with pytest.raises(PolicyTimingError, match=r"control_hz=30.*recorded at 10"):
        asyncio.run(
            pi05.check_pi05_policy_timing(wrong, pi05._lerobot_env_for_dataset(wrong), "e")
        )


# ---------------------------------------------------------------------------
# The probe subprocess
# ---------------------------------------------------------------------------

def test_probe_reports_unreadable_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real subprocess: a probe that fails surfaces as OpenpiProbeError."""
    monkeypatch.setattr(openpi_dataset, "_TRAINING_CONFIG_PROBE", "import sys; sys.exit(3)")
    with pytest.raises(OpenpiProbeError, match="exit 3"):
        asyncio.run(openpi_dataset.probe_openpi_training_config("cfg", "e", [], {}))


def test_probe_parses_last_json_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        openpi_dataset,
        "_TRAINING_CONFIG_PROBE",
        "print('loading...'); "
        "print('{\"action_horizon\": 16, \"repo_id\": \"r\", \"dataset_dir\": \"/h/r\"}')",
    )
    assert asyncio.run(
        openpi_dataset.probe_openpi_training_config("cfg", "e", [], {})
    ) == OpenpiTrainingConfig(16, "r", "/h/r")


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
    dataset = _dataset(tmp_path, fps=10)
    _fake_probe(monkeypatch, _effective(repo_id="my_dataset", dataset_dir=dataset.ref))
    train = _task(
        control_hz=30,
        dataset=dataset,
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

_SLEEPING_PROBE = "import time; time.sleep(60)"


def _capture_probe_proc(monkeypatch: pytest.MonkeyPatch) -> list[asyncio.subprocess.Process]:
    """Run the real probe subprocess, but sleeping, and record its handle."""
    monkeypatch.setattr(openpi_dataset, "_TRAINING_CONFIG_PROBE", _SLEEPING_PROBE)
    procs: list[asyncio.subprocess.Process] = []
    real_exec = asyncio.create_subprocess_exec

    async def _exec(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        proc = await real_exec(*args, **kwargs)
        procs.append(proc)
        return proc

    monkeypatch.setattr(openpi_dataset.asyncio, "create_subprocess_exec", _exec)
    return procs


async def _until_started(procs: list[asyncio.subprocess.Process]) -> None:
    while not procs:
        await asyncio.sleep(0.01)


def test_pi05_probe_reaped_when_coroutine_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    procs = _capture_probe_proc(monkeypatch)

    async def _scenario() -> None:
        probe = asyncio.create_task(openpi_dataset.probe_openpi_training_config("cfg", "e", [], {}))
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
            openpi_dataset.probe_openpi_training_config("cfg", "e", [], {}, cancel_event)
        )
        await _until_started(procs)
        cancel_event.set()
        return await asyncio.wait_for(probe, timeout=10)

    assert asyncio.run(_scenario()) is None
    assert procs[0].returncode is not None


def test_pi05_probe_reaped_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    procs = _capture_probe_proc(monkeypatch)
    monkeypatch.setattr(openpi_dataset, "_PROBE_TIMEOUT_S", 0.5)
    with pytest.raises(OpenpiProbeError, match="timed out"):
        asyncio.run(openpi_dataset.probe_openpi_training_config("cfg", "e", [], {}))
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
