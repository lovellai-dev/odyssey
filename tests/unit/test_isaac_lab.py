"""Tests for the Isaac Lab evaluation runner.

No Isaac Sim here — the runner's testable pieces are the launch
contract (argv, script/launcher resolution), the ODYSSEY_* stdout
protocol collector, and summary scoring. One integration-ish test runs
a fake eval script (plain python printing protocol lines) through the
real subprocess machinery end-to-end.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from odyssey.engine import TaskStatus
from odyssey.engine.records import MissionRun
from odyssey.runners.base import TaskContext
from odyssey.runners.evals.isaac_lab import (
    _EPISODE_PREFIX,
    _REASONING_PREFIX,
    _RESULT_PREFIX,
    EvalProtocolCollector,
    IsaacLabRunner,
    build_isaac_lab_argv,
    resolve_eval_script,
    resolve_launcher,
    summarize,
)
from odyssey.runners.subprocess import TrainingProcessSpec
from odyssey.spec import (
    AgentRole,
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


def _eval_task(**overrides: Any) -> EvaluationTask:
    fields: dict[str, Any] = {
        "name": "eval-isaac",
        "evaluation_type": EvaluationType.ISAAC_LAB,
        "benchmark_name": "Isaac-Lift-Cube-Franka-v0",
        "num_episodes": 10,
    }
    fields.update(overrides)
    return EvaluationTask(**fields)


# ---------------------------------------------------------------------------
# Launch contract
# ---------------------------------------------------------------------------

def test_argv_contains_contract_flags(tmp_path: Path) -> None:
    argv = build_isaac_lab_argv(task=_eval_task(), checkpoint=tmp_path / "ckpt")
    assert argv[argv.index("--task") + 1] == "Isaac-Lift-Cube-Franka-v0"
    assert argv[argv.index("--num_episodes") + 1] == "10"
    assert argv[argv.index("--checkpoint") + 1] == str(tmp_path / "ckpt")
    assert "--headless" in argv


def test_argv_headless_false_omits_flag(tmp_path: Path) -> None:
    task = _eval_task(config={"headless": False})
    argv = build_isaac_lab_argv(task=task, checkpoint=tmp_path)
    assert "--headless" not in argv


def test_argv_passthrough_keeps_snake_case(tmp_path: Path) -> None:
    task = _eval_task(config={"num_envs": 4})
    argv = build_isaac_lab_argv(task=task, checkpoint=tmp_path)
    assert argv[argv.index("--num_envs") + 1] == "4"


def test_argv_excludes_runner_keys(tmp_path: Path) -> None:
    task = _eval_task(config={"eval_script": "/x.py", "runner": "isaac_lab"})
    argv = build_isaac_lab_argv(task=task, checkpoint=tmp_path)
    assert "--eval_script" not in argv
    assert "--runner" not in argv


def test_eval_script_from_config() -> None:
    assert resolve_eval_script({"eval_script": "/opt/eval.py"}) == "/opt/eval.py"


def test_eval_script_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISAACLAB_EVAL_SCRIPT", "/env/eval.py")
    assert resolve_eval_script({}) == "/env/eval.py"


def test_eval_script_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ISAACLAB_EVAL_SCRIPT", raising=False)
    with pytest.raises(RuntimeError, match="eval_script"):
        resolve_eval_script({})


def test_launcher_from_isaaclab_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISAACLAB_PATH", "/opt/IsaacLab")
    assert resolve_launcher() == ["/opt/IsaacLab/isaaclab.sh", "-p"]


def test_launcher_none_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ISAACLAB_PATH", raising=False)
    assert resolve_launcher() is None


def test_process_spec_rejects_launcher_with_entry_module() -> None:
    with pytest.raises(ValueError, match="launcher"):
        TrainingProcessSpec(entry_module="some.module", launcher=["x.sh", "-p"])


# ---------------------------------------------------------------------------
# Protocol collector
# ---------------------------------------------------------------------------

def test_collector_records_episode_and_emits_progress() -> None:
    collector = EvalProtocolCollector()
    event = collector.parse(
        'ODYSSEY_EPISODE {"index": 3, "total": 10, "success": true, "return": 1.5}'
    )
    assert event is not None
    assert event["step"] == "episode_complete"
    assert event["step_index"] == 3
    assert event["step_total"] == 10
    assert collector.episodes[0]["success"] is True


def test_collector_records_result_line() -> None:
    collector = EvalProtocolCollector()
    event = collector.parse('ODYSSEY_RESULT {"success_rate": 0.4}')
    assert event is not None
    assert collector.result == {"success_rate": 0.4}


def test_collector_skips_malformed_protocol_line() -> None:
    collector = EvalProtocolCollector()
    assert collector.parse("ODYSSEY_EPISODE {not json") is None
    assert collector.episodes == []


def test_collector_ignores_boot_noise() -> None:
    collector = EvalProtocolCollector()
    assert collector.parse("[INFO] Simulation App Startup Complete") is None


def test_collector_records_reasoning_and_emits_progress() -> None:
    collector = EvalProtocolCollector()
    event = collector.parse(
        'ODYSSEY_REASONING {"episode": 2, "instruction": "stack the cubes", '
        '"reasoning": "locate, grasp, place"}'
    )
    assert event is not None
    assert event["step"] == "reasoning_received"
    assert event["step_index"] == 2
    assert collector.reasoning[0]["reasoning"] == "locate, grasp, place"


def test_collector_skips_malformed_reasoning_line() -> None:
    collector = EvalProtocolCollector()
    assert collector.parse("ODYSSEY_REASONING {not json") is None
    assert collector.reasoning == []


def test_collector_distinguishes_result_from_reasoning() -> None:
    # ODYSSEY_RESULT and ODYSSEY_REASONING share the ODYSSEY_RE prefix; each
    # must land in its own bucket and not be mistaken for the other.
    collector = EvalProtocolCollector()
    collector.parse('ODYSSEY_RESULT {"success_rate": 0.5}')
    collector.parse(
        'ODYSSEY_REASONING {"episode": 1, "instruction": "x", "reasoning": "y"}'
    )
    assert collector.result == {"success_rate": 0.5}
    assert len(collector.reasoning) == 1
    assert collector.reasoning[0]["episode"] == 1


def test_prefixes_are_mutually_exclusive() -> None:
    # Lock in the structural invariant the RESULT-before-REASONING parse order
    # relies on: no ODYSSEY_* prefix may be a prefix of another (else a line
    # could match the wrong bucket depending on check order). Guards a future
    # rename/reorder from silently breaking disambiguation.
    prefixes = [_EPISODE_PREFIX, _RESULT_PREFIX, _REASONING_PREFIX]
    for a in prefixes:
        for b in prefixes:
            if a is not b:
                assert not a.startswith(b), f"{a!r} starts with {b!r}"


# ---------------------------------------------------------------------------
# Summary scoring
# ---------------------------------------------------------------------------

def _collector_with_episodes(*successes: bool) -> EvalProtocolCollector:
    collector = EvalProtocolCollector()
    total = len(successes)
    for i, success in enumerate(successes, start=1):
        collector.parse(
            f'ODYSSEY_EPISODE {{"index": {i}, "total": {total}, '
            f'"success": {str(success).lower()}, "return": {1.0 if success else 0.0}}}'
        )
    return collector


def test_summary_computed_from_episodes(tmp_path: Path) -> None:
    collector = _collector_with_episodes(True, True, False, False)
    summary = summarize(
        collector=collector,
        spec=_eval_task(),
        checkpoint=tmp_path / "ckpt",
        eval_script="/opt/eval.py",
    )
    assert summary["num_episodes"] == 4
    assert summary["success_rate"] == 0.5
    assert summary["passed"] is True
    assert summary["metrics"]["successes"] == 2
    assert summary["metrics"]["benchmark"] == "Isaac-Lift-Cube-Franka-v0"


def test_summary_prefers_explicit_result(tmp_path: Path) -> None:
    collector = _collector_with_episodes(True, False)
    collector.parse(
        'ODYSSEY_RESULT {"success_rate": 0.9, "performance_score": 0.8, '
        '"metrics": {"sim_steps": 1234}}'
    )
    summary = summarize(
        collector=collector,
        spec=_eval_task(),
        checkpoint=tmp_path,
        eval_script="/opt/eval.py",
    )
    assert summary["success_rate"] == 0.9
    assert summary["performance_score"] == 0.8
    assert summary["letter_grade"] == "A"
    assert summary["metrics"]["sim_steps"] == 1234


def test_summary_includes_reasoning_when_present(tmp_path: Path) -> None:
    collector = _collector_with_episodes(True, False)
    collector.parse(
        'ODYSSEY_REASONING {"episode": 1, "instruction": "lift the cube", '
        '"reasoning": "locate the cube, grasp it, lift"}'
    )
    summary = summarize(
        collector=collector,
        spec=_eval_task(),
        checkpoint=tmp_path,
        eval_script="/opt/eval.py",
    )
    traces = summary["metrics"]["reasoning"]
    assert len(traces) == 1
    assert traces[0]["episode"] == 1
    assert "grasp" in traces[0]["reasoning"]


def test_summary_omits_reasoning_when_absent(tmp_path: Path) -> None:
    summary = summarize(
        collector=_collector_with_episodes(True, True),
        spec=_eval_task(),
        checkpoint=tmp_path,
        eval_script="/opt/eval.py",
    )
    assert "reasoning" not in summary["metrics"]


def test_summary_without_protocol_output_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="protocol"):
        summarize(
            collector=EvalProtocolCollector(),
            spec=_eval_task(),
            checkpoint=tmp_path,
            eval_script="/opt/eval.py",
        )


# ---------------------------------------------------------------------------
# End-to-end through the real subprocess machinery
# ---------------------------------------------------------------------------

class _NullPublisher(EventPublisher):
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        pass


FAKE_EVAL_SCRIPT = """\
import json, sys
args = sys.argv[1:]
num_episodes = int(args[args.index("--num_episodes") + 1])
for i in range(1, num_episodes + 1):
    print("ODYSSEY_REASONING " + json.dumps(
        {"episode": i, "instruction": "lift the cube", "reasoning": "grasp then lift"}
    ))
    print("ODYSSEY_EPISODE " + json.dumps(
        {"index": i, "total": num_episodes, "success": i % 2 == 1, "return": 1.0}
    ))
print("ODYSSEY_RESULT " + json.dumps({"success_rate": 0.5}))
"""


def _context_for(spec_task: EvaluationTask, tmp_path: Path) -> TaskContext:
    mission = Mission(
        metadata=MissionMetadata(name="msn-isaac"),
        objective="objective",
        acceptance_criteria="acceptance",
        robot=RobotSpec(
            embodiment="franka_panda",
            agents=[
                AgentSpec(
                    id="pilot",
                    role=AgentRole.PILOT,
                    model=HFModelRef(base="nvidia/GR00T-N1.7-3B"),
                ),
            ],
        ),
        tasks=[
            TrainingTask(
                name="train",
                training_type=TrainingType.DEMONSTRATION,
                agent_id="pilot",
            ),
            spec_task,
        ],
    )
    run = MissionRun.from_spec(mission)
    # Mark the training task complete with a checkpoint so the eval's
    # loadout walk finds one.
    train_run = run.tasks[0]
    train_run.status = TaskStatus.COMPLETED
    train_run.result_summary = {"checkpoint_path": str(tmp_path / "ckpt")}
    eval_run = run.tasks[1]
    return TaskContext(
        task=eval_run,
        mission=run,
        publisher=_NullPublisher(),
        output_dir=tmp_path / "out",
    )


def test_runner_end_to_end_with_fake_script(tmp_path: Path) -> None:
    script = tmp_path / "fake_eval.py"
    script.write_text(FAKE_EVAL_SCRIPT)
    task = _eval_task(num_episodes=4, config={"eval_script": str(script)})
    context = _context_for(task, tmp_path)

    runner = IsaacLabRunner()
    summary = asyncio.run(runner.run(context))

    assert summary["num_episodes"] == 4
    assert summary["success_rate"] == 0.5
    assert summary["passed"] is True
    assert summary["metrics"]["successes"] == 2
    # The ODYSSEY_REASONING lines flowed through the real subprocess +
    # collector and landed in the summary, one trace per episode.
    traces = summary["metrics"]["reasoning"]
    assert len(traces) == 4
    assert traces[0] == {
        "episode": 1, "instruction": "lift the cube", "reasoning": "grasp then lift"
    }
