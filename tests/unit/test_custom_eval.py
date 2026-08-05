"""Tests for the custom evaluation runner (``evaluation_type: custom``).

No GPU and no real policy here — the runner's testable pieces are the launch
contract (argv, script/interpreter resolution), the out-json metrics parsing,
and summary scoring for both the graded (``success_rate``) and metric-only
paths. One integration-ish test runs a fake eval script (plain python that
writes an out-json) through the real subprocess machinery end-to-end.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from odyssey.engine import TaskStatus
from odyssey.engine.records import MissionRun
from odyssey.runners.base import TaskContext
from odyssey.runners.evals.custom import (
    CustomEvalRunner,
    build_custom_argv,
    read_metrics,
    resolve_eval_script,
    resolve_interpreter,
    summarize,
)
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
        "name": "eval-custom",
        "evaluation_type": EvaluationType.CUSTOM,
        "benchmark_name": "openloop-gt",
        "num_episodes": 50,
    }
    fields.update(overrides)
    return EvaluationTask(**fields)


# ---------------------------------------------------------------------------
# Launch contract
# ---------------------------------------------------------------------------

def test_argv_contains_owned_flags(tmp_path: Path) -> None:
    argv = build_custom_argv(
        checkpoint=tmp_path / "ckpt", out_json=tmp_path / "m.json", config={}
    )
    assert argv[argv.index("--checkpoint") + 1] == str(tmp_path / "ckpt")
    assert argv[argv.index("--out-json") + 1] == str(tmp_path / "m.json")


def test_argv_passthrough_keeps_snake_case(tmp_path: Path) -> None:
    argv = build_custom_argv(
        checkpoint=tmp_path,
        out_json=tmp_path / "m.json",
        config={"replay_dataset": "/data/ur10e", "max_frames": 200},
    )
    assert argv[argv.index("--replay_dataset") + 1] == "/data/ur10e"
    assert argv[argv.index("--max_frames") + 1] == "200"


def test_argv_omits_isaac_specific_flags(tmp_path: Path) -> None:
    # Custom is decoupled from Isaac Lab: no --task / --headless, and
    # num_episodes is not auto-forwarded (the script asks for what it needs).
    argv = build_custom_argv(checkpoint=tmp_path, out_json=tmp_path / "m.json", config={})
    assert "--task" not in argv
    assert "--headless" not in argv
    assert "--num_episodes" not in argv


def test_argv_excludes_handled_keys(tmp_path: Path) -> None:
    argv = build_custom_argv(
        checkpoint=tmp_path,
        out_json=tmp_path / "m.json",
        config={
            "eval_script": "/x.py",
            "eval_python": "/venv/bin/python",
            "runner": "custom",
            "checkpoint": "/should/not/leak",
            "out_json": "/nope",
        },
    )
    for handled in ("--eval_script", "--eval_python", "--runner", "--out_json"):
        assert handled not in argv
    # checkpoint is resolved by the runner; the config key must not double up
    # as a passthrough flag pointing at a different path.
    assert argv.count("--checkpoint") == 1
    assert "/should/not/leak" not in argv


def test_eval_script_from_config() -> None:
    assert resolve_eval_script({"eval_script": "/opt/eval.py"}) == "/opt/eval.py"


def test_eval_script_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ODYSSEY_EVAL_SCRIPT", "/env/eval.py")
    assert resolve_eval_script({}) == "/env/eval.py"


def test_eval_script_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ODYSSEY_EVAL_SCRIPT", raising=False)
    with pytest.raises(RuntimeError, match="eval script"):
        resolve_eval_script({})


def test_interpreter_defaults_to_sys_executable() -> None:
    assert resolve_interpreter({}) == [sys.executable]


def test_interpreter_from_config() -> None:
    assert resolve_interpreter({"eval_python": "/venv/bin/python"}) == ["/venv/bin/python"]


# ---------------------------------------------------------------------------
# Metrics parsing
# ---------------------------------------------------------------------------

def test_read_metrics_loads_object(tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    out.write_text(json.dumps({"success_rate": 0.4, "metrics": {"mae": 0.1}}))
    assert read_metrics(out) == {"success_rate": 0.4, "metrics": {"mae": 0.1}}


def test_read_metrics_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="wrote no metrics file"):
        read_metrics(tmp_path / "absent.json")


def test_read_metrics_invalid_json_raises(tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    out.write_text("{not json")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        read_metrics(out)


def test_read_metrics_non_object_raises(tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    out.write_text("[1, 2, 3]")
    with pytest.raises(RuntimeError, match="must contain a JSON object"):
        read_metrics(out)


# ---------------------------------------------------------------------------
# Summary scoring
# ---------------------------------------------------------------------------

def test_summary_graded_when_success_rate_present(tmp_path: Path) -> None:
    summary = summarize(
        payload={
            "success_rate": 0.9,
            "performance_score": 0.8,
            "num_episodes": 20,
            "metrics": {"lift": 18, "seat": 16},
        },
        spec=_eval_task(),
        checkpoint=tmp_path / "ckpt",
        eval_script="/opt/eval.py",
    )
    assert summary["success_rate"] == 0.9
    assert summary["performance_score"] == 0.8
    assert summary["letter_grade"] == "A"
    assert summary["passed"] is True
    assert summary["num_episodes"] == 20
    assert summary["metrics"]["lift"] == 18
    assert summary["metrics"]["eval_script"] == "/opt/eval.py"


def test_summary_performance_defaults_to_success_rate(tmp_path: Path) -> None:
    summary = summarize(
        payload={"success_rate": 0.5},
        spec=_eval_task(),
        checkpoint=tmp_path,
        eval_script="/opt/eval.py",
    )
    assert summary["performance_score"] == 0.5


def test_summary_metric_only_has_no_pass_fail(tmp_path: Path) -> None:
    # The open-loop GT case: per-joint MAE, no success_rate. The summary must
    # carry the metrics without fabricating a 0.0 / grade-F pass/fail.
    summary = summarize(
        payload={"metrics": {"joint_mae": [0.01, 0.02], "gripper_agreement": 0.97}},
        spec=_eval_task(),
        checkpoint=tmp_path / "ckpt",
        eval_script="/opt/eval_openloop_gt.py",
    )
    assert "letter_grade" not in summary
    assert "passed" not in summary
    assert "success_rate" not in summary
    assert summary["metrics"]["gripper_agreement"] == 0.97
    assert summary["metrics"]["benchmark"] == "openloop-gt"
    assert summary["num_episodes"] == 50  # falls back to spec.num_episodes


def test_summary_folds_unknown_top_level_keys_into_metrics(tmp_path: Path) -> None:
    summary = summarize(
        payload={"joint_mae": 0.03, "notes": "replay of 200 frames"},
        spec=_eval_task(),
        checkpoint=tmp_path,
        eval_script="/opt/eval.py",
    )
    assert summary["metrics"]["joint_mae"] == 0.03
    assert summary["metrics"]["notes"] == "replay of 200 frames"


# ---------------------------------------------------------------------------
# End-to-end through the real subprocess machinery
# ---------------------------------------------------------------------------

class _NullPublisher(EventPublisher):
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        pass


# A fake eval script: reads --checkpoint / --out-json, writes a metrics object.
# Mirrors what scripts/eval_openloop_gt.py does, minus the model.
FAKE_EVAL_SCRIPT = """\
import argparse, json
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--out-json", required=True)
p.add_argument("--replay_dataset", required=True)
args = p.parse_args()
with open(args.out_json, "w") as f:
    json.dump(
        {"metrics": {"joint_mae": 0.012, "gripper_agreement": 0.98,
                     "checkpoint_seen": args.checkpoint,
                     "replay": args.replay_dataset}},
        f,
    )
"""

FAKE_EVAL_SCRIPT_GRADED = """\
import argparse, json
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--out-json", required=True)
args = p.parse_args()
with open(args.out_json, "w") as f:
    json.dump({"success_rate": 0.75, "num_episodes": 8}, f)
"""

FAKE_EVAL_SCRIPT_NO_OUTPUT = """\
import argparse
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--out-json", required=True)
p.parse_args()  # exits 0 but writes nothing
"""


def _context_for(spec_task: EvaluationTask, tmp_path: Path) -> TaskContext:
    mission = Mission(
        metadata=MissionMetadata(name="msn-custom"),
        objective="objective",
        acceptance_criteria="acceptance",
        robot=RobotSpec(
            embodiment="ur10e",
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


def test_runner_end_to_end_metric_only(tmp_path: Path) -> None:
    script = tmp_path / "fake_eval.py"
    script.write_text(FAKE_EVAL_SCRIPT)
    task = _eval_task(
        config={"eval_script": str(script), "replay_dataset": "/data/ur10e"}
    )
    context = _context_for(task, tmp_path)

    summary = asyncio.run(CustomEvalRunner().run(context))

    # Metric-only: metrics captured from the out-json the script wrote, no grade.
    assert "letter_grade" not in summary
    assert summary["metrics"]["joint_mae"] == 0.012
    assert summary["metrics"]["gripper_agreement"] == 0.98
    assert summary["metrics"]["checkpoint_seen"] == str(tmp_path / "ckpt")
    assert summary["metrics"]["replay"] == "/data/ur10e"


def test_runner_end_to_end_graded(tmp_path: Path) -> None:
    script = tmp_path / "fake_eval_graded.py"
    script.write_text(FAKE_EVAL_SCRIPT_GRADED)
    task = _eval_task(config={"eval_script": str(script)})
    context = _context_for(task, tmp_path)

    summary = asyncio.run(CustomEvalRunner().run(context))

    assert summary["success_rate"] == 0.75
    assert summary["num_episodes"] == 8
    assert summary["letter_grade"] == "C"
    assert summary["passed"] is True


def test_runner_raises_when_script_writes_no_metrics(tmp_path: Path) -> None:
    script = tmp_path / "silent_eval.py"
    script.write_text(FAKE_EVAL_SCRIPT_NO_OUTPUT)
    task = _eval_task(config={"eval_script": str(script)})
    context = _context_for(task, tmp_path)

    with pytest.raises(RuntimeError, match="wrote no metrics file"):
        asyncio.run(CustomEvalRunner().run(context))


# ---------------------------------------------------------------------------
# Registry dispatch — the reason this runner exists
# ---------------------------------------------------------------------------

def test_custom_type_dispatches_to_custom_runner_not_mock() -> None:
    # Acceptance criterion: `evaluation_type: custom` must resolve to the real
    # runner, not fall through to the wildcard CPUMockRunner. Registration order
    # mirrors _build_runners (custom before mock).
    from odyssey.runners.cpu_mock import CPUMockRunner
    from odyssey.runners.registry import RunnerRegistry

    registry = RunnerRegistry()
    registry.register(CustomEvalRunner())
    registry.register(CPUMockRunner())

    selected = registry.select(_eval_task())
    assert isinstance(selected, CustomEvalRunner)
    assert selected.name == "custom"
