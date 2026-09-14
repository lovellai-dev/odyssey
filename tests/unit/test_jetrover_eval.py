"""Tests for the shipped JetRover real-arm eval script (examples/quickstart-jetrover).

No hardware, no GPU, no policy server: the mock arm + mock policy backends
exercise the rollout loop, the operator/timeout scoring, and the out-json
contract; one end-to-end test drives the shipped script through the real
``CustomEvalRunner`` subprocess machinery. The script must import with stdlib
only (numpy/zmq/rclpy are lazy inside the real backends), which is what lets
these tests run under the ``dev`` extra alone.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from odyssey.engine import TaskStatus
from odyssey.engine.records import MissionRun
from odyssey.runners.base import TaskContext
from odyssey.runners.evals.custom import CustomEvalRunner
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

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "examples" / "quickstart-jetrover" / "eval_jetrover.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("eval_jetrover", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def jetrover() -> Any:
    return _load_module()


# ---------------------------------------------------------------------------
# Import + argv contract
# ---------------------------------------------------------------------------

def test_module_imports_with_stdlib_only(jetrover: Any) -> None:
    # Loading the module must not pull in the heavy backend deps — they are
    # imported lazily inside connect()/__init__ of the real backends.
    assert callable(jetrover.main)
    # Vendor docs: servos 1-5 position the arm, ID 10 is the gripper — the
    # marketed "6DOF" counts the gripper, so the default arm chain is 5.
    assert jetrover.DEFAULT_ARM_DOF == 5
    assert jetrover.MockArm(5).action_dim == 6  # 5 joints + gripper


def test_parse_args_accepts_runner_contract_flags(jetrover: Any) -> None:
    # --checkpoint/--out-json are runner-owned; everything else arrives as the
    # verbatim snake_case passthrough of build_custom_argv.
    args = jetrover.parse_args(
        [
            "--checkpoint", "/ckpt",
            "--out-json", "/tmp/m.json",
            "--arm_backend", "mock",
            "--policy_backend", "mock",
            "--policy_host", "10.0.0.2",
            "--policy_port", "5561",
            "--arm_dof", "5",
            "--num_episodes", "3",
            "--max_steps", "50",
            "--task_description", "pick up the vial",
            "--scorer", "none",
            "--auto_timeout", "2.5",
        ]
    )
    assert args.checkpoint == "/ckpt"
    assert args.out_json == "/tmp/m.json"
    assert args.policy_port == 5561
    assert args.arm_dof == 5
    assert args.task_description == "pick up the vial"


def test_unknown_arm_backend_rejected(jetrover: Any) -> None:
    with pytest.raises(SystemExit):
        jetrover.parse_args(
            ["--checkpoint", "/c", "--out-json", "/o", "--arm_backend", "bogus"]
        )


def test_unknown_policy_backend_rejected(jetrover: Any) -> None:
    with pytest.raises(SystemExit):
        jetrover.parse_args(
            ["--checkpoint", "/c", "--out-json", "/o", "--policy_backend", "bogus"]
        )


# ---------------------------------------------------------------------------
# Rollout loop + scoring
# ---------------------------------------------------------------------------

def _run_main(
    jetrover: Any,
    tmp_path: Path,
    *extra: str,
    input_fn: Any = None,
) -> dict[str, Any]:
    out_json = tmp_path / "metrics.json"
    argv = [
        "--checkpoint", "/fake/ckpt",
        "--out-json", str(out_json),
        "--arm_backend", "mock",
        "--policy_backend", "mock",
        "--num_episodes", "2",
        "--max_steps", "6",
        *extra,
    ]
    kwargs = {} if input_fn is None else {"input_fn": input_fn}
    assert jetrover.main(argv, **kwargs) == 0
    return json.loads(out_json.read_text())


def test_mock_run_writes_metric_only_json(jetrover: Any, tmp_path: Path) -> None:
    payload = _run_main(jetrover, tmp_path, "--scorer", "none")
    assert "success_rate" not in payload  # metric-only: no fabricated grade
    assert payload["num_episodes"] == 2
    episodes = payload["metrics"]["episodes"]
    assert len(episodes) == 2
    assert all(e["steps"] == 6 for e in episodes)
    assert payload["metrics"]["arm_backend"] == "mock"
    assert payload["metrics"]["checkpoint"] == "/fake/ckpt"


def test_operator_scorer_counts_stdin_confirmations(
    jetrover: Any, tmp_path: Path
) -> None:
    answers = iter(["y", "n", "yes"])
    payload = _run_main(
        jetrover,
        tmp_path,
        "--scorer", "operator",
        "--num_episodes", "3",
        input_fn=lambda prompt: next(answers),
    )
    assert payload["success_rate"] == pytest.approx(2 / 3)
    successes = [e["success"] for e in payload["metrics"]["episodes"]]
    assert successes == [True, False, True]


def test_auto_timeout_scores_failure(jetrover: Any, tmp_path: Path) -> None:
    def slow_operator(prompt: str) -> str:
        time.sleep(0.5)
        return "y"

    payload = _run_main(
        jetrover,
        tmp_path,
        "--scorer", "operator",
        "--num_episodes", "1",
        "--auto_timeout", "0.05",
        input_fn=slow_operator,
    )
    assert payload["success_rate"] == 0.0
    assert payload["metrics"]["episodes"][0]["success"] is False


def test_clamp_action_rate_limits_joints(jetrover: Any) -> None:
    # 5 joints + gripper (the JetRover default)
    current = [0.0] * 6
    action = [1.0, -1.0, 0.05, 0.0, 0.0, 0.8]
    clamped = jetrover.clamp_action(action, current, max_delta=0.1)
    assert clamped[0] == pytest.approx(0.1)    # capped upward
    assert clamped[1] == pytest.approx(-0.1)   # capped downward
    assert clamped[2] == pytest.approx(0.05)   # within limit: untouched
    assert clamped[5] == pytest.approx(0.8)    # gripper passes through


# ---------------------------------------------------------------------------
# End-to-end through the real CustomEvalRunner
# ---------------------------------------------------------------------------

class _NullPublisher(EventPublisher):
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        pass


def _context_for(spec_task: EvaluationTask, tmp_path: Path) -> TaskContext:
    mission = Mission(
        metadata=MissionMetadata(name="msn-jetrover"),
        objective="objective",
        acceptance_criteria="acceptance",
        robot=RobotSpec(
            embodiment="jetrover",
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
    return TaskContext(
        task=run.tasks[1],
        mission=run,
        publisher=_NullPublisher(),
        output_dir=tmp_path / "out",
    )


def test_e2e_through_custom_eval_runner(tmp_path: Path) -> None:
    # The shipped script, launched by the real runner subprocess machinery with
    # both backends mocked: the metric-only summary must surface the episodes.
    task = EvaluationTask(
        name="eval-jetrover-real-arm",
        evaluation_type=EvaluationType.CUSTOM,
        benchmark_name="jetrover-real-arm",
        num_episodes=2,
        config={
            "eval_script": str(SCRIPT),
            "eval_python": sys.executable,
            "arm_backend": "mock",
            "policy_backend": "mock",
            "num_episodes": 2,
            "max_steps": 6,
            "scorer": "none",
        },
    )
    context = _context_for(task, tmp_path)

    summary = asyncio.run(CustomEvalRunner().run(context))

    assert "letter_grade" not in summary  # scorer=none → metric-only
    assert summary["num_episodes"] == 2
    assert len(summary["metrics"]["episodes"]) == 2
    assert summary["metrics"]["arm_backend"] == "mock"
