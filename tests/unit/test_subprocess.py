"""Tests for runners/subprocess.py.

Drives real Python subprocesses using ``python -c "..."`` so we exercise
the full launch / stream / cancel path without requiring any external
training framework.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from odyssey.engine.records import MissionRun
from odyssey.runners.base import TaskContext
from odyssey.runners.subprocess import (
    TrainingProcessSpec,
    output_path,
    run_training_subprocess,
)
from odyssey.spec import (
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


class _CapturingPublisher(EventPublisher):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))


def _spec() -> Mission:
    return Mission(
        metadata=MissionMetadata(name="sub-tests"),
        objective="o",
        acceptance_criteria="a",
        robot=RobotSpec(embodiment="franka_panda"),
        tasks=[
            TrainingTask(
                name="train",
                training_type=TrainingType.DEMONSTRATION,
                model=HFModelRef(base="openvla/openvla-7b"),
                target_agent_id="pilot",
            ),
            EvaluationTask(
                name="eval",
                evaluation_type=EvaluationType.ROBOSUITE,
                benchmark_name="Lift",
                model=HFModelRef(base="openvla/openvla-7b"),
                target_agent_id="pilot",
            ),
        ],
    )


def _ctx(tmp_path: Path) -> tuple[TaskContext, _CapturingPublisher]:
    spec = _spec()
    mission = MissionRun.from_spec(spec)
    publisher = _CapturingPublisher()
    ctx = TaskContext(
        task=mission.tasks[0],
        mission=mission,
        publisher=publisher,
        output_dir=tmp_path,
    )
    return ctx, publisher


# ---------------------------------------------------------------------------
# TrainingProcessSpec validation
# ---------------------------------------------------------------------------

def test_spec_requires_exactly_one_entry() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        TrainingProcessSpec()
    with pytest.raises(ValueError, match="exactly one"):
        TrainingProcessSpec(entry_module="m", script_path="/x")


# ---------------------------------------------------------------------------
# output_path helper
# ---------------------------------------------------------------------------

def test_output_path_returns_under_output_dir(tmp_path: Path) -> None:
    ctx, _ = _ctx(tmp_path)
    assert output_path(ctx, "sub", "file.bin") == tmp_path / "sub" / "file.bin"


def test_output_path_requires_output_dir() -> None:
    spec = _spec()
    mission = MissionRun.from_spec(spec)
    ctx = TaskContext(
        task=mission.tasks[0],
        mission=mission,
        publisher=_CapturingPublisher(),
        output_dir=None,
    )
    with pytest.raises(RuntimeError, match="output_dir is None"):
        output_path(ctx)


# ---------------------------------------------------------------------------
# Subprocess execution — happy path
# ---------------------------------------------------------------------------

async def test_clean_exit_returns_zero(tmp_path: Path) -> None:
    ctx, _ = _ctx(tmp_path)
    script_file = tmp_path / "tiny.py"
    script_file.write_text("print('hello')\n")
    spec = TrainingProcessSpec(script_path=str(script_file))
    rc = await run_training_subprocess(ctx, spec)
    assert rc == 0


async def test_nonzero_exit_propagates(tmp_path: Path) -> None:
    ctx, _ = _ctx(tmp_path)
    script_file = tmp_path / "fail.py"
    script_file.write_text("import sys; sys.exit(7)\n")
    spec = TrainingProcessSpec(script_path=str(script_file))
    rc = await run_training_subprocess(ctx, spec)
    assert rc == 7


# ---------------------------------------------------------------------------
# stdout streaming — generic step regex fallback
# ---------------------------------------------------------------------------

async def test_generic_step_regex_emits_progress(tmp_path: Path) -> None:
    ctx, pub = _ctx(tmp_path)
    script_file = tmp_path / "steps.py"
    script_file.write_text(
        "import sys\n"
        "for s in range(1, 4):\n"
        "    print(f'Step {s}: loss=0.{s}', flush=True)\n"
    )
    spec = TrainingProcessSpec(script_path=str(script_file))
    rc = await run_training_subprocess(ctx, spec)
    assert rc == 0

    step_events = [
        e for _, e in pub.events
        if e.get("step") == "training_step"
    ]
    assert [e["step_index"] for e in step_events] == [1, 2, 3]


async def test_line_parser_overrides_fallback(tmp_path: Path) -> None:
    ctx, pub = _ctx(tmp_path)
    script_file = tmp_path / "lines.py"
    script_file.write_text(
        "print('epoch 1', flush=True)\n"
        "print('Step 5: loss=0.1', flush=True)\n"
    )

    def parser(line: str) -> dict[str, Any] | None:
        if line.startswith("epoch"):
            return {"stage": "executing", "step": "epoch", "step_label": line}
        return None  # let fallback handle "Step N"

    spec = TrainingProcessSpec(
        script_path=str(script_file), line_parser=parser
    )
    rc = await run_training_subprocess(ctx, spec)
    assert rc == 0

    stages = [e.get("step") for _, e in pub.events]
    # Parser fired for "epoch 1", fallback fired for "Step 5".
    assert "epoch" in stages
    assert "training_step" in stages


async def test_subprocess_start_event_fires(tmp_path: Path) -> None:
    ctx, pub = _ctx(tmp_path)
    script_file = tmp_path / "quick.py"
    script_file.write_text("pass\n")
    spec = TrainingProcessSpec(script_path=str(script_file))
    await run_training_subprocess(ctx, spec)
    assert any(
        e.get("step") == "subprocess_start" for _, e in pub.events
    )


# ---------------------------------------------------------------------------
# Cancellation — runs an infinite-loop subprocess and signals cancel.
# Skip on platforms without os.setsid (Windows); we use POSIX process
# groups for the SIGTERM behavior.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not hasattr(__import__("os"), "setsid"),
    reason="cancellation test requires POSIX process groups",
)
async def test_cancel_terminates_subprocess(tmp_path: Path) -> None:
    ctx, _ = _ctx(tmp_path)
    script_file = tmp_path / "loop.py"
    script_file.write_text("import time\nwhile True:\n    time.sleep(0.1)\n")
    spec = TrainingProcessSpec(
        script_path=str(script_file),
        sigterm_grace_seconds=2.0,
    )

    async def cancel_after(delay: float) -> None:
        await asyncio.sleep(delay)
        ctx.request_cancel()

    cancel_task = asyncio.create_task(cancel_after(0.2))
    try:
        rc = await asyncio.wait_for(
            run_training_subprocess(ctx, spec), timeout=5.0
        )
    finally:
        cancel_task.cancel()
    # SIGTERM produces -15 in asyncio's returncode convention; SIGKILL is -9.
    # Both are acceptable terminations for this test.
    assert rc < 0
