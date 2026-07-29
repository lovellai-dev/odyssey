"""Wiring tests for the π0.5 LIBERO eval path (issue #74).

These pin the *cabling* that makes ``evaluation_type: libero`` + ``pilot: pi05``
reach the right code — WITHOUT a GPU, an openpi server, LIBERO, or a real
simulator. Every heavy edge (the subprocess recipe, the scorer) is mocked the
same way the existing eval tests mock the runner/sim (see
``test_robosuite_runner.py`` / ``test_gr00t_libero_eval.py``):

  * the RunnerRegistry routes a LIBERO eval task to ``LiberoRunner`` (π0.5 runs
    under ``evaluation_type: libero``, same as GR00T);
  * ``LiberoRunner`` dispatches ``pilot: pi05`` to its π0.5 subprocess path and
    nowhere else;
  * the model-ref (``config.checkpoint`` HF id / local path, or the trained
    PILOT checkpoint) resolves via the shared ``resolve_eval_checkpoint``;
  * ``build_pi05_libero_argv`` forwards the launch contract + config passthrough
    and drops the keys the runner consumes itself (the GR00T-LIBERO bridge
    contract, reused verbatim);
  * ``_run_pi05_pilot`` launches the ``pi05_libero_eval.py`` recipe through the
    shared subprocess helper and scores it through the shared ``summarize``.

All tests are named ``test_pi05_eval_*`` (the ``-k pi05_eval`` gate).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from odyssey.engine.lifecycle import TaskStatus
from odyssey.engine.records import MissionRun, TaskRun
from odyssey.runners.base import TaskContext
from odyssey.runners.evals._common import resolve_eval_checkpoint
from odyssey.runners.evals.isaac_lab import EvalProtocolCollector, summarize
from odyssey.runners.evals.libero import LiberoRunner, build_pi05_libero_argv
from odyssey.runners.evals.robosuite import RobosuiteRunner
from odyssey.runners.registry import RunnerRegistry
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

# A published π0.5-LIBERO checkpoint stands in for the model-ref throughout.
PI05_MODEL_REF = "lerobot/pi05_libero"


# ---------------------------------------------------------------------------
# Fixtures / builders — an eval-only mission + a TaskContext, no engine.
# ---------------------------------------------------------------------------

class _NullPublisher(EventPublisher):
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        return


def _eval_spec(**config: Any) -> EvaluationTask:
    """A LIBERO EvaluationTask spec whose ``config`` carries the pilot wiring."""
    return EvaluationTask(
        name="pi05-libero-eval",
        evaluation_type=EvaluationType.LIBERO,
        benchmark_name="libero_object",
        num_episodes=4,
        config=dict(config),
    )


def _mission(
    *,
    pilot: str = "pi05",
    checkpoint: str | None = PI05_MODEL_REF,
    trained_checkpoint: str | None = None,
    extra_config: dict[str, Any] | None = None,
) -> MissionRun:
    """Eval-only mission: one PILOT agent + a LIBERO eval task.

    ``checkpoint`` sets ``config.checkpoint`` (the eval-only model-ref path).
    ``trained_checkpoint`` instead simulates a completed training task that
    produced the PILOT's checkpoint (the fall-back resolution branch).
    """
    config: dict[str, Any] = {"pilot": pilot}
    if checkpoint is not None:
        config["checkpoint"] = checkpoint
    if extra_config:
        config.update(extra_config)

    spec = Mission(
        metadata=MissionMetadata(name="msn-pi05"),
        objective="o",
        acceptance_criteria="a",
        robot=RobotSpec(
            embodiment="franka_panda",
            agents=[
                AgentSpec(
                    id="pilot",
                    role=AgentRole.PILOT,
                    model=HFModelRef(base="lerobot/pi05_base"),
                ),
            ],
        ),
        tasks=[
            TrainingTask(
                name="train",
                training_type=TrainingType.DEMONSTRATION,
                agent_id="pilot",
            ),
            EvaluationTask(
                name="pi05-libero-eval",
                evaluation_type=EvaluationType.LIBERO,
                benchmark_name="libero_object",
                num_episodes=4,
                config=config,
            ),
        ],
    )
    mission = MissionRun.from_spec(spec)
    if trained_checkpoint is not None:
        mission.tasks[0].status = TaskStatus.COMPLETED
        mission.tasks[0].result_summary = {
            "checkpoint_path": trained_checkpoint,
            "agent_id": "pilot",
        }
    return mission


def _eval_task(mission: MissionRun) -> TaskRun:
    return next(t for t in mission.tasks if isinstance(t.spec, EvaluationTask))


def _ctx(mission: MissionRun, tmp_path: Path | None = None) -> TaskContext:
    return TaskContext(
        task=_eval_task(mission),
        mission=mission,
        publisher=_NullPublisher(),
        output_dir=tmp_path,
    )


# ---------------------------------------------------------------------------
# 1. RunnerRegistry — evaluation_type routes a π0.5 eval to LiberoRunner.
# ---------------------------------------------------------------------------

def test_pi05_eval_registry_routes_libero_task_to_libero_runner() -> None:
    registry = RunnerRegistry()
    registry.register(RobosuiteRunner())
    registry.register(LiberoRunner())

    # π0.5 rides on evaluation_type: libero — the pilot lives in config, so the
    # registry key is the same LIBERO route GR00T/OpenVLA take.
    spec = _eval_spec(pilot="pi05", checkpoint=PI05_MODEL_REF)
    runner = registry.select(spec)
    assert isinstance(runner, LiberoRunner)
    assert runner.name == "libero"


def test_pi05_eval_registry_keeps_libero_and_robosuite_distinct() -> None:
    registry = RunnerRegistry()
    registry.register(RobosuiteRunner())
    registry.register(LiberoRunner())

    libero_spec = _eval_spec(pilot="pi05")
    robosuite_spec = EvaluationTask(
        name="rs",
        evaluation_type=EvaluationType.ROBOSUITE,
        benchmark_name="Lift",
        num_episodes=1,
    )
    assert registry.select(libero_spec).name == "libero"
    assert registry.select(robosuite_spec).name == "robosuite"


# ---------------------------------------------------------------------------
# 2. LiberoRunner — `pilot: pi05` dispatches to the π0.5 path, and only that.
# ---------------------------------------------------------------------------

async def test_pi05_eval_pilot_dispatches_to_pi05_path(monkeypatch, tmp_path) -> None:
    runner = LiberoRunner()
    calls: list[str] = []

    async def _fake_pi05(context, spec):
        calls.append("pi05")
        return {"dispatched": "pi05"}

    async def _fake_gr00t(context, spec):
        calls.append("gr00t")
        return {"dispatched": "gr00t"}

    monkeypatch.setattr(runner, "_run_pi05_pilot", _fake_pi05)
    monkeypatch.setattr(runner, "_run_gr00t_pilot", _fake_gr00t)

    result = await runner.run(_ctx(_mission(pilot="pi05"), tmp_path))

    assert result == {"dispatched": "pi05"}
    assert calls == ["pi05"]  # gr00t path untouched, in-process path never reached


async def test_pi05_eval_gr00t_pilot_not_hijacked_by_pi05(monkeypatch, tmp_path) -> None:
    # Reverse guard: pilot: gr00t still lands on the GR00T path (the new pi05
    # branch must not swallow it).
    runner = LiberoRunner()
    calls: list[str] = []

    async def _fake_pi05(context, spec):
        calls.append("pi05")
        return {"dispatched": "pi05"}

    async def _fake_gr00t(context, spec):
        calls.append("gr00t")
        return {"dispatched": "gr00t"}

    monkeypatch.setattr(runner, "_run_pi05_pilot", _fake_pi05)
    monkeypatch.setattr(runner, "_run_gr00t_pilot", _fake_gr00t)

    result = await runner.run(_ctx(_mission(pilot="gr00t"), tmp_path))

    assert result == {"dispatched": "gr00t"}
    assert calls == ["gr00t"]


async def test_pi05_eval_pilot_value_is_case_insensitive(monkeypatch, tmp_path) -> None:
    runner = LiberoRunner()
    seen: list[str] = []

    async def _fake_pi05(context, spec):
        seen.append("pi05")
        return {"ok": True}

    monkeypatch.setattr(runner, "_run_pi05_pilot", _fake_pi05)
    await runner.run(_ctx(_mission(pilot="PI05"), tmp_path))
    assert seen == ["pi05"]


# ---------------------------------------------------------------------------
# 3. Model-ref resolution — shared resolve_eval_checkpoint, both branches.
# ---------------------------------------------------------------------------

def test_pi05_eval_resolves_model_ref_from_config_checkpoint(tmp_path) -> None:
    # Eval-only mission: the model-ref is a HF repo id in config.checkpoint.
    ctx = _ctx(_mission(pilot="pi05", checkpoint=PI05_MODEL_REF), tmp_path)
    assert resolve_eval_checkpoint(ctx) == Path(PI05_MODEL_REF)


def test_pi05_eval_resolves_model_ref_from_trained_pilot_checkpoint(tmp_path) -> None:
    # No config.checkpoint -> fall back to the PILOT's latest trained checkpoint.
    mission = _mission(pilot="pi05", checkpoint=None, trained_checkpoint="/tmp/pi05-ft")
    assert resolve_eval_checkpoint(_ctx(mission, tmp_path)) == Path("/tmp/pi05-ft")


def test_pi05_eval_resolution_raises_without_any_checkpoint(tmp_path) -> None:
    mission = _mission(pilot="pi05", checkpoint=None, trained_checkpoint=None)
    with pytest.raises(ValueError, match="No completed training task"):
        resolve_eval_checkpoint(_ctx(mission, tmp_path))


# ---------------------------------------------------------------------------
# 4. build_pi05_libero_argv — launch contract + passthrough, handled keys out.
# ---------------------------------------------------------------------------

def test_pi05_eval_build_argv_contract_and_passthrough() -> None:
    task = _eval_spec(
        pilot="pi05",                 # handled — not forwarded
        checkpoint=PI05_MODEL_REF,    # handled — passed explicitly as --checkpoint
        runner="libero",              # handled — not forwarded
        capture_video=True,           # handled — video via --video_dir
        task_id=2,
        host="10.0.0.2",
        port=8000,
        n_action_steps=10,
    )
    argv = build_pi05_libero_argv(
        spec=task, checkpoint=Path(PI05_MODEL_REF), video_dir=Path("/out/videos"),
    )
    # Contract flags first, model-ref threaded through as --checkpoint.
    assert argv[:6] == [
        "--task", "libero_object", "--num_episodes", "4",
        "--checkpoint", PI05_MODEL_REF,
    ]
    assert "--video_dir" in argv and "/out/videos" in argv
    # openpi server address + chunk length pass through verbatim (snake_case).
    assert argv[argv.index("--host") + 1] == "10.0.0.2"
    assert argv[argv.index("--port") + 1] == "8000"
    assert argv[argv.index("--n_action_steps") + 1] == "10"
    assert argv[argv.index("--task_id") + 1] == "2"
    # Keys the runner consumes itself are never forwarded as flags.
    for handled in ("--pilot", "--runner", "--capture_video"):
        assert handled not in argv


def test_pi05_eval_build_argv_omits_video_dir_when_none() -> None:
    argv = build_pi05_libero_argv(
        spec=_eval_spec(pilot="pi05", checkpoint=PI05_MODEL_REF, task_id=1),
        checkpoint=Path(PI05_MODEL_REF),
        video_dir=None,
    )
    assert "--video_dir" not in argv
    assert argv[argv.index("--task_id") + 1] == "1"


def test_pi05_eval_build_argv_uses_resolved_checkpoint_not_config() -> None:
    # The runner passes the *resolved* checkpoint (may be a trained local path);
    # the argv --checkpoint must be that, independent of config.checkpoint.
    task = _eval_spec(pilot="pi05", checkpoint=PI05_MODEL_REF)
    argv = build_pi05_libero_argv(
        spec=task, checkpoint=Path("/local/ft/pi05"), video_dir=None,
    )
    assert argv[argv.index("--checkpoint") + 1] == "/local/ft/pi05"


# ---------------------------------------------------------------------------
# 5. _run_pi05_pilot — launches the recipe + scores, with a mocked subprocess.
# ---------------------------------------------------------------------------

def _patch_bridge(monkeypatch, *, rc: int, summary: dict[str, Any]) -> dict[str, Any]:
    """Stub the shared subprocess + scorer; capture what the runner built."""
    captured: dict[str, Any] = {}

    async def _fake_subprocess(context, process_spec):
        captured["process_spec"] = process_spec
        return rc

    def _fake_summarize(*, collector, spec, checkpoint, eval_script):
        captured["summarize"] = {
            "checkpoint": checkpoint,
            "eval_script": eval_script,
            "collector": collector,
        }
        return summary

    monkeypatch.setattr(
        "odyssey.runners.evals.libero.run_training_subprocess", _fake_subprocess
    )
    monkeypatch.setattr("odyssey.runners.evals.libero.summarize", _fake_summarize)
    return captured


async def test_pi05_eval_run_pi05_pilot_launches_recipe_and_scores(
    monkeypatch, tmp_path
) -> None:
    summary = {"success_rate": 0.5, "num_episodes": 4, "passed": True}
    captured = _patch_bridge(monkeypatch, rc=0, summary=summary)

    runner = LiberoRunner()
    mission = _mission(
        pilot="pi05", checkpoint=PI05_MODEL_REF,
        extra_config={"capture_video": True, "n_action_steps": 10, "port": 8000},
    )
    result = await runner.run(_ctx(mission, tmp_path))

    # The runner returned exactly what the shared scorer produced.
    assert result is summary

    spec = captured["process_spec"]
    # Launched the π0.5 recipe (not the GR00T one) through the subprocess helper.
    assert spec.script_path.endswith("pi05_libero_eval.py")
    assert spec.line_parser is not None  # the shared EvalProtocolCollector.parse

    argv = spec.argv_extra
    assert argv[:6] == [
        "--task", "libero_object", "--num_episodes", "4",
        "--checkpoint", PI05_MODEL_REF,
    ]
    # capture_video -> resolved --video_dir under the task output dir.
    assert "--video_dir" in argv
    assert str(tmp_path) in argv[argv.index("--video_dir") + 1]
    assert argv[argv.index("--n_action_steps") + 1] == "10"

    # Scored through the shared summarize, with the resolved model-ref.
    assert captured["summarize"]["checkpoint"] == Path(PI05_MODEL_REF)
    assert captured["summarize"]["eval_script"].endswith("pi05_libero_eval.py")


async def test_pi05_eval_run_pi05_pilot_raises_on_nonzero_exit(
    monkeypatch, tmp_path
) -> None:
    _patch_bridge(monkeypatch, rc=3, summary={})
    runner = LiberoRunner()
    with pytest.raises(RuntimeError, match="pi05_libero_eval exited with code 3"):
        await runner.run(_ctx(_mission(pilot="pi05"), tmp_path))


async def test_pi05_eval_run_pi05_pilot_reports_cancelled(monkeypatch, tmp_path) -> None:
    # A user cancel mid-subprocess short-circuits to a clean cancelled summary
    # (nonzero rc from the killed child must not raise over the cancel).
    captured: dict[str, Any] = {}

    async def _fake_subprocess(context, process_spec):
        context.request_cancel()
        return 137  # SIGKILL-style exit from the cancelled child

    monkeypatch.setattr(
        "odyssey.runners.evals.libero.run_training_subprocess", _fake_subprocess
    )
    captured["hit_summarize"] = False

    def _fake_summarize(**_kwargs):
        captured["hit_summarize"] = True
        return {}

    monkeypatch.setattr("odyssey.runners.evals.libero.summarize", _fake_summarize)

    runner = LiberoRunner()
    result = await runner.run(_ctx(_mission(pilot="pi05"), tmp_path))
    assert result == {"cancelled": True}
    assert captured["hit_summarize"] is False  # never scored a cancelled run


async def test_pi05_eval_run_pi05_pilot_omits_video_dir_without_capture(
    monkeypatch, tmp_path
) -> None:
    captured = _patch_bridge(monkeypatch, rc=0, summary={"ok": True})
    runner = LiberoRunner()
    # capture_video defaults off -> no --video_dir even with an output dir.
    await runner.run(_ctx(_mission(pilot="pi05"), tmp_path))
    assert "--video_dir" not in captured["process_spec"].argv_extra


# ---------------------------------------------------------------------------
# 6. Protocol contract — π0.5 reuses the shared GR00T-LIBERO collector+scorer.
# ---------------------------------------------------------------------------

def test_pi05_eval_protocol_roundtrips_through_shared_summarize() -> None:
    # The pi05 recipe speaks the same ODYSSEY_* protocol as GR00T/Isaac, so the
    # runner's own collector + summarize score it unchanged (bridge reuse).
    collector = EvalProtocolCollector()
    for i in range(1, 5):
        ok = i <= 2  # 2/4 pass
        collector.parse(
            "ODYSSEY_EPISODE "
            + json.dumps({"index": i, "total": 4, "success": ok, "return": 1.0 if ok else 0.0})
        )
    collector.parse(
        "ODYSSEY_RESULT "
        + json.dumps({"success_rate": 0.5, "performance_score": 0.5, "metrics": {}})
    )
    summary = summarize(
        collector=collector,
        spec=_eval_spec(pilot="pi05", checkpoint=PI05_MODEL_REF),
        checkpoint=Path(PI05_MODEL_REF),
        eval_script="pi05_libero_eval.py",
    )
    assert summary["success_rate"] == 0.5
    assert summary["num_episodes"] == 4
