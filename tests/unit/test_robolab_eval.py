"""Wiring tests for the first-class RoboLab eval (``evaluation_type: robolab``).

These pin the cabling WITHOUT Isaac Sim, RoboLab, or a policy server — the
same mocking pattern as ``test_pi05_eval.py`` / the custom-eval tests:

  * the new ``EvaluationType.ROBOLAB`` enum value round-trips through the spec;
  * the RunnerRegistry routes a robolab eval task to ``RobolabRunner`` (and
    keeps isaac_lab/libero/custom distinct);
  * ``build_robolab_argv`` maps ``num_episodes`` to ``--num-runs``, dashes the
    passthrough keys, and drops the keys the runner consumes itself;
  * ``read_episode_results`` / ``summarize`` fold RoboLab's
    ``episode_results.jsonl`` into the shared eval summary (per-task rates +
    failure reasons), with contract violations raising clearly;
  * ``RobolabRunner.run`` launches run.py through the shared subprocess helper
    (cwd = the checkout, Isaac interpreter as launcher), copies the results
    artifact, and errors on nonzero exit / missing results.

All tests are named ``test_robolab_*`` (the ``-k robolab`` gate).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from odyssey.engine.records import MissionRun
from odyssey.runners.base import TaskContext
from odyssey.runners.evals.robolab import (
    RobolabRunner,
    build_robolab_argv,
    read_episode_results,
    resolve_robolab_root,
    summarize,
)
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
)
from odyssey.telemetry import EventPublisher

CHECKPOINT = "nvidia/Cosmos3-Nano-Policy-DROID"


class _NullPublisher(EventPublisher):
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        return


def _eval_spec(**config: Any) -> EvaluationTask:
    return EvaluationTask(
        name="robolab-eval",
        evaluation_type=EvaluationType.ROBOLAB,
        benchmark_name="BananaInBowlTask",
        num_episodes=10,
        config=dict(config),
    )


def _mission(**config: Any) -> MissionRun:
    config.setdefault("checkpoint", CHECKPOINT)
    spec = Mission(
        metadata=MissionMetadata(name="msn-robolab"),
        objective="o",
        acceptance_criteria="a",
        robot=RobotSpec(
            embodiment="franka_panda",
            agents=[AgentSpec(id="pilot", role=AgentRole.PILOT,
                              model=HFModelRef(base=CHECKPOINT))],
        ),
        tasks=[_eval_spec(**config)],
    )
    return MissionRun.from_spec(spec)


def _ctx(mission: MissionRun, tmp_path: Path | None = None) -> TaskContext:
    return TaskContext(
        task=mission.tasks[0],
        mission=mission,
        publisher=_NullPublisher(),
        output_dir=tmp_path,
    )


def _episodes_jsonl(records: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(r) for r in records) + "\n"


# ---------------------------------------------------------------------------
# 1. Spec + registry — the new enum value routes to RobolabRunner.
# ---------------------------------------------------------------------------

def test_robolab_enum_value_round_trips() -> None:
    assert EvaluationType("robolab") is EvaluationType.ROBOLAB
    spec = _eval_spec(robolab_root="/rl")
    assert spec.evaluation_type is EvaluationType.ROBOLAB


def test_robolab_registry_routes_to_robolab_runner() -> None:
    registry = RunnerRegistry()
    registry.register(RobolabRunner())
    mission = _mission(robolab_root="/rl")
    runner = registry.select(mission.tasks[0].spec)
    assert isinstance(runner, RobolabRunner)


def test_robolab_registry_keeps_sibling_evals_distinct() -> None:
    from odyssey.runners.evals.isaac_lab import IsaacLabRunner
    from odyssey.runners.evals.libero import LiberoRunner

    registry = RunnerRegistry()
    registry.register(IsaacLabRunner())
    registry.register(LiberoRunner())
    registry.register(RobolabRunner())
    runner = registry.select(_mission(robolab_root="/rl").tasks[0].spec)
    assert isinstance(runner, RobolabRunner)


# ---------------------------------------------------------------------------
# 2. Argv builder — contract flags, episode mapping, dashed passthrough.
# ---------------------------------------------------------------------------

def test_robolab_argv_contract_and_episode_mapping() -> None:
    spec = _eval_spec(robolab_root="/rl", num_envs=5, remote_host="10.0.0.5",
                      remote_port=8902, checkpoint=CHECKPOINT, runner="robolab")
    argv = build_robolab_argv(spec=spec, folder="odyssey_t1")

    assert argv[:4] == ["--task", "BananaInBowlTask", "--num-runs", "2"]  # ceil(10/5)
    assert argv[argv.index("--num-envs") + 1] == "5"
    assert argv[argv.index("--output-folder-name") + 1] == "odyssey_t1"
    assert "--headless" in argv
    # Passthrough keys are dashed (RoboLab's backend flags are kebab-only).
    assert argv[argv.index("--remote-host") + 1] == "10.0.0.5"
    assert argv[argv.index("--remote-port") + 1] == "8902"
    # Keys the runner consumes never leak as flags.
    for handled in ("--robolab-root", "--checkpoint", "--runner", "--num_envs"):
        assert handled not in argv


def test_robolab_argv_defaults_single_env_run_per_episode() -> None:
    argv = build_robolab_argv(spec=_eval_spec(robolab_root="/rl"), folder="f")
    assert argv[argv.index("--num-runs") + 1] == "10"  # ceil(10/1)
    assert argv[argv.index("--num-envs") + 1] == "1"


def test_robolab_argv_booleans_become_bare_flags() -> None:
    # RoboLab's store_true/store_false flags take NO value (--disable-subtask,
    # --enable-gt-state); True -> bare flag, False -> omitted entirely.
    spec = _eval_spec(robolab_root="/rl", disable_subtask=True, enable_gt_state=False)
    argv = build_robolab_argv(spec=spec, folder="f")
    assert "--disable-subtask" in argv
    assert argv[argv.index("--disable-subtask") + 1].startswith("--")  # no value after it
    assert "--enable-gt-state" not in argv
    assert "True" not in argv and "False" not in argv


def test_robolab_root_is_required() -> None:
    with pytest.raises(RuntimeError, match="robolab_root"):
        resolve_robolab_root({})


# ---------------------------------------------------------------------------
# 3. Results parsing + summary.
# ---------------------------------------------------------------------------

def test_robolab_reads_episode_results_jsonl(tmp_path) -> None:
    f = tmp_path / "episode_results.jsonl"
    f.write_text(_episodes_jsonl([
        {"task": "BananaInBowlTask", "success": True},
        {"task": "BananaInBowlTask", "success": False, "reason": "missed grasp"},
    ]))
    episodes = read_episode_results(f)
    assert len(episodes) == 2


def test_robolab_missing_results_raises(tmp_path) -> None:
    with pytest.raises(RuntimeError, match=r"episode_results\.jsonl"):
        read_episode_results(tmp_path / "episode_results.jsonl")


def test_robolab_empty_results_raises(tmp_path) -> None:
    f = tmp_path / "episode_results.jsonl"
    f.write_text("\n")
    with pytest.raises(RuntimeError, match="no episodes"):
        read_episode_results(f)


def test_robolab_summary_scores_success_rate_and_reasons() -> None:
    episodes = [
        {"task": "BananaInBowlTask", "success": True},
        {"task": "BananaInBowlTask", "success": False, "reason": "missed grasp"},
        {"task": "HammerInBinTask", "success": False, "reason": "missed grasp"},
        {"task": "HammerInBinTask", "success": True},
    ]
    summary = summarize(episodes=episodes, spec=_eval_spec(robolab_root="/rl"),
                        checkpoint=Path(CHECKPOINT))
    assert summary["num_episodes"] == 4
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["metrics"]["per_task"] == {
        "BananaInBowlTask": "1/2", "HammerInBinTask": "1/2",
    }
    assert summary["metrics"]["failure_reasons"] == {"missed grasp": 2}


# ---------------------------------------------------------------------------
# 4. Runner.run — mocked subprocess; launch shape, artifact copy, failures.
# ---------------------------------------------------------------------------

def _patch_subprocess(monkeypatch, *, rc: int, write_results=None):
    captured: dict[str, Any] = {}

    async def _fake_subprocess(context, process_spec):
        captured["process_spec"] = process_spec
        if write_results is not None:
            write_results()
        return rc

    monkeypatch.setattr(
        "odyssey.runners.evals.robolab.run_training_subprocess", _fake_subprocess
    )
    return captured


async def test_robolab_run_launches_and_scores(monkeypatch, tmp_path) -> None:
    robolab_root = tmp_path / "RoboLab"
    (robolab_root / "policies" / "cosmos3").mkdir(parents=True)

    mission = _mission(robolab_root=str(robolab_root), num_envs=2,
                       remote_port=8902, eval_python="/isaac/bin/python")
    task_id = mission.tasks[0].id
    results_dir = robolab_root / "output" / f"odyssey_{task_id}"

    def _write_results() -> None:
        results_dir.mkdir(parents=True)
        (results_dir / "episode_results.jsonl").write_text(_episodes_jsonl(
            [{"success": True}] * 3 + [{"success": False, "reason": "timeout"}]
        ))

    captured = _patch_subprocess(monkeypatch, rc=0, write_results=_write_results)
    out_dir = tmp_path / "task-out"
    summary = await RobolabRunner().run(_ctx(mission, out_dir))

    spec = captured["process_spec"]
    assert spec.script_path.endswith("policies/cosmos3/run.py")
    assert spec.launcher == ["/isaac/bin/python"]  # the Isaac interpreter
    assert spec.cwd == str(robolab_root)
    assert "--headless" in spec.argv_extra

    assert summary["num_episodes"] == 4
    assert summary["success_rate"] == pytest.approx(0.75)
    # The raw RoboLab record is preserved as a task artifact.
    assert (out_dir / "episode_results.jsonl").is_file()


async def test_robolab_run_raises_on_nonzero_exit(monkeypatch, tmp_path) -> None:
    robolab_root = tmp_path / "RoboLab"
    robolab_root.mkdir()
    _patch_subprocess(monkeypatch, rc=7)
    with pytest.raises(RuntimeError, match="exited with code 7"):
        await RobolabRunner().run(_ctx(_mission(robolab_root=str(robolab_root)), tmp_path))


async def test_robolab_run_raises_when_results_missing(monkeypatch, tmp_path) -> None:
    robolab_root = tmp_path / "RoboLab"
    robolab_root.mkdir()
    _patch_subprocess(monkeypatch, rc=0)  # exits 0 but writes nothing
    with pytest.raises(RuntimeError, match="wrote no results"):
        await RobolabRunner().run(_ctx(_mission(robolab_root=str(robolab_root)), tmp_path))


async def test_robolab_run_reports_cancelled(monkeypatch, tmp_path) -> None:
    robolab_root = tmp_path / "RoboLab"
    robolab_root.mkdir()

    async def _fake_subprocess(context, process_spec):
        context.request_cancel()
        return 137

    monkeypatch.setattr(
        "odyssey.runners.evals.robolab.run_training_subprocess", _fake_subprocess
    )
    result = await RobolabRunner().run(_ctx(_mission(robolab_root=str(robolab_root)), tmp_path))
    assert result == {"cancelled": True}
