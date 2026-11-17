"""Wiring tests for the WALL-OSS LIBERO eval path.

These pin the *cabling* that makes ``evaluation_type: libero`` + ``pilot:
wall_oss`` reach the right code — WITHOUT a GPU, a wall-x server,
LIBERO, or a real simulator (the same mocking pattern as ``test_cosmos3_eval.py``):

  * ``LiberoRunner`` dispatches ``pilot: wall_oss`` to its WALL-OSS subprocess
    path and nowhere else;
  * ``build_wall_oss_libero_argv`` forwards the launch contract + config
    passthrough and drops the keys the runner consumes itself;
  * ``_run_wall_oss_pilot`` launches the ``wall_oss_libero_eval.py`` recipe
    through the shared subprocess helper and scores via the shared
    ``summarize``;
  * the eval recipe's argv surface (parser defaults) and ``ODYSSEY_*`` protocol
    emitters, imported under the bare stdlib.

All tests are named ``test_wall_oss_eval_*`` (the ``-k wall_oss_eval`` gate).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from odyssey.engine.records import MissionRun
from odyssey.runners.base import TaskContext
from odyssey.runners.evals.libero import LiberoRunner, build_wall_oss_libero_argv
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

# A published WALL-OSS policy checkpoint stands in for the model-ref throughout.
WALL_OSS_MODEL_REF = "x-square-robot/wall-oss-0.5"

_EVAL_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src" / "odyssey" / "runners" / "evals" / "wall_oss_libero_eval.py"
)


# ---------------------------------------------------------------------------
# Fixtures / builders — an eval-only mission + a TaskContext, no engine.
# ---------------------------------------------------------------------------

class _NullPublisher(EventPublisher):
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        return


def _mission(*, pilot: str = "wall_oss",
             extra_config: dict[str, Any] | None = None) -> MissionRun:
    config: dict[str, Any] = {"pilot": pilot, "checkpoint": WALL_OSS_MODEL_REF}
    if extra_config:
        config.update(extra_config)
    spec = Mission(
        metadata=MissionMetadata(name="msn-wall-oss"),
        objective="o",
        acceptance_criteria="a",
        robot=RobotSpec(
            embodiment="franka_panda",
            agents=[AgentSpec(id="pilot", role=AgentRole.PILOT,
                              model=HFModelRef(base=WALL_OSS_MODEL_REF))],
        ),
        tasks=[
            EvaluationTask(
                name="wall-oss-libero-eval",
                evaluation_type=EvaluationType.LIBERO,
                benchmark_name="libero_object",
                num_episodes=4,
                config=config,
            ),
        ],
    )
    return MissionRun.from_spec(spec)


def _ctx(mission: MissionRun, tmp_path: Path | None = None) -> TaskContext:
    return TaskContext(
        task=mission.tasks[0],
        mission=mission,
        publisher=_NullPublisher(),
        output_dir=tmp_path,
    )


def _eval_spec(**config: Any) -> EvaluationTask:
    return EvaluationTask(
        name="wall-oss-libero-eval",
        evaluation_type=EvaluationType.LIBERO,
        benchmark_name="libero_object",
        num_episodes=4,
        config=dict(config),
    )


# ---------------------------------------------------------------------------
# 1. LiberoRunner — `pilot: wall_oss` dispatches to the WALL-OSS path, and only that.
# ---------------------------------------------------------------------------

async def test_wall_oss_eval_pilot_dispatches_to_wall_oss_path(
    monkeypatch, tmp_path
) -> None:
    runner = LiberoRunner()
    seen: list[str] = []

    async def _fake_wall_oss(context, spec):
        seen.append("wall_oss")
        return {"dispatched": "wall_oss"}

    async def _fake_cosmos3(context, spec):
        seen.append("cosmos3")
        return {"dispatched": "cosmos3"}

    monkeypatch.setattr(runner, "_run_wall_oss_pilot", _fake_wall_oss)
    monkeypatch.setattr(runner, "_run_cosmos3_pilot", _fake_cosmos3)

    result = await runner.run(_ctx(_mission(pilot="wall_oss"), tmp_path))
    assert result == {"dispatched": "wall_oss"}
    assert seen == ["wall_oss"]


async def test_wall_oss_eval_pilot_value_is_case_insensitive(
    monkeypatch, tmp_path
) -> None:
    runner = LiberoRunner()

    async def _fake_wall_oss(context, spec):
        return {"dispatched": "wall_oss"}

    monkeypatch.setattr(runner, "_run_wall_oss_pilot", _fake_wall_oss)
    result = await runner.run(_ctx(_mission(pilot="Wall_OSS"), tmp_path))
    assert result == {"dispatched": "wall_oss"}


@pytest.mark.parametrize("alias", ["walloss", "wall-oss", "WALL_OSS"])
async def test_wall_oss_eval_pilot_aliases_dispatch(monkeypatch, tmp_path, alias) -> None:
    runner = LiberoRunner()

    async def _fake_wall_oss(context, spec):
        return {"dispatched": "wall_oss"}

    monkeypatch.setattr(runner, "_run_wall_oss_pilot", _fake_wall_oss)
    result = await runner.run(_ctx(_mission(pilot=alias), tmp_path))
    assert result == {"dispatched": "wall_oss"}


# ---------------------------------------------------------------------------
# 2. build_wall_oss_libero_argv — contract flags + passthrough, handled keys dropped.
# ---------------------------------------------------------------------------

def test_wall_oss_eval_build_argv_contract_and_passthrough() -> None:
    spec = _eval_spec(
        pilot="wall_oss", checkpoint=WALL_OSS_MODEL_REF, runner="libero",
        capture_video=True,  # handled by the runner (resolved --video_dir)
        task_id=3, host="10.0.0.5", port=8100,
        n_action_steps=8, invert_gripper=True,
    )
    argv = build_wall_oss_libero_argv(
        spec=spec, checkpoint=Path("/ckpt"), video_dir=Path("/videos"),
    )

    assert argv[:6] == [
        "--task", "libero_object", "--num_episodes", "4", "--checkpoint", "/ckpt",
    ]
    assert argv[argv.index("--video_dir") + 1] == "/videos"
    # Passthrough keys arrive verbatim (snake_case flags).
    for flag, value in [
        ("--task_id", "3"), ("--host", "10.0.0.5"), ("--port", "8100"),
        ("--n_action_steps", "8"), ("--invert_gripper", "True"),
    ]:
        assert argv[argv.index(flag) + 1] == value
    # Keys the runner consumes itself never leak as flags.
    for handled in ("--pilot", "--checkpoint_", "--runner", "--capture_video"):
        assert handled not in argv


def test_wall_oss_eval_build_argv_omits_video_dir_when_none() -> None:
    argv = build_wall_oss_libero_argv(
        spec=_eval_spec(pilot="wall_oss"), checkpoint=Path("/ckpt"), video_dir=None,
    )
    assert "--video_dir" not in argv


# ---------------------------------------------------------------------------
# 3. _run_wall_oss_pilot — launches the recipe + scores, with a mocked subprocess.
# ---------------------------------------------------------------------------

def _patch_bridge(monkeypatch, *, rc: int, summary: dict[str, Any]) -> dict[str, Any]:
    """Stub the shared subprocess + scorer; capture what the runner built."""
    captured: dict[str, Any] = {}

    async def _fake_subprocess(context, process_spec):
        captured["process_spec"] = process_spec
        return rc

    def _fake_summarize(*, collector, spec, checkpoint, eval_script):
        captured["summarize"] = {"checkpoint": checkpoint, "eval_script": eval_script}
        return summary

    monkeypatch.setattr(
        "odyssey.runners.evals.libero.run_training_subprocess", _fake_subprocess
    )
    monkeypatch.setattr("odyssey.runners.evals.libero.summarize", _fake_summarize)
    return captured


async def test_wall_oss_eval_run_pilot_launches_recipe_and_scores(
    monkeypatch, tmp_path
) -> None:
    summary = {"success_rate": 0.25, "num_episodes": 4, "passed": False}
    captured = _patch_bridge(monkeypatch, rc=0, summary=summary)

    runner = LiberoRunner()
    mission = _mission(extra_config={"capture_video": True, "port": 8000})
    result = await runner.run(_ctx(mission, tmp_path))

    assert result is summary
    spec = captured["process_spec"]
    assert spec.script_path.endswith("wall_oss_libero_eval.py")
    assert spec.line_parser is not None  # the shared EvalProtocolCollector.parse
    argv = spec.argv_extra
    assert argv[:6] == [
        "--task", "libero_object", "--num_episodes", "4",
        "--checkpoint", WALL_OSS_MODEL_REF,
    ]
    assert "--video_dir" in argv
    assert captured["summarize"]["eval_script"].endswith("wall_oss_libero_eval.py")


async def test_wall_oss_eval_run_pilot_raises_on_nonzero_exit(
    monkeypatch, tmp_path
) -> None:
    _patch_bridge(monkeypatch, rc=3, summary={})
    runner = LiberoRunner()
    with pytest.raises(RuntimeError, match="wall_oss_libero_eval exited with code 3"):
        await runner.run(_ctx(_mission(), tmp_path))


# ---------------------------------------------------------------------------
# 4. The eval recipe's stdlib surface — argv defaults + ODYSSEY_* protocol.
# ---------------------------------------------------------------------------

def _load_recipe():
    spec = importlib.util.spec_from_file_location("wall_oss_libero_eval", _EVAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wall_oss_eval_recipe_parser_defaults() -> None:
    recipe = _load_recipe()
    args = recipe.build_parser().parse_args(["--task", "libero_object"])

    assert args.host == "127.0.0.1"
    assert args.port == 32195  # wall-x run_serving.sh default
    assert args.n_action_steps == 10  # --env LIBERO ACTION_HORIZON
    assert args.jpeg_quality == 95
    assert args.binarize_gripper is False
    assert args.invert_gripper is False
    assert args.flip_images is True
    assert args.translation_only is False


def test_wall_oss_eval_recipe_protocol_lines_are_parseable_json() -> None:
    recipe = _load_recipe()
    ep = recipe.episode_line(index=2, total=4, success=True, ret=1.0)
    res = recipe.result_line(success_rate=0.5, performance_score=0.5,
                             metrics={"successes": 2})

    assert ep.startswith("ODYSSEY_EPISODE ")
    assert json.loads(ep.removeprefix("ODYSSEY_EPISODE ")) == {
        "index": 2, "total": 4, "success": True, "return": 1.0,
    }
    assert res.startswith("ODYSSEY_RESULT ")
    assert json.loads(res.removeprefix("ODYSSEY_RESULT "))["success_rate"] == 0.5


def test_wall_oss_eval_recipe_imports_under_bare_stdlib() -> None:
    # Loading the module must not pull numpy/libero/torch (they're run-path lazy).
    before = set(sys.modules)
    _load_recipe()
    leaked = {m for m in ("numpy", "torch", "libero") if m in sys.modules} - before
    assert not leaked, f"recipe imported heavy deps at load: {leaked}"
