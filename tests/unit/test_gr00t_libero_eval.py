"""Tests for the GR00T LIBERO eval recipe + LiberoRunner's gr00t dispatch.

The recipe (``odyssey/runners/evals/gr00t_libero_eval.py``) is the eval script
the subprocess LiberoRunner spawns for ``pilot: gr00t``. These tests pin the
interop surface WITHOUT booting LIBERO or a GR00T server:

  * the module imports under the bare stdlib (heavy deps deferred to the run path);
  * the launch-contract argv it accepts + config passthrough;
  * that its ODYSSEY_* protocol lines are consumed by the runner's own
    EvalProtocolCollector + summarize (the real contract, shared with Isaac);
  * that LiberoRunner.build_gr00t_libero_argv forwards config correctly and drops
    the keys it consumes itself.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from odyssey.runners.evals.isaac_lab import EvalProtocolCollector, summarize
from odyssey.runners.evals.libero import build_gr00t_libero_argv
from odyssey.spec import EvaluationTask, EvaluationType

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "odyssey", "runners", "evals"))
import gr00t_libero_eval as E


def _eval_task(**overrides: Any) -> EvaluationTask:
    fields: dict[str, Any] = {
        "name": "gr00t-libero-eval",
        "evaluation_type": EvaluationType.LIBERO,
        "benchmark_name": "libero_object",
        "num_episodes": 4,
    }
    fields.update(overrides)
    return EvaluationTask(**fields)


# ---------------------------------------------------------------------------
# Heavy deps are deferred — the module imports under the bare stdlib.
# ---------------------------------------------------------------------------

def test_module_imports_without_heavy_deps() -> None:
    evals_dir = Path(__file__).resolve().parents[2] / "src" / "odyssey" / "runners" / "evals"
    heavy_deps = ("numpy", "libero", "gr00t", "torch", "robosuite")
    script = (
        "import importlib, json, sys\n"
        f"sys.path.insert(0, {str(evals_dir)!r})\n"
        "importlib.import_module('gr00t_libero_eval')\n"
        f"print(json.dumps([m for m in {heavy_deps!r} if m in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"gr00t_libero_eval failed to import in a clean interpreter:\n{result.stderr}"
    )
    leaked = json.loads(result.stdout)
    assert leaked == [], f"gr00t_libero_eval imported heavy deps at module load: {leaked}"


# ---------------------------------------------------------------------------
# Launch contract (matches build_gr00t_libero_argv in the runner)
# ---------------------------------------------------------------------------

def test_parser_accepts_contract_flags() -> None:
    args = E.build_parser().parse_args(
        ["--task", "libero_object", "--num_episodes", "7", "--checkpoint", "/tmp/ckpt"]
    )
    assert args.task == "libero_object"
    assert args.num_episodes == 7
    assert args.checkpoint == "/tmp/ckpt"
    assert args.task_id == 0  # default


def test_parser_accepts_passthrough_config() -> None:
    args = E.build_parser().parse_args(
        ["--task", "libero_object", "--num_episodes", "1", "--checkpoint", "/c",
         "--task_id", "3", "--host", "10.0.0.2", "--port", "6000",
         "--n_action_steps", "8", "--pos_scale", "0.5",
         "--serve_checkpoint", "true", "--embodiment_tag", "new_embodiment"]
    )
    assert args.task_id == 3
    assert args.host == "10.0.0.2"
    assert args.port == 6000
    assert args.n_action_steps == 8
    assert args.pos_scale == 0.5
    assert args.serve_checkpoint is True
    assert args.embodiment_tag == "new_embodiment"


def test_parser_bool_flags_are_value_style() -> None:
    # The runner forwards config keys as "--flag value", so booleans must parse
    # a trailing value (not store_true).
    args = E.build_parser().parse_args(
        ["--task", "x", "--num_episodes", "1", "--checkpoint", "/c",
         "--serve_checkpoint", "false", "--flip_images", "0", "--translation_only", "yes"]
    )
    assert args.serve_checkpoint is False
    assert args.flip_images is False
    assert args.translation_only is True
    assert E._bool("true") and E._bool("1") and E._bool("on")
    assert not E._bool("false") and not E._bool("")


def test_serve_checkpoint_defaults_off_and_flip_on() -> None:
    args = E.build_parser().parse_args(
        ["--task", "x", "--num_episodes", "1", "--checkpoint", "/c"])
    assert args.serve_checkpoint is False
    assert args.flip_images is True  # LIBERO frames are stored rotated


# ---------------------------------------------------------------------------
# Served-path resolution: GR00T-N1.7-LIBERO ships one subdir per suite.
# ---------------------------------------------------------------------------

def test_resolve_served_path_joins_suite_subdir(tmp_path: Path) -> None:
    (tmp_path / "libero_object").mkdir()
    resolved = E._resolve_served_path(str(tmp_path), "libero_object")
    assert resolved == os.path.join(str(tmp_path), "libero_object")


def test_resolve_served_path_falls_back_when_no_subdir(tmp_path: Path) -> None:
    # No suite subdir on disk (e.g. an HF id) -> serve the checkpoint as-is.
    assert E._resolve_served_path("nvidia/GR00T-N1.7-LIBERO", "libero_object") == (
        "nvidia/GR00T-N1.7-LIBERO"
    )


# ---------------------------------------------------------------------------
# Protocol emission is consumed by the runner's OWN collector + scorer.
# ---------------------------------------------------------------------------

def test_episode_and_result_lines_parsed_by_runner_collector() -> None:
    collector = EvalProtocolCollector()
    event = collector.parse(E.episode_line(index=3, total=10, success=True, ret=1.5))
    assert event is not None and event["step"] == "episode_complete"
    assert event["step_index"] == 3 and event["step_total"] == 10
    assert collector.episodes[0]["success"] is True

    collector.parse(E.result_line(success_rate=0.4, performance_score=0.4,
                                  metrics={"successes": 4}))
    assert collector.result["success_rate"] == 0.4


def test_protocol_roundtrips_through_summarize() -> None:
    collector = EvalProtocolCollector()
    for i in range(1, 5):
        ok = i <= 2  # 2/4 pass
        collector.parse(E.episode_line(index=i, total=4, success=ok, ret=1.0 if ok else 0.0))
    collector.parse(E.result_line(success_rate=0.5, performance_score=0.5, metrics={}))
    summary = summarize(
        collector=collector, spec=_eval_task(),
        checkpoint=Path("nvidia/GR00T-N1.7-LIBERO"), eval_script="gr00t_libero_eval.py",
    )
    assert summary["success_rate"] == 0.5
    assert summary["num_episodes"] == 4


# ---------------------------------------------------------------------------
# LiberoRunner argv builder: contract flags + passthrough, minus handled keys.
# ---------------------------------------------------------------------------

def test_build_argv_contract_and_passthrough() -> None:
    task = _eval_task(
        num_episodes=10,
        config={
            "pilot": "gr00t",                     # handled — not forwarded
            "checkpoint": "nvidia/GR00T-N1.7-LIBERO",  # handled — passed explicitly
            "runner": "libero",                   # handled — not forwarded
            "capture_video": True,                # handled — video via --video_dir
            "task_id": 0,
            "serve_checkpoint": "true",
            "embodiment_tag": "new_embodiment",
            "n_action_steps": 16,
        },
    )
    argv = build_gr00t_libero_argv(
        spec=task, checkpoint=Path("nvidia/GR00T-N1.7-LIBERO"), video_dir=Path("/out/videos"),
    )
    # Contract flags present.
    assert argv[:6] == [
        "--task", "libero_object", "--num_episodes", "10",
        "--checkpoint", "nvidia/GR00T-N1.7-LIBERO",
    ]
    assert "--video_dir" in argv and "/out/videos" in argv
    # Passthrough config forwarded verbatim (snake_case).
    assert "--task_id" in argv and "--serve_checkpoint" in argv
    assert argv[argv.index("--embodiment_tag") + 1] == "new_embodiment"
    assert argv[argv.index("--n_action_steps") + 1] == "16"
    # Handled keys NOT forwarded as flags.
    for handled in ("--pilot", "--runner", "--capture_video"):
        assert handled not in argv


def test_build_argv_omits_video_dir_when_none() -> None:
    argv = build_gr00t_libero_argv(
        spec=_eval_task(config={"task_id": 1}), checkpoint=Path("/c"), video_dir=None,
    )
    assert "--video_dir" not in argv
    assert argv[argv.index("--task_id") + 1] == "1"


# ---------------------------------------------------------------------------
# The parsed argv wires into a valid TrainingProcessSpec (script_path form).
# ---------------------------------------------------------------------------

def test_argv_parses_back_into_the_recipe() -> None:
    # End-to-end: what the runner builds, the script's parser accepts.
    task = _eval_task(config={"task_id": 2, "serve_checkpoint": "true",
                              "embodiment_tag": "new_embodiment", "n_action_steps": 8})
    argv = build_gr00t_libero_argv(
        spec=task, checkpoint=Path("/ckpt"), video_dir=None,
    )
    args = E.build_parser().parse_args(argv)
    assert args.task == "libero_object"
    assert args.task_id == 2
    assert args.serve_checkpoint is True
    assert args.n_action_steps == 8
