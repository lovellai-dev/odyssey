"""Tests for the GR00T Isaac Lab eval recipe (odyssey#17).

The recipe (``odyssey/runners/evals/gr00t_isaac_eval.py``) is the pluggable
eval script that the subprocess ``IsaacLabRunner`` spawns. These tests pin the
things that make it interoperate WITHOUT booting Isaac Sim or GR00T:

  * the launch-contract argv it accepts (--task/--num_episodes/--checkpoint/...);
  * the closed-loop auto-serve argv + flags (deploy the finetuned checkpoint);
  * that the ODYSSEY_* protocol lines it emits are consumed correctly by the
    runner's own ``EvalProtocolCollector`` + ``summarize`` (the real contract).

Heavy deps (isaaclab, gymnasium, gr00t, torch, numpy) are deferred into the run
path, so importing the module here needs only the stdlib — that deferral is
itself asserted below.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from odyssey.runners.evals.isaac_lab import EvalProtocolCollector, summarize
from odyssey.spec import EvaluationTask, EvaluationType

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "odyssey", "runners", "evals"))
import gr00t_isaac_eval as E


def _eval_task(**overrides: Any) -> EvaluationTask:
    fields: dict[str, Any] = {
        "name": "gr00t-isaac-eval",
        "evaluation_type": EvaluationType.ISAAC_LAB,
        "benchmark_name": "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0",
        "num_episodes": 4,
    }
    fields.update(overrides)
    return EvaluationTask(**fields)


# ---------------------------------------------------------------------------
# Heavy deps are deferred — the module imports under the bare stdlib.
# ---------------------------------------------------------------------------

def test_module_imports_without_sim_deps() -> None:
    # isaaclab / gymnasium / gr00t / torch are not installed in the core env;
    # a clean import proves they are imported lazily in the run path, never at
    # module load (otherwise this file would have failed to import at all).
    for mod in ("isaaclab", "isaaclab_tasks", "gymnasium", "gr00t", "torch"):
        assert mod not in sys.modules, f"{mod} leaked into module-load imports"


# ---------------------------------------------------------------------------
# Launch contract (matches build_isaac_lab_argv in the runner)
# ---------------------------------------------------------------------------

def test_parser_accepts_contract_flags() -> None:
    args = E.build_parser().parse_args(
        ["--task", "Isaac-Foo-v0", "--num_episodes", "7",
         "--checkpoint", "/tmp/ckpt", "--headless"]
    )
    assert args.task == "Isaac-Foo-v0"
    assert args.num_episodes == 7
    assert args.checkpoint == "/tmp/ckpt"
    assert args.headless is True


def test_parser_accepts_passthrough_config() -> None:
    # Extra config keys the runner forwards verbatim (snake_case).
    args = E.build_parser().parse_args(
        ["--task", "X", "--num_episodes", "1", "--checkpoint", "/c",
         "--host", "10.0.0.2", "--port", "6000",
         "--instruction", "stack the red cube", "--pos_scale", "0.05"]
    )
    assert args.host == "10.0.0.2"
    assert args.port == 6000
    assert args.instruction == "stack the red cube"
    assert args.pos_scale == 0.05


def test_parser_headless_defaults_off_without_flag() -> None:
    args = E.build_parser().parse_args(
        ["--task", "X", "--num_episodes", "1", "--checkpoint", "/c"])
    assert args.headless is False


# ---------------------------------------------------------------------------
# Protocol emission is consumed by the runner's OWN collector + scorer.
# ---------------------------------------------------------------------------

def test_episode_line_parsed_by_runner_collector() -> None:
    collector = EvalProtocolCollector()
    event = collector.parse(E.episode_line(index=3, total=10, success=True, ret=1.5))
    assert event is not None and event["step"] == "episode_complete"
    assert event["step_index"] == 3 and event["step_total"] == 10
    assert collector.episodes[0]["success"] is True
    assert collector.episodes[0]["return"] == 1.5


def test_result_line_parsed_by_runner_collector() -> None:
    collector = EvalProtocolCollector()
    event = collector.parse(
        E.result_line(success_rate=0.4, performance_score=0.7, metrics={"sim_steps": 99}))
    assert event is not None
    assert collector.result["success_rate"] == 0.4
    assert collector.result["performance_score"] == 0.7
    assert collector.result["metrics"]["sim_steps"] == 99


def test_full_protocol_scored_by_runner_summarize(tmp_path: Path) -> None:
    # Emit a full rollout the way the recipe does, then score it with the
    # runner's summarize() — the exact path Odyssey takes in production.
    collector = EvalProtocolCollector()
    outcomes = [True, True, False, False]
    for i, ok in enumerate(outcomes, start=1):
        collector.parse(E.episode_line(index=i, total=4, success=ok, ret=1.0 if ok else 0.0))
    collector.parse(E.result_line(
        success_rate=0.5, performance_score=0.5, metrics={"benchmark": "cosmos"}))
    summary = summarize(
        collector=collector, spec=_eval_task(),
        checkpoint=tmp_path / "ckpt",
        eval_script="src/odyssey/runners/evals/gr00t_isaac_eval.py")
    assert summary["num_episodes"] == 4
    assert summary["success_rate"] == 0.5
    assert summary["passed"] is True
    assert summary["letter_grade"] in {"C", "D", "F", "B", "A"}


# ---------------------------------------------------------------------------
# Closed-loop auto-serve: deploy the finetuned checkpoint as a policy server.
# ---------------------------------------------------------------------------

def test_serve_flags_parse_through_config_passthrough() -> None:
    # The runner forwards task.config keys verbatim as `--key value`; the
    # boolean must survive that path (value-style, not store_true).
    args = E.build_parser().parse_args(
        ["--task", "X", "--num_episodes", "1", "--checkpoint", "/ckpt",
         "--serve_checkpoint", "true", "--embodiment_tag", "new_embodiment",
         "--server_python", "/venv/bin/python", "--server_device", "cuda:0"])
    assert args.serve_checkpoint is True
    assert args.embodiment_tag == "new_embodiment"
    assert args.server_python == "/venv/bin/python"
    assert args.server_device == "cuda:0"


def test_serve_checkpoint_defaults_off() -> None:
    args = E.build_parser().parse_args(
        ["--task", "X", "--num_episodes", "1", "--checkpoint", "/c"])
    assert args.serve_checkpoint is False


def test_bool_type() -> None:
    assert E._bool("true") and E._bool("1") and E._bool("YES") and E._bool("on")
    assert not E._bool("false") and not E._bool("0") and not E._bool("")


def test_build_server_command_shape() -> None:
    argv = E.build_server_command(
        checkpoint="/work/ckpt", embodiment_tag="new_embodiment", port=5555,
        server_python="/venv/bin/python", device="cuda:0",
        modality_config_path="/repo/examples/SO100/so100_config.py")
    assert argv[:3] == ["/venv/bin/python", "-m", "gr00t.eval.run_gr00t_server"]
    assert argv[argv.index("--model-path") + 1] == "/work/ckpt"
    assert argv[argv.index("--embodiment-tag") + 1] == "new_embodiment"
    assert argv[argv.index("--port") + 1] == "5555"
    assert argv[argv.index("--device") + 1] == "cuda:0"
    assert argv[argv.index("--modality-config-path") + 1].endswith("so100_config.py")
    # raw nested obs -> the server must NOT wrap in a sim policy
    assert "--use-sim-policy-wrapper" not in argv


def test_build_server_command_defaults_and_omits_optional() -> None:
    argv = E.build_server_command(
        checkpoint="/c", embodiment_tag="libero_sim", port=6000)
    assert argv[0] == sys.executable          # defaults to current interpreter
    assert "--modality-config-path" not in argv   # omitted when not given
    assert "--denoising-steps" not in argv         # omitted when 0


def test_wait_for_server_ready_when_listening() -> None:
    import socket
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()
    try:
        assert E._wait_for_server(host, port, timeout_s=5) is True
    finally:
        srv.close()


def test_wait_for_server_fails_fast_when_proc_dead() -> None:
    class _DeadProc:
        def poll(self):  # already exited non-zero
            return 1
    # Nothing is listening on this port; the dead proc short-circuits the wait.
    assert E._wait_for_server("127.0.0.1", 1, timeout_s=30, proc=_DeadProc()) is False


def test_served_model_path_overrides_checkpoint_for_serving() -> None:
    # --served_model_path lets the server serve an explicit checkpoint even when
    # the runner's --checkpoint is a stub (e.g. a Command-Center-delegated eval,
    # where training ran as a separate task and the reconstructed mission's
    # training stub yields a mock checkpoint). It parses, and is what gets served.
    args = E.build_parser().parse_args(
        ["--task", "X", "--num_episodes", "1", "--checkpoint", "/stub",
         "--serve_checkpoint", "true", "--embodiment_tag", "libero_sim",
         "--served_model_path", "/real/checkpoint-500"])
    assert args.served_model_path == "/real/checkpoint-500"
    argv = E.build_server_command(
        checkpoint=(args.served_model_path or args.checkpoint),
        embodiment_tag=args.embodiment_tag, port=5555)
    assert argv[argv.index("--model-path") + 1] == "/real/checkpoint-500"
