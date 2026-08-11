"""Tests for the π0.5 LIBERO eval recipe (``pi05_libero_eval.py``).

The recipe is the eval script the subprocess LiberoRunner spawns for
``pilot: pi05``. These tests pin the interop surface WITHOUT booting LIBERO or an
openpi server (mirror ``test_gr00t_libero_eval.py``):

  * the module imports under the bare stdlib (heavy deps deferred to the run path);
  * the launch-contract argv it accepts + config passthrough (open-loop: host/port,
    NO GR00T ``--serve_checkpoint``/``--embodiment_tag``/server flags);
  * that its ODYSSEY_* protocol lines are consumed by the runner's own
    EvalProtocolCollector + summarize (the real contract, shared with GR00T/Isaac);
  * that LiberoRunner.build_pi05_libero_argv forwards config correctly and drops
    the keys it consumes itself, and round-trips back into this parser.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from odyssey.runners.evals.isaac_lab import EvalProtocolCollector, summarize
from odyssey.runners.evals.libero import build_pi05_libero_argv
from odyssey.spec import EvaluationTask, EvaluationType

sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "odyssey", "runners", "evals"))
import pi05_libero_eval as E


def _eval_task(**overrides: Any) -> EvaluationTask:
    fields: dict[str, Any] = {
        "name": "pi05-libero-eval",
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
    heavy_deps = ("numpy", "libero", "openpi_client", "torch", "robosuite", "jax")
    script = (
        "import importlib, json, sys\n"
        f"sys.path.insert(0, {str(evals_dir)!r})\n"
        "importlib.import_module('pi05_libero_eval')\n"
        f"print(json.dumps([m for m in {heavy_deps!r} if m in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"pi05_libero_eval failed to import in a clean interpreter:\n{result.stderr}"
    )
    leaked = json.loads(result.stdout)
    assert leaked == [], f"pi05_libero_eval imported heavy deps at module load: {leaked}"


# ---------------------------------------------------------------------------
# Launch contract (matches build_pi05_libero_argv in the runner)
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
         "--n_action_steps", "8", "--api_key", "secret"]
    )
    assert args.task_id == 3
    assert args.host == "10.0.0.2"
    assert args.port == 6000
    assert args.n_action_steps == 8
    assert args.api_key == "secret"


def test_parser_defaults_are_openpi_flavoured() -> None:
    # π0.5 defaults differ from GR00T: port 8000 (openpi) and chunk 10 (pi05_libero).
    args = E.build_parser().parse_args(
        ["--task", "x", "--num_episodes", "1", "--checkpoint", "/c"])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.n_action_steps == 10
    assert args.translation_only is False


def test_parser_bool_flags_are_value_style() -> None:
    # The runner forwards config keys as "--flag value", so booleans must parse
    # a trailing value (not store_true).
    args = E.build_parser().parse_args(
        ["--task", "x", "--num_episodes", "1", "--checkpoint", "/c",
         "--flip_images", "0", "--translation_only", "yes"]
    )
    assert args.flip_images is False
    assert args.translation_only is True
    assert E._bool("true") and E._bool("1") and E._bool("on")
    assert not E._bool("false") and not E._bool("")


def test_parser_has_no_gr00t_server_flags() -> None:
    # π0.5 is open-loop (server pre-started); the GR00T auto-serve flags must not
    # exist here (they'd silently accept config that does nothing).
    for gr00t_only in ("--serve_checkpoint", "--embodiment_tag", "--sim_policy_wrapper"):
        try:
            E.build_parser().parse_args(
                ["--task", "x", "--num_episodes", "1", "--checkpoint", "/c",
                 gr00t_only, "true"])
        except SystemExit:
            continue  # unknown flag -> argparse exits, as intended
        raise AssertionError(f"{gr00t_only} should not be a pi05 recipe flag")
    # a plain valid parse still works
    assert E.build_parser().parse_args(
        ["--task", "x", "--num_episodes", "1", "--checkpoint", "/c"]) is not None


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
        checkpoint=Path("lerobot/pi05_libero"), eval_script="pi05_libero_eval.py",
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
            "pilot": "pi05",                 # handled — not forwarded
            "checkpoint": "lerobot/pi05_libero",  # handled — passed explicitly
            "runner": "libero",              # handled — not forwarded
            "capture_video": True,           # handled — video via --video_dir
            "task_id": 0,
            "host": "10.0.0.9",
            "port": 8000,
            "n_action_steps": 10,
        },
    )
    argv = build_pi05_libero_argv(
        spec=task, checkpoint=Path("lerobot/pi05_libero"), video_dir=Path("/out/videos"),
    )
    # Contract flags present.
    assert argv[:6] == [
        "--task", "libero_object", "--num_episodes", "10",
        "--checkpoint", "lerobot/pi05_libero",
    ]
    assert "--video_dir" in argv and "/out/videos" in argv
    # Passthrough config forwarded verbatim (snake_case).
    assert argv[argv.index("--host") + 1] == "10.0.0.9"
    assert argv[argv.index("--port") + 1] == "8000"
    assert argv[argv.index("--n_action_steps") + 1] == "10"
    # Handled keys NOT forwarded as flags.
    for handled in ("--pilot", "--runner", "--capture_video"):
        assert handled not in argv


def test_build_argv_omits_video_dir_when_none() -> None:
    argv = build_pi05_libero_argv(
        spec=_eval_task(config={"task_id": 1}), checkpoint=Path("/c"), video_dir=None,
    )
    assert "--video_dir" not in argv
    assert argv[argv.index("--task_id") + 1] == "1"


# ---------------------------------------------------------------------------
# The parsed argv wires back into the recipe (what the runner builds, it accepts).
# ---------------------------------------------------------------------------

def test_argv_parses_back_into_the_recipe() -> None:
    task = _eval_task(config={"task_id": 2, "host": "1.2.3.4", "port": 9000,
                              "n_action_steps": 10, "translation_only": "true"})
    argv = build_pi05_libero_argv(
        spec=task, checkpoint=Path("/ckpt"), video_dir=None,
    )
    args = E.build_parser().parse_args(argv)
    assert args.task == "libero_object"
    assert args.task_id == 2
    assert args.host == "1.2.3.4"
    assert args.port == 9000
    assert args.n_action_steps == 10
    assert args.translation_only is True


# ---------------------------------------------------------------------------
# Recovery wiring — flags parse, round-trip through the builder, shadow mode
# logs without intervening.
# ---------------------------------------------------------------------------


def test_parser_accepts_recovery_flags_defaulting_off() -> None:
    args = E.build_parser().parse_args(
        ["--task", "libero_object", "--checkpoint", "/ckpt"]
    )
    assert args.recovery is False and args.shadow_mode is False
    assert args.recovery_mode == "command" and args.recovery_dir == ""


def test_argv_roundtrips_recovery_keys_through_builder() -> None:
    task = _eval_task(config={
        "pilot": "pi05",
        "checkpoint": "lerobot/pi05_libero",
        "shadow_mode": "true",
        "specialist_base_url": "http://judge:8002/v1",
        "poll_every_chunks": 3,
        "log_actions": "true",
    })
    argv = build_pi05_libero_argv(
        spec=task, checkpoint=Path("lerobot/pi05_libero"), video_dir=None,
    )
    args = E.build_parser().parse_args(argv)
    assert args.shadow_mode is True
    assert args.specialist_base_url == "http://judge:8002/v1"
    assert args.poll_every_chunks == 3 and args.log_actions is True


class _RecordingChunkPilot:
    """make_pi05_pilot stand-in: constant-motion chunks + call recording."""

    def __init__(self, n: int = 4) -> None:
        self._n = n
        self._cursor = n  # empty buffer: first act "re-queries"
        self.flush_calls = 0
        self.reset_calls = 0

    @property
    def steps_remaining(self) -> int:
        return 0 if self._cursor >= self._n else self._n - self._cursor

    def reset(self) -> None:
        self.reset_calls += 1
        self._cursor = self._n

    def flush(self) -> None:
        self.flush_calls += 1
        self._cursor = self._n

    def act(self, raw_obs: Any, instruction: str) -> Any:
        import numpy as np
        if self._cursor >= self._n:
            self._cursor = 0
        self._cursor += 1
        return np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


class _StaticLiberoEnv:
    def _obs(self) -> dict[str, Any]:
        import numpy as np
        return {
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
            "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((4, 4, 3), dtype=np.uint8),
        }

    def reset(self) -> dict[str, Any]:
        return self._obs()

    def set_init_state(self, state: Any) -> None:
        pass

    def step(self, action: Any):
        return self._obs(), 0.0, False, {}

    def close(self) -> None:
        pass


def test_run_eval_shadow_mode_logs_but_never_intervenes(
    monkeypatch: Any, tmp_path: Path
) -> None:
    import odyssey.runners.models.pi05 as pi05_mod
    from odyssey.runners.evals import libero as runner_mod

    pilot = _RecordingChunkPilot(n=4)
    monkeypatch.setattr(pi05_mod, "make_pi05_pilot", lambda **kwargs: pilot)
    monkeypatch.setattr(
        runner_mod, "_make_libero_env", lambda *a, **k: (_StaticLiberoEnv(), object(), [0])
    )
    monkeypatch.setattr(runner_mod, "_resolve_libero_instruction", lambda *a, **k: "pick")

    args = E.build_parser().parse_args([
        "--task", "libero_object", "--checkpoint", "/ckpt",
        "--num_episodes", "1", "--max_steps_per_episode", "30",
        "--num_warmup_steps", "0", "--n_action_steps", "4",
        "--shadow_mode", "true", "--stuck_window_steps", "6",
        "--recovery_dir", str(tmp_path),
    ])
    summary = E.run_eval(args)

    recovery = summary["metrics"]["recovery"]
    assert recovery["shadow_triggers"] >= 1  # the static arm was detected
    assert recovery["flushes"] == 0
    assert pilot.flush_calls == 0  # never intervened
    events = (tmp_path / "recovery_events.jsonl").read_text().splitlines()
    assert events and all(json.loads(e)["applied"] is False
                          for e in events if "applied" in json.loads(e))
