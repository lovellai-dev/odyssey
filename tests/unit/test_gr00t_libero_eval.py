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
         "--serve_checkpoint", "true", "--embodiment_tag", "LIBERO_PANDA"]
    )
    assert args.task_id == 3
    assert args.host == "10.0.0.2"
    assert args.port == 6000
    assert args.n_action_steps == 8
    assert args.pos_scale == 0.5
    assert args.serve_checkpoint is True
    assert args.embodiment_tag == "LIBERO_PANDA"


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


def test_sim_policy_wrapper_defaults_on_and_toggles() -> None:
    # GR00T-N1.7-LIBERO is served through the sim policy wrapper (NVIDIA's recipe).
    on = E.build_parser().parse_args(["--task", "x", "--num_episodes", "1", "--checkpoint", "/c"])
    assert on.sim_policy_wrapper is True
    off = E.build_parser().parse_args(
        ["--task", "x", "--num_episodes", "1", "--checkpoint", "/c",
         "--sim_policy_wrapper", "false"])
    assert off.sim_policy_wrapper is False


def test_build_server_command_wraps_sim_policy_for_libero() -> None:
    from odyssey.runners.evals import _gr00t_server as S
    on = S.build_server_command(
        checkpoint="/c/libero_object", embodiment_tag="LIBERO_PANDA", port=5555,
        sim_policy_wrapper=True)
    assert "--use-sim-policy-wrapper" in on
    assert on[on.index("--embodiment-tag") + 1] == "LIBERO_PANDA"
    # Default stays off (the Isaac/DROID nested-obs path relies on this).
    off = S.build_server_command(checkpoint="/c", embodiment_tag="LIBERO_PANDA", port=5555)
    assert "--use-sim-policy-wrapper" not in off


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
            "embodiment_tag": "LIBERO_PANDA",
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
    assert argv[argv.index("--embodiment_tag") + 1] == "LIBERO_PANDA"
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
                              "embodiment_tag": "LIBERO_PANDA", "n_action_steps": 8})
    argv = build_gr00t_libero_argv(
        spec=task, checkpoint=Path("/ckpt"), video_dir=None,
    )
    args = E.build_parser().parse_args(argv)
    assert args.task == "libero_object"
    assert args.task_id == 2
    assert args.serve_checkpoint is True
    assert args.n_action_steps == 8


# ---------------------------------------------------------------------------
# ChunkPilotAdapter migration — the adapter-driven loop is action-for-action
# and query-for-query identical to the recipe's historical inline chunk loop.
# ---------------------------------------------------------------------------


class _FakeChunkClient:
    """msgpack-client stand-in: records wire obs, returns tagged (tuple) chunks."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def get_action(self, observation: Any) -> Any:
        self.calls.append(observation)
        return (f"chunk-{len(self.calls)}", {"latency_ms": 1})  # tuple, like the real client


class _CountingEnv:
    """Gym-4-tuple env: obs is a step counter; done fires at ``done_at`` steps."""

    def __init__(self, *, done_at: int | None = None) -> None:
        self._done_at = done_at
        self._steps = 0

    def reset(self) -> dict[str, Any]:
        self._steps = 0
        return {"tick": 0}

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._steps += 1
        done = self._done_at is not None and self._steps >= self._done_at
        return {"tick": self._steps}, 0.0, done, {}


def _fake_decode(chunk: Any, k: int, *, translation_only: bool = False) -> Any:
    return (chunk, k, translation_only)


def _patched_pilot(monkeypatch: Any, client: _FakeChunkClient, *, n: int) -> Any:
    class _T:
        gr00t_action_to_libero = staticmethod(_fake_decode)

    monkeypatch.setattr(E, "_transforms", lambda: _T)
    monkeypatch.setattr(
        E, "_build_obs",
        lambda obs, instr, *, image_key, wrist_image_key, flip: ("wire", obs["tick"], instr),
    )
    args = E.build_parser().parse_args(
        ["--task", "libero_object", "--checkpoint", "/ckpt", "--n_action_steps", str(n)]
    )
    return E._make_gr00t_pilot(client, args)


def _drive_adapter(pilot: Any, env: _CountingEnv, *, max_steps: int) -> list[Any]:
    """The migrated run_eval loop shape (per-step act, break on done/max)."""
    actions: list[Any] = []
    obs = env.reset()
    pilot.reset()
    step, success = 0, False
    while step < max_steps and not success:
        action = pilot.act(obs, "pick up the milk")
        actions.append(action)
        obs, _r, done, _i = env.step(action)
        step += 1
        if done:
            success = True
    return actions


def _drive_inline(client: _FakeChunkClient, env: _CountingEnv, *, n: int,
                  max_steps: int) -> list[Any]:
    """The recipe's PRE-migration inline chunk loop, reimplemented verbatim."""
    actions: list[Any] = []
    obs = env.reset()
    step, success = 0, False
    while step < max_steps and not success:
        observation = ("wire", obs["tick"], "pick up the milk")
        result = client.get_action(observation)
        chunk = result[0] if isinstance(result, tuple) else result
        for k in range(n):
            if step >= max_steps:
                break
            action = _fake_decode(chunk, k)
            actions.append(action)
            obs, _r, done, _i = env.step(action)
            step += 1
            if done:
                success = True
                break
    return actions


def _assert_equivalent(monkeypatch: Any, *, n: int, max_steps: int,
                       done_at: int | None) -> None:
    old_client = _FakeChunkClient()
    old = _drive_inline(old_client, _CountingEnv(done_at=done_at), n=n, max_steps=max_steps)

    new_client = _FakeChunkClient()
    pilot = _patched_pilot(monkeypatch, new_client, n=n)
    new = _drive_adapter(pilot, _CountingEnv(done_at=done_at), max_steps=max_steps)

    assert new == old  # identical action sequence (chunk tag, cursor, flags)
    assert new_client.calls == old_client.calls  # identical query count AND wire obs


def test_gr00t_pilot_adapter_matches_inline_loop_clean_drain(monkeypatch: Any) -> None:
    _assert_equivalent(monkeypatch, n=4, max_steps=8, done_at=None)


def test_gr00t_pilot_adapter_matches_inline_loop_done_midchunk(monkeypatch: Any) -> None:
    _assert_equivalent(monkeypatch, n=4, max_steps=20, done_at=6)


def test_gr00t_pilot_adapter_matches_inline_loop_max_steps_midchunk(monkeypatch: Any) -> None:
    _assert_equivalent(monkeypatch, n=4, max_steps=6, done_at=None)


def test_gr00t_pilot_reset_between_episodes_drops_partial_chunk(monkeypatch: Any) -> None:
    client = _FakeChunkClient()
    pilot = _patched_pilot(monkeypatch, client, n=4)

    _drive_adapter(pilot, _CountingEnv(done_at=2), max_steps=10)  # ends mid-chunk
    assert len(client.calls) == 1
    _drive_adapter(pilot, _CountingEnv(done_at=2), max_steps=10)  # fresh episode

    # Episode 2 re-queried instead of replaying episode 1's leftover actions.
    assert len(client.calls) == 2
    assert [obs for _w, obs, _i in client.calls] == [0, 0]  # both queries at reset obs


# ---------------------------------------------------------------------------
# Recovery wiring — flags default off; enabled path runs end-to-end on fakes.
# ---------------------------------------------------------------------------


def _base_argv(*extra: str) -> list[str]:
    return ["--task", "libero_object", "--checkpoint", "/ckpt", *extra]


def test_parser_accepts_recovery_flags_defaulting_off() -> None:
    args = E.build_parser().parse_args(_base_argv())
    assert args.recovery is False and args.shadow_mode is False
    assert args.recovery_mode == "command"
    assert args.log_actions is False and args.recovery_dir == ""

    on = E.build_parser().parse_args(_base_argv(
        "--recovery", "true", "--recovery_mode", "teleport",
        "--specialist_base_url", "http://judge:8002/v1",
        "--stuck_window_steps", "6", "--log_actions", "true",
        "--recovery_dir", "/out/recovery",
    ))
    assert on.recovery is True and on.recovery_mode == "teleport"
    assert on.specialist_base_url == "http://judge:8002/v1"
    assert on.stuck_window_steps == 6 and on.log_actions is True


class _StaticLiberoEnv:
    """LIBERO-shaped env whose arm never moves — a guaranteed kinematic stuck."""

    def __init__(self) -> None:
        self.steps = 0
        self.restored: list[Any] = []

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
        self.restored.append(state)

    def step(self, action: Any):
        self.steps += 1
        return self._obs(), 0.0, False, {}

    def close(self) -> None:
        pass


def _run_recovery_eval(monkeypatch: Any, tmp_path: Path, *extra: str) -> tuple[dict, Any]:
    """Drive run_eval end-to-end on fakes; return (summary, fake client)."""
    import numpy as np

    from odyssey.runners.evals import libero as runner_mod

    def fake_decode(chunk: Any, k: int, *, translation_only: bool = False) -> Any:
        return np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

    class _T:
        gr00t_action_to_libero = staticmethod(fake_decode)

    client = _FakeChunkClient()

    class _S:
        @staticmethod
        def connect_policy_client(*, host: str, port: int, timeout_ms: int) -> Any:
            return client

    env = _StaticLiberoEnv()
    monkeypatch.setattr(E, "_transforms", lambda: _T)
    monkeypatch.setattr(E, "_server", lambda: _S)
    monkeypatch.setattr(
        E, "_build_obs",
        lambda obs, instr, *, image_key, wrist_image_key, flip: {"wire": instr},
    )
    monkeypatch.setattr(
        runner_mod, "_make_libero_env", lambda *a, **k: (env, object(), [0])
    )
    monkeypatch.setattr(runner_mod, "_resolve_libero_instruction", lambda *a, **k: "pick")

    args = E.build_parser().parse_args(_base_argv(
        "--num_episodes", "1", "--max_steps_per_episode", "30",
        "--num_warmup_steps", "0", "--n_action_steps", "4", *extra,
    ))
    return E.run_eval(args), client


def test_run_eval_recovery_off_is_inert(monkeypatch: Any, tmp_path: Path) -> None:
    summary, client = _run_recovery_eval(monkeypatch, tmp_path)
    assert "recovery" not in summary["metrics"]
    assert len(client.calls) == 8  # ceil(30 / 4) chunks, no extra queries
    assert not list(tmp_path.iterdir())  # nothing written anywhere


def test_run_eval_kinematic_stuck_flushes_and_writes_events(
    monkeypatch: Any, tmp_path: Path
) -> None:
    summary, client = _run_recovery_eval(
        monkeypatch, tmp_path,
        "--recovery", "true", "--stuck_window_steps", "6",
        "--recovery_settle_steps", "0", "--max_recoveries", "2",
        "--recovery_dir", str(tmp_path),
    )
    recovery = summary["metrics"]["recovery"]
    assert recovery["flushes"] >= 1  # the static arm tripped tier 1
    assert len(client.calls) > 8  # each flush forces an extra re-query

    events_file = tmp_path / "recovery_events.jsonl"
    assert events_file.exists()
    events = [json.loads(line) for line in events_file.read_text().splitlines()]
    assert any(e["kind"] == "flush" and e["cause"] == "kinematic" for e in events)


def test_run_eval_log_actions_writes_npz_corpus(monkeypatch: Any, tmp_path: Path) -> None:
    import numpy as np

    summary, _client = _run_recovery_eval(
        monkeypatch, tmp_path,
        "--shadow_mode", "true", "--log_actions", "true",
        "--recovery_dir", str(tmp_path),
    )
    assert "recovery" in summary["metrics"]
    npz_files = sorted((tmp_path / "rollouts").glob("*.npz"))
    assert len(npz_files) == 1 and npz_files[0].name == "episode_01_FAIL.npz"
    data = np.load(npz_files[0])
    assert data["frames"].shape == (30, 4, 4, 3)
    assert data["actions"].shape == (30, 7)
