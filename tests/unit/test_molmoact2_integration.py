"""Integration tests for ``pilot: molmoact2`` — the layers wired together.

``test_molmoact2_pilot.py`` and ``test_molmoact2_eval.py`` pin each piece in
isolation (codec, transforms, adapter wiring, runner dispatch, argv builder).
These tests run the pieces *together*, still WITHOUT a GPU, LIBERO, torch or a
served checkpoint:

  * the eval recipe's real ``run_eval`` loop driving a fake LIBERO env against a
    real local ``/act`` HTTP server (stdlib ``http.server`` + our json_numpy
    codec): warmup, 180° image flip on the wire, 8-D state, chunk replay +
    re-query, per-episode ``pilot.reset()``, success detection, video naming,
    and a mid-sweep server error that must not drop the final ``ODYSSEY_RESULT``;
  * the recipe's stdout round-tripping through the runner's
    ``EvalProtocolCollector`` + ``summarize`` (what ``LiberoRunner`` scores);
  * runner -> recipe argv: the shipped example ``mission.yaml`` (and every
    documented knob) produces an argv the recipe's parser accepts;
  * the shipped ``host_server_libero.py`` reads exactly the wire keys the client
    sends (AST-level, so no torch/fastapi needed).

All tests are named ``test_molmoact2_integration_*`` (the ``-k molmoact2`` gate).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from odyssey.runners.evals.isaac_lab import EvalProtocolCollector, summarize
from odyssey.runners.evals.libero import build_molmoact2_libero_argv
from odyssey.spec import EvaluationTask, EvaluationType
from odyssey.spec.loader import load_mission

np = pytest.importorskip("numpy")

_REPO = Path(__file__).resolve().parents[2]
_EVAL_SCRIPT = _REPO / "src" / "odyssey" / "runners" / "evals" / "molmoact2_libero_eval.py"
_EXAMPLE_DIR = _REPO / "examples" / "quickstart-molmoact2"

_INSTRUCTION = "pick up the black bowl and place it on the plate"
_HORIZON = 10  # what the fake server advertises on GET /act
_IMG = 6  # tiny frames keep the json_numpy payloads small


def _load_recipe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("molmoact2_libero_eval", _EVAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fake /act server — speaks the host_server_libero.py wire protocol.
# ---------------------------------------------------------------------------

@dataclass
class _ActServer:
    host: str
    port: int
    requests: list[dict[str, Any]] = field(default_factory=list)
    fail_queries: set[int] = field(default_factory=set)  # 0-based POST indices -> 500

    @staticmethod
    def chunk(query: int) -> Any:
        """Chunk for POST #``query``: dx = query*100 + k pins which row was replayed."""
        c = np.zeros((_HORIZON, 7), np.float32)
        c[:, 0] = query * 100 + np.arange(_HORIZON)
        c[:, 3:6] = 0.01
        c[:, 6] = 1.0  # close
        return c


@pytest.fixture
def act_server() -> Iterator[_ActServer]:
    from odyssey.runners.evals import molmoact2_transforms as t

    state = _ActServer(host="127.0.0.1", port=0)

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            return

        def _reply(self, code: int, body: dict[str, Any]) -> None:
            data = t.json_numpy_dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._reply(200, {"status": "ok", "norm_tag": "libero",
                              "action_horizon": _HORIZON})

        def do_POST(self) -> None:
            body = t.json_numpy_loads(self.rfile.read(int(self.headers["Content-Length"])))
            query = len(state.requests)
            state.requests.append(body)
            if query in state.fail_queries:
                self._reply(500, {"error": "inference failed: CUDA OOM"})
                return
            self._reply(200, {"actions": state.chunk(query), "dt_ms": 1.0})

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    state.port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Fake LIBERO env — the robosuite/LIBERO surface run_eval touches.
# ---------------------------------------------------------------------------

def _asym_frame(offset: int) -> Any:
    """An asymmetric HxWx3 frame, so a missing/extra 180° flip is detectable."""
    return (np.arange(_IMG * _IMG * 3, dtype=np.int64).reshape(_IMG, _IMG, 3) + offset) % 256


class _FakeLiberoEnv:
    """Records every env.step; solves episode ``e`` after ``solve_at[e]`` policy steps."""

    def __init__(self, *, warmup: int, solve_at: list[int | None]) -> None:
        self.warmup = warmup
        self.solve_at = solve_at
        self.episode = -1
        self.t = 0
        self.actions: list[list[list[float]]] = []  # per episode, every env.step action
        self.init_states: list[Any] = []
        self.closed = False

    def _obs(self) -> dict[str, Any]:
        return {
            "agentview_image": _asym_frame(0).astype(np.uint8),
            "robot0_eye_in_hand_image": _asym_frame(7).astype(np.uint8),
            "robot0_eef_pos": np.array([0.1, -0.2, 1.05]),
            # 90° about z, xyzw (robosuite convention)
            "robot0_eef_quat": np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
            "robot0_gripper_qpos": np.array([0.039, -0.039]),
        }

    def reset(self) -> dict[str, Any]:
        self.episode += 1
        self.t = 0
        self.actions.append([])
        return self._obs()

    def set_init_state(self, init_state: Any) -> dict[str, Any]:
        self.init_states.append(init_state)
        return self._obs()

    def step(self, action: list[float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.actions[self.episode].append(list(action))
        self.t += 1
        target = self.solve_at[self.episode]
        done = target is not None and self.t == self.warmup + target
        return self._obs(), float(done), done, {}

    def close(self) -> None:
        self.closed = True


def _run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    server: _ActServer,
    *,
    solve_at: list[int | None],
    extra_argv: list[str] | None = None,
    max_steps: int = 25,
    warmup: int = 2,
) -> tuple[dict[str, Any], _FakeLiberoEnv, list[str]]:
    recipe = _load_recipe()
    env = _FakeLiberoEnv(warmup=warmup, solve_at=solve_at)
    task = SimpleNamespace(language=_INSTRUCTION)
    init_states = ["init-0", "init-1"]

    def _fake_make_env(suite: str, task_id: int, cfg: dict[str, Any], *, horizon: int):
        # robosuite's horizon must cover warmup + the whole step budget.
        assert horizon >= warmup + max_steps
        return env, task, init_states

    monkeypatch.setattr("odyssey.runners.evals.libero._make_libero_env", _fake_make_env)
    args = recipe.build_parser().parse_args([
        "--task", "libero_spatial",
        "--num_episodes", str(len(solve_at)),
        "--checkpoint", "allenai/MolmoAct2-LIBERO",
        "--host", server.host,
        "--port", str(server.port),
        "--max_steps_per_episode", str(max_steps),
        "--num_warmup_steps", str(warmup),
        *(extra_argv or []),
    ])
    summary = recipe.run_eval(args)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("ODYSSEY_")]
    return summary, env, lines


# ---------------------------------------------------------------------------
# 1. run_eval end-to-end: fake env <-> recipe <-> adapter <-> HTTP <-> fake /act.
# ---------------------------------------------------------------------------

def test_molmoact2_integration_run_eval_end_to_end(monkeypatch, capsys, act_server) -> None:
    # ep1 solves after 13 policy steps (2 queries); ep2 never solves (25 steps, 3 queries).
    summary, env, lines = _run(monkeypatch, capsys, act_server, solve_at=[13, None])

    # --- protocol + scoring ---
    assert [ln.split(" ", 1)[0] for ln in lines] == [
        "ODYSSEY_EPISODE", "ODYSSEY_EPISODE", "ODYSSEY_RESULT",
    ]
    assert summary["success_rate"] == 0.5
    assert summary["metrics"]["instruction"] == _INSTRUCTION
    assert summary["metrics"]["benchmark"] == "libero_spatial[task=0]"
    assert env.closed
    assert env.init_states == ["init-0", "init-1"]  # one init state per episode, cycled

    # --- query cadence: horizon adopted from GET /act, reset() flushes per episode ---
    # Without the per-episode reset, ep2 would first drain ep1's 7 leftover rows
    # and only need 2 fresh queries (4 total); the rows below pin it too.
    assert len(act_server.requests) == 2 + 3
    ep1, ep2 = env.actions
    assert all(a == [0, 0, 0, 0, 0, 0, -1.0] for a in ep1[:2] + ep2[:2])  # warmup no-ops
    ep1_dx = [a[0] for a in ep1[2:]]
    ep2_dx = [a[0] for a in ep2[2:]]
    assert ep1_dx == [*range(10), 100, 101, 102]
    # ep2 starts on a FRESH chunk (query #2, row 0), not ep1's leftover row 103.
    assert ep2_dx[:1] == [200] and ep2_dx[10:11] == [300] and len(ep2_dx) == 25

    # --- wire: what the /act server actually received ---
    req = act_server.requests[0]
    assert set(req) == {"image", "wrist_image", "state", "instruction", "num_steps"}
    assert req["instruction"] == _INSTRUCTION
    assert req["num_steps"] == 10
    assert req["image"].dtype == np.uint8 and req["image"].shape == (_IMG, _IMG, 3)
    np.testing.assert_array_equal(req["image"], _asym_frame(0)[::-1, ::-1])  # 180° flip
    np.testing.assert_array_equal(req["wrist_image"], _asym_frame(7)[::-1, ::-1])
    assert req["state"].dtype == np.float32 and req["state"].shape == (8,)
    np.testing.assert_allclose(
        req["state"], [0.1, -0.2, 1.05, 0.0, 0.0, np.pi / 2, 0.039, -0.039], atol=1e-5
    )
    # The 7-D action reaches env.step untouched (gripper +1 = close, rotation kept).
    np.testing.assert_allclose(ep1[2], [0, 0, 0, 0.01, 0.01, 0.01, 1.0], atol=1e-6)


def test_molmoact2_integration_stdout_scores_through_runner_collector(
    monkeypatch, capsys, act_server
) -> None:
    _, _, lines = _run(monkeypatch, capsys, act_server, solve_at=[3, None, 5], max_steps=8)

    collector = EvalProtocolCollector()
    events = [collector.parse(ln) for ln in lines]
    assert [e["step"] for e in events if e] == [
        "episode_complete", "episode_complete", "episode_complete", "result_received",
    ]
    assert [ep["success"] for ep in collector.episodes] == [True, False, True]

    spec = EvaluationTask(
        name="molmoact2-libero-eval",
        evaluation_type=EvaluationType.LIBERO,
        benchmark_name="libero_spatial",
        num_episodes=3,
        config={"pilot": "molmoact2"},
    )
    result = summarize(
        collector=collector, spec=spec, checkpoint=Path("allenai/MolmoAct2-LIBERO"),
        eval_script=str(_EVAL_SCRIPT),
    )
    assert result["success_rate"] == pytest.approx(2 / 3, abs=1e-4)  # summary rounds to 4dp
    assert result["num_episodes"] == 3


def test_molmoact2_integration_server_error_fails_episode_not_sweep(
    monkeypatch, capsys, act_server
) -> None:
    # The very first POST /act 500s: ep1 must be recorded as a fail, ep2 must
    # still run (fresh chunk) and the final ODYSSEY_RESULT must still be emitted.
    act_server.fail_queries = {0}
    summary, env, lines = _run(monkeypatch, capsys, act_server, solve_at=[4, 4])

    episodes = [json.loads(ln.split(" ", 1)[1]) for ln in lines if "EPISODE" in ln]
    assert [ep["success"] for ep in episodes] == [False, True]
    assert lines[-1].startswith("ODYSSEY_RESULT ")
    assert summary["success_rate"] == 0.5
    assert len(env.actions[0]) == 2  # only the warmup ran before the failed query
    assert [a[0] for a in env.actions[1][2:]] == [100, 101, 102, 103]


def test_molmoact2_integration_server_down_fails_fast(monkeypatch, capsys) -> None:
    # Forgot to start host_server_libero.py: the recipe must raise (rc != 0 -> task
    # FAILED), NOT sweep every episode as a "fail" and report a fake 0% success.
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead_port = s.getsockname()[1]  # closed on exit -> nothing listening

    with pytest.raises(RuntimeError, match=r"unreachable.*host_server_libero\.py"):
        _run(monkeypatch, capsys, _ActServer(host="127.0.0.1", port=dead_port),
             solve_at=[3, 3])
    assert not [ln for ln in capsys.readouterr().out.splitlines()
                if ln.startswith("ODYSSEY_")]


def test_molmoact2_integration_all_episodes_aborted_fails_task(
    monkeypatch, capsys, act_server
) -> None:
    # Server passes the startup probe, then every POST /act errors: that is an
    # infra failure, not a 0% policy — no ODYSSEY_RESULT, and the recipe raises.
    act_server.fail_queries = set(range(100))
    with pytest.raises(RuntimeError, match=r"all 2 episodes aborted.*CUDA OOM"):
        _run(monkeypatch, capsys, act_server, solve_at=[3, 3])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("ODYSSEY_")]
    assert [ln.split(" ", 1)[0] for ln in lines] == ["ODYSSEY_EPISODE", "ODYSSEY_EPISODE"]


def test_molmoact2_integration_config_knobs_reach_env(monkeypatch, capsys, act_server) -> None:
    summary, env, _ = _run(
        monkeypatch, capsys, act_server, solve_at=[None], max_steps=12,
        extra_argv=["--n_action_steps", "4", "--num_steps", "3",
                    "--invert_gripper", "True", "--flip_images", "false"],
    )
    assert summary["success_rate"] == 0.0
    # n_action_steps=4 overrides the server's horizon of 10: 12 steps -> 3 queries.
    assert len(act_server.requests) == 3
    assert [a[0] for a in env.actions[0][2:]] == [0, 1, 2, 3, 100, 101, 102, 103,
                                                  200, 201, 202, 203]
    assert all(a[6] == -1.0 for a in env.actions[0][2:])  # invert_gripper: +1 -> -1
    assert act_server.requests[0]["num_steps"] == 3
    # flip_images=false: frames go out exactly as the env rendered them.
    np.testing.assert_array_equal(act_server.requests[0]["image"], _asym_frame(0))


def test_molmoact2_integration_video_per_episode_named_by_outcome(
    monkeypatch, capsys, act_server, tmp_path
) -> None:
    saved: list[tuple[str, int, Any]] = []

    def _fake_save(frames: list[Any], path: Path, fps: int = 24) -> Path:
        saved.append((path.name, len(frames), frames[0]))
        return path

    monkeypatch.setattr("odyssey.runners.video.save_rollout_video", _fake_save)
    _run(monkeypatch, capsys, act_server, solve_at=[3, None], max_steps=5,
         extra_argv=["--video_dir", str(tmp_path / "videos")])

    assert [(name, n) for name, n, _ in saved] == [
        ("episode_01_PASS.mp4", 3), ("episode_02_FAIL.mp4", 5),
    ]
    # Video frames get the same 180° flip as the pilot input (else upside down).
    np.testing.assert_array_equal(saved[0][2], _asym_frame(0)[::-1, ::-1])


# ---------------------------------------------------------------------------
# 2. Runner -> recipe argv: what LiberoRunner builds, the recipe must accept.
# ---------------------------------------------------------------------------

def test_molmoact2_integration_example_mission_argv_parses_in_recipe(tmp_path) -> None:
    mission = load_mission(_EXAMPLE_DIR / "mission.yaml")
    [task] = mission.tasks
    assert isinstance(task, EvaluationTask)
    assert (task.config or {}).get("pilot") == "molmoact2"

    argv = build_molmoact2_libero_argv(
        spec=task, checkpoint=Path("allenai/MolmoAct2-LIBERO"), video_dir=tmp_path / "v",
    )
    args = _load_recipe().build_parser().parse_args(argv)  # SystemExit on unknown flags

    assert args.task == "libero_spatial"
    assert args.num_episodes == 5
    assert args.task_id == 0
    assert (args.host, args.port) == ("127.0.0.1", 8000)
    assert args.n_action_steps == 0  # adopt the server's action_horizon
    assert args.video_dir == str(tmp_path / "v")


def test_molmoact2_integration_all_documented_knobs_parse_in_recipe() -> None:
    # Every knob the example/README documents, typed the way YAML hands them over
    # (bools as Python bools -> "True"/"False" strings on the argv).
    spec = EvaluationTask(
        name="molmoact2-libero-eval",
        evaluation_type=EvaluationType.LIBERO,
        benchmark_name="libero_object",
        num_episodes=2,
        config={
            "pilot": "molmoact2", "checkpoint": "allenai/MolmoAct2-LIBERO",
            "capture_video": True, "task_id": 3, "host": "10.0.0.5", "port": 8123,
            "n_action_steps": 8, "num_steps": 5, "invert_gripper": True,
            "translation_only": False, "flip_images": False,
            "max_steps_per_episode": 300, "num_warmup_steps": 5,
            "camera_height": 224, "camera_width": 224,
            "task_instruction": "put the bowl on the plate", "strict_instruction": True,
        },
    )
    argv = build_molmoact2_libero_argv(spec=spec, checkpoint=Path("ckpt"), video_dir=None)
    assert "--pilot" not in argv and "--capture_video" not in argv

    args = _load_recipe().build_parser().parse_args(argv)
    assert (args.task_id, args.host, args.port) == (3, "10.0.0.5", 8123)
    assert (args.n_action_steps, args.num_steps) == (8, 5)
    assert args.invert_gripper is True and args.translation_only is False
    assert args.flip_images is False and args.strict_instruction is True
    assert (args.camera_height, args.camera_width) == (224, 224)
    assert (args.max_steps_per_episode, args.num_warmup_steps) == (300, 5)


# ---------------------------------------------------------------------------
# 3. The shipped /act server reads exactly what the client sends.
# ---------------------------------------------------------------------------

def _server_contract() -> tuple[set[str], set[str], dict[str, Any], set[str]]:
    """(required POST keys, optional POST keys, module constants, GET /act keys)."""
    tree = ast.parse((_EXAMPLE_DIR / "host_server_libero.py").read_text())
    required: set[str] = set()
    optional: set[str] = set()
    consts: dict[str, Any] = {}
    get_keys: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "payload" and isinstance(node.slice, ast.Constant)):
            required.add(node.slice.value)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "get" and isinstance(node.func.value, ast.Name)
              and node.func.value.id == "payload" and isinstance(node.args[0], ast.Constant)):
            optional.add(node.args[0].value)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)
              and isinstance(node.value, ast.Constant)):
            consts[node.targets[0].id] = node.value.value
        elif isinstance(node, ast.Dict) and any(
            isinstance(k, ast.Constant) and k.value == "action_horizon" for k in node.keys
        ):
            get_keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
    return required, optional, consts, get_keys


def test_molmoact2_integration_host_server_matches_client_contract() -> None:
    from odyssey.runners.evals import molmoact2_transforms as t

    required, optional, consts, get_keys = _server_contract()
    req = t.build_molmoact2_libero_request(
        image=np.zeros((4, 4, 3)), wrist_image=np.zeros((4, 4, 3)),
        eef_pos=[0, 0, 0], eef_quat_xyzw=[0, 0, 0, 1], gripper_qpos=[0, 0],
        instruction="x",
    )
    # Every key the server requires is sent; every key sent is one it reads.
    assert required == {t.MOLMOACT2_IMAGE_KEY, t.MOLMOACT2_WRIST_IMAGE_KEY,
                        t.MOLMOACT2_STATE_KEY, t.MOLMOACT2_INSTRUCTION_KEY}
    assert set(req) <= required | optional
    assert consts["STATE_DIM"] == req[t.MOLMOACT2_STATE_KEY].shape[0] == 8
    assert consts["NORM_TAG"] == "libero"
    assert consts["DEFAULT_NUM_STEPS"] == t.MOLMOACT2_DEFAULT_FLOW_STEPS
    # GET /act advertises the horizon make_molmoact2_pilot adopts.
    assert "action_horizon" in get_keys
