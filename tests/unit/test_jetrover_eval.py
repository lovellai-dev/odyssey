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
import contextlib
import importlib.util
import json
import queue
import struct
import sys
import threading
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
    # @dataclass resolves the defining module through sys.modules.
    sys.modules[spec.name] = module
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


def test_timed_out_prompt_does_not_consume_next_answer(jetrover: Any) -> None:
    # Review #101: the orphaned read of an expired prompt used to take the
    # answer typed for the next prompt. Simulate a shared terminal: input_fn
    # blocks until the "operator" types a line.
    import queue
    import threading

    stdin: queue.Queue[str] = queue.Queue()
    console = jetrover.OperatorConsole(lambda prompt: stdin.get())

    assert jetrover.prompt_operator(0, 0.05, console) is False  # nobody answered

    # The operator answers episode 1 only after its prompt is up.
    threading.Timer(0.1, stdin.put, args=("y",)).start()
    assert jetrover.prompt_operator(1, 5.0, console) is True


def test_answer_typed_between_prompts_is_discarded(jetrover: Any) -> None:
    # A late answer to an expired prompt must not leak into the next one.
    import queue
    import threading

    stdin: queue.Queue[str] = queue.Queue()
    console = jetrover.OperatorConsole(lambda prompt: stdin.get())

    assert jetrover.prompt_operator(0, 0.05, console) is False
    stdin.put("y")  # too late for episode 0, typed while no prompt is open
    time.sleep(0.05)
    threading.Timer(0.1, stdin.put, args=("n",)).start()
    assert jetrover.prompt_operator(1, 5.0, console) is False


def test_closed_stdin_fails_loudly(jetrover: Any) -> None:
    def closed(prompt: str) -> str:
        raise EOFError

    console = jetrover.OperatorConsole(closed)
    with pytest.raises(RuntimeError, match="stdin closed"):
        console.ask("? ", timeout=1.0)
    with pytest.raises(RuntimeError, match="stdin is closed"):
        console.ask("? ", timeout=1.0)


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
# Episode reset (review #101 P1: reset bypassed the motion limit)
# ---------------------------------------------------------------------------

BUS_SERVO_FUNC = 5  # ros_robot_controller_sdk.PacketFunction.PACKET_FUNC_BUS_SERVO


class _FakeBoard:
    """Stand-in for ros_robot_controller_sdk.Board: servos jump to commands.

    Models the SDK's receive path (Hiwonder/JetRover@ef96538): reception starts
    disabled, a read request is answered through a 1-slot ``bus_servo_queue``
    only once ``enable_reception()`` was called, and reads share
    ``servo_read_lock``. ``silent`` servos never answer.
    """

    def __init__(
        self,
        pulses: dict[int, int],
        stuck: set[int] | None = None,
        silent: set[int] | None = None,
    ) -> None:
        self.pulses = dict(pulses)
        self.stuck = stuck or set()
        self.silent = silent or set()
        self.commands: list[tuple[float, list[list[int]]]] = []
        self.enable_recv = False
        self.servo_read_lock = threading.Lock()
        self.bus_servo_queue: queue.Queue[bytes] = queue.Queue(maxsize=1)

    def enable_reception(self, enable: bool = True) -> None:
        self.enable_recv = enable

    def bus_servo_set_position(self, duration: float, targets: list[list[int]]) -> None:
        self.commands.append((duration, [list(t) for t in targets]))
        for servo_id, pulse in targets:
            if servo_id not in self.stuck:
                self.pulses[servo_id] = pulse

    def buf_write(self, func: int, data: list[int]) -> None:
        assert func == BUS_SERVO_FUNC
        cmd, servo_id = data
        if not self.enable_recv or servo_id in self.silent:
            return  # the SDK's receive thread drops the reply
        reply = struct.pack("<BBbh", servo_id, cmd, 0, self.pulses[servo_id])
        with contextlib.suppress(queue.Full):  # as packet_report_serial_servo does
            self.bus_servo_queue.put_nowait(reply)


def _hiwonder_arm_at(jetrover: Any, radians: float, **board_kwargs: Any) -> tuple[Any, Any]:
    arm = jetrover.HiwonderArm(5)
    pulse = arm._to_pulse(radians)
    board = _FakeBoard({i: pulse for i in [1, 2, 3, 4, 5, 10]}, **board_kwargs)
    board.enable_reception()
    arm._board = board
    arm._bus_servo_func = BUS_SERVO_FUNC
    return arm, board


def test_reset_from_nonzero_pose_is_rate_limited(jetrover: Any) -> None:
    # The reviewer's repro: arm at 1.2 rad, max_steps=0. Before the fix the
    # board received a single all-zero target (a full 1.2 rad jump).
    arm, board = _hiwonder_arm_at(jetrover, 1.2)
    homing = jetrover.HomingConfig(max_joint_delta=0.05)

    steps = jetrover.run_episode(
        arm,
        jetrover.MockPolicy(),
        task_description="t",
        max_steps=0,
        action_horizon=8,
        max_joint_delta=0.01,
        step_delay=0.0,
        homing=homing,
    )

    assert steps == 0
    assert len(board.commands) > 1  # walked home, not jumped
    one_pulse = 1 / arm.PULSE_PER_RAD
    previous = {i: 1.2 for i in [1, 2, 3, 4, 5, 10]}
    for _duration, targets in board.commands:
        for servo_id, pulse in targets:
            rad = (pulse - arm.PULSE_CENTER) / arm.PULSE_PER_RAD
            # every axis, gripper included, moves at most one homing step
            assert abs(rad - previous[servo_id]) <= 0.05 + 2 * one_pulse
            previous[servo_id] = rad
    obs = arm.get_observation()
    assert max(abs(v) for v in [*obs["joints"], obs["gripper"]]) <= homing.tolerance


def test_reset_gives_up_on_a_stalled_servo(jetrover: Any) -> None:
    arm, board = _hiwonder_arm_at(jetrover, 0.6, stuck={3})
    homing = jetrover.HomingConfig(max_joint_delta=0.1, max_steps=20)

    with pytest.raises(RuntimeError, match="did not reach the home pose"):
        jetrover.home_arm(arm, homing, step_delay=0.0, console=None)
    assert len(board.commands) == 20  # bounded, then stopped


def test_operator_reset_sends_no_motion_and_needs_explicit_token(jetrover: Any) -> None:
    # A bare Enter or a stray "y" (e.g. a late answer to an expired score
    # prompt) must not confirm that the arm has been placed.
    arm, board = _hiwonder_arm_at(jetrover, 1.2)
    lines = iter(["", "y", "READY"])
    console = jetrover.OperatorConsole(lambda prompt: next(lines))

    sent = jetrover.home_arm(
        arm, jetrover.HomingConfig(mode="operator"), step_delay=0.0, console=console
    )

    assert sent == 0
    assert board.commands == []
    assert next(lines, None) is None  # all three lines were needed


def test_non_finite_policy_action_aborts_before_sending(jetrover: Any) -> None:
    class NanPolicy(jetrover.PolicyClient):  # type: ignore[name-defined]
        def get_action(self, obs: dict[str, Any], task_description: str) -> list[list[float]]:
            return [[float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0]]

    arm = jetrover.MockArm(5)
    with pytest.raises(RuntimeError, match="non-finite"):
        jetrover.run_episode(
            arm,
            NanPolicy(),
            task_description="t",
            max_steps=5,
            action_horizon=8,
            max_joint_delta=0.1,
            step_delay=0.0,
        )
    assert arm.actions_received == []


def test_aborted_run_still_writes_completed_episodes(
    jetrover: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_run_episode = jetrover.run_episode
    calls = {"n": 0}

    def flaky_run_episode(*args: Any, **kwargs: Any) -> int:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("Arm did not reach the home pose")
        return int(real_run_episode(*args, **kwargs))

    monkeypatch.setattr(jetrover, "run_episode", flaky_run_episode)
    out_json = tmp_path / "metrics.json"
    with pytest.raises(RuntimeError, match="home pose"):
        jetrover.main(
            [
                "--checkpoint", "/c", "--out-json", str(out_json),
                "--arm_backend", "mock", "--policy_backend", "mock",
                "--num_episodes", "3", "--max_steps", "4", "--scorer", "none",
            ]
        )
    payload = json.loads(out_json.read_text())
    assert len(payload["metrics"]["episodes"]) == 1
    assert "home pose" in payload["metrics"]["aborted"]


def test_hiwonder_gripper_is_measured_not_remembered(jetrover: Any) -> None:
    arm, _board = _hiwonder_arm_at(jetrover, 0.4)
    assert arm.get_observation()["gripper"] == pytest.approx(0.4, abs=0.01)


# ---------------------------------------------------------------------------
# Hiwonder SDK receive path (review #101 P1: reception disabled, unbounded read)
# ---------------------------------------------------------------------------

def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch, board: _FakeBoard) -> None:
    import types

    sdk = types.ModuleType("ros_robot_controller_sdk")
    sdk.Board = lambda: board  # type: ignore[attr-defined]
    sdk.PacketFunction = types.SimpleNamespace(  # type: ignore[attr-defined]
        PACKET_FUNC_BUS_SERVO=BUS_SERVO_FUNC
    )
    monkeypatch.setitem(sys.modules, "ros_robot_controller_sdk", sdk)


def test_hiwonder_connect_enables_reception(
    jetrover: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Board() defaults to reception disabled; without enabling it no read
    # reply is ever delivered.
    board = _FakeBoard({i: 500 for i in [1, 2, 3, 4, 5, 10]})
    _install_fake_sdk(monkeypatch, board)
    arm = jetrover.HiwonderArm(5)

    arm.connect()

    assert board.enable_recv
    obs = arm.get_observation()
    assert obs["joints"] == pytest.approx([0.0] * 5)
    assert obs["gripper"] == pytest.approx(0.0)


def test_hiwonder_read_without_reception_times_out(jetrover: Any) -> None:
    arm, board = _hiwonder_arm_at(jetrover, 0.0)
    board.enable_reception(False)  # the SDK default
    arm.READ_TIMEOUT_S = 0.05

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="did not reply"):
        arm.get_observation()
    assert time.monotonic() - start < 1.0  # bounded, not an indefinite wait


def test_hiwonder_missing_reply_is_bounded_and_releases_the_lock(jetrover: Any) -> None:
    arm, board = _hiwonder_arm_at(jetrover, 0.2, silent={3})
    arm.READ_TIMEOUT_S = 0.05

    with pytest.raises(RuntimeError, match="Bus servo 3 did not reply"):
        arm.get_observation()

    # No thread is left holding the read lock: later reads still work.
    assert not board.servo_read_lock.locked()
    assert arm._read_radians(4) == pytest.approx(0.2, abs=0.01)


def test_hiwonder_homing_aborts_on_missing_reply(jetrover: Any) -> None:
    # home_max_steps cannot bound a read that never returns; the read deadline must.
    arm, board = _hiwonder_arm_at(jetrover, 0.6, silent={10})
    arm.READ_TIMEOUT_S = 0.05

    with pytest.raises(RuntimeError, match="did not reply"):
        jetrover.home_arm(arm, jetrover.HomingConfig(), step_delay=0.0, console=None)
    assert board.commands == []  # never moved from an unread pose


def test_hiwonder_discards_stale_and_foreign_replies(jetrover: Any) -> None:
    # A late reply to an earlier timed-out read must not be taken as this one.
    arm, board = _hiwonder_arm_at(jetrover, 0.3)
    board.bus_servo_queue.put_nowait(struct.pack("<BBbh", 2, 0x05, 0, 0))

    assert arm._read_radians(1) == pytest.approx(0.3, abs=0.01)


def test_hiwonder_read_error_status_is_not_a_position(jetrover: Any) -> None:
    arm, board = _hiwonder_arm_at(jetrover, 0.3)
    board.buf_write = lambda func, data: board.bus_servo_queue.put_nowait(  # type: ignore[method-assign]
        struct.pack("<BBbh", data[1], data[0], -1, 0)
    )

    with pytest.raises(RuntimeError, match="read error"):
        arm._read_radians(1)


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--max_joint_delta", "0"),
        ("--max_joint_delta", "15"),
        ("--home_max_joint_delta", "-0.1"),
        ("--home_max_joint_delta", "1.0"),
        ("--home_tolerance", "0"),
        ("--home_max_steps", "0"),
        ("--control_hz", "0"),
        ("--action_horizon", "0"),  # would re-query the policy forever
        ("--arm_dof", "0"),
        ("--num_episodes", "0"),
        ("--max_steps", "-1"),
    ],
)
def test_motion_limits_are_validated(jetrover: Any, flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        jetrover.parse_args(["--checkpoint", "/c", "--out-json", "/o", flag, value])


# ---------------------------------------------------------------------------
# ROS 2 backend wire contract (review #101 P1: wrong message type)
# ---------------------------------------------------------------------------

class _Msg:
    """Attribute bag standing in for a generated ROS message class."""


class _ServoPosition(_Msg):
    pass


class _ServosPosition(_Msg):
    pass


class _JointState(_Msg):
    pass


class _Image(_Msg):
    pass


SENSOR_DATA_QOS = object()  # stands in for rclpy.qos.qos_profile_sensor_data
CAMERA_TOPIC = "/depth_cam/rgb/image_raw"


def _publish_image(arm: Any, value: int = 7) -> None:
    _, callback = arm._node.subscriptions[CAMERA_TOPIC]
    msg = _Image()
    msg.height, msg.width = 2, 3
    msg.data = bytes([value]) * (2 * 3 * 3)
    callback(msg)


class _FakeNode:
    def __init__(self, name: str) -> None:
        self.publishers: list[tuple[Any, str, _FakePublisher]] = []
        self.subscriptions: dict[str, tuple[Any, Any]] = {}
        self.qos: dict[str, Any] = {}

    def create_publisher(self, msg_type: Any, topic: str, qos: int) -> _FakePublisher:
        pub = _FakePublisher()
        self.publishers.append((msg_type, topic, pub))
        return pub

    def create_subscription(self, msg_type: Any, topic: str, cb: Any, qos: Any) -> None:
        self.subscriptions[topic] = (msg_type, cb)
        self.qos[topic] = qos

    def destroy_node(self) -> None:
        pass


class _FakePublisher:
    def __init__(self) -> None:
        self.published: list[Any] = []

    def publish(self, msg: Any) -> None:
        self.published.append(msg)


@pytest.fixture
def fake_ros(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import types

    state: dict[str, Any] = {"on_spin": None}

    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda: None  # type: ignore[attr-defined]
    rclpy.shutdown = lambda: None  # type: ignore[attr-defined]

    def spin_once(node: Any, timeout_sec: float = 0.0) -> None:
        if state["on_spin"] is not None:
            state["on_spin"]()

    rclpy.spin_once = spin_once  # type: ignore[attr-defined]
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = _FakeNode  # type: ignore[attr-defined]
    rclpy_qos = types.ModuleType("rclpy.qos")
    rclpy_qos.qos_profile_sensor_data = SENSOR_DATA_QOS  # type: ignore[attr-defined]
    sensor = types.ModuleType("sensor_msgs")
    sensor_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msg.Image = _Image  # type: ignore[attr-defined]
    sensor_msg.JointState = _JointState  # type: ignore[attr-defined]
    servo = types.ModuleType("servo_controller_msgs")
    servo_msg = types.ModuleType("servo_controller_msgs.msg")
    servo_msg.ServoPosition = _ServoPosition  # type: ignore[attr-defined]
    servo_msg.ServosPosition = _ServosPosition  # type: ignore[attr-defined]
    for name, module in {
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "rclpy.qos": rclpy_qos,
        "sensor_msgs": sensor,
        "sensor_msgs.msg": sensor_msg,
        "servo_controller_msgs": servo,
        "servo_controller_msgs.msg": servo_msg,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state


def test_ros2_publishes_vendor_servos_position(jetrover: Any, fake_ros: Any) -> None:
    arm = jetrover.Ros2Arm(5)
    arm.command_duration = 0.2
    arm.connect()

    msg_type, topic, pub = arm._node.publishers[0]
    assert msg_type is _ServosPosition  # what the stock controller_manager subscribes to
    assert topic == "/servo_controller"
    assert arm._node.subscriptions["/controller_manager/joint_states"][0] is _JointState

    arm.send_action([0.1, -0.2, 0.3, -0.4, 0.5, 0.6])

    (msg,) = pub.published
    assert isinstance(msg, _ServosPosition)
    assert msg.position_unit == "rad"
    assert msg.duration == pytest.approx(0.2)
    assert [(p.id, p.position) for p in msg.position] == [
        (1, pytest.approx(0.1)),
        (2, pytest.approx(-0.2)),
        (3, pytest.approx(0.3)),
        (4, pytest.approx(-0.4)),
        (5, pytest.approx(0.5)),
        (10, pytest.approx(0.6)),  # gripper servo
    ]
    assert all(isinstance(p, _ServoPosition) for p in msg.position)


def test_ros2_reads_measured_joint_state(jetrover: Any, fake_ros: Any) -> None:
    arm = jetrover.Ros2Arm(5)
    arm.connect()
    _, callback = arm._node.subscriptions["/controller_manager/joint_states"]

    def publish_state() -> None:
        msg = _JointState()
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "r_joint"]
        msg.position = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        callback(msg)
        _publish_image(arm)

    fake_ros["on_spin"] = publish_state
    obs = arm.get_observation()
    assert obs["joints"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert obs["gripper"] == pytest.approx(0.6)
    assert obs["image"].shape == (2, 3, 3)


def test_ros2_requires_a_fresh_joint_state_per_observation(
    jetrover: Any, fake_ros: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Measured state must postdate the previous command: a cached message is
    # not accepted as the arm's current pose.
    arm = jetrover.Ros2Arm(5)
    arm.connect()
    _, callback = arm._node.subscriptions["/controller_manager/joint_states"]
    monkeypatch.setattr(arm, "FRESH_STATE_TIMEOUT_S", 0.05)
    poses = iter([0.1, 0.2])

    def publish_next() -> None:
        value = next(poses, None)
        if value is None:
            return  # the controller went silent
        msg = _JointState()
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "r_joint"]
        msg.position = [value] * 6
        callback(msg)
        _publish_image(arm)

    fake_ros["on_spin"] = publish_next
    assert arm.get_observation()["joints"][0] == pytest.approx(0.1)
    assert arm.get_observation()["joints"][0] == pytest.approx(0.2)
    with pytest.raises(RuntimeError, match="No fresh joint state"):
        arm.get_observation()


def test_ros2_rejects_missing_joint(jetrover: Any, fake_ros: Any) -> None:
    arm = jetrover.Ros2Arm(5)
    arm.connect()
    _, callback = arm._node.subscriptions["/controller_manager/joint_states"]

    def publish_partial() -> None:
        msg = _JointState()
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5"]  # no gripper
        msg.position = [0.0] * 5
        callback(msg)

    fake_ros["on_spin"] = publish_partial
    with pytest.raises(RuntimeError, match="r_joint"):
        arm.get_observation()


# ---------------------------------------------------------------------------
# GR00T N1.7 modality config (review #101 P1: old API, no registration)
# ---------------------------------------------------------------------------

MODALITY_CONFIG = REPO_ROOT / "examples" / "quickstart-jetrover" / "jetrover_modality_config.py"


def _install_fake_gr00t(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub exactly the N1.7 names so100_config.py imports (Isaac-GR00T@51d4c89).

    Any import of a name N1.7 does not export (e.g. the old
    ``gr00t.data.dataset.ModalityConfig``) fails here as it would upstream.
    """
    import dataclasses
    import enum
    import types

    registry: dict[str, Any] = {}

    @dataclasses.dataclass
    class ModalityConfig:
        delta_indices: list[int]
        modality_keys: list[str]
        action_configs: list[Any] | None = None

    @dataclasses.dataclass
    class ActionConfig:
        rep: Any
        type: Any
        format: Any

    class EmbodimentTag(enum.Enum):
        NEW_EMBODIMENT = "new_embodiment"

    def register_modality_config(config: dict, embodiment_tag: Any) -> None:
        assert embodiment_tag.value not in registry
        registry[embodiment_tag.value] = config

    types_mod = types.ModuleType("gr00t.data.types")
    types_mod.ModalityConfig = ModalityConfig  # type: ignore[attr-defined]
    types_mod.ActionConfig = ActionConfig  # type: ignore[attr-defined]
    types_mod.ActionRepresentation = enum.Enum(  # type: ignore[attr-defined]
        "ActionRepresentation", "RELATIVE ABSOLUTE"
    )
    types_mod.ActionType = enum.Enum("ActionType", "NON_EEF EEF")  # type: ignore[attr-defined]
    types_mod.ActionFormat = enum.Enum("ActionFormat", "DEFAULT")  # type: ignore[attr-defined]
    tags_mod = types.ModuleType("gr00t.data.embodiment_tags")
    tags_mod.EmbodimentTag = EmbodimentTag  # type: ignore[attr-defined]
    emb_mod = types.ModuleType("gr00t.configs.data.embodiment_configs")
    emb_mod.register_modality_config = register_modality_config  # type: ignore[attr-defined]
    dataset_mod = types.ModuleType("gr00t.data.dataset")  # N1.7: empty package
    for name, module in {
        "gr00t": types.ModuleType("gr00t"),
        "gr00t.data": types.ModuleType("gr00t.data"),
        "gr00t.data.dataset": dataset_mod,
        "gr00t.data.types": types_mod,
        "gr00t.data.embodiment_tags": tags_mod,
        "gr00t.configs": types.ModuleType("gr00t.configs"),
        "gr00t.configs.data": types.ModuleType("gr00t.configs.data"),
        "gr00t.configs.data.embodiment_configs": emb_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return registry


def _import_like_launch_finetune(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    # gr00t/experiment/launch_finetune.py: sys.path.append(parent);
    # importlib.import_module(stem) — imported for its registration side effect.
    import importlib

    monkeypatch.syspath_prepend(str(path.parent))
    monkeypatch.delitem(sys.modules, path.stem, raising=False)
    importlib.import_module(path.stem)
    monkeypatch.delitem(sys.modules, path.stem, raising=False)


def test_modality_config_registers_new_embodiment(
    jetrover: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _install_fake_gr00t(monkeypatch)

    _import_like_launch_finetune(monkeypatch, MODALITY_CONFIG)

    config = registry["new_embodiment"]
    # Keys must line up with what the eval client sends to the policy server.
    assert config["video"].modality_keys == [jetrover.VIDEO_KEY]
    assert config["state"].modality_keys == ["single_arm", "gripper"]
    assert config["action"].modality_keys == ["single_arm", "gripper"]
    assert config["language"].modality_keys == [jetrover.LANGUAGE_KEY]
    assert len(config["action"].action_configs) == 2  # one per action key
    assert config["action"].delta_indices == list(range(16))


@pytest.mark.skipif(
    not __import__("os").getenv("ISAAC_GR00T_REPO_PATH"),
    reason="ISAAC_GR00T_REPO_PATH not set — real GR00T registration skipped",
)
def test_modality_config_registers_against_real_gr00t(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("gr00t.configs.data.embodiment_configs")
    from gr00t.configs.data import embodiment_configs

    monkeypatch.setattr(embodiment_configs, "MODALITY_CONFIGS", {})
    _import_like_launch_finetune(monkeypatch, MODALITY_CONFIG)
    assert "new_embodiment" in embodiment_configs.MODALITY_CONFIGS


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


# ---------------------------------------------------------------------------
# ROS 2 camera (review #101: a missing camera fed the policy black frames)
# ---------------------------------------------------------------------------

def _ros2_arm_with_joint_states(jetrover: Any, fake_ros: Any, *, camera: Any) -> Any:
    """Ros2Arm whose controller publishes on every spin; ``camera()`` decides
    whether a frame arrives on that spin."""
    arm = jetrover.Ros2Arm(5)
    arm.connect()
    _, callback = arm._node.subscriptions["/controller_manager/joint_states"]

    def spin() -> None:
        msg = _JointState()
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "r_joint"]
        msg.position = [0.0] * 6
        callback(msg)
        if camera():
            _publish_image(arm)

    fake_ros["on_spin"] = spin
    return arm


def test_ros2_camera_uses_sensor_data_qos(jetrover: Any, fake_ros: Any) -> None:
    # Camera drivers publish best-effort; a reliable subscriber never receives.
    arm = jetrover.Ros2Arm(5)
    arm.connect()
    assert arm._node.qos[CAMERA_TOPIC] is SENSOR_DATA_QOS


def test_ros2_without_camera_frame_fails_instead_of_going_blind(
    jetrover: Any, fake_ros: Any
) -> None:
    # Wrong/namespaced topic or QoS mismatch: no frame ever arrives.
    arm = _ros2_arm_with_joint_states(jetrover, fake_ros, camera=lambda: False)
    arm.FIRST_IMAGE_TIMEOUT_S = 0.05

    with pytest.raises(RuntimeError, match="No live camera frame"):
        arm.get_observation()


def test_ros2_stalled_camera_fails(jetrover: Any, fake_ros: Any) -> None:
    # A frame that stopped updating is a frozen view, not the arm's camera.
    frames = iter([True])
    arm = _ros2_arm_with_joint_states(
        jetrover, fake_ros, camera=lambda: next(frames, False)
    )
    arm.IMAGE_MAX_AGE_S = 0.05

    assert arm.get_observation()["image"] is not None
    time.sleep(0.1)
    with pytest.raises(RuntimeError, match="No live camera frame"):
        arm.get_observation()


def test_ros2_eval_never_sends_black_frames(jetrover: Any, fake_ros: Any) -> None:
    # The policy is only ever queried with a real frame on ros2.
    arm = _ros2_arm_with_joint_states(jetrover, fake_ros, camera=lambda: False)
    arm.FIRST_IMAGE_TIMEOUT_S = 0.05
    queried: list[Any] = []

    class RecordingPolicy(jetrover.PolicyClient):  # type: ignore[name-defined]
        def get_action(self, obs: dict[str, Any], task_description: str) -> list[list[float]]:
            queried.append(obs)
            return [[0.0] * 6]

    with pytest.raises(RuntimeError, match="No live camera frame"):
        jetrover.run_episode(
            arm,
            RecordingPolicy(),
            task_description="t",
            max_steps=5,
            action_horizon=1,
            max_joint_delta=0.1,
            step_delay=0.0,
            homing=jetrover.HomingConfig(mode="bounded"),
        )
    assert queried == []
