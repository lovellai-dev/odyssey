#!/usr/bin/env python3
"""Real-hardware eval for the Hiwonder JetRover arm (``evaluation_type: custom``).

Implements the launch contract of ``src/odyssey/runners/evals/custom.py``::

    <python> eval_jetrover.py --checkpoint <path> --out-json <path> \\
        --arm_backend {mock,ros2,hiwonder} --policy_backend {mock,zmq} [...]

and writes the out-json metrics object the runner expects (``success_rate`` when
the operator scorer ran, metric-only otherwise).

Split of responsibilities:

* ``ArmBackend`` — talks to the arm: joint positions + gripper in, camera
  frame + *measured* joint state out. ``mock`` is deterministic and
  dependency-free (CI); ``ros2`` / ``hiwonder`` are thin adapters over the stock
  JetRover ``controller_manager`` (``servo_controller_msgs/ServosPosition``) or
  Hiwonder's on-board bus-servo SDK.
* ``home_arm`` — the episode reset. It never sends an unbounded move: it either
  walks the arm home from the measured pose in ``--home_max_joint_delta`` steps
  (``--reset_mode bounded``) or asks the operator to place the arm
  (``--reset_mode operator``).
* ``OperatorConsole`` — the one stdin reader shared by every prompt, so an
  expired prompt can never swallow the answer meant for the next one.
* ``PolicyClient`` — produces action chunks. ``zmq`` is a msgpack/ZMQ REQ client
  against a pre-started GR00T policy server (``python -m gr00t.eval.run_gr00t_server
  --embodiment_tag new_embodiment ...`` on the GPU box — GR00T inference does not
  fit on the Jetson, so serving is always remote); ``mock`` emits fixed deltas.

The script never loads the checkpoint itself: ``--checkpoint`` is recorded into
the metrics so runs are attributable, and the README tells the user to serve the
same path. Import-time deps are stdlib-only; numpy/msgpack/zmq/rclpy and the
Hiwonder SDK are imported lazily inside the backends that need them.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import queue
import struct
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Hiwonder markets the JetRover arm as "6DOF" *counting the gripper*: per the
# vendor docs (Robot Arm Control Course), servo IDs 1-5 position the arm
# (pan-tilt base + 3 body joints + wrist) and ID 10 is the gripper — a 5-joint
# kinematic chain + 1 gripper, i.e. 6-dim actions. If your unit (or dataset)
# differs, override with --arm_dof; every backend adapts.
DEFAULT_ARM_DOF = 5

# GR00T-server observation keys; must match the modality config the checkpoint
# was finetuned with (jetrover_modality_config.py in this directory).
VIDEO_KEY = "front"
LANGUAGE_KEY = "annotation.human.task_description"

# Hard ceilings on the per-step joint limits accepted from the command line (rad).
# A typo like ``--max_joint_delta 15`` must fail at parse time, not on the arm.
MAX_JOINT_DELTA_CEILING = 0.5
HOME_JOINT_DELTA_CEILING = 0.2
OPERATOR_READY_TOKEN = "ready"


# --------------------------------------------------------------------------- arms
class ArmBackend:
    """Minimal arm interface: observations out, (arm_dof + 1)-float actions in."""

    # Duration the controller is given to reach each commanded target. ``main``
    # sets it to the step period so a step never asks for a faster move than
    # the control rate implies.
    command_duration = 0.2

    def __init__(self, arm_dof: int) -> None:
        self.arm_dof = arm_dof
        self.action_dim = arm_dof + 1  # joints + gripper

    def connect(self) -> None:
        raise NotImplementedError

    def home_pose(self) -> list[float]:
        """Episode start target: ``arm_dof`` joints + gripper (rad).

        Zeros = every servo at its calibrated centre, which is what the vendor
        controller treats as 0 rad. Reached by ``home_arm``, never sent raw.
        """
        return [0.0] * self.action_dim

    def get_observation(self) -> dict[str, Any]:
        """Return the *measured* state:
        ``{"joints": [arm_dof floats], "gripper": float, "image": ndarray|None}``."""
        raise NotImplementedError

    def send_action(self, action: list[float]) -> None:
        """Execute one action step: ``arm_dof`` joint positions (rad) + gripper command."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class MockArm(ArmBackend):
    """Deterministic, dependency-free arm for CI and dry runs."""

    def __init__(self, arm_dof: int) -> None:
        super().__init__(arm_dof)
        self.joints = [0.0] * arm_dof
        self.gripper = 0.0
        self.actions_received: list[list[float]] = []

    def connect(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        return {"joints": list(self.joints), "gripper": self.gripper, "image": None}

    def send_action(self, action: list[float]) -> None:
        self.actions_received.append(list(action))
        self.joints = list(action[: self.arm_dof])
        self.gripper = float(action[self.arm_dof])

    def close(self) -> None:
        pass


class Ros2Arm(ArmBackend):
    """ROS 2 adapter for the stock JetRover ``controller_manager``.

    The vendor node (``servo_controller`` package) subscribes to
    ``servo_controller_msgs/ServosPosition`` on ``servo_controller`` and
    publishes measured positions as ``sensor_msgs/JointState`` on
    ``~/joint_states`` (= ``/controller_manager/joint_states``), both in rad
    when ``position_unit == "rad"`` — the controller applies each joint's own
    calibration (centre, direction, limits) to convert to servo pulses, so no
    pulse mapping is duplicated here. A ``JointTrajectory`` publisher would not
    reach it: ROS 2 topics are typed. Verify the topics with ``ros2 topic list``
    if your image namespaces them.
    """

    JOINT_STATE_TOPIC = "/controller_manager/joint_states"
    ARM_COMMAND_TOPIC = "/servo_controller"
    CAMERA_TOPIC = "/depth_cam/rgb/image_raw"
    POSITION_UNIT = "rad"
    GRIPPER_JOINT_NAME = "r_joint"
    GRIPPER_SERVO_ID = 10
    FIRST_STATE_TIMEOUT_S = 5.0
    FRESH_STATE_TIMEOUT_S = 1.0  # the vendor node publishes at ~50 Hz
    FIRST_IMAGE_TIMEOUT_S = 5.0
    IMAGE_MAX_AGE_S = 1.0  # older than this = camera stalled, not a live view

    def __init__(self, arm_dof: int) -> None:
        super().__init__(arm_dof)
        self.arm_joint_names = [f"joint{i}" for i in range(1, arm_dof + 1)]
        self.arm_servo_ids = list(range(1, arm_dof + 1))
        self._node: Any = None
        self._last_joint_state: Any = None
        self._joint_state_count = 0
        self._last_image: Any = None
        self._last_image_time = 0.0

    def connect(self) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, JointState
        from servo_controller_msgs.msg import ServoPosition, ServosPosition

        rclpy.init()
        self._rclpy = rclpy
        self._ServoPosition = ServoPosition
        self._ServosPosition = ServosPosition
        self._node = Node("odyssey_jetrover_eval")
        self._pub = self._node.create_publisher(ServosPosition, self.ARM_COMMAND_TOPIC, 1)
        self._node.create_subscription(
            JointState, self.JOINT_STATE_TOPIC, self._on_joint_state, 1
        )
        # Camera drivers publish best-effort (sensor-data QoS); a default
        # reliable subscription would silently never receive from them.
        self._node.create_subscription(
            Image, self.CAMERA_TOPIC, self._on_image, qos_profile_sensor_data
        )

    def _on_joint_state(self, msg: Any) -> None:
        self._last_joint_state = msg
        self._joint_state_count += 1

    def _on_image(self, msg: Any) -> None:
        self._last_image = msg
        self._last_image_time = time.monotonic()

    def _image_is_fresh(self) -> bool:
        return (
            self._last_image is not None
            and time.monotonic() - self._last_image_time <= self.IMAGE_MAX_AGE_S
        )

    def _wait_for_camera_frame(self) -> None:
        # The policy is visuomotor: running it on no frame (black) or on a frozen
        # one would score a blind policy as a normal eval. Fail instead.
        timeout = (
            self.FIRST_IMAGE_TIMEOUT_S if self._last_image is None else self.IMAGE_MAX_AGE_S
        )
        deadline = time.monotonic() + timeout
        while not self._image_is_fresh() and time.monotonic() < deadline:
            self._spin()
        if not self._image_is_fresh():
            raise RuntimeError(
                f"No live camera frame on {self.CAMERA_TOPIC} within {timeout}s — "
                "is the depth camera driver running? Check the topic name with "
                "`ros2 topic list` (it may be namespaced)."
            )

    def _spin(self, seconds: float = 0.05) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=seconds)

    def _wait_for_fresh_joint_state(self) -> None:
        # Measured state must postdate the previous command, or homing and the
        # action clamp would reason about where the arm *was*.
        seen = self._joint_state_count
        timeout = (
            self.FIRST_STATE_TIMEOUT_S
            if self._last_joint_state is None
            else self.FRESH_STATE_TIMEOUT_S
        )
        deadline = time.monotonic() + timeout
        while self._joint_state_count == seen and time.monotonic() < deadline:
            self._spin()
        if self._joint_state_count == seen:
            raise RuntimeError(
                f"No fresh joint state on {self.JOINT_STATE_TOPIC} within "
                f"{timeout}s — is the jetrover controller_manager running?"
            )

    def get_observation(self) -> dict[str, Any]:
        import numpy as np

        self._wait_for_fresh_joint_state()
        name_to_pos = dict(
            zip(self._last_joint_state.name, self._last_joint_state.position, strict=False)
        )
        # A missing joint must not read as 0 rad: homing would think it is
        # already home and the clamp would rate-limit from a fictional pose.
        missing = [
            n for n in [*self.arm_joint_names, self.GRIPPER_JOINT_NAME] if n not in name_to_pos
        ]
        if missing:
            raise RuntimeError(
                f"{self.JOINT_STATE_TOPIC} has no position for {missing} — check "
                "--arm_dof against the controller's joints"
            )
        joints = [float(name_to_pos[n]) for n in self.arm_joint_names]
        gripper = float(name_to_pos[self.GRIPPER_JOINT_NAME])

        self._wait_for_camera_frame()
        msg = self._last_image
        image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, -1
        )[..., :3]
        return {"joints": joints, "gripper": gripper, "image": image}

    def send_action(self, action: list[float]) -> None:
        msg = self._ServosPosition()
        msg.duration = float(self.command_duration)
        msg.position_unit = self.POSITION_UNIT
        servo_ids = [*self.arm_servo_ids, self.GRIPPER_SERVO_ID]
        positions = []
        for servo_id, value in zip(servo_ids, action[: self.action_dim], strict=True):
            position = self._ServoPosition()
            position.id = servo_id
            position.position = float(value)
            positions.append(position)
        msg.position = positions
        self._pub.publish(msg)
        self._spin(0.0)

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._rclpy.shutdown()


class HiwonderArm(ArmBackend):
    """Direct bus-servo adapter via Hiwonder's on-board controller SDK.

    Uses the ``ros_robot_controller_sdk`` module that ships on JetRover images
    (the same board API the vendor demos use). Servo IDs follow the vendor docs
    (arm = IDs 1..arm_dof, gripper = ID 10); direction signs and the rad→pulse
    mapping are the vendor defaults — verify against your unit before running
    with the workspace occupied.
    """

    GRIPPER_SERVO_ID = 10
    PULSE_CENTER = 500
    PULSE_PER_RAD = 500 / 2.094  # Hiwonder bus servos: 0..1000 pulses over ±120°
    # Wire constants of the SDK's own position read (``bus_servo_read_position``):
    # sub-command 0x05, reply ``(servo_id, cmd, status, pulse)``.
    READ_POSITION_CMD = 0x05
    READ_POSITION_REPLY = "<BBbh"
    READ_TIMEOUT_S = 0.5  # per servo; a healthy bus answers in a few ms

    def __init__(self, arm_dof: int) -> None:
        super().__init__(arm_dof)
        self.arm_servo_ids = list(range(1, arm_dof + 1))
        self._board: Any = None
        self._bus_servo_func: Any = None

    def connect(self) -> None:
        from ros_robot_controller_sdk import (  # type: ignore[import-not-found]
            Board,
            PacketFunction,
        )

        self._bus_servo_func = PacketFunction.PACKET_FUNC_BUS_SERVO
        self._board = Board()
        # Board() starts with reception disabled: its receive thread drops every
        # reply, so no servo read could ever complete. The vendor demos enable it.
        self._board.enable_reception()

    def _to_pulse(self, radians: float) -> int:
        pulse = self.PULSE_CENTER + radians * self.PULSE_PER_RAD
        return int(min(1000, max(0, round(pulse))))

    def _read_radians(self, servo_id: int) -> float:
        """One position-read transaction with a deadline.

        Mirrors the SDK's ``bus_servo_read_and_unpack`` (same lock, request and
        reply format) but never blocks indefinitely: the SDK waits on its reply
        queue with no timeout, so a missing reply would hang the eval — and
        ``home_max_steps`` cannot bound a read that never returns. Doing the
        transaction here also means a timeout leaves no thread holding the read
        lock. A late reply to a timed-out read is drained before the next
        request, and replies for another servo are ignored.
        """
        board = self._board
        deadline = time.monotonic() + self.READ_TIMEOUT_S
        if not board.servo_read_lock.acquire(timeout=self.READ_TIMEOUT_S):
            raise RuntimeError(f"Bus servo read lock busy — servo {servo_id} not read")
        try:
            with contextlib.suppress(queue.Empty):
                board.bus_servo_queue.get_nowait()  # stale reply from a timed-out read
            board.buf_write(self._bus_servo_func, [self.READ_POSITION_CMD, servo_id])
            while True:
                remaining = deadline - time.monotonic()
                try:
                    if remaining <= 0:
                        raise queue.Empty
                    data = board.bus_servo_queue.get(timeout=remaining)
                except queue.Empty:
                    raise RuntimeError(
                        f"Bus servo {servo_id} did not reply within "
                        f"{self.READ_TIMEOUT_S}s — check the servo cable and that "
                        "no other process (e.g. controller_manager) owns the bus"
                    ) from None
                if len(data) != struct.calcsize(self.READ_POSITION_REPLY):
                    continue
                reply_id, cmd, status, pulse = struct.unpack(self.READ_POSITION_REPLY, data)
                if reply_id == servo_id and cmd == self.READ_POSITION_CMD:
                    break
        finally:
            board.servo_read_lock.release()
        if status != 0:
            raise RuntimeError(f"Bus servo {servo_id} reported a read error ({status})")
        return (float(pulse) - self.PULSE_CENTER) / self.PULSE_PER_RAD

    def get_observation(self) -> dict[str, Any]:
        # The bus-servo protocol reports positions on request; camera access goes
        # through ROS/OpenCV and is out of scope for this backend (state-only).
        # The gripper is read back too: a remembered command would be wrong on
        # the first episode and after any stall.
        return {
            "joints": [self._read_radians(servo_id) for servo_id in self.arm_servo_ids],
            "gripper": self._read_radians(self.GRIPPER_SERVO_ID),
            "image": None,
        }

    def send_action(self, action: list[float]) -> None:
        targets = [
            [servo_id, self._to_pulse(action[i])]
            for i, servo_id in enumerate(self.arm_servo_ids)
        ]
        targets.append([self.GRIPPER_SERVO_ID, self._to_pulse(action[self.arm_dof])])
        self._board.bus_servo_set_position(self.command_duration, targets)

    def close(self) -> None:
        self._board = None


# ------------------------------------------------------------------------ policies
class PolicyClient:
    def get_action(self, obs: dict[str, Any], task_description: str) -> list[list[float]]:
        """Return a chunk of action steps, each ``[j1..j<arm_dof>, gripper]``."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class MockPolicy(PolicyClient):
    """Fixed small joint deltas — exercises the loop without a server."""

    CHUNK_LEN = 4
    DELTA = 0.01

    def get_action(self, obs: dict[str, Any], task_description: str) -> list[list[float]]:
        joints = list(obs["joints"])
        chunk = []
        for step in range(1, self.CHUNK_LEN + 1):
            chunk.append([j + self.DELTA * step for j in joints] + [0.0])
        return chunk


class ZmqGrootPolicy(PolicyClient):
    """msgpack/ZMQ REQ client for a running GR00T policy server.

    Wire format mirrors ``gr00t.eval.run_gr00t_server``'s client (see
    ``scripts/test_init_noise_determinism.py`` for the same pattern): a nested
    B=1,T=1 observation with ``state.single_arm`` (arm_dof) + ``state.gripper``
    (1), matching jetrover_modality_config.py.
    """

    IMAGE_SIZE = 256

    def __init__(self, host: str, port: int, arm_dof: int, timeout_ms: int = 180_000) -> None:
        import msgpack
        import msgpack_numpy as mnp
        import numpy as np
        import zmq

        self._arm_dof = arm_dof
        self._warned_no_image = False
        self._msgpack, self._mnp, self._np = msgpack, mnp, np
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._sock.connect(f"tcp://{host}:{port}")

    def _call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        req: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self._sock.send(self._msgpack.packb(req, default=self._mnp.encode))
        resp = self._msgpack.unpackb(
            self._sock.recv(), object_hook=self._mnp.decode, raw=False
        )
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"GR00T server error: {resp['error']}")
        return resp

    def _build_obs(self, obs: dict[str, Any], task_description: str) -> dict[str, Any]:
        np = self._np
        image = obs.get("image")
        if image is None:
            if not self._warned_no_image:
                # Only the state-only backends (hiwonder, mock) get here: ros2
                # refuses to return an observation without a live frame.
                print(
                    "[warning] no camera frame — sending black frames to the policy "
                    "(this arm backend is state-only; use ros2 for a visuomotor eval)",
                    flush=True,
                )
                self._warned_no_image = True
            image = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE, 3), dtype=np.uint8)
        image = np.asarray(image, dtype=np.uint8)[None, None]  # (1, 1, H, W, 3)
        joints = np.asarray(obs["joints"], dtype=np.float32).reshape(1, 1, self._arm_dof)
        gripper = np.asarray([obs["gripper"]], dtype=np.float32).reshape(1, 1, 1)
        return {
            "video": {VIDEO_KEY: image},
            "state": {"single_arm": joints, "gripper": gripper},
            "language": {LANGUAGE_KEY: [[task_description]]},
        }

    def get_action(self, obs: dict[str, Any], task_description: str) -> list[list[float]]:
        np = self._np
        resp = self._call(
            "get_action",
            {"observation": self._build_obs(obs, task_description), "options": None},
        )
        action = resp[0] if isinstance(resp, (list, tuple)) else resp
        arm = np.asarray(action["single_arm"], dtype=np.float32).reshape(-1, self._arm_dof)
        grip = np.asarray(action["gripper"], dtype=np.float32).reshape(-1, 1)
        steps = min(len(arm), len(grip))
        return [
            [*map(float, arm[i]), float(grip[i, 0])] for i in range(steps)
        ]


ARM_BACKENDS: dict[str, Callable[[int], ArmBackend]] = {
    "mock": MockArm,
    "ros2": Ros2Arm,
    "hiwonder": HiwonderArm,
}


# -------------------------------------------------------------------------- scoring
class OperatorConsole:
    """Single owner of the operator's input stream, shared by every prompt.

    ``input()`` cannot be cancelled, so a prompt that times out leaves its read
    blocked. With one reader thread per prompt, that orphaned read would take
    the answer typed for the *next* prompt and drop it into an abandoned queue.
    Here at most one read is ever in flight, and its line goes to whichever
    prompt is open when it arrives; lines typed while no prompt is open are
    stale and discarded when the next prompt opens.
    """

    def __init__(self, input_fn: Callable[[str], str]) -> None:
        self._input_fn = input_fn
        self._lines: queue.Queue[str | None] = queue.Queue()  # None = stdin closed
        self._lock = threading.Lock()
        self._read_in_flight = False
        self._closed = False

    def _read_one_line(self) -> None:
        try:
            line: str | None = self._input_fn("")
        except EOFError:
            line = None
        # Publishing the line and clearing the flag happen atomically, so ask()
        # can never see "no read in flight" while this line is still unqueued.
        with self._lock:
            self._read_in_flight = False
            if line is None:
                self._closed = True
            self._lines.put(line)

    def ask(self, prompt: str, timeout: float | None) -> str | None:
        """Show ``prompt`` and return the next line, or ``None`` after ``timeout``.

        ``timeout=None`` waits indefinitely.
        """
        with self._lock:
            while True:  # drop answers that arrived while no prompt was open
                try:
                    self._lines.get_nowait()
                except queue.Empty:
                    break
            closed = self._closed
        if closed:
            raise RuntimeError(
                "stdin is closed — operator prompts need an interactive terminal "
                "(use --scorer none / --reset_mode bounded for unattended runs)"
            )
        print(prompt, end="", flush=True)
        with self._lock:
            if not self._read_in_flight:
                self._read_in_flight = True
                threading.Thread(target=self._read_one_line, daemon=True).start()
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is None:
            raise RuntimeError("stdin closed while waiting for the operator")
        return line


def prompt_operator(
    episode_index: int,
    auto_timeout: float,
    console: OperatorConsole,
) -> bool:
    """Ask the operator whether the episode succeeded.

    With ``auto_timeout > 0`` an unanswered prompt scores the episode as a
    FAILURE after the timeout — honest-pessimistic, so unattended runs terminate
    without inventing successes.
    """
    answer = console.ask(
        f"Episode {episode_index} success? [y/n] ",
        timeout=auto_timeout if auto_timeout > 0 else None,
    )
    if answer is None:
        print(f"\n[auto_timeout] no answer in {auto_timeout}s — scoring as failure")
        return False
    return answer.strip().lower().startswith("y")


# ----------------------------------------------------------------------------- loop
def _step_toward(target: float, current: float, max_delta: float) -> float:
    return current + max(-max_delta, min(max_delta, target - current))


def clamp_action(action: list[float], current: list[float], max_delta: float) -> list[float]:
    """Rate-limit joint targets to ``max_delta`` rad/step; gripper passes through."""
    arm_dof = len(action) - 1
    clamped = [_step_toward(action[i], current[i], max_delta) for i in range(arm_dof)]
    clamped.append(action[arm_dof])
    return clamped


@dataclass(frozen=True)
class HomingConfig:
    """How each episode reaches its start pose.

    ``bounded``: closed-loop walk from the measured pose to ``arm.home_pose()``,
    every axis (gripper included) limited to ``max_joint_delta`` per step; gives
    up with an error after ``max_steps`` rather than start from an unknown pose.
    ``operator``: no motion — the operator places the arm and confirms.
    """

    mode: str = "bounded"
    max_joint_delta: float = 0.05
    tolerance: float = 0.03
    max_steps: int = 200


def home_arm(
    arm: ArmBackend,
    homing: HomingConfig,
    *,
    step_delay: float,
    console: OperatorConsole | None,
) -> int:
    """Bring the arm to its episode start pose. Returns homing commands sent."""
    if homing.mode == "operator":
        if console is None:
            raise RuntimeError("--reset_mode operator needs an operator console")
        # A safety confirmation needs an explicit token: a bare Enter or a late
        # "y" meant for an expired score prompt must not start an episode.
        while True:
            answer = console.ask(
                f"Place the arm in the start pose, then type '{OPERATOR_READY_TOKEN}' ",
                timeout=None,
            )
            if answer is not None and answer.strip().lower() == OPERATOR_READY_TOKEN:
                break
        obs = arm.get_observation()
        current = [*obs["joints"], obs["gripper"]]
        offset = max(abs(t - c) for t, c in zip(arm.home_pose(), current, strict=True))
        print(
            f"[reset] operator start pose: {[round(v, 3) for v in current]} "
            f"(max {offset:.3f} rad from home)",
            flush=True,
        )
        return 0

    target = arm.home_pose()
    error = float("inf")
    for sent in range(homing.max_steps + 1):
        obs = arm.get_observation()
        current = [*obs["joints"], obs["gripper"]]
        error = max(abs(t - c) for t, c in zip(target, current, strict=True))
        if error <= homing.tolerance:
            return sent
        if sent == homing.max_steps:
            break
        arm.send_action(
            [
                _step_toward(t, c, homing.max_joint_delta)
                for t, c in zip(target, current, strict=True)
            ]
        )
        if step_delay > 0:
            time.sleep(step_delay)
    raise RuntimeError(
        f"Arm did not reach the home pose within {homing.max_steps} bounded steps "
        f"(max joint error {error:.3f} rad > {homing.tolerance}) — check for an "
        "obstruction or a stalled servo. Refusing to start an episode from an "
        "unknown pose."
    )


def run_episode(
    arm: ArmBackend,
    policy: PolicyClient,
    *,
    task_description: str,
    max_steps: int,
    action_horizon: int,
    max_joint_delta: float,
    step_delay: float,
    homing: HomingConfig | None = None,
    console: OperatorConsole | None = None,
) -> int:
    """One rollout: home → obs → policy chunk → execute → repeat. Returns steps taken."""
    home_arm(arm, homing or HomingConfig(), step_delay=step_delay, console=console)
    steps = 0
    while steps < max_steps:
        obs = arm.get_observation()
        chunk = policy.get_action(obs, task_description)
        if not chunk:
            raise RuntimeError("Policy returned an empty action chunk")
        current = [*obs["joints"], obs["gripper"]]
        for action in chunk[:action_horizon]:
            if len(action) != arm.action_dim:
                raise RuntimeError(
                    f"Expected {arm.action_dim}-dim action ({arm.arm_dof} joints "
                    "+ gripper), got "
                    f"{len(action)} — check the served checkpoint's modality "
                    "config and --arm_dof"
                )
            # NaN slips through min/max (min(d, nan) == d): it would become a
            # full max_joint_delta step on every axis it touches.
            if not all(math.isfinite(v) for v in action):
                raise RuntimeError(f"Policy returned a non-finite action: {action}")
            safe = clamp_action(action, current, max_joint_delta)
            arm.send_action(safe)
            current = safe
            steps += 1
            if steps >= max_steps:
                break
            if step_delay > 0:
                time.sleep(step_delay)
    return steps


def make_policy(args: argparse.Namespace) -> PolicyClient:
    if args.policy_backend == "mock":
        return MockPolicy()
    return ZmqGrootPolicy(args.policy_host, args.policy_port, args.arm_dof)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    # Runner-owned flags (see build_custom_argv in runners/evals/custom.py).
    ap.add_argument("--checkpoint", required=True,
                    help="Trained checkpoint (recorded in metrics; the policy "
                         "server is what actually loads it)")
    ap.add_argument("--out-json", required=True, help="Metrics output path")
    # Passthrough config flags — snake_case, matching the runner's verbatim style.
    ap.add_argument("--arm_backend", choices=sorted(ARM_BACKENDS), default="mock")
    ap.add_argument("--policy_backend", choices=["mock", "zmq"], default="zmq")
    ap.add_argument("--policy_host", default="127.0.0.1")
    ap.add_argument("--policy_port", type=int, default=5555)
    ap.add_argument("--arm_dof", type=int, default=DEFAULT_ARM_DOF,
                    help="Positional arm joints, gripper excluded. JetRover: 5 "
                         "(Hiwonder's '6DOF' marketing counts the gripper)")
    ap.add_argument("--num_episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--action_horizon", type=int, default=8,
                    help="Steps executed per policy chunk before re-querying")
    ap.add_argument("--task_description", default="pick up the object")
    ap.add_argument("--scorer", choices=["operator", "none"], default="operator")
    ap.add_argument("--auto_timeout", type=float, default=0.0,
                    help="Seconds to wait for the operator prompt before scoring "
                         "the episode as a failure (0 = wait forever)")
    ap.add_argument("--control_hz", type=float, default=5.0,
                    help="Action step rate on the real arm (mock runs unthrottled)")
    ap.add_argument("--max_joint_delta", type=float, default=0.15,
                    help="Per-step joint clamp in rad — the safety rate limit "
                         f"(0 < x <= {MAX_JOINT_DELTA_CEILING})")
    ap.add_argument("--reset_mode", choices=["bounded", "operator"], default="bounded",
                    help="Episode reset: 'bounded' walks home from the measured "
                         "pose; 'operator' sends no motion and waits for the "
                         "operator to place the arm")
    ap.add_argument("--home_max_joint_delta", type=float,
                    default=HomingConfig.max_joint_delta,
                    help="Per-step clamp (rad) for the bounded homing walk, all "
                         f"axes incl. gripper (0 < x <= {HOME_JOINT_DELTA_CEILING})")
    ap.add_argument("--home_tolerance", type=float, default=HomingConfig.tolerance,
                    help="Max per-axis error (rad) at which the arm counts as home")
    ap.add_argument("--home_max_steps", type=int, default=HomingConfig.max_steps,
                    help="Homing commands before giving up with an error")
    args = ap.parse_args(argv)

    if not 0 < args.max_joint_delta <= MAX_JOINT_DELTA_CEILING:
        ap.error(f"--max_joint_delta must be in (0, {MAX_JOINT_DELTA_CEILING}] rad")
    if not 0 < args.home_max_joint_delta <= HOME_JOINT_DELTA_CEILING:
        ap.error(f"--home_max_joint_delta must be in (0, {HOME_JOINT_DELTA_CEILING}] rad")
    if args.home_tolerance <= 0:
        ap.error("--home_tolerance must be > 0")
    if args.home_max_steps < 1:
        ap.error("--home_max_steps must be >= 1")
    if args.control_hz <= 0:
        ap.error("--control_hz must be > 0")
    for flag, minimum in (
        ("action_horizon", 1),  # 0 would re-query the policy forever without moving
        ("arm_dof", 1),
        ("num_episodes", 1),
        ("max_steps", 0),
    ):
        if getattr(args, flag) < minimum:
            ap.error(f"--{flag} must be >= {minimum}")
    return args


def main(
    argv: list[str] | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    args = parse_args(argv)
    arm = ARM_BACKENDS[args.arm_backend](args.arm_dof)
    arm.command_duration = 1.0 / args.control_hz
    policy = make_policy(args)
    step_delay = 0.0 if args.arm_backend == "mock" else 1.0 / args.control_hz
    console = OperatorConsole(input_fn)
    homing = HomingConfig(
        mode=args.reset_mode,
        max_joint_delta=args.home_max_joint_delta,
        tolerance=args.home_tolerance,
        max_steps=args.home_max_steps,
    )

    episodes: list[dict[str, Any]] = []
    successes = 0
    error: BaseException | None = None
    try:
        arm.connect()
        for index in range(args.num_episodes):
            start = time.monotonic()
            steps = run_episode(
                arm,
                policy,
                task_description=args.task_description,
                max_steps=args.max_steps,
                action_horizon=args.action_horizon,
                max_joint_delta=args.max_joint_delta,
                step_delay=step_delay,
                homing=homing,
                console=console,
            )
            record: dict[str, Any] = {
                "index": index,
                "steps": steps,
                "duration_s": round(time.monotonic() - start, 3),
            }
            if args.scorer == "operator":
                success = prompt_operator(index, args.auto_timeout, console)
                record["success"] = success
                successes += int(success)
            episodes.append(record)
            print(f"episode {index}: {record}", flush=True)
    except BaseException as exc:
        error = exc  # still write what completed, then re-raise below
    finally:
        arm.close()
        policy.close()

    payload: dict[str, Any] = {
        "num_episodes": args.num_episodes,
        "metrics": {
            "episodes": episodes,
            "mean_steps": (
                sum(e["steps"] for e in episodes) / len(episodes) if episodes else 0.0
            ),
            "arm_backend": args.arm_backend,
            "policy_backend": args.policy_backend,
            "arm_dof": args.arm_dof,
            "reset_mode": args.reset_mode,
            "checkpoint": args.checkpoint,
        },
    }
    if args.scorer == "operator":
        # Over the requested episodes: an aborted run never inflates the rate.
        payload["success_rate"] = successes / args.num_episodes
    if error is not None:
        payload["metrics"]["aborted"] = f"{type(error).__name__}: {error}"

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote metrics to {out}", flush=True)
    if error is not None:
        raise error
    return 0


if __name__ == "__main__":
    sys.exit(main())
