#!/usr/bin/env python3
"""Real-hardware eval for the Hiwonder JetRover 6DoF arm (``evaluation_type: custom``).

Implements the launch contract of ``src/odyssey/runners/evals/custom.py``::

    <python> eval_jetrover.py --checkpoint <path> --out-json <path> \\
        --arm_backend {mock,ros2,hiwonder} --policy_backend {mock,zmq} [...]

and writes the out-json metrics object the runner expects (``success_rate`` when
the operator scorer ran, metric-only otherwise).

Split of responsibilities:

* ``ArmBackend`` — talks to the arm: 6 joint positions + 1 gripper in, camera
  frame + joint state out. ``mock`` is deterministic and dependency-free (CI);
  ``ros2`` / ``hiwonder`` are thin adapters over the JetRover's ROS 2 graph or
  Hiwonder's on-board bus-servo SDK.
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
import json
import queue
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

ARM_DOF = 6
ACTION_DIM = ARM_DOF + 1  # 6 arm joints + 1 gripper — the JetRover arm's action space

# GR00T-server observation keys; must match the modality config the checkpoint
# was finetuned with (jetrover_modality_config.py in this directory).
VIDEO_KEY = "front"
LANGUAGE_KEY = "annotation.human.task_description"


# --------------------------------------------------------------------------- arms
class ArmBackend:
    """Minimal arm interface: observations out, 7-float actions in."""

    def connect(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        """Move to the home pose (start of every episode)."""
        raise NotImplementedError

    def get_observation(self) -> dict[str, Any]:
        """Return ``{"joints": [6 floats], "gripper": float, "image": ndarray|None}``."""
        raise NotImplementedError

    def send_action(self, action: list[float]) -> None:
        """Execute one action step: 6 joint positions (rad) + gripper command."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class MockArm(ArmBackend):
    """Deterministic, dependency-free arm for CI and dry runs."""

    def __init__(self) -> None:
        self.joints = [0.0] * ARM_DOF
        self.gripper = 0.0
        self.actions_received: list[list[float]] = []

    def connect(self) -> None:
        pass

    def reset(self) -> None:
        self.joints = [0.0] * ARM_DOF
        self.gripper = 0.0

    def get_observation(self) -> dict[str, Any]:
        return {"joints": list(self.joints), "gripper": self.gripper, "image": None}

    def send_action(self, action: list[float]) -> None:
        self.actions_received.append(list(action))
        self.joints = list(action[:ARM_DOF])
        self.gripper = float(action[ARM_DOF])

    def close(self) -> None:
        pass


class Ros2Arm(ArmBackend):
    """ROS 2 adapter for the JetRover arm controller.

    Topic names below match Hiwonder's shipped jetrover ROS 2 packages; verify
    against your image (``ros2 topic list``) and adjust if your namespace differs.
    """

    JOINT_STATE_TOPIC = "/joint_states"
    ARM_COMMAND_TOPIC = "/servo_controller"
    CAMERA_TOPIC = "/depth_cam/rgb/image_raw"
    ARM_JOINT_NAMES: ClassVar[list[str]] = [f"joint{i}" for i in range(1, ARM_DOF + 1)]
    GRIPPER_JOINT_NAME = "r_joint"

    def __init__(self) -> None:
        self._node: Any = None
        self._last_joint_state: Any = None
        self._last_image: Any = None

    def connect(self) -> None:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image, JointState
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        rclpy.init()
        self._rclpy = rclpy
        self._JointTrajectory = JointTrajectory
        self._JointTrajectoryPoint = JointTrajectoryPoint
        self._node = Node("odyssey_jetrover_eval")
        self._pub = self._node.create_publisher(JointTrajectory, self.ARM_COMMAND_TOPIC, 1)
        self._node.create_subscription(
            JointState, self.JOINT_STATE_TOPIC, self._on_joint_state, 1
        )
        self._node.create_subscription(Image, self.CAMERA_TOPIC, self._on_image, 1)

    def _on_joint_state(self, msg: Any) -> None:
        self._last_joint_state = msg

    def _on_image(self, msg: Any) -> None:
        self._last_image = msg

    def _spin(self, seconds: float = 0.05) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=seconds)

    def reset(self) -> None:
        self.send_action([0.0] * ARM_DOF + [0.0])
        time.sleep(2.0)  # give the servos time to reach home before the episode

    def get_observation(self) -> dict[str, Any]:
        import numpy as np

        deadline = time.monotonic() + 5.0
        while self._last_joint_state is None and time.monotonic() < deadline:
            self._spin()
        if self._last_joint_state is None:
            raise RuntimeError(
                f"No joint state on {self.JOINT_STATE_TOPIC} after 5s — is the "
                "jetrover controller running?"
            )
        name_to_pos = dict(
            zip(self._last_joint_state.name, self._last_joint_state.position, strict=False)
        )
        joints = [float(name_to_pos.get(n, 0.0)) for n in self.ARM_JOINT_NAMES]
        gripper = float(name_to_pos.get(self.GRIPPER_JOINT_NAME, 0.0))

        image = None
        if self._last_image is not None:
            msg = self._last_image
            image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1
            )[..., :3]
        return {"joints": joints, "gripper": gripper, "image": image}

    def send_action(self, action: list[float]) -> None:
        traj = self._JointTrajectory()
        traj.joint_names = [*self.ARM_JOINT_NAMES, self.GRIPPER_JOINT_NAME]
        point = self._JointTrajectoryPoint()
        point.positions = [float(v) for v in action[:ACTION_DIM]]
        traj.points = [point]
        self._pub.publish(traj)
        self._spin(0.0)

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._rclpy.shutdown()


class HiwonderArm(ArmBackend):
    """Direct bus-servo adapter via Hiwonder's on-board controller SDK.

    Uses the ``ros_robot_controller_sdk`` module that ships on JetRover images
    (the same board API the vendor demos use). Servo IDs, direction signs and
    the rad→pulse mapping below are the vendor defaults — verify against your
    unit before running with the workspace occupied.
    """

    ARM_SERVO_IDS: ClassVar[list[int]] = [1, 2, 3, 4, 5, 6]
    GRIPPER_SERVO_ID = 10
    PULSE_CENTER = 500
    PULSE_PER_RAD = 500 / 2.094  # Hiwonder bus servos: 0..1000 pulses over ±120°
    MOVE_SECONDS = 0.2

    def __init__(self) -> None:
        self._board: Any = None
        self._last_command = [0.0] * ACTION_DIM

    def connect(self) -> None:
        from ros_robot_controller_sdk import Board  # type: ignore[import-not-found]

        self._board = Board()

    def _to_pulse(self, radians: float) -> int:
        pulse = self.PULSE_CENTER + radians * self.PULSE_PER_RAD
        return int(min(1000, max(0, round(pulse))))

    def reset(self) -> None:
        self.send_action([0.0] * ARM_DOF + [0.0])
        time.sleep(2.0)

    def get_observation(self) -> dict[str, Any]:
        # The bus-servo protocol reports positions on request; camera access goes
        # through ROS/OpenCV and is out of scope for this backend (state-only).
        positions = []
        for servo_id in self.ARM_SERVO_IDS:
            reading = self._board.bus_servo_read_position(servo_id)
            pulse = reading[0] if isinstance(reading, (list, tuple)) else reading
            positions.append((float(pulse) - self.PULSE_CENTER) / self.PULSE_PER_RAD)
        return {"joints": positions, "gripper": self._last_command[ARM_DOF], "image": None}

    def send_action(self, action: list[float]) -> None:
        targets = [
            [servo_id, self._to_pulse(action[i])]
            for i, servo_id in enumerate(self.ARM_SERVO_IDS)
        ]
        targets.append([self.GRIPPER_SERVO_ID, self._to_pulse(action[ARM_DOF])])
        self._board.bus_servo_set_position(self.MOVE_SECONDS, targets)
        self._last_command = list(action[:ACTION_DIM])

    def close(self) -> None:
        self._board = None


# ------------------------------------------------------------------------ policies
class PolicyClient:
    def get_action(self, obs: dict[str, Any], task_description: str) -> list[list[float]]:
        """Return a chunk of action steps, each ``[j1..j6, gripper]``."""
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
    B=1,T=1 observation with ``state.single_arm`` (6) + ``state.gripper`` (1),
    matching jetrover_modality_config.py.
    """

    IMAGE_SIZE = 256

    def __init__(self, host: str, port: int, timeout_ms: int = 180_000) -> None:
        import msgpack
        import msgpack_numpy as mnp
        import numpy as np
        import zmq

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
            image = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE, 3), dtype=np.uint8)
        image = np.asarray(image, dtype=np.uint8)[None, None]  # (1, 1, H, W, 3)
        joints = np.asarray(obs["joints"], dtype=np.float32).reshape(1, 1, ARM_DOF)
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
        arm = np.asarray(action["single_arm"], dtype=np.float32).reshape(-1, ARM_DOF)
        grip = np.asarray(action["gripper"], dtype=np.float32).reshape(-1, 1)
        steps = min(len(arm), len(grip))
        return [
            [*map(float, arm[i]), float(grip[i, 0])] for i in range(steps)
        ]


ARM_BACKENDS: dict[str, Callable[[], ArmBackend]] = {
    "mock": MockArm,
    "ros2": Ros2Arm,
    "hiwonder": HiwonderArm,
}


# -------------------------------------------------------------------------- scoring
def prompt_operator(
    episode_index: int,
    auto_timeout: float,
    input_fn: Callable[[str], str],
) -> bool:
    """Ask the operator whether the episode succeeded.

    With ``auto_timeout > 0`` an unanswered prompt scores the episode as a
    FAILURE after the timeout — honest-pessimistic, so unattended runs terminate
    without inventing successes.
    """
    prompt = f"Episode {episode_index} success? [y/n] "
    if auto_timeout <= 0:
        answer = input_fn(prompt)
        return answer.strip().lower().startswith("y")

    answers: queue.Queue[str] = queue.Queue()
    thread = threading.Thread(
        target=lambda: answers.put(input_fn(prompt)), daemon=True
    )
    thread.start()
    try:
        answer = answers.get(timeout=auto_timeout)
    except queue.Empty:
        print(f"\n[auto_timeout] no answer in {auto_timeout}s — scoring as failure")
        return False
    return answer.strip().lower().startswith("y")


# ----------------------------------------------------------------------------- loop
def clamp_action(action: list[float], current: list[float], max_delta: float) -> list[float]:
    """Rate-limit joint targets to ``max_delta`` rad/step; gripper passes through."""
    clamped = []
    for i in range(ARM_DOF):
        delta = max(-max_delta, min(max_delta, action[i] - current[i]))
        clamped.append(current[i] + delta)
    clamped.append(action[ARM_DOF])
    return clamped


def run_episode(
    arm: ArmBackend,
    policy: PolicyClient,
    *,
    task_description: str,
    max_steps: int,
    action_horizon: int,
    max_joint_delta: float,
    step_delay: float,
) -> int:
    """One rollout: obs → policy chunk → execute → repeat. Returns steps taken."""
    arm.reset()
    steps = 0
    while steps < max_steps:
        obs = arm.get_observation()
        chunk = policy.get_action(obs, task_description)
        if not chunk:
            raise RuntimeError("Policy returned an empty action chunk")
        current = [*obs["joints"], obs["gripper"]]
        for action in chunk[:action_horizon]:
            if len(action) != ACTION_DIM:
                raise RuntimeError(
                    f"Expected {ACTION_DIM}-dim action (6 joints + gripper), "
                    f"got {len(action)} — check the served checkpoint's modality config"
                )
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
    return ZmqGrootPolicy(args.policy_host, args.policy_port)


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
                    help="Per-step joint clamp in rad — the safety rate limit")
    return ap.parse_args(argv)


def main(
    argv: list[str] | None = None,
    input_fn: Callable[[str], str] = input,
) -> int:
    args = parse_args(argv)
    arm = ARM_BACKENDS[args.arm_backend]()
    policy = make_policy(args)
    step_delay = 0.0 if args.arm_backend == "mock" else 1.0 / args.control_hz

    episodes: list[dict[str, Any]] = []
    successes = 0
    arm.connect()
    try:
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
            )
            record: dict[str, Any] = {
                "index": index,
                "steps": steps,
                "duration_s": round(time.monotonic() - start, 3),
            }
            if args.scorer == "operator":
                success = prompt_operator(index, args.auto_timeout, input_fn)
                record["success"] = success
                successes += int(success)
            episodes.append(record)
            print(f"episode {index}: {record}", flush=True)
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
            "checkpoint": args.checkpoint,
        },
    }
    if args.scorer == "operator":
        payload["success_rate"] = successes / args.num_episodes if args.num_episodes else 0.0

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote metrics to {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
