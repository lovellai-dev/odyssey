"""Scripted pick-and-place policy for the UR5e / Robotiq 2F-85 drug-sorting cell.

This is a **pure** (mujoco-free) Python port of the browser controller in
``lai-agent`` (``src/assets/js/aseptipack-pickplace.js``). It reproduces the
finite-state waypoint machine that servos the UR5e to a sequence of joint
targets while opening/closing the Robotiq gripper, so we can replay the exact
same demonstration head-lessly and record it as GR00T training data.

The port keeps the JS semantics 1:1:

* ``PHASES`` — the same 10 waypoints (arm joint targets in radians, gripper
  command, and a ``hold`` that is either ``"conv"`` = advance on convergence or
  an integer dwell in physics steps).
* Per physics step: interpolate the arm command from the previous waypoint to
  the current one over ``MOVE_STEPS`` with an ease-in/out profile, hold the
  gripper command, then advance the phase once the arm has converged
  (``max joint error < CONV_TOL``, capped at ``CONV_MAX`` steps) or the dwell
  has elapsed.

Nothing here imports mujoco: :class:`PickPlaceController` consumes the measured
arm joint positions each step and emits the actuator command to apply. The
mujoco stepping + rendering lives in ``scripts/gen_ur5e_drugsort_demos.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Gripper actuator command range (Robotiq 2F-85 tendon actuator) ---------
# ``gr_fingers_actuator`` ctrlrange is [0, 255]: 0 = fully open, 255 = closed.
GRIP_OPEN: float = 0.0
GRIP_CLOSE: float = 255.0

# --- Arm actuator / joint order (matches the MJCF <actuator> block) ----------
ARM_ACTUATOR_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"{n}_joint" for n in ARM_ACTUATOR_NAMES)
GRIPPER_ACTUATOR_NAME: str = "gr_fingers_actuator"

# --- FSM tuning (ported verbatim from the JS controller) ---------------------
MOVE_STEPS: int = 800
CONV_TOL: float = 6e-3
CONV_MAX: int = 2200
SETTLE_STEPS: int = 180

# --- Iteration-3 grasp densification -----------------------------------------
# The iteration-2 policy fired the gripper but closed ~7 cm short on the final
# descent (closed-loop eval 0/20, every episode lifted=False). Two levers add
# demonstration frames exactly where the policy was under-resolved:
#   * DESCEND_MOVE_STEPS slows the final ``descend`` interpolation (vs the
#     default MOVE_STEPS) so ~2x more frames are recorded lowering onto the vial.
#     The ease-in/out profile naturally clusters the extra frames near the bottom
#     of the descent — the exact stretch the policy stopped short of.
#   * CLOSE_DWELL lengthens the closed-gripper dwell before the lift so the
#     Robotiq fully seats on the vial and more grasp-hold frames are recorded.
DESCEND_MOVE_STEPS: int = 1600
CLOSE_DWELL: int = 600


@dataclass(frozen=True)
class Waypoint:
    """One FSM phase: arm joint targets (rad), gripper command, hold rule."""

    name: str
    q: tuple[float, float, float, float, float, float]
    grip: float
    # ``"conv"`` -> advance once the arm converges; ``int`` -> dwell that many
    # physics steps after the interpolated move completes.
    hold: str | int
    # Optional per-phase interpolation length (physics steps); ``None`` falls
    # back to the controller's global ``move_steps``. Set on ``descend`` to slow
    # the final approach so the grasp region is densely sampled (iteration 3).
    move_steps: int | None = None


# The precomputed waypoint solutions (radians) + gripper ctrl, verbatim from
# ``aseptipack-pickplace.js`` PH[]. home -> approach -> descend -> close (grasp)
# -> lift -> transport -> lower -> release -> retract -> home.
PHASES: tuple[Waypoint, ...] = (
    Waypoint("home", (2.850, -1.670, 1.580, -1.413, -1.590, -0.291), GRIP_OPEN, "conv"),
    Waypoint("approach", (3.174, -1.617, 1.582, -1.466, -1.567, 0.032), GRIP_OPEN, "conv"),
    Waypoint("descend", (3.176, -1.572, 1.871, -1.798, -1.568, 0.034), GRIP_OPEN, "conv",
             move_steps=DESCEND_MOVE_STEPS),
    Waypoint("close", (3.176, -1.572, 1.871, -1.798, -1.568, 0.034), GRIP_CLOSE, CLOSE_DWELL),
    Waypoint("lift", (3.175, -1.616, 1.581, -1.466, -1.567, 0.033), GRIP_CLOSE, "conv"),
    Waypoint("transport", (2.436, -1.729, 1.686, -1.475, -1.617, -0.704), GRIP_CLOSE, "conv"),
    Waypoint("lower", (2.435, -1.683, 1.992, -1.824, -1.616, -0.705), GRIP_CLOSE, 140),
    Waypoint("release", (2.435, -1.683, 1.992, -1.824, -1.616, -0.705), GRIP_OPEN, 360),
    Waypoint("retract", (2.434, -1.729, 1.686, -1.474, -1.616, -0.706), GRIP_OPEN, "conv"),
    Waypoint("home2", (2.850, -1.670, 1.580, -1.413, -1.590, -0.291), GRIP_OPEN, "conv"),
)


def ease(t: float) -> float:
    """Ease-in/out cubic, matching the JS ``ease``."""
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - (-2.0 * t + 2.0) ** 2 / 2.0


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def arm_error(target: tuple[float, ...], measured: tuple[float, ...]) -> float:
    """Max absolute per-joint error between target and measured arm positions."""
    return max(abs(t - m) for t, m in zip(target, measured, strict=True))


@dataclass
class ControllerOutput:
    arm_ctrl: tuple[float, float, float, float, float, float]
    grip_ctrl: float
    phase_index: int
    phase_name: str
    phase_advanced: bool
    finished: bool


@dataclass
class PickPlaceController:
    """Stateful, mujoco-free replay of the scripted pick-and-place FSM.

    Call :meth:`reset` with the initial arm joint positions, then :meth:`step`
    once per physics step, passing the *measured* arm joint positions; it
    returns the actuator command to apply for that step.
    """

    move_steps: int = MOVE_STEPS
    conv_tol: float = CONV_TOL
    conv_max: int = CONV_MAX
    phases: tuple[Waypoint, ...] = PHASES

    _phase: int = field(default=0, init=False)
    _s: int = field(default=0, init=False)
    _prev: tuple[float, float, float, float, float, float] = field(
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), init=False
    )

    def reset(self, arm_qpos: tuple[float, float, float, float, float, float]) -> None:
        self._phase = 0
        self._s = 0
        self._prev = arm_qpos

    @property
    def finished(self) -> bool:
        return self._phase >= len(self.phases)

    @property
    def phase_index(self) -> int:
        return self._phase

    def step(
        self, measured_arm_qpos: tuple[float, float, float, float, float, float]
    ) -> ControllerOutput:
        """Advance the FSM one physics step; return the command to apply now."""
        if self.finished:
            last = self.phases[-1]
            return ControllerOutput(
                arm_ctrl=last.q,
                grip_ctrl=last.grip,
                phase_index=self._phase,
                phase_name="finished",
                phase_advanced=False,
                finished=True,
            )

        ph = self.phases[self._phase]
        # Same target as prev? (a pure grip change) -> no interpolation.
        same = all(abs(x - p) < 1e-3 for x, p in zip(ph.q, self._prev, strict=True))
        # Per-phase move length overrides the global default (slower descend).
        phase_move = ph.move_steps if ph.move_steps is not None else self.move_steps
        move_dur = 0 if same else phase_move

        if self._s < move_dur:
            frac = ease(self._s / move_dur)
            q = tuple(lerp(p, x, frac) for p, x in zip(self._prev, ph.q, strict=True))
        else:
            q = ph.q
        arm_ctrl: tuple[float, float, float, float, float, float] = (
            q[0], q[1], q[2], q[3], q[4], q[5],
        )

        self._s += 1
        advanced = False
        if self._s >= move_dur:
            held = self._s - move_dur
            if ph.hold == "conv":
                done = arm_error(ph.q, measured_arm_qpos) < self.conv_tol or held > self.conv_max
            else:
                done = held >= int(ph.hold)
            if done:
                self._prev = ph.q
                self._phase += 1
                self._s = 0
                advanced = True

        return ControllerOutput(
            arm_ctrl=arm_ctrl,
            grip_ctrl=ph.grip,
            phase_index=self._phase if not advanced else self._phase - 1,
            phase_name=ph.name,
            phase_advanced=advanced,
            finished=self.finished,
        )
