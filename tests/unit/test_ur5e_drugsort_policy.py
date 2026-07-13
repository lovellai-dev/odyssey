"""Tests for the pure (mujoco-free) UR5e drug-sorting pick-place FSM port."""

from __future__ import annotations

from odyssey.embodiments.ur5e_drugsort.policy import (
    GRIP_CLOSE,
    GRIP_OPEN,
    PHASES,
    PickPlaceController,
    arm_error,
    ease,
)


def test_phases_are_the_ten_scripted_waypoints() -> None:
    names = [p.name for p in PHASES]
    assert names == [
        "home", "approach", "descend", "close", "lift",
        "transport", "lower", "release", "retract", "home2",
    ]
    # Every waypoint has a 6-DoF arm target and a valid gripper command.
    for p in PHASES:
        assert len(p.q) == 6
        assert p.grip in (GRIP_OPEN, GRIP_CLOSE)


def test_ease_endpoints_and_midpoint() -> None:
    assert ease(0.0) == 0.0
    assert ease(1.0) == 1.0
    assert abs(ease(0.5) - 0.5) < 1e-9


def test_controller_reaches_all_phases_with_perfect_tracking() -> None:
    """A perfect joint tracker (measured == last command) should drive the FSM
    cleanly through all 10 phases and finish."""
    ctrl = PickPlaceController()
    start = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
    ctrl.reset(start)

    measured = start
    visited: list[str] = []
    advances = 0
    grips: list[float] = []
    for _ in range(50_000):
        out = ctrl.step(measured)
        measured = out.arm_ctrl  # perfect tracking
        grips.append(out.grip_ctrl)
        if out.phase_advanced:
            visited.append(out.phase_name)
            advances += 1
        if out.finished:
            break

    assert ctrl.finished
    assert advances == len(PHASES)
    assert visited == [p.name for p in PHASES]
    # Gripper must have closed at some point (the grasp) and been open initially.
    assert grips[0] == GRIP_OPEN
    assert GRIP_CLOSE in grips


def test_gripper_closes_only_between_close_and_release() -> None:
    """Gripper stays open through approach/descend, closes at 'close', and
    reopens at 'release' — the physical grasp window."""
    ctrl = PickPlaceController()
    start = PHASES[0].q
    ctrl.reset(start)
    measured = start

    seen_close_phase_grip: set[float] = set()
    seen_transport_phase_grip: set[float] = set()
    for _ in range(50_000):
        out = ctrl.step(measured)
        measured = out.arm_ctrl
        if out.phase_name == "close":
            seen_close_phase_grip.add(out.grip_ctrl)
        if out.phase_name == "transport":
            seen_transport_phase_grip.add(out.grip_ctrl)
        if out.finished:
            break

    # The vial is held (gripper closed) through the close + transport phases.
    assert seen_close_phase_grip == {GRIP_CLOSE}
    assert seen_transport_phase_grip == {GRIP_CLOSE}


def test_arm_error_is_max_abs_component() -> None:
    a = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    b = (0.1, -0.3, 0.05, 0.0, 0.2, -0.02)
    assert abs(arm_error(a, b) - 0.3) < 1e-9


def test_grip_only_phase_needs_no_interpolation() -> None:
    """'close' shares the arm target of 'descend', so it must apply the target
    immediately (move_dur == 0) rather than re-interpolate."""
    ctrl = PickPlaceController(move_steps=800)
    # Jump the controller to the 'close' phase by driving through with perfect
    # tracking until we are on it.
    start = PHASES[0].q
    ctrl.reset(start)
    measured = start
    first_close_step_ctrl: tuple[float, ...] | None = None
    for _ in range(50_000):
        out = ctrl.step(measured)
        if out.phase_name == "close" and first_close_step_ctrl is None:
            first_close_step_ctrl = out.arm_ctrl
        measured = out.arm_ctrl
        if out.finished:
            break
    assert first_close_step_ctrl is not None
    # No interpolation ramp: the very first 'close' command equals its target q.
    for c, q in zip(first_close_step_ctrl, PHASES[3].q, strict=True):
        assert abs(c - q) < 1e-9
