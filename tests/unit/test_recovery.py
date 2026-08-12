"""Tests for the closed-loop recovery core (``runners/agents/recovery.py``).

Everything is pure Python driven with plain fakes — no env, no GPU, no HTTP.
The specialist is a scripted fake satisfying ``SpecialistGateLike``; verdicts
are ``SimpleNamespace`` objects (the policy reads attributes, not a class).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from odyssey.runners.agents.recovery import (
    BoundarySnapshot,
    ChunkLedger,
    RecoveryController,
    RecoveryPolicy,
    RolloutLog,
    StuckMonitor,
    _quat_error_axis_angle,
)

# ---------------------------------------------------------------------------
# Fakes + tiny builders.
# ---------------------------------------------------------------------------

IDENTITY = (0.0, 0.0, 0.0, 1.0)
OPEN_GRIPPER = (0.04, -0.04)
STILL = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # pushes +x, arm won't move
IDLE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _snapshot(chunk: int, *, step: int | None = None, pos=(0.0, 0.0, 0.0)) -> BoundarySnapshot:
    return BoundarySnapshot(
        step=chunk * 4 if step is None else step,
        chunk_index=chunk,
        eef_pos=pos,
        eef_quat_xyzw=IDENTITY,
        gripper_qpos=OPEN_GRIPPER,
    )


class _FakeGate:
    """SpecialistGateLike: records submits, replays scripted poll verdicts."""

    def __init__(self, verdicts: list[Any] | None = None) -> None:
        self.verdicts = list(verdicts or [])
        self.submits: list[tuple[int, Any, str]] = []
        self.episodes: list[int] = []
        self.closed = False

    def begin_episode(self, episode: int) -> None:
        self.episodes.append(episode)

    def submit(self, *, chunk_index: int, frame: Any, instruction: str) -> bool:
        self.submits.append((chunk_index, frame, instruction))
        return True

    def poll(self) -> Any | None:
        return self.verdicts.pop(0) if self.verdicts else None

    def close(self) -> None:
        self.closed = True


def _policy(
    *,
    specialist: _FakeGate | None = None,
    shadow: bool = False,
    mode: str = "command",
    max_recoveries: int = 3,
    settle_steps: int = 0,
    window_steps: int = 3,
    frame_pair_gap: int = 1,
) -> RecoveryPolicy:
    return RecoveryPolicy(
        ledger=ChunkLedger(),
        monitor=StuckMonitor(window_steps=window_steps, eps_m=0.005, min_commanded=0.05),
        controller=RecoveryController(max_steps=8),
        specialist=specialist,
        mode=mode,
        shadow=shadow,
        max_recoveries=max_recoveries,
        settle_steps=settle_steps,
        frame_pair_gap=frame_pair_gap,
    )


def _boundary(policy: RecoveryPolicy, *, step: int, pos=(0.0, 0.0, 0.0)) -> None:
    policy.on_chunk_boundary(
        step=step,
        frame=f"f{step}",  # distinct per boundary so pair contents are assertable
        instruction="pick up the bowl",
        eef_pos=pos,
        eef_quat_xyzw=IDENTITY,
        gripper_qpos=OPEN_GRIPPER,
    )


def _drive_stuck(policy: RecoveryPolicy, *, steps: int = 3, start_step: int = 0):
    """Feed `steps` non-moving-but-commanded steps; return the last decision."""
    decision = None
    for i in range(steps):
        decision = policy.after_step(
            step=start_step + i,
            action=STILL,
            eef_pos=(0.0, 0.0, 0.0),
            chunk_start=(i == 0),
        )
    assert decision is not None
    return decision


# ---------------------------------------------------------------------------
# ChunkLedger.
# ---------------------------------------------------------------------------


def test_ledger_last_good_skips_flagged_boundaries() -> None:
    ledger = ChunkLedger()
    for chunk in range(3):
        ledger.record(_snapshot(chunk))
    ledger.mark(2, tier="kinematic", ok=False)
    good = ledger.last_good()
    assert good is not None and good.chunk_index == 1


def test_ledger_mark_from_taints_everything_since() -> None:
    ledger = ChunkLedger()
    for chunk in range(3):
        ledger.record(_snapshot(chunk))
    ledger.mark_from(1, tier="specialist", ok=False)
    good = ledger.last_good()
    assert good is not None and good.chunk_index == 0


def test_ledger_late_specialist_verdict_lands_on_the_right_chunk() -> None:
    ledger = ChunkLedger()
    ledger.record(_snapshot(0))
    ledger.record(_snapshot(1))
    ledger.mark(0, tier="specialist", ok=True)  # late clean verdict
    ledger.mark(1, tier="specialist", ok=False)
    good = ledger.last_good()
    assert good is not None and good.chunk_index == 0
    assert good.specialist_ok is True


def test_ledger_capacity_evicts_oldest() -> None:
    ledger = ChunkLedger(capacity=2)
    for chunk in range(3):
        ledger.record(_snapshot(chunk))
    latest = ledger.latest()
    assert latest is not None and latest.chunk_index == 2
    ledger.mark_from(1, tier="kinematic", ok=False)  # taints both survivors
    assert ledger.last_good() is None  # chunk 0 was evicted


def test_ledger_reset_and_unknown_tier() -> None:
    ledger = ChunkLedger()
    ledger.record(_snapshot(0))
    ledger.reset()
    assert ledger.latest() is None
    with pytest.raises(ValueError, match="tier"):
        ledger.mark(0, tier="vibes", ok=False)


# ---------------------------------------------------------------------------
# StuckMonitor.
# ---------------------------------------------------------------------------


def test_monitor_kinematic_trigger_when_commanded_but_no_displacement() -> None:
    monitor = StuckMonitor(window_steps=3, eps_m=0.005, min_commanded=0.05)
    signals = [
        monitor.update(action=STILL, eef_pos=(0.1, 0.2, 0.3), chunk_start=False)
        for _ in range(3)
    ]
    assert [s.kinematic for s in signals] == [False, False, True]
    assert signals[-1].cause == "kinematic"


def test_monitor_silent_during_warmup_window() -> None:
    monitor = StuckMonitor(window_steps=5)
    for _ in range(4):  # window never fills
        signal = monitor.update(action=STILL, eef_pos=(0.0, 0.0, 0.0), chunk_start=False)
    assert not signal.triggered


def test_monitor_no_trigger_when_arm_moves() -> None:
    monitor = StuckMonitor(window_steps=3, eps_m=0.005)
    for i in range(5):
        signal = monitor.update(
            action=STILL, eef_pos=(0.01 * i, 0.0, 0.0), chunk_start=False
        )
        assert not signal.kinematic


def test_monitor_no_trigger_when_no_motion_commanded() -> None:
    monitor = StuckMonitor(window_steps=3, min_commanded=0.05)
    for _ in range(5):  # arm static, but nothing was commanded either
        signal = monitor.update(action=IDLE, eef_pos=(0.0, 0.0, 0.0), chunk_start=False)
        assert not signal.kinematic


def test_monitor_continuity_fires_on_chunk_start_discrepancy_only() -> None:
    monitor = StuckMonitor(window_steps=3, continuity_eps=0.5, continuity_tail=2)
    head = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]
    # The arm keeps moving throughout so the kinematic tier stays silent and
    # the continuity flag is observed in isolation.
    for i in range(2):  # previous chunk's tail: idle pose deltas
        monitor.update(action=IDLE, eef_pos=(0.01 * i, 0.0, 0.0), chunk_start=False)

    mid = monitor.update(action=head, eef_pos=(0.02, 0.0, 0.0), chunk_start=False)
    assert not mid.continuity  # same discrepancy mid-chunk: not the tier's business

    monitor.reset()
    for i in range(2):
        monitor.update(action=IDLE, eef_pos=(0.01 * i, 0.0, 0.0), chunk_start=False)
    boundary = monitor.update(action=head, eef_pos=(0.02, 0.0, 0.0), chunk_start=True)
    assert boundary.continuity and boundary.cause == "continuity"


def test_monitor_continuity_disabled_at_zero_eps() -> None:
    monitor = StuckMonitor(window_steps=3)  # continuity_eps defaults to 0
    for _ in range(3):
        monitor.update(action=IDLE, eef_pos=(0.0, 0.0, 0.0), chunk_start=False)
    signal = monitor.update(
        action=[9.0] * 6 + [-1.0], eef_pos=(0.0, 0.0, 0.0), chunk_start=True
    )
    assert not signal.continuity


# ---------------------------------------------------------------------------
# RecoveryController + quaternion helpers.
# ---------------------------------------------------------------------------


def test_quat_error_axis_angle_identity_and_known_rotation() -> None:
    assert _quat_error_axis_angle(IDENTITY, IDENTITY) == (0.0, 0.0, 0.0)
    quarter_z = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # 90° about z
    rx, ry, rz = _quat_error_axis_angle(quarter_z, IDENTITY)
    assert (rx, ry) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert rz == pytest.approx(math.pi / 2, abs=1e-9)


def test_controller_action_points_toward_target_and_clips() -> None:
    controller = RecoveryController(kp_pos=8.0, max_action=1.0)
    controller.start(_snapshot(0, pos=(1.0, 0.0, 0.0)))  # 1 m away in +x
    action = controller.action(eef_pos=(0.0, 0.0, 0.0), eef_quat_xyzw=IDENTITY)
    assert action is not None
    assert action[0] == pytest.approx(1.0)  # kp*1.0 = 8 clipped to +1
    assert action[1:6] == pytest.approx([0.0] * 5)
    assert action[6] == -1.0  # snapshot gripper open (width 0.08)


def test_controller_arrives_within_tolerance() -> None:
    controller = RecoveryController(pos_tol_m=0.01, rot_tol_rad=0.15)
    controller.start(_snapshot(0, pos=(0.0, 0.0, 0.0)))
    assert controller.action(eef_pos=(0.001, 0.0, 0.0), eef_quat_xyzw=IDENTITY) is None
    assert controller.outcome == "arrived"
    assert not controller.active


def test_controller_exhausts_at_max_steps() -> None:
    controller = RecoveryController(max_steps=2)
    controller.start(_snapshot(0, pos=(5.0, 0.0, 0.0)))  # unreachable
    assert controller.action(eef_pos=(0.0, 0.0, 0.0), eef_quat_xyzw=IDENTITY) is not None
    assert controller.action(eef_pos=(0.0, 0.0, 0.0), eef_quat_xyzw=IDENTITY) is not None
    assert controller.action(eef_pos=(0.0, 0.0, 0.0), eef_quat_xyzw=IDENTITY) is None
    assert controller.outcome == "exhausted"
    assert controller.steps_used == 2


def test_controller_gripper_matches_snapshot_width() -> None:
    controller = RecoveryController(gripper_closed_below=0.03)
    closed = _snapshot(0, pos=(1.0, 0.0, 0.0))
    closed.gripper_qpos = (0.01, -0.01)  # width 0.02 -> was holding something
    controller.start(closed)
    action = controller.action(eef_pos=(0.0, 0.0, 0.0), eef_quat_xyzw=IDENTITY)
    assert action is not None and action[6] == 1.0  # +1 closes in LIBERO


# ---------------------------------------------------------------------------
# RecoveryPolicy.
# ---------------------------------------------------------------------------


def test_policy_tier1_trigger_yields_flush_decision() -> None:
    policy = _policy()
    policy.begin_episode(1)
    _boundary(policy, step=0)
    decision = _drive_stuck(policy)
    assert decision.kind == "flush" and decision.cause == "kinematic"
    assert policy.metrics()["flushes"] == 1


def test_policy_specialist_stuck_yields_rollback_to_last_good() -> None:
    # Two-frame timeline: the first boundary has no pair (no submit); verdicts
    # about chunk k land one boundary late. Chunk 1 is vouched clean; chunk 2
    # is judged stuck -> roll back to 1.
    clean1 = SimpleNamespace(episode=1, chunk_index=1, stuck=False)
    stuck2 = SimpleNamespace(episode=1, chunk_index=2, stuck=True)
    gate = _FakeGate(verdicts=[None, None, clean1, stuck2])
    policy = _policy(specialist=gate)
    policy.begin_episode(1)

    _boundary(policy, step=0, pos=(0.0, 0.0, 0.0))  # chunk 0: no pair yet
    policy.after_step(step=0, action=IDLE, eef_pos=(0.0, 0.0, 0.0), chunk_start=True)
    _boundary(policy, step=1, pos=(0.4, 0.0, 0.0))  # chunk 1 (the clean one)
    policy.after_step(step=1, action=IDLE, eef_pos=(0.4, 0.0, 0.0), chunk_start=True)
    _boundary(policy, step=2, pos=(0.5, 0.0, 0.0))  # chunk 2: clean1 drained
    policy.after_step(step=2, action=IDLE, eef_pos=(0.5, 0.0, 0.0), chunk_start=True)
    _boundary(policy, step=3, pos=(0.6, 0.0, 0.0))  # chunk 3: stuck2 drained
    decision = policy.after_step(
        step=3, action=IDLE, eef_pos=(0.6, 0.0, 0.0), chunk_start=True
    )

    assert decision.kind == "rollback" and decision.cause == "specialist"
    assert decision.snapshot is not None and decision.snapshot.chunk_index == 1
    assert policy.recovering  # command mode started the controller
    assert [chunk for chunk, _f, _i in gate.submits] == [1, 2, 3]
    assert policy.metrics()["recoveries_triggered"] == 1


def test_policy_first_boundary_submits_nothing_then_pairs() -> None:
    gate = _FakeGate()
    policy = _policy(specialist=gate)
    policy.begin_episode(1)

    _boundary(policy, step=0)
    assert gate.submits == []  # no pair to compare yet

    _boundary(policy, step=1)
    _boundary(policy, step=2)
    assert [(chunk, frame) for chunk, frame, _i in gate.submits] == [
        (1, ("f0", "f1")),  # (earlier boundary frame, current frame)
        (2, ("f1", "f2")),  # window slides
    ]


def test_policy_frame_pair_gap_widens_the_compare_window() -> None:
    gate = _FakeGate()
    policy = _policy(specialist=gate, frame_pair_gap=2)
    policy.begin_episode(1)

    for step in range(4):
        _boundary(policy, step=step)

    assert [(chunk, frame) for chunk, frame, _i in gate.submits] == [
        (2, ("f0", "f2")),  # first pair spans two boundaries
        (3, ("f1", "f3")),
    ]


def test_policy_rollback_without_good_snapshot_degrades_to_flush() -> None:
    # Stuck since the very first chunk: everything is tainted, nothing to
    # return to -> the rollback degrades to a flush.
    verdict = SimpleNamespace(episode=1, chunk_index=0, stuck=True)
    gate = _FakeGate(verdicts=[verdict])
    policy = _policy(specialist=gate)
    policy.begin_episode(1)
    _boundary(policy, step=0)  # drains the verdict about chunk 0 immediately
    decision = policy.after_step(
        step=0, action=IDLE, eef_pos=(0.0, 0.0, 0.0), chunk_start=True
    )
    assert decision.kind == "flush"
    assert any(e.get("reason") == "no_good_snapshot" for e in policy.events)


def test_policy_shadow_mode_logs_but_never_intervenes() -> None:
    policy = _policy(shadow=True)
    policy.begin_episode(1)
    _boundary(policy, step=0)
    decision = _drive_stuck(policy)
    assert decision.kind == "none"
    assert not policy.recovering
    assert policy.metrics()["shadow_triggers"] == 1
    assert policy.metrics()["flushes"] == 0
    assert [e["applied"] for e in policy.events] == [False]


def test_policy_budget_exhaustion_stops_intervening() -> None:
    policy = _policy(max_recoveries=1)
    policy.begin_episode(1)
    _boundary(policy, step=0)
    first = _drive_stuck(policy)
    assert first.kind == "flush"
    second = _drive_stuck(policy, start_step=10)
    assert second.kind == "none"
    assert any(e.get("reason") == "recovery_budget_exhausted" for e in policy.events)


def test_policy_muting_during_command_recovery() -> None:
    verdict = SimpleNamespace(episode=1, chunk_index=1, stuck=True)
    gate = _FakeGate(verdicts=[None, verdict])
    policy = _policy(specialist=gate, settle_steps=2)
    policy.begin_episode(1)
    _boundary(policy, step=0, pos=(0.0, 0.0, 0.0))
    policy.after_step(step=0, action=IDLE, eef_pos=(0.0, 0.0, 0.0), chunk_start=True)
    _boundary(policy, step=1, pos=(5.0, 0.0, 0.0))
    decision = policy.after_step(
        step=1, action=IDLE, eef_pos=(5.0, 0.0, 0.0), chunk_start=True
    )
    assert decision.kind == "rollback" and policy.recovering

    # While recovering, stuck-looking steps yield no new decisions.
    muted = policy.after_step(
        step=2, action=STILL, eef_pos=(5.0, 0.0, 0.0), chunk_start=False
    )
    assert muted.kind == "none"

    # Drive to arrival (target is chunk 0 at origin; current already there).
    assert policy.recovery_action(eef_pos=(0.0, 0.0, 0.0), eef_quat_xyzw=IDENTITY) is None
    assert not policy.recovering
    assert any(e.get("outcome") == "arrived" for e in policy.events)

    # Settle window: two more stuck-ish steps stay muted.
    for step in (3, 4):
        assert (
            policy.after_step(
                step=step, action=STILL, eef_pos=(0.0, 0.0, 0.0), chunk_start=False
            ).kind
            == "none"
        )


def test_policy_teleport_mode_returns_rollback_without_driving() -> None:
    verdict = SimpleNamespace(episode=1, chunk_index=1, stuck=True)
    gate = _FakeGate(verdicts=[None, verdict])
    policy = _policy(specialist=gate, mode="teleport")
    policy.begin_episode(1)
    _boundary(policy, step=0)
    policy.after_step(step=0, action=IDLE, eef_pos=(0.0, 0.0, 0.0), chunk_start=True)
    _boundary(policy, step=1)
    decision = policy.after_step(
        step=1, action=IDLE, eef_pos=(0.0, 0.0, 0.0), chunk_start=True
    )
    assert decision.kind == "rollback"
    assert not policy.recovering  # the recipe performs the state restore
    policy.note_teleport(applied=True)
    assert any(e.get("outcome") == "teleported" for e in policy.events)


def test_policy_stale_verdict_after_rollback_discarded() -> None:
    stuck1 = SimpleNamespace(episode=1, chunk_index=1, stuck=True)
    stale = SimpleNamespace(episode=1, chunk_index=1, stuck=True)  # pre-rollback frame
    wrong_ep = SimpleNamespace(episode=99, chunk_index=5, stuck=True)
    gate = _FakeGate(verdicts=[None, stuck1, stale, wrong_ep])
    policy = _policy(specialist=gate, mode="teleport")
    policy.begin_episode(1)
    _boundary(policy, step=0)
    policy.after_step(step=0, action=IDLE, eef_pos=(0.0, 0.0, 0.0), chunk_start=True)
    _boundary(policy, step=1)
    decision = policy.after_step(
        step=1, action=IDLE, eef_pos=(0.0, 0.0, 0.0), chunk_start=True
    )
    assert decision.kind == "rollback"  # rolled back to chunk 0

    _boundary(policy, step=2)  # drains `stale` (chunk 1 <= rollback target horizon? no: 1 > 0)
    _boundary(policy, step=3)  # drains `wrong_ep`
    assert policy.metrics()["stale_verdicts"] >= 1
    assert any(e["kind"] == "stale_verdict" for e in policy.events)


def test_policy_events_are_json_serializable_and_metrics_counts() -> None:
    policy = _policy()
    policy.begin_episode(1)
    _boundary(policy, step=0)
    _drive_stuck(policy)
    policy.end_episode(success=True)

    json.dumps(policy.events)  # must not raise
    metrics = policy.metrics()
    assert metrics["flushes"] == 1
    assert metrics["episodes_with_recovery"] == 1
    assert metrics["post_recovery_successes"] == 1


def test_policy_write_events_appends_jsonl(tmp_path: Path) -> None:
    policy = _policy()
    policy.begin_episode(1)
    _boundary(policy, step=0)
    _drive_stuck(policy)
    out = policy.write_events(tmp_path / "recovery" / "recovery_events.jsonl")
    assert out is not None and out.exists()
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines and lines[0]["kind"] == "flush"


def test_policy_close_closes_specialist() -> None:
    gate = _FakeGate()
    policy = _policy(specialist=gate)
    policy.close()
    assert gate.closed


# ---------------------------------------------------------------------------
# RolloutLog.
# ---------------------------------------------------------------------------


def test_rollout_log_saves_npz_frames_and_actions(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    log = RolloutLog(tmp_path / "corpus")
    log.begin_episode(3)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    log.add(frame=frame, action=[0.1] * 7)
    log.add(frame=frame, action=[0.2] * 7)

    path = log.save_episode(success=True)

    assert path is not None and path.name == "episode_03_PASS.npz"
    data = np.load(path)
    assert data["frames"].shape == (2, 4, 4, 3)
    assert data["actions"].shape == (2, 7)
    assert data["actions"].dtype == np.float32


def test_rollout_log_empty_episode_saves_nothing(tmp_path: Path) -> None:
    log = RolloutLog(tmp_path)
    log.begin_episode(1)
    assert log.save_episode(success=False) is None
