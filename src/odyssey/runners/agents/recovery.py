"""Closed-loop recovery core — ledger, tiered stuck detection, rollback control.

The recovery experiment (``docs/experiment-recovery-design-vla.md``) runs a
tiered stuck detector inside the LIBERO eval loop of a chunk-emitting pilot
and, on trigger, either logs the event (shadow mode) or intervenes:

* tiers 1-2 (kinematic / inter-chunk continuity, both training-free) trigger a
  **flush** — the recipe calls ``ChunkPilotAdapter.flush()`` so the stale chunk
  is truncated and the policy re-queried from the current state (VLA-Corrector's
  event-triggered truncation);
* tier 4 (a SPECIALIST VLM verdict, delivered asynchronously by
  ``specialist_gate.SpecialistGate``) triggers a **rollback** — return the arm
  to the last chunk boundary whose verdicts were clean, then flush + re-query.

Everything here is pure Python over sequences of floats (math/dataclasses
only — numpy appears lazily inside ``RolloutLog.save_episode`` alone), so the
module is strict-mypy clean, imports under bare stdlib, and is unit-testable
with plain fakes — the same bet as ``chunk_pilot.py``. The mypy-exempt eval
recipes convert ndarrays to lists at the call sites.
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

_TIERS = ("kinematic", "continuity", "specialist")


# ---------------------------------------------------------------------------
# Quaternion helpers (xyzw order, matching robosuite's ``robot0_eef_quat``).
# ---------------------------------------------------------------------------


def _quat_conj(q: Sequence[float]) -> tuple[float, float, float, float]:
    return (-q[0], -q[1], -q[2], q[3])


def _quat_mul(
    a: Sequence[float], b: Sequence[float]
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_error_axis_angle(
    target_xyzw: Sequence[float], current_xyzw: Sequence[float]
) -> tuple[float, float, float]:
    """Axis-angle rotation carrying ``current`` onto ``target`` (shortest path)."""
    err = _quat_mul(target_xyzw, _quat_conj(current_xyzw))
    x, y, z, w = err
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return (0.0, 0.0, 0.0)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    if w < 0.0:  # shortest path
        x, y, z, w = -x, -y, -z, -w
    vec_norm = math.sqrt(x * x + y * y + z * z)
    if vec_norm < 1e-9:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(vec_norm, w)
    scale = angle / vec_norm
    return (x * scale, y * scale, z * scale)


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=False)))


def _clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


# ---------------------------------------------------------------------------
# Ledger — chunk-boundary snapshots; ``last_good()`` defines "last successful
# action chunk".
# ---------------------------------------------------------------------------


@dataclass
class BoundarySnapshot:
    """State the arm actually held at a chunk boundary (recorded before act)."""

    step: int
    chunk_index: int
    eef_pos: tuple[float, float, float]
    eef_quat_xyzw: tuple[float, float, float, float]
    gripper_qpos: tuple[float, float]
    sim_state: Any | None = None  # opaque; only teleport mode touches it
    kinematic_ok: bool = True
    continuity_ok: bool = True
    specialist_ok: bool | None = None  # None = no verdict yet (counts as clean)

    @property
    def good(self) -> bool:
        return self.kinematic_ok and self.continuity_ok and self.specialist_ok is not False


class ChunkLedger:
    """Ring of boundary snapshots + per-tier verdicts landing after the fact."""

    def __init__(self, *, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._snapshots: deque[BoundarySnapshot] = deque(maxlen=capacity)

    def record(self, snapshot: BoundarySnapshot) -> None:
        self._snapshots.append(snapshot)

    def mark(self, chunk_index: int, *, tier: str, ok: bool) -> None:
        """Apply a (possibly late) verdict to the snapshot for ``chunk_index``."""
        if tier not in _TIERS:
            raise ValueError(f"unknown tier {tier!r}; expected one of {_TIERS}")
        for snapshot in self._snapshots:
            if snapshot.chunk_index == chunk_index:
                if tier == "kinematic":
                    snapshot.kinematic_ok = ok
                elif tier == "continuity":
                    snapshot.continuity_ok = ok
                else:
                    snapshot.specialist_ok = ok
                return

    def mark_from(self, chunk_index: int, *, tier: str, ok: bool) -> None:
        """Apply a verdict to every snapshot at or after ``chunk_index``.

        A stuck state persists: a specialist verdict about chunk *k* taints all
        boundaries recorded since — the rollback target must predate it.
        """
        for snapshot in list(self._snapshots):
            if snapshot.chunk_index >= chunk_index:
                self.mark(snapshot.chunk_index, tier=tier, ok=ok)

    def last_good(self) -> BoundarySnapshot | None:
        for snapshot in reversed(self._snapshots):
            if snapshot.good:
                return snapshot
        return None

    def latest(self) -> BoundarySnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def reset(self) -> None:
        self._snapshots.clear()


# ---------------------------------------------------------------------------
# Tiered stuck detection (tiers 1-2; the specialist is tier 4, delivered via
# the gate; the trained LVM tier 3 arrives in a later PR).
# ---------------------------------------------------------------------------


@dataclass
class StuckSignal:
    kinematic: bool = False
    continuity: bool = False

    @property
    def triggered(self) -> bool:
        return self.kinematic or self.continuity

    @property
    def cause(self) -> str:
        if self.kinematic:
            return "kinematic"
        if self.continuity:
            return "continuity"
        return ""


class StuckMonitor:
    """Training-free stuck detection over the (action, EE-pos) stream.

    Tier 1 (kinematic, every step): over the last ``window_steps`` steps the
    commanded translation summed to >= ``min_commanded`` while the EE moved
    less than ``eps_m`` — the arm is pushing but not going anywhere.

    Tier 2 (continuity, chunk starts only): the new chunk's first action
    diverges from the mean of the previous chunk's last ``continuity_tail``
    actions by more than ``continuity_eps`` over the 6 pose dims — the policy
    changed its mind sharply between chunks (Rewind-IL's TIDE signal at our
    query cadence). ``continuity_eps=0`` disables the tier.
    """

    def __init__(
        self,
        *,
        window_steps: int = 12,
        eps_m: float = 0.005,
        min_commanded: float = 0.05,
        continuity_eps: float = 0.0,
        continuity_tail: int = 3,
    ) -> None:
        if window_steps < 2:
            raise ValueError(f"window_steps must be >= 2, got {window_steps}")
        self._window = int(window_steps)
        self._eps_m = float(eps_m)
        self._min_commanded = float(min_commanded)
        self._continuity_eps = float(continuity_eps)
        self._continuity_tail = max(1, int(continuity_tail))
        self._positions: deque[tuple[float, ...]] = deque(maxlen=self._window)
        self._commanded: deque[float] = deque(maxlen=self._window)
        self._recent_actions: deque[tuple[float, ...]] = deque(maxlen=self._continuity_tail)

    def update(
        self,
        *,
        action: Sequence[float],
        eef_pos: Sequence[float],
        chunk_start: bool,
    ) -> StuckSignal:
        signal = StuckSignal()

        if (
            chunk_start
            and self._continuity_eps > 0.0
            and len(self._recent_actions) == self._continuity_tail
        ):
            tail = [
                sum(a[i] for a in self._recent_actions) / len(self._recent_actions)
                for i in range(6)
            ]
            head = [float(action[i]) for i in range(6)]
            if _l2(head, tail) > self._continuity_eps:
                signal.continuity = True

        self._positions.append(tuple(float(v) for v in eef_pos[:3]))
        commanded = math.sqrt(sum(float(action[i]) ** 2 for i in range(3)))
        self._commanded.append(commanded)
        self._recent_actions.append(tuple(float(v) for v in action[:6]))

        if len(self._positions) == self._window:
            displacement = _l2(self._positions[-1], self._positions[0])
            if sum(self._commanded) >= self._min_commanded and displacement < self._eps_m:
                signal.kinematic = True

        return signal

    def reset(self) -> None:
        self._positions.clear()
        self._commanded.clear()
        self._recent_actions.clear()


# ---------------------------------------------------------------------------
# Command-back recovery controller — EE-space P-control emitting the env's
# native 7-D OSC delta actions toward a boundary snapshot.
# ---------------------------------------------------------------------------


class RecoveryController:
    """Drive the arm back to a snapshot with clipped EE-delta actions."""

    def __init__(
        self,
        *,
        kp_pos: float = 8.0,
        kp_rot: float = 2.0,
        max_steps: int = 24,
        pos_tol_m: float = 0.01,
        rot_tol_rad: float = 0.15,
        max_action: float = 1.0,
        gripper_closed_below: float = 0.03,
    ) -> None:
        self._kp_pos = float(kp_pos)
        self._kp_rot = float(kp_rot)
        self._max_steps = int(max_steps)
        self._pos_tol = float(pos_tol_m)
        self._rot_tol = float(rot_tol_rad)
        self._max_action = float(max_action)
        self._gripper_closed_below = float(gripper_closed_below)
        self._target: BoundarySnapshot | None = None
        self._steps = 0
        self._outcome = "idle"

    @property
    def active(self) -> bool:
        return self._target is not None

    @property
    def outcome(self) -> str:
        return self._outcome

    @property
    def steps_used(self) -> int:
        return self._steps

    def start(self, target: BoundarySnapshot) -> None:
        self._target = target
        self._steps = 0
        self._outcome = "driving"

    def action(
        self, *, eef_pos: Sequence[float], eef_quat_xyzw: Sequence[float]
    ) -> list[float] | None:
        """One 7-D OSC delta toward the target; None = arrived/exhausted/idle."""
        target = self._target
        if target is None:
            return None

        pos_err = [target.eef_pos[i] - float(eef_pos[i]) for i in range(3)]
        rot_err = _quat_error_axis_angle(target.eef_quat_xyzw, eef_quat_xyzw)
        pos_dist = math.sqrt(sum(e * e for e in pos_err))
        rot_dist = math.sqrt(sum(e * e for e in rot_err))

        if pos_dist < self._pos_tol and rot_dist < self._rot_tol:
            self._target = None
            self._outcome = "arrived"
            return None
        if self._steps >= self._max_steps:
            self._target = None
            self._outcome = "exhausted"
            return None

        self._steps += 1
        gripper_width = abs(target.gripper_qpos[0]) + abs(target.gripper_qpos[1])
        # LIBERO gripper action: -1.0 opens, +1.0 closes (warmup no-op uses -1).
        gripper_cmd = 1.0 if gripper_width < self._gripper_closed_below else -1.0
        return [
            *(_clip(self._kp_pos * e, self._max_action) for e in pos_err),
            *(_clip(self._kp_rot * e, self._max_action) for e in rot_err),
            gripper_cmd,
        ]

    def reset(self) -> None:
        self._target = None
        self._steps = 0
        self._outcome = "idle"


# ---------------------------------------------------------------------------
# Orchestrator — glues ledger + monitor + controller + specialist gate; owns
# shadow mode, the recovery budget, staleness rules, events and metrics.
# ---------------------------------------------------------------------------


@runtime_checkable
class SpecialistGateLike(Protocol):
    """Structural surface of ``specialist_gate.SpecialistGate`` (or a fake)."""

    def begin_episode(self, episode: int) -> None: ...

    def submit(self, *, chunk_index: int, frame: Any, instruction: str) -> bool: ...

    def poll(self) -> Any | None: ...

    def close(self) -> None: ...


@dataclass
class RecoveryDecision:
    kind: str = "none"  # "none" | "flush" | "rollback"
    snapshot: BoundarySnapshot | None = None
    cause: str = ""  # "kinematic" | "continuity" | "specialist"


@dataclass
class _EpisodeState:
    number: int = 0
    chunk_index: int = -1
    interventions: int = 0
    settle_remaining: int = 0
    pending_specialist: Any | None = None
    stale_horizon: int = -1  # verdicts about chunks <= this predate a rollback
    had_intervention: bool = False


class RecoveryPolicy:
    """Per-episode recovery orchestration for one eval rollout loop.

    Call order per step (see the recipes): ``on_chunk_boundary`` before the
    boundary ``act`` (records the held pose, drains + re-arms the specialist),
    then ``after_step`` after ``env.step`` (feeds the monitor, returns the
    decision). While ``recovering`` the recipe pulls ``recovery_action``
    instead of ``pilot.act`` and the monitors stay muted (plus
    ``settle_steps`` afterwards).
    """

    def __init__(
        self,
        *,
        ledger: ChunkLedger,
        monitor: StuckMonitor,
        controller: RecoveryController,
        specialist: SpecialistGateLike | None = None,
        mode: str = "command",
        shadow: bool = False,
        max_recoveries: int = 3,
        settle_steps: int = 5,
    ) -> None:
        if mode not in ("command", "teleport"):
            raise ValueError(f"mode must be 'command' or 'teleport', got {mode!r}")
        self._ledger = ledger
        self._monitor = monitor
        self._controller = controller
        self._specialist = specialist
        self._mode = mode
        self._shadow = bool(shadow)
        self._max_recoveries = int(max_recoveries)
        self._settle_steps = int(settle_steps)
        self._ep = _EpisodeState()
        self._events: list[dict[str, Any]] = []
        self._metrics = {
            "recoveries_triggered": 0,
            "flushes": 0,
            "shadow_triggers": 0,
            "specialist_polls": 0,
            "stale_verdicts": 0,
            "episodes_with_recovery": 0,
            "post_recovery_successes": 0,
            "budget_exhausted_episodes": 0,
        }

    # -- episode lifecycle --------------------------------------------------

    def begin_episode(self, episode: int) -> None:
        self._ep = _EpisodeState(number=episode)
        self._ledger.reset()
        self._monitor.reset()
        self._controller.reset()
        if self._specialist is not None:
            self._specialist.begin_episode(episode)

    def end_episode(self, *, success: bool) -> None:
        if self._ep.had_intervention:
            self._metrics["episodes_with_recovery"] += 1
            if success:
                self._metrics["post_recovery_successes"] += 1
        if self._ep.interventions >= self._max_recoveries and self._ep.had_intervention:
            self._metrics["budget_exhausted_episodes"] += 1

    # -- chunk boundaries ---------------------------------------------------

    def on_chunk_boundary(
        self,
        *,
        step: int,
        frame: Any,
        instruction: str,
        eef_pos: Sequence[float],
        eef_quat_xyzw: Sequence[float],
        gripper_qpos: Sequence[float],
        sim_state: Any | None = None,
    ) -> None:
        self._ep.chunk_index += 1
        self._ledger.record(
            BoundarySnapshot(
                step=step,
                chunk_index=self._ep.chunk_index,
                eef_pos=(float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2])),
                eef_quat_xyzw=(
                    float(eef_quat_xyzw[0]),
                    float(eef_quat_xyzw[1]),
                    float(eef_quat_xyzw[2]),
                    float(eef_quat_xyzw[3]),
                ),
                gripper_qpos=(float(gripper_qpos[0]), float(gripper_qpos[1])),
                sim_state=sim_state,
            )
        )
        if self._specialist is None:
            return
        verdict = self._specialist.poll()
        if verdict is not None:
            self._consume_verdict(verdict)
        if self._specialist.submit(
            chunk_index=self._ep.chunk_index, frame=frame, instruction=instruction
        ):
            self._metrics["specialist_polls"] += 1

    def _consume_verdict(self, verdict: Any) -> None:
        episode = int(getattr(verdict, "episode", -1))
        chunk_index = int(getattr(verdict, "chunk_index", -1))
        stuck = bool(getattr(verdict, "stuck", False))
        if episode != self._ep.number or chunk_index <= self._ep.stale_horizon:
            self._metrics["stale_verdicts"] += 1
            self._events.append(
                {
                    "kind": "stale_verdict",
                    "episode": self._ep.number,
                    "verdict_episode": episode,
                    "verdict_chunk": chunk_index,
                }
            )
            return
        if stuck:
            # A stuck state persists: taint every boundary since the frame.
            self._ledger.mark_from(chunk_index, tier="specialist", ok=False)
            self._ep.pending_specialist = verdict
        else:
            # A clean verdict vouches only for the chunk it actually saw.
            self._ledger.mark(chunk_index, tier="specialist", ok=True)

    # -- per-step decision --------------------------------------------------

    def after_step(
        self,
        *,
        step: int,
        action: Sequence[float],
        eef_pos: Sequence[float],
        chunk_start: bool,
    ) -> RecoveryDecision:
        if self._controller.active:
            return RecoveryDecision()
        if self._ep.settle_remaining > 0:
            self._ep.settle_remaining -= 1
            return RecoveryDecision()

        signal = self._monitor.update(
            action=action, eef_pos=eef_pos, chunk_start=chunk_start
        )

        if self._ep.pending_specialist is not None:
            self._ep.pending_specialist = None
            target = self._ledger.last_good()
            return self._decide(
                step=step, cause="specialist", kind="rollback", snapshot=target
            )

        if signal.triggered:
            self._ledger.mark(self._ep.chunk_index, tier=signal.cause, ok=False)
            return self._decide(step=step, cause=signal.cause, kind="flush", snapshot=None)

        return RecoveryDecision()

    def _decide(
        self,
        *,
        step: int,
        cause: str,
        kind: str,
        snapshot: BoundarySnapshot | None,
    ) -> RecoveryDecision:
        event: dict[str, Any] = {
            "kind": kind,
            "cause": cause,
            "episode": self._ep.number,
            "step": step,
            "chunk_index": self._ep.chunk_index,
            "mode": self._mode,
            "shadow": self._shadow,
        }
        if snapshot is not None:
            event["target_chunk"] = snapshot.chunk_index
            event["target_step"] = snapshot.step

        if self._shadow:
            event["applied"] = False
            self._metrics["shadow_triggers"] += 1
            self._events.append(event)
            self._reset_after_trigger()
            return RecoveryDecision()

        if self._ep.interventions >= self._max_recoveries:
            event["applied"] = False
            event["reason"] = "recovery_budget_exhausted"
            self._events.append(event)
            self._reset_after_trigger()
            return RecoveryDecision()

        if kind == "rollback" and snapshot is None:
            # Nothing clean to return to: degrade to a flush.
            kind = "flush"
            event["kind"] = "flush"
            event["reason"] = "no_good_snapshot"

        event["applied"] = True
        self._events.append(event)
        self._ep.interventions += 1
        self._ep.had_intervention = True
        self._reset_after_trigger()

        if kind == "flush":
            self._metrics["flushes"] += 1
            return RecoveryDecision(kind="flush", cause=cause)

        assert snapshot is not None
        self._metrics["recoveries_triggered"] += 1
        # Any in-flight verdict was captured at or before the current chunk —
        # it describes the pre-rollback world and must not re-trigger.
        self._ep.stale_horizon = self._ep.chunk_index
        if self._mode == "command":
            self._controller.start(snapshot)
        return RecoveryDecision(kind="rollback", snapshot=snapshot, cause=cause)

    def _reset_after_trigger(self) -> None:
        self._monitor.reset()
        self._ep.settle_remaining = self._settle_steps

    # -- recovery drive (command mode) --------------------------------------

    @property
    def recovering(self) -> bool:
        return self._controller.active

    def recovery_action(
        self, *, eef_pos: Sequence[float], eef_quat_xyzw: Sequence[float]
    ) -> list[float] | None:
        action = self._controller.action(eef_pos=eef_pos, eef_quat_xyzw=eef_quat_xyzw)
        if action is None and self._controller.outcome in ("arrived", "exhausted"):
            self._events.append(
                {
                    "kind": "recovery_outcome",
                    "episode": self._ep.number,
                    "outcome": self._controller.outcome,
                    "steps_used": self._controller.steps_used,
                }
            )
            self._ep.settle_remaining = self._settle_steps
        return action

    def note_teleport(self, *, applied: bool) -> None:
        """Record the recipe-side teleport outcome (state restore or degrade)."""
        self._events.append(
            {
                "kind": "recovery_outcome",
                "episode": self._ep.number,
                "outcome": "teleported" if applied else "teleport_unavailable",
            }
        )
        self._ep.settle_remaining = self._settle_steps

    # -- reporting -----------------------------------------------------------

    @property
    def events(self) -> list[dict[str, Any]]:
        return self._events

    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    def write_events(self, path: str | Path) -> Path | None:
        """Append events as JSON lines; best-effort like video encoding."""
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                for event in self._events:
                    fh.write(json.dumps(event) + "\n")
            return target
        except OSError:
            logger.warning("recovery: failed to write events to %s", path, exc_info=True)
            return None

    def close(self) -> None:
        if self._specialist is not None:
            self._specialist.close()


# ---------------------------------------------------------------------------
# Rollout corpus logging — (frame, action) pairs per step, one npz per
# episode. This is the LVM tier's future training corpus (PR 2).
# ---------------------------------------------------------------------------


class RolloutLog:
    """Accumulate per-step (frame, action) pairs; save one npz per episode."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._episode = 0
        self._frames: list[Any] = []
        self._actions: list[Sequence[float]] = []

    def begin_episode(self, episode: int) -> None:
        self._episode = episode
        self._frames = []
        self._actions = []

    def add(self, *, frame: Any, action: Sequence[float]) -> None:
        if frame is not None:
            self._frames.append(frame)
            self._actions.append([float(v) for v in action])

    def save_episode(self, *, success: bool) -> Path | None:
        """Write ``episode_{ep:02d}_{PASS|FAIL}.npz`` (frames + actions); None on error."""
        if not self._frames:
            return None
        try:
            import numpy as np

            self._dir.mkdir(parents=True, exist_ok=True)
            tag = "PASS" if success else "FAIL"
            path = self._dir / f"episode_{self._episode:02d}_{tag}.npz"
            np.savez_compressed(
                path,
                frames=np.asarray(self._frames, dtype=np.uint8),
                actions=np.asarray(self._actions, dtype=np.float32),
            )
            return path
        except Exception:
            logger.warning(
                "recovery: failed to save rollout log for episode %d",
                self._episode,
                exc_info=True,
            )
            return None
        finally:
            self._frames = []
            self._actions = []
