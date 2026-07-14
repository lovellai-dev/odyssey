"""Grasp-target **Observer** for the UR5e / Robotiq drug-sorting cell.

This module is the home of the **Observer agent** — the specialist that OWNS
deploy-time visual perception *and* the closed-loop correction it drives. GR00T
stays the *coarse pilot* (it reaches, approaches, and holds actuator authority);
the Observer is the *fine* eyes-on-the-target servo that composes a small,
advisory correction into GR00T's command. It owns three responsibilities:

1. **Perception** — :class:`GraspTargetObserver` maps the two deploy frames to
   the 3D vial-cap grasp target (below).
2. **Visual servoing** — :meth:`ObserverAgent.servo` closes the loop: it measures
   the gripper↔target Cartesian error from the live frames and nudges GR00T's
   joint targets toward the vial (and, on the place descent, the nest) through
   damped-least-squares IK, gated to the approach/place window.
3. **Grasp verification + re-grasp** — :meth:`ObserverAgent.verify_grasp` decides
   from vision whether the closed grip is secure, and owns the re-grasp trigger
   that recovers a marginal/missed grasp before the doomed transport.

GR00T keeps action authority (the Observer never actuates directly — it returns a
corrected command GR00T's bridge applies), so the pilot/specialist boundary is
clean and the toggle can A/B raw GR00T against GR00T+Observer.

The iteration-2 policy fired the gripper but closed ~7 cm short on the final
descent (closed-loop eval 0/20, every episode ``lifted=False``): a single
overhead camera cannot resolve descent *depth*. This Observer attacks that
failure directly. It maps the two deploy-time camera frames (``exterior`` +
``wrist`` eye-in-hand) to the **3D grasp target** — the vial cap/neck point in
the robot **base** frame (metres). That absolute 3D target is:

* a **reward signal** for residual RL (penalise the gripper-target gap on the
  descent, exactly where the policy stalls), and
* a **deployable conditioning input**: fed alongside the images it tells the
  policy where to descend, with no ground truth needed at inference (it is
  regressed from RGB, so it runs from real cameras).

Design (mirrors the repo's lazy-heavy-deps convention, e.g. ``cosmos_reason``):
a **frozen** vision backbone + a small trained regression head. DINOv3 is the
first-choice backbone; it is gated on HuggingFace, so the loader falls back to
the open, compact **DINOv2-small** (22 M params, frozen) — a ViT whose dense
features localise the cap/neck well with only a tiny head to train. Only the
head is optimised, so training is cheap (CPU or a brief GPU touch) and never
competes with the big ``launch_finetune`` runs.

The pure helpers (coordinate/metric math, backbone selection) use only
numpy/stdlib and are unit-testable without torch; ``torch``/``transformers`` are
imported lazily inside :meth:`GraspTargetObserver.load`. The class deliberately
does **not** subclass ``torch.nn.Module`` (it composes ``nn.Sequential`` +
a backbone as attributes) so the typed surface stays ``mypy --strict`` clean.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Backbone preference order: DINOv3 (gated) -> DINOv2-small (open, compact).
DEFAULT_BACKBONES: tuple[str, ...] = (
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov2-small",
)
# ImageNet normalisation (both DINOv2 and DINOv3 expect it).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
DEFAULT_IMG_SIZE = 224
TARGET_DIM = 3  # (x, y, z) grasp target in the robot base frame (metres)
VIEW_KEYS: tuple[str, ...] = ("exterior", "wrist")


# ---------------------------------------------------------------------------
# Pure helpers (numpy/stdlib only -> unit-testable without torch)
# ---------------------------------------------------------------------------

def keypoint_error_cm(pred: Any, gt: Any) -> Any:
    """Per-sample Euclidean grasp-target error in **centimetres**.

    ``pred`` / ``gt`` are ``(N, 3)`` (or ``(3,)``) arrays of metres. Returns the
    per-sample error in cm — the headline Observer accuracy metric.
    """
    p = np.asarray(pred, dtype=np.float64).reshape(-1, TARGET_DIM)
    g = np.asarray(gt, dtype=np.float64).reshape(-1, TARGET_DIM)
    return np.linalg.norm(p - g, axis=1) * 100.0


def world_to_base(p_world: Any, base_pos: Any, base_mat: Any) -> Any:
    """Express a world point in the robot base frame: ``R_base^T (p - base_pos)``.

    ``base_mat`` is the 3x3 base rotation (columns = base axes in world), as
    given by MuJoCo ``data.xmat``. Kept here so the deploy client and the
    label generator share one convention.
    """
    r = np.asarray(base_mat, dtype=np.float64).reshape(3, 3)
    return r.T @ (np.asarray(p_world, dtype=np.float64) - np.asarray(base_pos, dtype=np.float64))


def select_backbone(candidates: tuple[str, ...]) -> tuple[str, Any]:
    """Return the first loadable ``(model_id, AutoModel)`` from ``candidates``.

    Tries each id in order (DINOv3 first, DINOv2 fallback); skips gated /
    unreachable weights. Raises ``RuntimeError`` if none load.
    """
    from transformers import AutoModel  # lazy heavy dep

    last_err: Exception | None = None
    for model_id in candidates:
        try:
            model = AutoModel.from_pretrained(model_id)
            return model_id, model
        except Exception as exc:  # gated repo / offline / missing — try the next
            last_err = exc
    raise RuntimeError(
        f"no backbone loadable from {candidates!r}: {last_err}"
    )


# ---------------------------------------------------------------------------
# Observer (heavy model deferred to load())
# ---------------------------------------------------------------------------

class GraspTargetObserver:
    """Frozen ViT backbone + small head: two RGB views -> 3D grasp target.

    Lazy: constructing is cheap (no torch import); the backbone + head load on
    :meth:`load` (or first :meth:`predict`). Pass ``backbone`` to inject a
    callable ``(N,3,H,W) float tensor -> (N, feat_dim) tensor`` (used by tests
    to avoid downloading DINO); otherwise it is resolved via
    :func:`select_backbone`.
    """

    def __init__(
        self,
        *,
        backbones: tuple[str, ...] = DEFAULT_BACKBONES,
        num_views: int = 2,
        head_hidden: int = 256,
        img_size: int = DEFAULT_IMG_SIZE,
        device: str = "cpu",
        feat_dim: int | None = None,
        backbone: Any = None,
    ) -> None:
        self.backbones = backbones
        self.num_views = num_views
        self.head_hidden = head_hidden
        self.img_size = img_size
        self.device = device
        self.backbone_id: str | None = "injected" if backbone is not None else None
        self._feat_dim = feat_dim
        self._backbone: Any = backbone
        self._head: Any = None

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        """Resolve the frozen backbone (if not injected) and build the head."""
        import torch
        from torch import nn

        if self._backbone is None:
            self.backbone_id, self._backbone = select_backbone(self.backbones)
            self._feat_dim = int(self._backbone.config.hidden_size)
            self._backbone = self._backbone.to(self.device).eval()
            for p in self._backbone.parameters():
                p.requires_grad_(False)
        if self._feat_dim is None:
            raise ValueError("feat_dim must be given when injecting a backbone")
        if self._head is None:
            self._head = nn.Sequential(
                nn.Linear(self._feat_dim * self.num_views, self.head_hidden),
                nn.GELU(),
                nn.Linear(self.head_hidden, self.head_hidden),
                nn.GELU(),
                nn.Linear(self.head_hidden, TARGET_DIM),
            ).to(self.device)
        # torch is only needed for the isinstance guard below in predict()
        _ = torch

    # -- tensor path (training / forward) -----------------------------------
    def _prep(self, frames: Any) -> Any:
        """``(N,H,W,3) uint8`` -> normalised ``(N,3,img,img)`` float tensor."""
        import torch
        import torch.nn.functional as fn

        arr = np.asarray(frames)
        if arr.ndim == 3:
            arr = arr[None]
        t = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        t = t.permute(0, 3, 1, 2) / 255.0
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        t = (t - mean) / std
        if t.shape[-1] != self.img_size or t.shape[-2] != self.img_size:
            t = fn.interpolate(
                t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )
        return t

    def _embed_view(self, view_t: Any) -> Any:
        """Frozen backbone features for one preprocessed view -> ``(N, feat_dim)``."""
        import torch

        with torch.no_grad():
            if callable(self._backbone) and not hasattr(self._backbone, "config"):
                return self._backbone(view_t)  # injected test backbone
            out = self._backbone(pixel_values=view_t)
            # DINOv2/DINOv3: CLS token is row 0 of the last hidden state.
            return out.last_hidden_state[:, 0]

    def forward(self, views: list[Any]) -> Any:
        """Preprocessed/raw views (list of ``(N,H,W,3)`` uint8) -> ``(N,3)`` target.

        The backbone runs frozen (no grad); the head is trainable.
        """
        import torch

        self.load()  # idempotent; ensures backbone + head are built
        feats = [self._embed_view(self._prep(v)) for v in views]
        cat = torch.cat(feats, dim=1)
        return self._head(cat)

    # -- deploy inference ---------------------------------------------------
    def predict(self, exterior: Any, wrist: Any) -> Any:
        """Real-camera inference: two ``(H,W,3)`` uint8 frames -> ``(3,)`` xyz (m).

        No ground truth needed — this is the deployable grasp-target signal.
        """
        import torch

        with torch.no_grad():  # forward() loads lazily
            out = self.forward([exterior[None], wrist[None]])
        return out.detach().cpu().numpy().reshape(TARGET_DIM)

    # -- optimisation surface ----------------------------------------------
    def head_parameters(self) -> Any:
        """The trainable head parameters (the backbone stays frozen)."""
        self.load()
        return self._head.parameters()

    def train_mode(self, flag: bool = True) -> None:
        if self._head is not None:
            self._head.train(flag)

    def save(self, path: str) -> None:
        import torch

        if self._head is None:
            raise RuntimeError("nothing to save: head not built")
        torch.save(
            {"head": self._head.state_dict(), "backbone_id": self.backbone_id,
             "feat_dim": self._feat_dim, "img_size": self.img_size,
             "num_views": self.num_views, "head_hidden": self.head_hidden},
            path,
        )

    def load_head(self, path: str) -> None:
        import torch

        ckpt = torch.load(path, map_location=self.device)
        self._feat_dim = int(ckpt["feat_dim"])
        self.img_size = int(ckpt["img_size"])
        self.num_views = int(ckpt["num_views"])
        self.head_hidden = int(ckpt["head_hidden"])
        self.load()
        self._head.load_state_dict(ckpt["head"])


# ===========================================================================
# Observer AGENT — perception + visual servoing + grasp-verify/re-grasp
# ===========================================================================
#
# The three responsibilities below are the Observer's, not GR00T's: GR00T issues
# a coarse action, the Observer *sees* the vial/nest and *corrects* the fine
# approach/place, and the Observer *judges* the grasp and re-grasps if it is bad.
# All the geometry is pure numpy; the heavy pieces (the DINOv2 perception head,
# the phase CNN, the mujoco FK/IK) are injected so the servo/verify surface is
# unit-testable without torch or mujoco.

# The Observer regresses the vial **cap** point (24 mm above the vial-body
# origin; see ``gen_ur5e_perception_data.VIAL_CAP_OFFSET_Z``). The scripted grasp
# puts the pinch 10 mm below the vial-body origin, so the pinch descends to
# ``cap_z - (0.024 + 0.010)`` = 34 mm below the observed cap. The place target is
# the (fixed) nest pocket, with the pinch 23 mm below it. These reuse the data
# expert's measured grasp geometry (``ik.py``) as the single source of truth.
from odyssey.embodiments.ur5e_drugsort.ik import (  # noqa: E402
    GRASP_PINCH_BELOW_VIAL,
    PLACE_PINCH_BELOW_POCKET,
)

VIAL_CAP_OFFSET_Z: float = 0.024
GRASP_PINCH_BELOW_CAP: float = VIAL_CAP_OFFSET_Z + GRASP_PINCH_BELOW_VIAL  # 0.034 m

# Phase-classifier class indices (mirrors success_classifier.PHASE_CLASSES).
PH_REACHING, PH_GRASPED, PH_LIFTED, PH_SEATED = 0, 1, 2, 3

Q6 = tuple[float, float, float, float, float, float]


def _as_q6(x: Any) -> Q6:
    """Coerce any length-6 sequence into a typed 6-tuple of floats."""
    return (float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]))


# ---------------------------------------------------------------------------
# Pure geometry helpers for the servo (numpy/stdlib only)
# ---------------------------------------------------------------------------

def base_to_world(pred_base: Any, base_pos: Any, base_mat: Any) -> Any:
    """Inverse of :func:`world_to_base`: base-frame point -> world.

    ``base_mat`` is the 3x3 base rotation (columns = base axes in world, MuJoCo
    ``data.xmat``); ``world = base_pos + R @ p_base``.
    """
    r = np.asarray(base_mat, dtype=np.float64).reshape(3, 3)
    return np.asarray(base_pos, dtype=np.float64) + r @ np.asarray(pred_base, dtype=np.float64)


def grasp_pinch_target(cap_world: Any) -> Any:
    """World pinch target that grasps the vial whose cap is at ``cap_world``."""
    p = np.asarray(cap_world, dtype=np.float64).reshape(3).copy()
    p[2] -= GRASP_PINCH_BELOW_CAP
    return p


def place_pinch_target(pocket_world: Any) -> Any:
    """World pinch target that seats the vial into the nest at ``pocket_world``."""
    p = np.asarray(pocket_world, dtype=np.float64).reshape(3).copy()
    p[2] -= PLACE_PINCH_BELOW_POCKET
    return p


def bounded_nudge(pinch_cmd: Any, goal: Any, gain: float, max_step: float) -> Any:
    """Move ``pinch_cmd`` a bounded fraction toward ``goal``.

    Returns ``pinch_cmd + clip(gain * (goal - pinch_cmd), max_step)`` with the
    step clipped in Euclidean norm to ``max_step`` (m) — small and local so IK
    stays in GR00T's solution branch and the correction never yanks.
    """
    c = np.asarray(pinch_cmd, dtype=np.float64).reshape(3)
    step = float(gain) * (np.asarray(goal, dtype=np.float64).reshape(3) - c)
    n = float(np.linalg.norm(step))
    if n > max_step and n > 0.0:
        step = step * (max_step / n)
    return c + step


def horiz_dist(a: Any, b: Any) -> float:
    """Horizontal (xy) Euclidean distance between two world points (metres)."""
    aa = np.asarray(a, dtype=np.float64).reshape(3)
    bb = np.asarray(b, dtype=np.float64).reshape(3)
    return float(np.hypot(aa[0] - bb[0], aa[1] - bb[1]))


def in_box(p: Any, lo: Sequence[float], hi: Sequence[float]) -> bool:
    """Sanity gate: is world point ``p`` inside the axis-aligned workspace box?"""
    pp = np.asarray(p, dtype=np.float64).reshape(3)
    return bool(np.all(pp >= np.asarray(lo)) and np.all(pp <= np.asarray(hi)))


# ---------------------------------------------------------------------------
# Config + typed I/O
# ---------------------------------------------------------------------------

@dataclass
class ObserverServoConfig:
    """Tunables for the Observer's servoing + verify/re-grasp (metres)."""

    servo_gain: float = 0.6
    servo_max_step: float = 0.05
    grasp_gate_xy: float = 0.15       # only servo the grasp once pinch is this close in xy
    place_gate_xy: float = 0.15       # only servo the place once pinch is this close in xy
    agree_max: float = 0.20           # skip if the Observer target disagrees with GR00T's cmd

    secure_xy: float = 0.045          # cap must sit within this of the pinch to be "secure"
    regrasp_max_attempts: int = 1
    ticks_per_subphase: int = 2       # queries spent in each re-grasp sub-phase
    regrasp_lift_clear: float = 0.06  # rise this far above the cap before re-descending

    grip_closed_thresh: float = 0.30

    # Workspace sanity box for the observed cap (world frame).
    ws_lo: tuple[float, float, float] = (0.20, -0.45, 0.16)
    ws_hi: tuple[float, float, float] = (0.75, 0.45, 0.45)


@dataclass
class ObserverReading:
    """One frame's Observer perception: where the vial is + what phase we're in."""

    cap_base: Any        # (3,) raw perception output, robot base frame
    cap_world: Any       # (3,) cap in world frame
    cap_ok: bool         # inside the workspace sanity box
    phase_label: int     # 0 reaching / 1 grasped / 2 lifted / 3 seated
    phase_probs: Any     # (4,) softmax probabilities


@dataclass
class CorrectedAction:
    """GR00T's action after the Observer's advisory correction is composed in."""

    chunk_q: list[Q6]
    chunk_grip: list[float]
    mode: str            # passthrough | servo_grasp | servo_place | regrasp
    corrected: bool
    debug: dict[str, Any] = field(default_factory=dict)


# Re-grasp sub-phases.
_RG_IDLE, _RG_OPEN, _RG_DESCEND, _RG_CLOSE = "idle", "open", "descend", "close"


@dataclass
class ObserverAgent:
    """The Observer as an agent: perceive -> servo GR00T's fine motion -> verify.

    The Observer OWNS the correction. GR00T proposes a coarse action chunk; the
    Observer perceives the vial/nest, and:

    * :meth:`servo` nudges the commanded joint targets so the Robotiq pinch
      converges on the observed cap during the grasp descent (and on the known
      nest during the place descent), and
    * :meth:`verify_grasp` judges the closed grip and triggers a bounded
      re-open/re-descend/re-close recovery when the grasp is marginal.

    GR00T retains actuator authority: the returned :class:`CorrectedAction` is
    applied through GR00T's own bridge. ``perception`` (:class:`GraspTargetObserver`)
    and ``verifier`` (:class:`SuccessClassifier`) are the torch models; ``ik_solve``
    / ``fk`` are the injected mujoco Cartesian<->joint maps. ``servo`` and
    ``verify_grasp`` never touch torch/mujoco directly (they consume a
    pre-computed :class:`ObserverReading` + the injected FK/IK), so they unit-test
    with analytic fakes.
    """

    ik_solve: Callable[[Any, Q6], Q6]
    fk: Callable[[Q6], Any]
    base_pos: Any
    base_mat: Any
    pocket_world: Any
    perception: Any = None      # GraspTargetObserver (torch); optional for servo-only tests
    verifier: Any = None        # SuccessClassifier (torch); optional for servo-only tests
    config: ObserverServoConfig = field(default_factory=ObserverServoConfig)

    _rg_phase: str = field(default=_RG_IDLE, init=False)
    _rg_ticks: int = field(default=0, init=False)
    _rg_attempts: int = field(default=0, init=False)

    def reset(self) -> None:
        """Reset the per-episode re-grasp latch (call at the start of a run)."""
        self._rg_phase, self._rg_ticks, self._rg_attempts = _RG_IDLE, 0, 0

    # -- 1) perception ------------------------------------------------------
    def perceive(self, exterior: Any, wrist: Any) -> ObserverReading:
        """Run the Observer head + phase classifier on the two live frames."""
        cap_base = np.asarray(self.perception.predict(exterior, wrist), dtype=np.float64)
        label, probs = self.verifier.predict([exterior, wrist])
        cap_world = base_to_world(cap_base, self.base_pos, self.base_mat)
        cap_ok = in_box(cap_world, self.config.ws_lo, self.config.ws_hi)
        return ObserverReading(cap_base, cap_world, cap_ok, int(label), probs)

    # -- 3) grasp verification ---------------------------------------------
    def verify_grasp(self, reading: ObserverReading, arm_now: Q6) -> bool:
        """Is the closed grip secure? (vision: holding phase AND cap under the pads)."""
        if reading.phase_label not in (PH_GRASPED, PH_LIFTED, PH_SEATED):
            return False
        pinch = self.fk(_as_q6(arm_now))
        return horiz_dist(pinch, reading.cap_world) <= self.config.secure_xy

    # -- 2) visual servoing + re-grasp (composed into GR00T's command) ------
    def servo(
        self,
        reading: ObserverReading,
        pilot_action: dict[str, Any],
        arm_now: Q6,
    ) -> CorrectedAction:
        """Compose the Observer's advisory correction into GR00T's action chunk."""
        cfg = self.config
        raw_q = [_as_q6(q) for q in pilot_action["chunk_q"]]
        raw_grip = [float(g) for g in pilot_action["chunk_grip"]]
        n = len(raw_q)

        grasp_tgt = grasp_pinch_target(reading.cap_world)
        place_tgt = place_pinch_target(self.pocket_world)
        pinch_now = np.asarray(self.fk(_as_q6(arm_now)), dtype=np.float64).reshape(3)
        dxy_cap = horiz_dist(pinch_now, reading.cap_world)
        dxy_pocket = horiz_dist(pinch_now, self.pocket_world)
        grip_closed = (len(raw_grip) > 0 and raw_grip[0] >= cfg.grip_closed_thresh)

        dbg: dict[str, Any] = {
            "cap_world": [round(float(x), 4) for x in reading.cap_world],
            "cap_ok": reading.cap_ok, "phase": reading.phase_label,
            "dxy_cap": round(dxy_cap, 4), "dxy_pocket": round(dxy_pocket, 4),
            "rg_phase": self._rg_phase, "rg_attempts": self._rg_attempts,
        }

        # (0) an active re-grasp latch runs the scripted recovery to completion.
        if self._rg_phase != _RG_IDLE:
            return self._run_regrasp(grasp_tgt, arm_now, n, dbg)

        # (1) verify: closed but not securely holding -> start a bounded re-grasp.
        secure = self.verify_grasp(reading, arm_now)
        missed = (
            reading.cap_ok and grip_closed
            and reading.phase_label in (PH_REACHING, PH_GRASPED)
            and not secure
        )
        if missed and self._rg_attempts < cfg.regrasp_max_attempts:
            self._rg_phase, self._rg_ticks = _RG_OPEN, 0
            dbg["trigger"] = "regrasp"
            return self._run_regrasp(grasp_tgt, arm_now, n, dbg)

        # (2) servo the pre-grasp descent onto the observed cap.
        if (
            reading.cap_ok and (not grip_closed)
            and reading.phase_label == PH_REACHING and dxy_cap < cfg.grasp_gate_xy
        ):
            new_q, touched = self._servo_chunk(raw_q, grasp_tgt)
            if touched:
                dbg["goal"] = [round(float(x), 4) for x in grasp_tgt]
                return CorrectedAction(new_q, raw_grip, "servo_grasp", True, dbg)

        # (3) servo the place descent onto the known nest pocket.
        if grip_closed and reading.phase_label == PH_LIFTED and dxy_pocket < cfg.place_gate_xy:
            new_q, touched = self._servo_chunk(raw_q, place_tgt)
            if touched:
                dbg["goal"] = [round(float(x), 4) for x in place_tgt]
                return CorrectedAction(new_q, raw_grip, "servo_place", True, dbg)

        # (4) passthrough: raw GR00T action, unmodified.
        return CorrectedAction(raw_q, raw_grip, "passthrough", False, dbg)

    def correct(
        self, *, exterior: Any, wrist: Any, arm_now: Q6, pilot_action: dict[str, Any]
    ) -> CorrectedAction:
        """Full per-tick path: perceive the frames, then servo GR00T's command."""
        return self.servo(self.perceive(exterior, wrist), pilot_action, arm_now)

    # -- internals ----------------------------------------------------------
    def _servo_chunk(self, chunk_q: Sequence[Q6], goal: Any) -> tuple[list[Q6], bool]:
        """Nudge every commanded pose toward ``goal`` via bounded IK, skipping any
        step where GR00T and the Observer disagree by more than ``agree_max``."""
        cfg = self.config
        g = np.asarray(goal, dtype=np.float64).reshape(3)
        out: list[Q6] = []
        touched = False
        for q in chunk_q:
            qq = _as_q6(q)
            pinch_cmd = np.asarray(self.fk(qq), dtype=np.float64).reshape(3)
            if float(np.linalg.norm(g - pinch_cmd)) > cfg.agree_max:
                out.append(qq)                # perception disagrees -> trust GR00T
                continue
            tgt = bounded_nudge(pinch_cmd, g, cfg.servo_gain, cfg.servo_max_step)
            out.append(_as_q6(self.ik_solve(tgt, qq)))
            touched = True
        return out, touched

    def _run_regrasp(self, grasp_tgt: Any, arm_now: Q6, n: int, dbg: dict[str, Any]) -> CorrectedAction:
        """Advance the bounded re-open -> re-descend -> re-close recovery."""
        cfg = self.config
        seed = _as_q6(arm_now)
        grip: float
        if self._rg_phase == _RG_OPEN:
            above = np.asarray(grasp_tgt, dtype=np.float64).reshape(3).copy()
            above[2] += cfg.regrasp_lift_clear
            q, grip = _as_q6(self.ik_solve(above, seed)), 0.0     # open + rise above cap
        elif self._rg_phase == _RG_DESCEND:
            q, grip = _as_q6(self.ik_solve(grasp_tgt, seed)), 0.0  # open + descend onto cap
        else:  # _RG_CLOSE
            q, grip = _as_q6(self.ik_solve(grasp_tgt, seed)), 1.0  # close on the cap

        self._rg_ticks += 1
        if self._rg_ticks >= cfg.ticks_per_subphase:
            self._rg_ticks = 0
            self._rg_phase = {
                _RG_OPEN: _RG_DESCEND, _RG_DESCEND: _RG_CLOSE, _RG_CLOSE: _RG_IDLE,
            }[self._rg_phase]
            if self._rg_phase == _RG_IDLE:
                self._rg_attempts += 1               # recovery done -> hand back to GR00T
        dbg["rg_phase"] = self._rg_phase
        return CorrectedAction([q] * n, [float(grip)] * n, "regrasp", True, dbg)
