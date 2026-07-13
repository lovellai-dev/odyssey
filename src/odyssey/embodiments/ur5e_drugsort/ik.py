"""Damped-least-squares IK expert for the UR5e / Robotiq drug-sorting cell.

The scripted :mod:`odyssey.embodiments.ur5e_drugsort.policy` replays a *fixed*
sequence of joint waypoints, so every episode's action/proprio trajectory is
byte-for-byte identical regardless of where the vial actually is — a policy
trained on that data would learn to ignore the camera. This module makes the
expert **adaptive**: each episode it reads the (randomised) ``vial_0`` and
``pocket_0`` poses and *solves* for the arm joint targets that put the Robotiq
pinch point over the real vial, so the recorded trajectory genuinely varies with
the scene.

How it works
------------
* :class:`DampedLeastSquaresIK` is a position-only Jacobian IK solver. It servos
  the ``gr_pinch`` site to a Cartesian target via the damped-least-squares (a.k.a.
  Levenberg-Marquardt) update ``dq = J^T (J J^T + lambda^2 I)^-1 e`` plus a small
  null-space pull back toward the seed pose. Seeding each solve from the corresponding
  *scripted* waypoint (already a good top-down grasp configuration) keeps the
  wrist orientation top-down without solving orientation explicitly — the vial
  randomisation is small relative to the seed, so the local solution stays in the
  same IK branch. The solve runs on a private scratch :class:`mujoco.MjData` and
  never touches the live simulation state.
* :func:`build_adaptive_phases` turns the scripted phase list into an
  episode-specific one: for every phase that has a Cartesian anchor (approach,
  descend/grasp, lift, transport, lower) it re-solves the joint target against
  the current vial/pocket pose; grip-only phases (close, release) inherit the
  preceding solved pose; the home/retract phases keep their scripted anchors.
  The result is a :class:`odyssey.embodiments.ur5e_drugsort.policy.Waypoint`
  tuple that drops straight into the unchanged :class:`PickPlaceController` FSM —
  so the interpolation, convergence and gripper open/close *timing* are identical
  to the scripted policy; only the joint *targets* adapt.

Nothing in :mod:`policy` or :mod:`embodiment` imports mujoco; this module does
(the Jacobian needs it), so it is imported lazily by the data generator and its
tests guard on ``mujoco`` being installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from odyssey.embodiments.ur5e_drugsort.policy import PHASES, Waypoint

# --- Grasp geometry, measured from the scripted waypoints (see the RUNBOOK) ---
# At the scripted grasp the pinch site sits 10 mm below the vial-body origin;
# the carry/approach clearance above that is 130 mm; at the place pose the pinch
# sits 23 mm below the nest ``pocket_0`` site. These reproduce the hand-tuned
# scripted Cartesian path, then translate with the randomised vial/pocket poses.
GRASP_PINCH_BELOW_VIAL: float = 0.010
APPROACH_CLEARANCE: float = 0.130
PLACE_PINCH_BELOW_POCKET: float = 0.023

# Site we drive (the Robotiq pinch point) and the free/anchor bodies we read.
PINCH_SITE_NAME: str = "gr_pinch"
VIAL_BODY_NAME: str = "vial_0"
POCKET_SITE_NAME: str = "pocket_0"


@dataclass
class IKSolveInfo:
    """Diagnostics for a single IK solve."""

    q: tuple[float, float, float, float, float, float]
    pos_error: float
    iters: int
    converged: bool


@dataclass
class DampedLeastSquaresIK:
    """Position-only damped-least-squares IK for the UR5e pinch site.

    Owns a private scratch :class:`mujoco.MjData`; :meth:`solve` never mutates
    the caller's simulation state.
    """

    model: Any
    site_id: int
    arm_qadr: tuple[int, ...]
    arm_dofadr: tuple[int, ...]
    damping: float = 0.08
    max_iters: int = 120
    tol: float = 1.0e-3
    step_scale: float = 1.0
    null_gain: float = 0.05
    _mj: Any = field(default=None, repr=False)
    _data: Any = field(default=None, repr=False)
    _lo: Any = field(default=None, repr=False)
    _hi: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        import mujoco as mj

        self._mj = mj
        self._data = mj.MjData(self.model)
        # Seed the scratch state from the home keyframe if present (only the arm
        # joints matter for the pinch FK, but a valid full qpos is tidiest).
        key = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, "home")
        if key >= 0:
            mj.mj_resetDataKeyframe(self.model, self._data, key)
        # Per-arm-joint clip bounds: use the MJCF limits where the joint is
        # limited, else a generous ±2π so the solver can wind without escaping.
        lo: list[float] = []
        hi: list[float] = []
        for dof in self.arm_dofadr:
            jid = int(np.asarray(self.model.dof_jntid)[dof])
            if bool(np.asarray(self.model.jnt_limited)[jid]):
                rng = np.asarray(self.model.jnt_range)[jid]
                lo.append(float(rng[0]))
                hi.append(float(rng[1]))
            else:
                lo.append(-2.0 * np.pi)
                hi.append(2.0 * np.pi)
        self._lo = np.asarray(lo, dtype=np.float64)
        self._hi = np.asarray(hi, dtype=np.float64)

    def solve(
        self,
        target_pos: Any,
        q_seed: tuple[float, float, float, float, float, float],
    ) -> IKSolveInfo:
        """Solve for arm joint angles putting the pinch site at ``target_pos``.

        Args:
            target_pos: desired ``gr_pinch`` world position ``(x, y, z)``.
            q_seed: initial (and null-space anchor) arm joint angles.
        """
        mj = self._mj
        data = self._data
        qadr = list(self.arm_qadr)
        dof = list(self.arm_dofadr)
        target = np.asarray(target_pos, dtype=np.float64).reshape(3)
        seed = np.asarray(q_seed, dtype=np.float64)

        q = np.clip(seed.copy(), self._lo, self._hi)
        jacp = np.zeros((3, int(self.model.nv)), dtype=np.float64)
        eye3 = np.eye(3, dtype=np.float64)
        eye_n = np.eye(len(qadr), dtype=np.float64)

        err_norm = float("inf")
        iters = 0
        for _ in range(self.max_iters):
            iters += 1
            for a, val in zip(qadr, q, strict=True):
                data.qpos[a] = float(val)
            mj.mj_kinematics(self.model, data)
            mj.mj_comPos(self.model, data)

            cur = np.asarray(data.site_xpos[self.site_id], dtype=np.float64)
            err = target - cur
            err_norm = float(np.linalg.norm(err))
            if err_norm < self.tol:
                break

            mj.mj_jacSite(self.model, data, jacp, None, self.site_id)
            jac = np.asarray(jacp)[:, dof]  # (3, n)
            jjt = jac @ jac.T + (self.damping**2) * eye3
            j_pinv = jac.T @ np.linalg.solve(jjt, eye3)  # (n, 3)
            dq = j_pinv @ err
            # Null-space term: pull unconstrained DoF back toward the seed so the
            # wrist keeps the seed's top-down orientation across the redundancy.
            null = (eye_n - j_pinv @ jac) @ (seed - q)
            dq = self.step_scale * dq + self.null_gain * null
            q = np.clip(q + dq, self._lo, self._hi)

        q_tuple = (
            float(q[0]), float(q[1]), float(q[2]),
            float(q[3]), float(q[4]), float(q[5]),
        )
        return IKSolveInfo(
            q=q_tuple, pos_error=err_norm, iters=iters, converged=err_norm < self.tol
        )


def _resolve_ids(model: Any) -> tuple[int, int, int]:
    """Return ``(pinch_site_id, vial_body_id, pocket_site_id)``."""
    import mujoco as mj

    obj = mj.mjtObj
    pinch = mj.mj_name2id(model, obj.mjOBJ_SITE, PINCH_SITE_NAME)
    vial = mj.mj_name2id(model, obj.mjOBJ_BODY, VIAL_BODY_NAME)
    pocket = mj.mj_name2id(model, obj.mjOBJ_SITE, POCKET_SITE_NAME)
    if pinch < 0 or vial < 0 or pocket < 0:
        raise RuntimeError("could not resolve gr_pinch / vial_0 / pocket_0 in the MJCF")
    return pinch, vial, pocket


@dataclass
class AdaptivePlan:
    """The per-episode adaptive phase list plus IK diagnostics."""

    phases: tuple[Waypoint, ...]
    approach_q: tuple[float, float, float, float, float, float]
    solves: dict[str, IKSolveInfo]
    max_pos_error: float
    all_converged: bool


def build_adaptive_phases(
    ik: DampedLeastSquaresIK,
    *,
    vial_xyz: Any,
    pocket_xyz: Any,
    home_q: tuple[float, float, float, float, float, float] | None = None,
) -> AdaptivePlan:
    """Build an episode-specific phase list by IK-solving each Cartesian anchor.

    The Cartesian anchors are derived from ``vial_xyz`` (grasp column) and
    ``pocket_xyz`` (place column) using the measured grasp geometry; each is
    solved with its scripted waypoint as the IK seed so orientation stays
    top-down. Grip-only phases (``close``, ``release``) copy the preceding solved
    pose; ``home``/``home2`` keep the scripted home; ``retract`` reuses the
    transport solution. Returns a plan whose ``phases`` drops into
    :class:`PickPlaceController` unchanged.
    """
    vial = np.asarray(vial_xyz, dtype=np.float64).reshape(3)
    pocket = np.asarray(pocket_xyz, dtype=np.float64).reshape(3)
    home = np.asarray(home_q if home_q is not None else PHASES[0].q, dtype=np.float64)
    home_t: tuple[float, float, float, float, float, float] = (
        float(home[0]), float(home[1]), float(home[2]),
        float(home[3]), float(home[4]), float(home[5]),
    )

    grasp_z = float(vial[2] - GRASP_PINCH_BELOW_VIAL)
    carry_z = grasp_z + APPROACH_CLEARANCE
    place_z = float(pocket[2] - PLACE_PINCH_BELOW_POCKET)

    grasp_xy = (float(vial[0]), float(vial[1]))
    pocket_xy = (float(pocket[0]), float(pocket[1]))

    # Cartesian target per named phase (None => keep the scripted joint target).
    targets: dict[str, np.ndarray | None] = {
        "home": None,
        "approach": np.array([grasp_xy[0], grasp_xy[1], carry_z]),
        "descend": np.array([grasp_xy[0], grasp_xy[1], grasp_z]),
        "close": None,      # grip-only: inherits descend
        "lift": np.array([grasp_xy[0], grasp_xy[1], carry_z]),
        "transport": np.array([pocket_xy[0], pocket_xy[1], carry_z]),
        "lower": np.array([pocket_xy[0], pocket_xy[1], place_z]),
        "release": None,    # grip-only: inherits lower
        "retract": np.array([pocket_xy[0], pocket_xy[1], carry_z]),
        "home2": None,
    }

    solves: dict[str, IKSolveInfo] = {}
    new_phases: list[Waypoint] = []
    prev_q: tuple[float, float, float, float, float, float] = home_t
    for ph in PHASES:
        tgt = targets[ph.name]
        q: tuple[float, float, float, float, float, float]
        if ph.name in ("home", "home2"):
            q = home_t
        elif tgt is None:
            q = prev_q  # grip-only phase: hold the previous solved arm target
        else:
            info = ik.solve(tgt, ph.q)
            solves[ph.name] = info
            q = info.q
        new_phases.append(Waypoint(name=ph.name, q=q, grip=ph.grip, hold=ph.hold))
        prev_q = q

    approach_q = solves["approach"].q if "approach" in solves else prev_q
    max_err = max((s.pos_error for s in solves.values()), default=0.0)
    all_conv = all(s.converged for s in solves.values())
    return AdaptivePlan(
        phases=tuple(new_phases),
        approach_q=approach_q,
        solves=solves,
        max_pos_error=max_err,
        all_converged=all_conv,
    )
