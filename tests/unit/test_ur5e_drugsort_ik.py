"""Tests for the adaptive damped-least-squares IK expert.

These exercise the real AseptiPack MJCF, so they need both ``mujoco`` installed
and the MJCF present; otherwise they skip (mujoco is not in the CI ``dev``
extra). The pure FSM/embodiment schema is covered by the sibling test modules —
here we prove the IK solver reaches the vial and that the resulting joint targets
genuinely *vary* with the vial pose (the whole point of the adaptive expert).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from odyssey.embodiments.ur5e_drugsort.ik import (  # noqa: E402
    DampedLeastSquaresIK,
    _resolve_ids,
    build_adaptive_phases,
)
from odyssey.embodiments.ur5e_drugsort.policy import PHASES  # noqa: E402

_ARM = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
_DEFAULT_XML = (
    "/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/"
    "aseptipack_description/aseptipack.xml"
)


def _xml_path() -> Path:
    return Path(os.environ.get("ASEPTIPACK_XML", _DEFAULT_XML))


@pytest.fixture(scope="module")
def model():  # type: ignore[no-untyped-def]
    xml = _xml_path()
    if not xml.is_file():
        pytest.skip(f"AseptiPack MJCF not found at {xml} (set ASEPTIPACK_XML)")
    return mujoco.MjModel.from_xml_path(str(xml))


@pytest.fixture
def ik(model):  # type: ignore[no-untyped-def]
    m2i = mujoco.mj_name2id
    obj = mujoco.mjtObj
    qadr = tuple(
        int(model.jnt_qposadr[m2i(model, obj.mjOBJ_JOINT, f"{n}_joint")]) for n in _ARM
    )
    dofadr = tuple(
        int(model.jnt_dofadr[m2i(model, obj.mjOBJ_JOINT, f"{n}_joint")]) for n in _ARM
    )
    pinch, _, _ = _resolve_ids(model)
    return DampedLeastSquaresIK(
        model=model, site_id=pinch, arm_qadr=qadr, arm_dofadr=dofadr
    )


def test_resolve_ids_finds_all_landmarks(model) -> None:  # type: ignore[no-untyped-def]
    pinch, vial, pocket = _resolve_ids(model)
    assert pinch >= 0 and vial >= 0 and pocket >= 0


def test_ik_reaches_the_nominal_grasp(ik) -> None:  # type: ignore[no-untyped-def]
    # The nominal vial grasp point (measured from the scripted waypoints).
    info = ik.solve(np.array([0.45, 0.15, 0.21]), PHASES[2].q)
    assert info.converged
    assert info.pos_error < 2.0e-3  # within 2 mm
    # Seeded from the scripted descend pose, it stays in the same IK branch.
    for got, want in zip(info.q, PHASES[2].q, strict=True):
        assert abs(got - want) < 0.05


def test_ik_solution_varies_with_target(ik) -> None:  # type: ignore[no-untyped-def]
    a = ik.solve(np.array([0.42, 0.10, 0.21]), PHASES[2].q)
    b = ik.solve(np.array([0.48, 0.20, 0.21]), PHASES[2].q)
    assert a.converged and b.converged
    # Different Cartesian targets must give materially different joint angles.
    delta = max(abs(x - y) for x, y in zip(a.q, b.q, strict=True))
    assert delta > 0.05


def test_ik_never_leaves_scratch_state_on_the_model(model, ik) -> None:  # type: ignore[no-untyped-def]
    # Solving uses a private scratch MjData; a fresh MjData is unaffected.
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    before = np.array(d.qpos, copy=True)
    ik.solve(np.array([0.47, 0.12, 0.21]), PHASES[1].q)
    after = np.array(d.qpos, copy=True)
    assert np.array_equal(before, after)


def test_build_adaptive_phases_preserves_fsm_structure(ik) -> None:  # type: ignore[no-untyped-def]
    home_q = tuple(PHASES[0].q)
    plan = build_adaptive_phases(
        ik,
        vial_xyz=np.array([0.45, 0.15, 0.22]),
        pocket_xyz=np.array([0.39, -0.174, 0.228]),
        home_q=home_q,
    )
    # Same 10 phases, same names / gripper commands / hold rules as the script.
    assert [p.name for p in plan.phases] == [p.name for p in PHASES]
    for new, old in zip(plan.phases, PHASES, strict=True):
        assert new.grip == old.grip
        assert new.hold == old.hold
        assert len(new.q) == 6
    names = [p.name for p in plan.phases]
    q_by = {p.name: p.q for p in plan.phases}
    # Grip-only phases inherit the preceding solved arm target (no re-interp).
    assert q_by["close"] == q_by["descend"]
    assert q_by["release"] == q_by["lower"]
    # Home phases keep the fixed home configuration.
    assert q_by["home"] == home_q
    assert q_by["home2"] == home_q
    assert names[0] == "home"


def test_build_adaptive_phases_approach_tracks_the_vial(ik) -> None:  # type: ignore[no-untyped-def]
    pocket = np.array([0.39, -0.174, 0.228])
    left = build_adaptive_phases(ik, vial_xyz=np.array([0.42, 0.10, 0.22]), pocket_xyz=pocket)
    right = build_adaptive_phases(ik, vial_xyz=np.array([0.48, 0.20, 0.22]), pocket_xyz=pocket)
    assert left.all_converged and right.all_converged
    # The first-approach joint angles must differ when the vial moves.
    delta = max(abs(x - y) for x, y in zip(left.approach_q, right.approach_q, strict=True))
    assert delta > 0.05
    assert left.max_pos_error < 3.0e-3 and right.max_pos_error < 3.0e-3
