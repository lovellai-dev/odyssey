"""TDD for the autonomous-loop Specialist decision core (pure, numpy-free)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import autonomous_loop as al  # noqa: E402


def _funnel(**rates):
    # rates given as points; n=45 default so wilson is realistic
    out = {}
    for s in al.FUNNEL_STAGES:
        p = rates.get(s, 0.0)
        k = round(p * 45)
        out[s] = {"rate": p, "wilson": al.wilson(k, 45), "k": k, "n": 45}
    return out


def test_wilson_zero_and_bounds():
    assert al.wilson(0, 0) == (0.0, 0.0, 0.0)
    p, lo, hi = al.wilson(2, 45)
    assert 0.0 < lo < p < hi < 0.3


def test_transitions_find_biggest_drop():
    # closure high, lift low -> the closed->lifted drop dominates
    f = _funnel(reached=0.95, centered=0.9, closed=0.89, lifted=0.09, seated=0.04)
    tr = dict(al.funnel_transitions(f))
    biggest = max(tr, key=tr.get)
    assert biggest == "closed->lifted"


def test_diagnose_routes_capture_bottleneck_to_dagger():
    # this is the ACTUAL measured v4 funnel: 89% closed, 9% lifted
    f = _funnel(reached=0.98, centered=0.6, closed=0.89, lifted=0.09,
                transported=0.05, seated=0.044)
    ladder = al.LadderState(reached_lever_idx=len(al.LEVER_ORDER) - 1)  # all unlocked
    d = al.diagnose(f, ladder)
    assert d["bottleneck"] == "closed->lifted"
    assert d["lever"] == "L3_dagger"
    assert "closed->lifted" in d["rationale"]


def test_diagnose_routes_reach_failure_to_base():
    f = _funnel(reached=0.05, centered=0.0, seated=0.0)
    ladder = al.LadderState(reached_lever_idx=len(al.LEVER_ORDER) - 1)
    d = al.diagnose(f, ladder)
    assert d["bottleneck"] == "start->reached" and d["lever"] == "L0_base"


def test_diagnose_advances_ladder_when_lever_locked():
    f = _funnel(reached=0.9, centered=0.5, closed=0.4, lifted=0.1, seated=0.05)
    ladder = al.LadderState(reached_lever_idx=0)  # only L0 unlocked
    d = al.diagnose(f, ladder)
    assert d["action"] == "unlock+run"
    assert d["lever"] == "L1_steering"  # next rung


def test_diagnose_skips_retired_lever():
    f = _funnel(reached=0.98, centered=0.6, closed=0.89, lifted=0.09, seated=0.04)
    ladder = al.LadderState(reached_lever_idx=len(al.LEVER_ORDER) - 1,
                            failures={"L3_dagger": 2})  # L3 retired
    d = al.diagnose(f, ladder)
    # closed->lifted still the bottleneck; L3 retired -> next lever for it (L5_servo)
    assert d["lever"] in ("L5_servo", "L4_distill")


def test_diagnose_escalates_when_all_retired():
    f = _funnel(reached=0.98, centered=0.6, closed=0.89, lifted=0.09, seated=0.04)
    ladder = al.LadderState(
        reached_lever_idx=len(al.LEVER_ORDER) - 1,
        failures={lev: 2 for lev in al.LEVER_ORDER})
    d = al.diagnose(f, ladder)
    assert d["action"] == "escalate" and d["lever"] is None


def test_gate_promotes_on_clear_improvement():
    before = {"lifted": {"wilson": al.wilson(4, 45)}}   # 8.9% [3.5,20.7]
    after = {"lifted": {"wilson": al.wilson(15, 45)}}   # 33% [21,48]
    g = al.gate_result("L3_dagger", before, after)
    assert g["promote"] is True and g["target_metric"] == "lifted"


def test_gate_rejects_within_noise():
    before = {"lifted": {"wilson": al.wilson(4, 45)}}
    after = {"lifted": {"wilson": al.wilson(5, 45)}}   # 11% — overlaps
    g = al.gate_result("L3_dagger", before, after)
    assert g["promote"] is False


def test_target_met_on_seated_lower_bound():
    lo_hi = _funnel(seated=0.7)      # 70% -> lo well above 0.5
    lo_lo = _funnel(seated=0.3)
    assert al.target_met(lo_hi, 0.5) is True
    assert al.target_met(lo_lo, 0.5) is False
