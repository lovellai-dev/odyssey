"""TDD for the RESUMABLE STEP MACHINE state transitions (pure ``advance``).

No GPU / VM / browser: every transition is driven by a mock ``probe`` (the
observations the executor would supply). We assert the next phase and the
effects (which driver is launched, what gets appended to the research log).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import autonomous_loop as al  # noqa: E402
import run_autonomous_loop as rl  # noqa: E402


def _funnel(**rates):
    """Point rates -> a full funnel dict (n=45) like compute_funnel produces."""
    out = {}
    for s in al.FUNNEL_STAGES:
        p = rates.get(s, 0.0)
        k = round(p * 45)
        pp, lo, hi = al.wilson(k, 45)
        out[s] = {"rate": pp, "wilson": [pp, lo, hi], "k": k, "n": 45}
    return out


def _launches(effects):
    return [e for e in effects if e[0] == "launch"]


def _logs(effects):
    return [e[1] for e in effects if e[0] == "research_log"]


# ---------------------------------------------------------------------------
# ensure_base
# ---------------------------------------------------------------------------
def test_ensure_base_launches_finish_base_when_no_ckpt():
    st = rl.default_state()
    st2, eff = rl.advance(st, {"driver_status": None, "base_ckpt": None})
    assert st2["phase"] == "ensure_base"           # stays until driver finishes
    launches = _launches(eff)
    assert launches and launches[0][1] == "finish_base"


def test_ensure_base_advances_when_vm_ckpt_present():
    st = rl.default_state()
    ck = "/home/ubuntu/ckpt/ur5e_drugsort_dr/checkpoint-15000"
    st2, _ = rl.advance(st, {"driver_status": None, "base_ckpt": ck})
    assert st2["phase"] == "eval"
    assert st2["eval_target"] == "incumbent"
    assert st2["current_best"]["policy_ckpt"] == ck
    assert st2["current_best"]["config"]["mode"] == "bare"


def test_ensure_base_done_marker_records_ckpt_and_evals():
    st = rl.default_state()
    ck = "/home/ubuntu/ckpt/ur5e_drugsort_dr/checkpoint-15000"
    st2, _ = rl.advance(st, {"driver_status": ("done", {"checkpoint": ck, "gpu_hours": 3.2})})
    assert st2["phase"] == "eval" and st2["eval_target"] == "incumbent"
    assert st2["current_best"]["policy_ckpt"] == ck
    assert st2["gpu_hours_used"] == 3.2


def test_ensure_base_failed_driver_blocks_gracefully():
    st = rl.default_state()
    st2, _ = rl.advance(st, {"driver_status": ("failed", "rsync")})
    assert st2["phase"] == "blocked"
    assert "finish_base" in st2["blocked_reason"]


def test_ensure_base_running_is_noop():
    st = rl.default_state()
    st2, eff = rl.advance(st, {"driver_status": "running"})
    assert st2["phase"] == "ensure_base"
    assert not _launches(eff)


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def test_eval_incumbent_launches_bare_powered_eval():
    st = rl.default_state()
    st["phase"] = "eval"
    st["eval_target"] = "incumbent"
    st["current_best"] = {"policy_ckpt": "/vm/ckpt",
                          "config": {"policy_ckpt": "/vm/ckpt", "mode": "bare"}}
    st2, eff = rl.advance(st, {"driver_status": None})
    launches = _launches(eff)
    assert launches and launches[0][1] == "powered_eval"
    assert launches[0][2]["mode"] == "bare"
    assert st2["phase"] == "eval"  # waits for driver


def test_eval_incumbent_done_goes_to_diagnose_and_stores_funnel():
    st = rl.default_state()
    st["phase"] = "eval"
    st["eval_target"] = "incumbent"
    f = _funnel(reached=0.9, centered=0.4, closed=0.3, lifted=0.05, seated=0.02)
    st2, _ = rl.advance(st, {"driver_status": ("done", {"funnel": f, "gpu_hours": 1.0})})
    assert st2["phase"] == "diagnose"
    assert st2["funnel_before"]["reached"]["rate"] == f["reached"]["rate"]


def test_eval_candidate_done_goes_to_gate():
    st = rl.default_state()
    st["phase"] = "eval"
    st["eval_target"] = "candidate"
    st["candidate"] = {"config": {"policy_ckpt": "/vm/ckpt", "mode": "v4"}}
    f = _funnel(reached=0.9, centered=0.6, closed=0.5, lifted=0.3, seated=0.1)
    st2, _ = rl.advance(st, {"driver_status": ("done", {"funnel": f})})
    assert st2["phase"] == "gate"
    assert st2["funnel_after"]["lifted"]["rate"] == f["lifted"]["rate"]


def test_eval_cached_funnel_skips_launch():
    st = rl.default_state()
    st["phase"] = "eval"
    st["eval_target"] = "incumbent"
    f = _funnel(reached=0.8, centered=0.3)
    st2, eff = rl.advance(st, {"driver_status": None, "cached_funnel": f})
    assert st2["phase"] == "diagnose"
    assert not _launches(eff)


# ---------------------------------------------------------------------------
# diagnose (delegates to autonomous_loop.diagnose)
# ---------------------------------------------------------------------------
def test_diagnose_reach_bottleneck_routes_to_L0_and_records():
    st = rl.default_state()
    st["phase"] = "diagnose"
    st["funnel_before"] = _funnel(reached=0.05, centered=0.0, seated=0.0)
    st2, eff = rl.advance(st, {})
    assert st2["phase"] == "execute"
    assert st2["pending_lever"] == "L0_base"
    recs = _logs(eff)
    assert recs and recs[0]["lever"] == "L0_base"
    assert recs[0]["bottleneck"] == "start->reached"
    assert "rationale" in recs[0]


def test_diagnose_capture_bottleneck_unlocks_ladder_to_dagger():
    st = rl.default_state()
    st["phase"] = "diagnose"
    # base competent, capture is the wall; ladder only has L0 unlocked
    st["funnel_before"] = _funnel(reached=0.98, centered=0.6, closed=0.89,
                                  lifted=0.09, transported=0.05, seated=0.04)
    st["ladder"]["reached_lever_idx"] = 0
    st2, _ = rl.advance(st, {})
    assert st2["phase"] == "execute"
    # unlock+run advances the ladder one rung toward the capture lever
    assert st2["ladder"]["reached_lever_idx"] >= 1
    assert st2["pending_lever"] in al.LEVER_ORDER


def test_diagnose_escalate_blocks():
    st = rl.default_state()
    st["phase"] = "diagnose"
    st["funnel_before"] = _funnel(reached=0.98, centered=0.6, closed=0.89, lifted=0.09, seated=0.04)
    st["ladder"]["reached_lever_idx"] = len(al.LEVER_ORDER) - 1
    st["ladder"]["failures"] = {lev: 2 for lev in al.LEVER_ORDER}
    st2, _ = rl.advance(st, {})
    assert st2["phase"] == "blocked"
    assert "escalate" in st2["blocked_reason"]


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------
def test_execute_L0_launches_finish_base():
    st = rl.default_state()
    st["phase"] = "execute"
    st["pending_lever"] = "L0_base"
    st["candidate"] = {"config": {"policy_ckpt": "/vm/ckpt", "mode": "bare"}}
    _, eff = rl.advance(st, {"driver_status": None})
    launches = _launches(eff)
    assert launches and launches[0][1] == "finish_base"


def test_execute_pending_lever_blocks_gracefully_and_records():
    st = rl.default_state()
    st["phase"] = "execute"
    st["pending_lever"] = "L1_steering"     # driver not wired
    st2, eff = rl.advance(st, {"driver_status": None})
    assert st2["phase"] == "blocked"
    assert st2["blocked_reason"] == "lever_pending:L1_steering"
    recs = _logs(eff)
    assert recs and recs[0]["status"] == "lever_pending"


def test_execute_done_goes_to_candidate_eval():
    st = rl.default_state()
    st["phase"] = "execute"
    st["pending_lever"] = "L0_base"
    st["candidate"] = {"config": {"policy_ckpt": "/old", "mode": "bare"}}
    ck = "/home/ubuntu/ckpt/ur5e_drugsort_dr/checkpoint-15000"
    st2, _ = rl.advance(st, {"driver_status": ("done", {"checkpoint": ck})})
    assert st2["phase"] == "eval" and st2["eval_target"] == "candidate"
    assert st2["candidate"]["config"]["policy_ckpt"] == ck


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------
def test_gate_promote_updates_best_and_loops_to_diagnose():
    st = rl.default_state()
    st["phase"] = "gate"
    st["pending_lever"] = "L3_dagger"
    st["candidate"] = {"config": {"policy_ckpt": "/vm/new", "mode": "v4"}}
    st["funnel_before"] = _funnel(lifted=0.09)
    st["funnel_after"] = _funnel(lifted=0.33)
    after_rate = st["funnel_after"]["lifted"]["rate"]
    st2, eff = rl.advance(st, {})
    recs = _logs(eff)
    assert recs and recs[0]["gate"]["promote"] is True
    assert st2["current_best"]["policy_ckpt"] == "/vm/new"
    assert st2["funnel_before"]["lifted"]["rate"] == after_rate   # promoted funnel is new incumbent
    assert st2["iteration"] == 1
    assert st2["phase"] == "diagnose"


def test_gate_reject_increments_failures():
    st = rl.default_state()
    st["phase"] = "gate"
    st["pending_lever"] = "L3_dagger"
    st["candidate"] = {"config": {"policy_ckpt": "/vm/new", "mode": "v4"}}
    st["funnel_before"] = _funnel(lifted=0.09)
    st["funnel_after"] = _funnel(lifted=0.11)     # within noise
    st2, _ = rl.advance(st, {})
    assert st2["ladder"]["failures"]["L3_dagger"] == 1
    assert st2["current_best"]["policy_ckpt"] is None   # unchanged
    assert st2["phase"] == "diagnose"


def test_gate_stops_at_max_iterations():
    st = rl.default_state()
    st["phase"] = "gate"
    st["pending_lever"] = "L0_base"
    st["candidate"] = {"config": {"mode": "bare"}}
    st["funnel_before"] = _funnel(reached=0.5)
    st["funnel_after"] = _funnel(reached=0.5)
    st["iteration"] = rl.MAX_ITERATIONS - 1
    st["budget"]["max_iterations"] = rl.MAX_ITERATIONS
    st2, _ = rl.advance(st, {})
    assert st2["phase"] == "done"


def test_gate_stops_when_target_met():
    st = rl.default_state()
    st["phase"] = "gate"
    st["pending_lever"] = "L6_place"
    st["candidate"] = {"config": {"mode": "v4"}}
    st["funnel_before"] = _funnel(seated=0.2)
    st["funnel_after"] = _funnel(seated=0.7)      # promotes; seated lo well > 0.5
    st2, _ = rl.advance(st, {})
    assert st2["phase"] == "done"


# ---------------------------------------------------------------------------
# funnel computation (OBSERVER instrument)
# ---------------------------------------------------------------------------
def test_compute_funnel_stage_predicates():
    per = [
        # reached+centered+closed+lifted+transported+seated
        {"min_pad_to_vial": 0.005, "gripMax": 0.9, "lifted": True, "place_dist": 0.02, "success": True},
        # reached+centered+closed, no lift
        {"min_pad_to_vial": 0.010, "gripMax": 0.7, "lifted": False, "place_dist": None, "success": False},
        # reached only (8cm threshold; 6cm included)
        {"min_pad_to_vial": 0.06, "gripMax": 0.1, "lifted": False, "place_dist": None, "success": False},
        # never reached
        {"min_pad_to_vial": None, "gripMax": 0.0, "lifted": False, "place_dist": None, "success": False},
    ]
    f = rl.compute_funnel(per)
    assert f["reached"]["k"] == 3
    assert f["centered"]["k"] == 2
    assert f["closed"]["k"] == 2
    assert f["lifted"]["k"] == 1
    assert f["transported"]["k"] == 1
    assert f["seated"]["k"] == 1
    assert f["reached"]["n"] == 4
    # wilson present and ordered
    p, lo, hi = f["reached"]["wilson"]
    assert lo < p < hi


def test_compute_funnel_feeds_diagnose_end_to_end():
    # a bare-base-ish funnel: reaches but rarely centers -> steering bottleneck
    per = [{"min_pad_to_vial": 0.05, "gripMax": 0.1, "lifted": False,
            "place_dist": None, "success": False} for _ in range(40)]
    per += [{"min_pad_to_vial": 0.01, "gripMax": 0.6, "lifted": False,
             "place_dist": None, "success": False} for _ in range(5)]
    f = rl.compute_funnel(per)
    ladder = al.LadderState(reached_lever_idx=len(al.LEVER_ORDER) - 1)
    d = al.diagnose(f, ladder)
    assert d["bottleneck"] == "reached->centered"
    assert d["lever"] == "L1_steering"
