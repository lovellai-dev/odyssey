#!/usr/bin/env python3
"""Autonomous multi-agent loop — the RESUMABLE STEP MACHINE.

This is the *orchestrator* around the pure SPECIALIST brain in
``autonomous_loop.py``. It performs ONE atomic, idempotent action per ``step``
invocation, persists all state to ``~/auto_loop/state.json``, and appends every
decision to ``~/auto_loop/research_log.jsonl``. A killed/loged-out session loses
nothing: re-running the wrapper re-attaches to ``state.json`` and continues from
the current phase (every long driver is itself idempotent and marker-gated).

Phases cycle::

    ensure_base -> eval -> diagnose -> execute -> eval -> gate
                                                    |
                          (loop back to diagnose | done | blocked)

The transition logic (``advance``) is PURE — it takes the current state and a
``probe`` of observations (base-ckpt present? driver DONE/FAILED/running? cached
funnel?) and returns ``(new_state, effects)``. Side effects (launch a detached
driver, append the research log) are performed by the executor (``run_step``).
That split is what makes the state machine unit-testable with no GPU/VM.

See examples/ur5e-drugsort/AUTONOMOUS_LOOP.md for the design; the levers, gates
and Wilson/diagnose math live in ``autonomous_loop`` (do not reimplement).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autonomous_loop as al  # noqa: E402

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ODY = Path(__file__).resolve().parents[1]
BC = ODY / "examples" / "ur5e-drugsort" / "browser_capture"
HOME = Path(os.environ.get("AUTO_LOOP_HOME", str(Path.home())))
STATE_DIR = Path(os.environ.get("AUTO_LOOP_STATE_DIR", str(HOME / "auto_loop")))
STATE_JSON = STATE_DIR / "state.json"
RESEARCH_LOG = STATE_DIR / "research_log.jsonl"

VM = os.environ.get("AUTO_LOOP_VM", "ubuntu@192.222.52.169")
SSH = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=20",
       "-o", "ServerAliveInterval=30"]
VM_DR_CKPT_ROOT = "/home/ubuntu/ckpt/ur5e_drugsort_dr"

# Funnel-stage thresholds (metres / grip). The OBSERVER turns each episode into
# these stages — the abstraction every decision is made over (task-agnostic).
REACH_M = float(os.environ.get("AUTO_LOOP_REACH_M", "0.08"))       # arm approached the vial
CENTER_M = float(os.environ.get("AUTO_LOOP_CENTER_M", "0.015"))    # centered on the cap
TRANSPORT_M = float(os.environ.get("AUTO_LOOP_TRANSPORT_M", "0.12"))  # carried toward the nest
GRIP_CLOSED = 0.5

# Budget defaults.
MAX_ITERATIONS = int(os.environ.get("AUTO_LOOP_MAX_ITER", "8"))
GPU_HOURS_CAP = float(os.environ.get("AUTO_LOOP_GPU_HOURS", "24"))

# ---------------------------------------------------------------------------
# Lever -> driver wiring (see the ladder table). A driver of ``None`` means the
# lever is NOT built yet: the loop records "lever_pending" and pauses at
# ``blocked`` for a human to extend it (graceful, NOT a crash).
# ---------------------------------------------------------------------------
LEVER_DRIVER: dict[str, str | None] = {
    "L0_base": "finish_base",   # WIRED — assemble->augment->rsync->SFT fresh dr ckpt
    "L1_steering": "run_l1_steering",  # WIRED — v4 image-cond head for the current base
    "L2_selection": None,       # lever_pending — K/sigma/CEM config sweep (eval itself is best-of-N)
    "L3_dagger": None,          # lever_pending — v4-visited DAgger round
    "L4_distill": None,         # lever_pending
    "L5_servo": "run_l5_servo",  # WIRED — enable the bounded final-cm visual servo (serving-time)
    "L6_place": None,           # lever_pending
}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def _bare_config(ckpt: str | None) -> dict[str, Any]:
    return {"policy_ckpt": ckpt, "mode": "bare"}


def default_state() -> dict[str, Any]:
    return {
        "version": 1,
        "phase": "ensure_base",
        "eval_target": "incumbent",
        "iteration": 0,
        "ladder": {"reached_lever_idx": 0, "failures": {},
                   "best_seated_rate": 0.0, "iteration": 0},
        "current_best": {"policy_ckpt": None, "config": _bare_config(None), "funnel": None},
        "candidate": None,
        "pending_lever": None,
        "funnel_before": None,
        "funnel_after": None,
        "running_driver": None,
        "relaunch_counts": {},
        "last_decision": None,
        "blocked_reason": None,
        "budget": {"max_iterations": MAX_ITERATIONS, "gpu_hours_cap": GPU_HOURS_CAP},
        "gpu_hours_used": 0.0,
        "created_at": time.time(),
        "updated_at": time.time(),
    }


def load_state() -> dict[str, Any]:
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text())
    return default_state()


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = time.time()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_JSON)  # atomic


def append_research_log(record: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with RESEARCH_LOG.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def _config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def _funnel_summary(funnel: dict[str, Any] | None) -> dict[str, Any]:
    if not funnel:
        return {}
    out = {}
    for s in al.FUNNEL_STAGES:
        st = funnel.get(s, {})
        w = st.get("wilson", (0.0, 0.0, 0.0))
        out[s] = {"k": st.get("k"), "n": st.get("n"),
                  "rate": round(float(st.get("rate", 0.0)), 3),
                  "ci": [round(float(w[1]), 3), round(float(w[2]), 3)]}
    return out


# ---------------------------------------------------------------------------
# Funnel computation from a powered-eval per-episode array (OBSERVER role)
# ---------------------------------------------------------------------------
def compute_funnel(per_episode: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn 45 per-episode records into the reached->...->seated funnel with
    Wilson 95% CIs. Fields come straight from eval_browser_groot.js /
    run_powered_eval.sh: min_pad_to_vial (m), gripMax, lifted, place_dist (m),
    success."""
    n = len(per_episode)

    def pad(e: dict[str, Any]) -> float:
        v = e.get("min_pad_to_vial")
        return float(v) if v is not None else 9.99

    def place(e: dict[str, Any]) -> float:
        v = e.get("place_dist")
        return float(v) if v is not None else 9.99

    counts = {
        "reached": sum(1 for e in per_episode if pad(e) < REACH_M),
        "centered": sum(1 for e in per_episode if pad(e) < CENTER_M),
        "closed": sum(1 for e in per_episode if float(e.get("gripMax") or 0.0) >= GRIP_CLOSED),
        "lifted": sum(1 for e in per_episode if bool(e.get("lifted"))),
        "transported": sum(1 for e in per_episode
                           if bool(e.get("lifted")) and place(e) < TRANSPORT_M),
        "seated": sum(1 for e in per_episode if bool(e.get("success"))),
    }
    funnel: dict[str, Any] = {}
    for s in al.FUNNEL_STAGES:
        k = counts[s]
        p, lo, hi = al.wilson(k, n)
        funnel[s] = {"rate": p, "wilson": [p, lo, hi], "k": k, "n": n}
    return funnel


def _per_episode_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """run_powered_eval writes 'per_episode'; a raw eval_browser writes 'results'."""
    return result.get("per_episode") or result.get("results") or []


# ---------------------------------------------------------------------------
# PURE transition core (unit-tested with a mock probe)
# ---------------------------------------------------------------------------
def _stop_reason(state: dict[str, Any]) -> str | None:
    if al.target_met(state.get("funnel_before") or {}):
        return "target_met"
    if state["iteration"] >= state["budget"]["max_iterations"]:
        return "max_iterations"
    if state["gpu_hours_used"] >= state["budget"]["gpu_hours_cap"]:
        return "gpu_hours_exhausted"
    return None


def _lever_candidate_config(lever: str, state: dict[str, Any]) -> dict[str, Any]:
    ckpt = (state.get("current_best") or {}).get("policy_ckpt")
    if lever == "L0_base":
        return {"policy_ckpt": ckpt, "mode": "bare"}
    if lever == "L1_steering":
        return {"policy_ckpt": ckpt, "mode": "v4"}
    if lever == "L2_selection":
        return {"policy_ckpt": ckpt, "mode": "selection"}
    if lever == "L5_servo":
        # servo is a serving-time lever: same V4 best-of-N eval, but the service
        # composes the bounded final-cm correction (STEER_SERVO=1 in _eval_env).
        return {"policy_ckpt": ckpt, "mode": "servo"}
    return {"policy_ckpt": ckpt, "mode": "v4"}


def advance(state: dict[str, Any], probe: dict[str, Any]) -> tuple[dict[str, Any], list[tuple]]:
    """PURE: one phase transition. ``probe`` supplies observations:
      base_ckpt      : str|None   — a competent base ckpt present on the VM
      driver_status  : None | 'running' | ('done', result) | ('failed', reason)
      cached_funnel  : dict|None  — a previously-computed funnel for this config
    Returns (new_state, effects) where effects is a list of:
      ('launch', driver_name, config_or_env)
      ('research_log', record)
      ('log', message)
    """
    effects: list[tuple] = []
    phase = state["phase"]
    now = probe.get("now", time.time())

    # ---- ensure_base -------------------------------------------------------
    if phase == "ensure_base":
        ck = (state["current_best"] or {}).get("policy_ckpt") or probe.get("base_ckpt")
        if ck:
            state["current_best"]["policy_ckpt"] = ck
            state["current_best"]["config"] = _bare_config(ck)
            state["phase"] = "eval"
            state["eval_target"] = "incumbent"
            effects.append(("log", f"base ckpt present ({ck}) -> eval(incumbent, bare)"))
            return state, effects
        ds = probe.get("driver_status")
        if ds is None:
            effects.append(("launch", "finish_base", {}))
            effects.append(("log", "no base ckpt -> launching finish-base (assemble->augment->rsync->SFT)"))
            return state, effects
        if ds == "running":
            return state, effects
        kind, payload = ds
        if kind == "done":
            ck = payload.get("checkpoint")
            if not ck:
                state["phase"] = "blocked"
                state["blocked_reason"] = "finish_base-no-checkpoint"
                effects.append(("log", "finish-base DONE but no checkpoint recorded -> blocked"))
                return state, effects
            state["current_best"]["policy_ckpt"] = ck
            state["current_best"]["config"] = _bare_config(ck)
            state["gpu_hours_used"] += float(payload.get("gpu_hours", 0.0))
            state["phase"] = "eval"
            state["eval_target"] = "incumbent"
            effects.append(("log", f"finish-base DONE ckpt={ck} -> eval(incumbent, bare)"))
            return state, effects
        state["phase"] = "blocked"
        state["blocked_reason"] = f"finish_base:{payload}"
        effects.append(("log", f"finish-base FAILED ({payload}) -> blocked"))
        return state, effects

    # ---- eval --------------------------------------------------------------
    if phase == "eval":
        target = state.get("eval_target", "incumbent")
        cfg = (state["current_best"]["config"] if target == "incumbent"
               else (state.get("candidate") or {}).get("config") or {})
        cached = probe.get("cached_funnel")
        if cached is not None:
            _store_funnel(state, target, cached)
            _after_eval(state, target, effects)
            effects.append(("log", f"eval({target}) reused cached funnel for {cfg.get('mode')}"))
            return state, effects
        ds = probe.get("driver_status")
        if ds is None:
            effects.append(("launch", "powered_eval", cfg))
            effects.append(("log", f"eval({target}) launching powered protocol (mode={cfg.get('mode')})"))
            return state, effects
        if ds == "running":
            return state, effects
        kind, payload = ds
        if kind == "done":
            funnel = payload.get("funnel") or {}
            _store_funnel(state, target, funnel)
            state["gpu_hours_used"] += float(payload.get("gpu_hours", 0.0))
            _after_eval(state, target, effects)
            effects.append(("log", f"eval({target}) DONE seated={funnel.get('seated', {}).get('k')}/"
                                   f"{funnel.get('seated', {}).get('n')}"))
            return state, effects
        state["phase"] = "blocked"
        state["blocked_reason"] = f"eval:{payload}"
        effects.append(("log", f"eval({target}) FAILED ({payload}) -> blocked"))
        return state, effects

    # ---- diagnose (SPECIALIST) --------------------------------------------
    if phase == "diagnose":
        ladder = al.LadderState(
            reached_lever_idx=state["ladder"]["reached_lever_idx"],
            failures=dict(state["ladder"]["failures"]),
            best_seated_rate=state["ladder"]["best_seated_rate"],
            iteration=state["ladder"]["iteration"],
        )
        funnel = state.get("funnel_before") or {}
        d = al.diagnose(funnel, ladder)
        record = {
            "ts": now, "iso": time.strftime("%FT%TZ", time.gmtime(now)),
            "iteration": state["iteration"], "phase": "diagnose",
            "funnel": _funnel_summary(funnel),
            "bottleneck": d["bottleneck"], "drop": d["drop"],
            "lever": d["lever"], "action": d["action"],
            "rationale": d["rationale"],
            "ladder": {"reached_lever_idx": ladder.reached_lever_idx,
                       "failures": ladder.failures},
        }
        if "considered" in d:
            record["considered"] = d["considered"]
        effects.append(("research_log", record))
        state["last_decision"] = d
        if d["lever"] is None:  # escalate
            state["phase"] = "blocked"
            state["blocked_reason"] = "escalate:all-levers-retired"
            effects.append(("log", f"SPECIALIST escalate: {d['rationale']}"))
            return state, effects
        if d["action"] == "unlock+run":
            idx = al.LEVER_ORDER.index(d["lever"])
            state["ladder"]["reached_lever_idx"] = max(state["ladder"]["reached_lever_idx"], idx)
        state["pending_lever"] = d["lever"]
        state["candidate"] = {"config": _lever_candidate_config(d["lever"], state)}
        state["phase"] = "execute"
        effects.append(("log", f"SPECIALIST: bottleneck={d['bottleneck']} -> lever={d['lever']} ({d['action']})"))
        return state, effects

    # ---- execute -----------------------------------------------------------
    if phase == "execute":
        lever = state["pending_lever"]
        driver = LEVER_DRIVER.get(lever)
        if driver is None:
            record = {
                "ts": now, "iso": time.strftime("%FT%TZ", time.gmtime(now)),
                "iteration": state["iteration"], "phase": "execute",
                "lever": lever, "status": "lever_pending",
                "note": (f"{lever} driver is not built/wired yet; the loop pauses at "
                         f"'blocked' for a human to extend this rung. All prior "
                         f"evidence (funnel + diagnosis) is in the research log."),
            }
            effects.append(("research_log", record))
            state["phase"] = "blocked"
            state["blocked_reason"] = f"lever_pending:{lever}"
            effects.append(("log", f"execute: {lever} lever_pending -> blocked (graceful)"))
            return state, effects
        ds = probe.get("driver_status")
        if ds is None:
            effects.append(("launch", driver, (state.get("candidate") or {}).get("config") or {}))
            effects.append(("log", f"execute: launching {driver} for {lever}"))
            return state, effects
        if ds == "running":
            return state, effects
        kind, payload = ds
        if kind == "done":
            if payload.get("checkpoint"):
                state["candidate"]["config"]["policy_ckpt"] = payload["checkpoint"]
            state["gpu_hours_used"] += float(payload.get("gpu_hours", 0.0))
            state["phase"] = "eval"
            state["eval_target"] = "candidate"
            effects.append(("log", f"execute: {driver} DONE -> eval(candidate)"))
            return state, effects
        state["phase"] = "blocked"
        state["blocked_reason"] = f"execute:{lever}:{payload}"
        effects.append(("log", f"execute: {driver} FAILED ({payload}) -> blocked"))
        return state, effects

    # ---- gate --------------------------------------------------------------
    if phase == "gate":
        lever = state["pending_lever"]
        before = state.get("funnel_before") or {}
        after = state.get("funnel_after") or {}
        g = al.gate_result(lever, before, after)
        record = {
            "ts": now, "iso": time.strftime("%FT%TZ", time.gmtime(now)),
            "iteration": state["iteration"], "phase": "gate",
            "lever": lever, "gate": g,
        }
        effects.append(("research_log", record))
        if g["promote"]:
            cand_cfg = (state.get("candidate") or {}).get("config") or {}
            if cand_cfg.get("policy_ckpt"):
                state["current_best"]["policy_ckpt"] = cand_cfg["policy_ckpt"]
            state["current_best"]["config"] = cand_cfg
            state["current_best"]["funnel"] = after
            seated_rate = float(after.get("seated", {}).get("rate", 0.0))
            state["ladder"]["best_seated_rate"] = max(state["ladder"]["best_seated_rate"], seated_rate)
            state["funnel_before"] = after
            effects.append(("log", f"gate: {lever} PROMOTE ({g['rationale']})"))
        else:
            state["ladder"]["failures"][lever] = state["ladder"]["failures"].get(lever, 0) + 1
            effects.append(("log", f"gate: {lever} REJECT (fails={state['ladder']['failures'][lever]})"))
        state["iteration"] += 1
        state["ladder"]["iteration"] = state["iteration"]
        state["pending_lever"] = None
        state["candidate"] = None
        state["funnel_after"] = None
        reason = _stop_reason(state)
        if reason:
            state["phase"] = "done"
            state["blocked_reason"] = None
            effects.append(("log", f"gate: stop ({reason}) -> done"))
        else:
            state["phase"] = "diagnose"
        return state, effects

    # terminal / unknown
    return state, effects


def _store_funnel(state: dict[str, Any], target: str, funnel: dict[str, Any]) -> None:
    if target == "incumbent":
        state["funnel_before"] = funnel
        state["current_best"]["funnel"] = funnel
    else:
        state["funnel_after"] = funnel


def _after_eval(state: dict[str, Any], target: str, effects: list[tuple]) -> None:
    state["phase"] = "diagnose" if target == "incumbent" else "gate"


# ---------------------------------------------------------------------------
# Side-effecting executor (real drivers, VM probes)
# ---------------------------------------------------------------------------
def _vm_base_ckpt() -> str | None:
    """Newest ur5e_drugsort_dr checkpoint on the VM, or None."""
    try:
        out = subprocess.run(
            [*SSH, VM, f"ls -d {VM_DR_CKPT_ROOT}/checkpoint-* 2>/dev/null | "
                       f"sort -t- -k2 -n | tail -1"],
            capture_output=True, text=True, timeout=40).stdout.strip()
        return out or None
    except Exception:
        return None


def _eval_env(cfg: dict[str, Any]) -> dict[str, str]:
    """Env for run_powered_eval.sh given a candidate config (mode/ckpt)."""
    mode = cfg.get("mode", "bare")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["SKIP_GPU_WAIT"] = "1"     # the loop GPU-gates itself; don't wait on dagger1s
    env["RESUME"] = "1"            # keep finished blocks across infra hiccups
    if cfg.get("policy_ckpt"):
        env["CKPT_OVERRIDE"] = cfg["policy_ckpt"]
    if mode in ("v4", "selection", "servo"):
        env["V4"] = "1"
        env["BARE"] = "0"
        # L5: same V4 best-of-N serving, plus the bounded final-cm visual servo
        # composed onto the selected chunk inside the service.
        env["STEER_SERVO"] = "1" if mode == "servo" else "0"
    else:  # "bare" (and any unknown mode) -> genuinely bare
        env["V4"] = "0"
        env["BARE"] = "1"
    return env


def _driver_spec(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the launch spec (script, log, markers, result parsing) for a driver."""
    if name == "finish_base":
        return {
            "cmd": ["bash", str(BC / "finish_base.sh")],
            "env": {k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            "log": str(HOME / "finish_base.log"),
            "marker_done": "FINISH_BASE DONE",
            "marker_fail": "FINISH_BASE FAILED",
            "result": str(HOME / "finish_base_result.json"),
            "parse": "checkpoint",
        }
    if name == "run_l1_steering":
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        if cfg.get("policy_ckpt"):
            env["CKPT_OVERRIDE"] = cfg["policy_ckpt"]   # build the head for the current base
        return {
            "cmd": ["bash", str(BC / "run_l1_steering.sh")],
            "env": env,
            "log": str(HOME / "l1_steering.log"),
            "marker_done": "L1_STEERING DONE",
            "marker_fail": "L1_STEERING FAILED",
            "result": str(HOME / "l1_steering_result.json"),
            "parse": "checkpoint",   # benign: result has no 'checkpoint' -> base ckpt kept
        }
    if name == "run_l5_servo":
        # L5 TRAINS nothing — it is a serving-time config. This driver just
        # records that the servo is enabled (result JSON + marker); the loop's
        # SUBSEQUENT candidate eval serves with STEER_SERVO=1 (mode "servo").
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        if cfg.get("policy_ckpt"):
            env["CKPT_OVERRIDE"] = cfg["policy_ckpt"]
        return {
            "cmd": ["bash", str(BC / "run_l5_servo.sh")],
            "env": env,
            "log": str(HOME / "l5_servo.log"),
            "marker_done": "L5_SERVO DONE",
            "marker_fail": "L5_SERVO FAILED",
            "result": str(HOME / "l5_servo_result.json"),
            "parse": "checkpoint",   # benign: result has no 'checkpoint' -> base ckpt kept
        }
    if name == "powered_eval":
        suf = "_v4" if _eval_env(cfg).get("V4") == "1" else ""
        return {
            "cmd": ["bash", str(BC / "run_powered_eval.sh")],
            "env": _eval_env(cfg),
            "log": str(HOME / f"powered_eval{suf}.log"),
            "marker_done": "POWERED_EVAL DONE",
            "marker_fail": "POWERED_EVAL FAILED",
            "result": str(HOME / f"powered_eval{suf}_result.json"),
            "parse": "funnel",
            "config_hash": _config_hash(cfg),
        }
    raise ValueError(f"unknown driver {name}")


def _launch_driver(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    spec = _driver_spec(name, cfg)
    logp = Path(spec["log"])
    logp.parent.mkdir(parents=True, exist_ok=True)
    fh = open(logp, "a")  # noqa: SIM115 — handle is inherited by the detached child; must outlive this fn
    # setsid detaches into its own session so it survives the wrapper/session death.
    proc = subprocess.Popen(
        spec["cmd"], stdout=fh, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, env=spec["env"],
        cwd=str(BC), start_new_session=True)
    return {
        "driver": name, "pid": proc.pid, "log": spec["log"],
        "marker_done": spec["marker_done"], "marker_fail": spec["marker_fail"],
        "result": spec["result"], "parse": spec["parse"],
        "config_hash": spec.get("config_hash"),
        "phase": None, "started_at": time.time(), "status": "running",
    }


def _tail_marker(logp: str, done: str, fail: str) -> str | None:
    """Scan the tail of the log for the driver's completion marker."""
    p = Path(logp)
    if not p.exists():
        return None
    try:
        data = p.read_text(errors="ignore")
    except Exception:
        return None
    # last occurrence wins (log is appended across resumes)
    di = data.rfind(done)
    fi = data.rfind(fail)
    if di < 0 and fi < 0:
        return None
    return "done" if di > fi else "fail"


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True  # exists, owned by another user
    except (ProcessLookupError, OSError):
        return False


def _poll_driver(rd: dict[str, Any]) -> tuple:
    """('running' | ('done', payload) | ('failed', reason)). Idempotent."""
    marker = _tail_marker(rd["log"], rd["marker_done"], rd["marker_fail"])
    gpu_hours = (time.time() - rd.get("started_at", time.time())) / 3600.0
    if marker == "done":
        payload: dict[str, Any] = {"gpu_hours": round(gpu_hours, 3)}
        res = _load_json(rd.get("result"))
        if rd["parse"] == "checkpoint":
            payload["checkpoint"] = (res or {}).get("checkpoint")
        elif rd["parse"] == "funnel":
            per = _per_episode_from_result(res or {})
            payload["funnel"] = compute_funnel(per) if per else {}
            # cache the funnel keyed by config for idempotent re-eval
            if rd.get("config_hash"):
                _write_cache(rd["config_hash"], payload["funnel"])
        return ("done", payload)
    if marker == "fail":
        return ("failed", "driver-marker-FAILED")
    if _pid_alive(rd.get("pid")):
        return "running"
    # pid gone and NO marker (neither DONE nor FAILED): the local driver process
    # vanished without recording an outcome — almost always a session/host reboot
    # while the VM-side SFT tmux keeps running. Every driver is idempotent (it
    # adopts an in-flight VM tmux / skips finished sub-steps), so ask the executor
    # to relaunch rather than declaring failure.
    return "relaunch"


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _cache_path(config_hash: str) -> Path:
    return STATE_DIR / f"eval_{config_hash}.json"


def _write_cache(config_hash: str, funnel: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(config_hash).write_text(json.dumps(funnel, indent=2))


def _read_cache(config_hash: str) -> dict[str, Any] | None:
    return _load_json(str(_cache_path(config_hash)))


def _build_probe(state: dict[str, Any], driver_status: Any) -> dict[str, Any]:
    probe: dict[str, Any] = {"now": time.time(), "driver_status": driver_status}
    phase = state["phase"]
    if phase == "ensure_base" and driver_status is None \
            and not (state["current_best"] or {}).get("policy_ckpt"):
        probe["base_ckpt"] = _vm_base_ckpt()
    if phase == "eval" and driver_status is None:
        target = state.get("eval_target", "incumbent")
        cfg = (state["current_best"]["config"] if target == "incumbent"
               else (state.get("candidate") or {}).get("config") or {})
        cached = _read_cache(_config_hash(cfg))
        if cached:
            probe["cached_funnel"] = cached
    return probe


def run_step() -> int:
    """Perform ONE atomic step. Prints a PHASE=<p> line for the wrapper."""
    state = load_state()
    if state["phase"] in ("done", "blocked"):
        print(f"PHASE={state['phase']}")
        return 0

    driver_status: Any = None
    rd = state.get("running_driver")
    if rd:
        status = _poll_driver(rd)
        if status == "running":
            print(f"[auto_loop] phase={state['phase']} driver={rd['driver']} running "
                  f"(pid={rd.get('pid')})")
            print(f"PHASE={state['phase']}")
            return 0
        if status == "relaunch":
            phase = state["phase"]
            counts = state.setdefault("relaunch_counts", {})
            counts[phase] = counts.get(phase, 0) + 1
            state["running_driver"] = None
            if counts[phase] > 3:
                driver_status = ("failed", f"{rd['driver']}-vanished-x{counts[phase]}")
                print(f"[auto_loop] {rd['driver']} vanished {counts[phase]}x without a "
                      f"marker -> giving up (blocked)")
            else:
                driver_status = None   # advance re-launches (drivers are idempotent)
                print(f"[auto_loop] {rd['driver']} vanished without a marker "
                      f"(reboot?) -> relaunch #{counts[phase]} (idempotent)")
        else:
            driver_status = status
            state["running_driver"] = None

    probe = _build_probe(state, driver_status)
    new_state, effects = advance(state, probe)

    for eff in effects:
        kind = eff[0]
        if kind == "launch":
            _, name, cfg = eff
            rd = _launch_driver(name, cfg or {})
            rd["phase"] = new_state["phase"]
            new_state["running_driver"] = rd
            print(f"[auto_loop] launched {name} pid={rd['pid']} log={rd['log']}")
        elif kind == "research_log":
            append_research_log(eff[1])
        elif kind == "log":
            print(f"[auto_loop] {eff[1]}")

    save_state(new_state)
    print(f"PHASE={new_state['phase']}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_init() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_JSON.exists():
        save_state(default_state())
        print(f"[auto_loop] initialized {STATE_JSON}")
    else:
        print(f"[auto_loop] state exists ({load_state()['phase']}) — resuming")
    return 0


def cmd_phase() -> int:
    print(load_state()["phase"])
    return 0


def cmd_status() -> int:
    s = load_state()
    print(json.dumps({
        "phase": s["phase"], "iteration": s["iteration"],
        "eval_target": s.get("eval_target"),
        "pending_lever": s.get("pending_lever"),
        "blocked_reason": s.get("blocked_reason"),
        "current_best_ckpt": (s.get("current_best") or {}).get("policy_ckpt"),
        "funnel_before": _funnel_summary(s.get("funnel_before")),
        "gpu_hours_used": round(s.get("gpu_hours_used", 0.0), 2),
        "running_driver": (s.get("running_driver") or {}).get("driver"),
    }, indent=2))
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "step"
    if cmd == "step":
        return run_step()
    if cmd == "init":
        return cmd_init()
    if cmd == "phase":
        return cmd_phase()
    if cmd == "status":
        return cmd_status()
    print(f"usage: {argv[0]} [step|init|phase|status]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
