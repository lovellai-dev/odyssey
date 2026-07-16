#!/usr/bin/env bash
# ============================================================================
# run_stageb_ab.sh — SELF-DRIVING FlowDAgger Stage-B honest A/B
# (PLAN_MULTIAGENT.md Phase 4 / FlowDAgger port). Launched DETACHED; writes
# ~/stageb_ab_result.json + a STAGEB_AB DONE/FAILED marker to $LOG.
#
# WHAT: deploy the A4 steering net into the REAL browser serving path and run
# the honest A/B on the SAME seed-7777 poses that produced every prior 0/15:
#   Run A — sidecar attaches init_noise (steering ON), 15 episodes
#   Run B — same sidecar restarted WITHOUT weights (stock),  15 episodes
# Path: browser eval_browser_groot.js -> agent :8032 -> conditioner :5604
# (Observer target + steering noise) -> tunnel :5596 -> VM bridge -> ZMQ :5558
# GR00T (obscond checkpoint-12000 — the EXACT checkpoint the net was trained
# against; steering is checkpoint-specific).
#
# PRE-EVAL SMOKE GATE (mandatory, fail-fast): through the FULL chain —
#   (1) two IDENTICAL steered requests -> 200, 6-D action, steered:true,
#       byte-identical actions (chain determinism);
#   (2) restart sidecar stock -> same request -> action DIFFERS from steered.
#
# ISOLATION: same conventions as run_probe_phase1.sh — own ports (8032/5604/
# 5596/5558), own VM tmux (groot_obscond_*), own Chrome by pattern-checked PID,
# NEVER pkill chrome; ccproxy/gateway/groot_browser_* untouched. Serving only.
# ============================================================================
set -uo pipefail

BC=/home/daniel/LovellAI/odyssey-ur5e/examples/ur5e-drugsort/browser_capture
ODY=/home/daniel/LovellAI/odyssey-ur5e
AGENT_DIR=/home/daniel/LovellAI/lai-agent-multiagent/agent_service
AGENT_PY=/home/daniel/LovellAI/lai-agent/agent_service/.venv/bin/python
PYUR5E=$ODY/.venv-ur5e/bin/python
OBSPY=/home/daniel/Isaac-GR00T/.venv/bin/python
VM=ubuntu@192.222.52.169
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30"
PUP=/home/daniel/.npm/_npx/23232c69e5d221f3/node_modules/puppeteer-core
CHROME=/usr/bin/google-chrome-stable
NODE=/home/daniel/.nvm/versions/node/v22.22.0/bin/node
XML=/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/aseptipack_description/aseptipack.xml

WORK=$HOME/stageb_ab
LOG=$HOME/stageb_ab.log
RESULT=$HOME/stageb_ab_result.json
OUTDIR=$WORK/out
# Fresh OUTDIR every run: stale eval jsons from an aborted attempt must never
# leak into a later verdict. Steering weights live in $WORK (outside OUTDIR).
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

AGENT_PORT=8032
ZMQ_PORT=5558
BRIDGE_PORT=5596
COND_PORT=5604
CKPT=/home/ubuntu/ckpt/ur5e_drugsort_obscond/full/checkpoint-12000
OBS_WEIGHTS=$BC/percep_weights_browser
STEER_NPZ=$WORK/steering_net_v0.npz
PLANS=$HOME/observer_cond/plans_eval.json
EVAL_N=${EVAL_N:-15}
N_ACTION_STEPS=${N_ACTION_STEPS:-8}
MAX_TICKS=${MAX_TICKS:-900}

export DISPLAY=${DISPLAY:-:1}
cd "$BC"
exec >>"$LOG" 2>&1
echo "" ; echo "#################################################################"
log(){ echo "=== [stageb] $* $(date -u +%FT%TZ) ==="; }

cleanup(){
  log "cleanup (own agent-service + conditioner + tunnel + VM obscond serving; leave everything else)"
  for pf in "$WORK/agent_pid.txt" "$WORK/cond_pid.txt"; do
    [ -f "$pf" ] || continue
    p=$(cat "$pf" 2>/dev/null || true); [ -n "${p:-}" ] && kill "$p" 2>/dev/null || true
  done
  ss -tlnp 2>/dev/null | grep -E ":($AGENT_PORT|$COND_PORT) " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | xargs -r kill 2>/dev/null || true
  pkill -f "127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT" 2>/dev/null || true
  $SSH "$VM" "tmux kill-session -t groot_obscond_server 2>/dev/null; tmux kill-session -t groot_obscond_bridge 2>/dev/null" 2>/dev/null || true
}
kill_chrome_pidfile(){
  local pf="$1"; [ -f "$pf" ] || return 0
  local cp; cp=$(cat "$pf" 2>/dev/null || true)
  if [ -n "${cp:-}" ] && kill -0 "$cp" 2>/dev/null; then
    if tr '\0' ' ' < "/proc/$cp/cmdline" 2>/dev/null | grep -q "chrome-udd-eval"; then
      echo "  [stageb] kill Chrome PID $cp"; kill "$cp" 2>/dev/null || true
    fi
  fi
  rm -f "$pf" 2>/dev/null || true
}
FAILING=0
fail(){
  [ "$FAILING" = "1" ] && exit 1   # re-entry guard (signal during fail())
  FAILING=1
  trap - TERM INT HUP
  log "FAILED at: $*"
  printf '{"stage":"flowdagger-stage-b-ab","status":"FAILED","failed_at":"%s"}\n' "$*" > "$RESULT"
  kill_chrome_pidfile "$OUTDIR/eval_A/eval_pid.txt"
  kill_chrome_pidfile "$OUTDIR/eval_B/eval_pid.txt"
  cleanup
  echo "STAGEB_AB FAILED: $*"
  exit 1
}
# A signal to the detached driver must still leave a RESULT + marker and tear
# the stack down.
trap 'fail "signal"' TERM INT HUP

# ---- sidecar management ------------------------------------------------------
start_sidecar(){   # $1 = ON|OFF (steering) ; $2 = logfile
  local mode="$1" logf="$2" sw=""
  local op; op=$(cat "$WORK/cond_pid.txt" 2>/dev/null || true)
  [ -n "${op:-}" ] && { kill "$op" 2>/dev/null || true; sleep 2; }
  ss -tlnp 2>/dev/null | grep ":$COND_PORT " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | xargs -r kill 2>/dev/null || true
  sleep 1
  [ "$mode" = "ON" ] && sw="$STEER_NPZ"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OBSERVER_WEIGHTS="$OBS_WEIGHTS" \
    CONDITION_BRIDGE_URL="http://127.0.0.1:$BRIDGE_PORT" OBSERVER_DEVICE=cuda \
    PYTHONPATH="$ODY/src" STEERING_WEIGHTS="$sw" STEERING_FK_XML="$XML" \
    setsid "$OBSPY" "$ODY/scripts/serve_observer_conditioning.py" --port "$COND_PORT" \
    >"$logf" 2>&1 &
  echo $! > "$WORK/cond_pid.txt"
  local i CH=""
  for i in $(seq 1 40); do
    CH=$(curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health 2>/dev/null || true)
    echo "$CH" | grep -q '"ok": *true' && break; sleep 3
  done
  { echo "$CH" | grep -q '"observer_ready": *true' && echo "$CH" | grep -q '"ok": *true'; } || return 1
  if [ "$mode" = "ON" ]; then
    echo "$CH" | grep -q '"steering_enabled": *true' || return 1
  else
    echo "$CH" | grep -q '"steering_enabled": *false' || return 1
  fi
  log "sidecar up (steering=$mode pid=$(cat "$WORK/cond_pid.txt"))"
  return 0
}

log "START pid=$$ ckpt=$CKPT eval_n=$EVAL_N n_action_steps=$N_ACTION_STEPS max_ticks=$MAX_TICKS"
echo "STAGEB_AB PID=$$"

# ---- 0. sanity ---------------------------------------------------------------
[ -f "$PLANS" ] || fail "no-plans($PLANS)"
[ -f "$OBS_WEIGHTS/observer_head.pt" ] || fail "no-observer-weights"
[ -f "$XML" ] || fail "no-fk-xml($XML)"
$SSH "$VM" "echo ok >/dev/null" || fail "vm-unreachable"
$SSH "$VM" "[ -d $CKPT ]" || fail "no-vm-ckpt($CKPT)"
# steering weights: local copy, else pull the A4 artifact from the VM
if [ ! -f "$STEER_NPZ" ]; then
  scp -q -o StrictHostKeyChecking=no "$VM:/home/ubuntu/steering_net_v0.npz" "$STEER_NPZ" || fail "no-steering-weights"
fi
"$PYUR5E" - "$STEER_NPZ" <<'PY' || fail "bad-steering-weights"
import json, sys
import numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
meta = json.loads(str(z["meta"]))
assert meta.get("in_dim") == 14 and meta.get("out_dim") in (112, 5280), meta
print("[stageb] steering meta:", {k: meta[k] for k in ("in_dim", "out_dim", "output_design")})
PY
# the init_noise patch must be live in the VM checkout the server runs from
$SSH "$VM" "grep -q 'init_noise' /home/ubuntu/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py" || fail "vm-groot-not-patched"
for port in $AGENT_PORT $COND_PORT; do
  ss -tln 2>/dev/null | grep -q ":$port " && fail "port-busy($port)"
done
ss -tln 2>/dev/null | grep -q ":$BRIDGE_PORT " && { pkill -f "127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT" 2>/dev/null || true; sleep 2; }
ss -tln 2>/dev/null | grep -q ":$BRIDGE_PORT " && fail "port-busy($BRIDGE_PORT)"

# ---- 1. VM serving + tunnel ----------------------------------------------------
log "STEP1 scp init_noise-aware bridge + deploy obscond ckpt (ZMQ :$ZMQ_PORT + bridge :$BRIDGE_PORT)"
FREE=$($SSH "$VM" "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" | head -1 | tr -d ' ')
[ -n "${FREE:-}" ] && [ "$FREE" -gt 20000 ] || fail "gpu-not-free(${FREE:-?}MiB)"
scp -q -o StrictHostKeyChecking=no "$ODY/scripts/serve_groot_http_bridge.py" "$VM:/home/ubuntu/serve_groot_http_bridge_obscond.py" || fail "scp-bridge"
$SSH "$VM" "bash /home/ubuntu/vm_deploy_obscond.sh $CKPT $ZMQ_PORT $BRIDGE_PORT" || fail "vm-deploy"
$SSH -f -N -o ExitOnForwardFailure=yes -L 127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT "$VM" || fail "tunnel"
for i in $(seq 1 30); do curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health | grep -q '"ok": *true' && break; sleep 5; done
curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health | grep -q '"ok": *true' || fail "bridge-health"
log "bridge tunnel green"

# ---- 2. sidecar (STEERING ON) + agent-service ---------------------------------
log "STEP2 start conditioner+steering sidecar :$COND_PORT"
start_sidecar ON "$WORK/conditioner_smoke_on.log" || fail "conditioner-steered-not-up"

log "STEP2 launch OWN agent-service :$AGENT_PORT -> conditioner :$COND_PORT"
cd "$AGENT_DIR"
env -u PYTHONPATH ENVIRONMENT=development \
    DATABASE_URL="sqlite+aiosqlite:///./stageb_agent${AGENT_PORT}.db" \
    GROOT_BRIDGE_URL="http://127.0.0.1:$COND_PORT" \
    GROOT_STATE_CONDITIONER_URL="http://127.0.0.1:$COND_PORT" \
    GROOT_OBSERVER_URL="http://127.0.0.1:$COND_PORT" \
    DISPLAY="$DISPLAY" "$AGENT_PY" -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$AGENT_PORT" --loop asyncio > "$WORK/agent${AGENT_PORT}.log" 2>&1 &
echo $! > "$WORK/agent_pid.txt"
cd "$BC"
for i in $(seq 1 20); do
  curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null | grep -q 200 && break
  sleep 3
done
curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null | grep -q 200 || fail "agent-service-not-serving"
curl -s --max-time 8 "http://127.0.0.1:$AGENT_PORT/api/groot/health" | grep -q '"ok": *true' || fail "agent-groot-health"
log "stack green (agent :$AGENT_PORT -> conditioner :$COND_PORT -> tunnel :$BRIDGE_PORT -> VM ZMQ :$ZMQ_PORT)"

# ---- 3. pre-eval end-to-end SMOKE GATE (mandatory) -----------------------------
log "STEP3 smoke gate: steered determinism through the FULL chain"
smoke_post(){   # $1 = out json (two identical POSTs through agent :8032)
  "$PYUR5E" - "$AGENT_PORT" "$1" <<'PY'
import base64, io, json, sys, urllib.request
port, outp = sys.argv[1], sys.argv[2]
from PIL import Image
buf = io.BytesIO(); Image.new("RGB", (256, 256), (127, 127, 127)).save(buf, format="PNG")
img = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
body = json.dumps({
    "image_b64": img, "image_b64_wrist": img,
    "state": [-1.57, -1.56, 1.58, -1.57, -1.57, 0.0, 0.05],
    "instruction": "pick up the vial and place it in the rack", "sid": "stageb-smoke",
}).encode()
def post():
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/groot/get_action", data=body,
                                 headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.status, json.loads(r.read().decode())
c1, r1 = post()
c2, r2 = post()
ok = (c1 == 200 and c2 == 200 and isinstance(r1.get("q"), list) and len(r1["q"]) == 6)
json.dump({"ok": bool(ok), "status": [c1, c2],
           "steered_flag": [bool(r1.get("steered")), bool(r2.get("steered"))],
           "identical": bool(json.dumps([r1.get("q"), r1.get("chunk_q"), r1.get("chunk_grip")])
                             == json.dumps([r2.get("q"), r2.get("chunk_q"), r2.get("chunk_grip")])),
           "q": r1.get("q"), "chunk_q": r1.get("chunk_q"), "chunk_grip": r1.get("chunk_grip"),
           "steering_phase": r1.get("steering_phase")}, open(outp, "w"), indent=2)
print("[smoke] status", c1, c2, "steered", r1.get("steered"), "phase", r1.get("steering_phase"))
PY
}
smoke_post "$OUTDIR/smoke_steered.json" || fail "smoke-steered-post"
grep -q '"ok": true' "$OUTDIR/smoke_steered.json" || fail "smoke-steered-bad-response"
grep -q '\[steer\]' "$WORK/conditioner_smoke_on.log" || fail "smoke-no-steer-log"
SMOKE_ATTACH=$(grep -c '\[steer\]' "$WORK/conditioner_smoke_on.log" || true)
log "smoke: steered POSTs ok, sidecar attached noise x$SMOKE_ATTACH"

log "STEP3 smoke gate: stock arm (sidecar restart WITHOUT weights)"
start_sidecar OFF "$WORK/conditioner_smoke_off.log" || fail "conditioner-stock-not-up"
smoke_post "$OUTDIR/smoke_stock.json" || fail "smoke-stock-post"
"$PYUR5E" - "$OUTDIR" <<'PY' || fail "smoke-gate"
import json, sys
out = sys.argv[1]
st = json.load(open(f"{out}/smoke_steered.json"))
sk = json.load(open(f"{out}/smoke_stock.json"))
det = bool(st["identical"] and st["steered_flag"] == [True, True])
differ = bool(json.dumps([st["q"], st["chunk_q"]]) != json.dumps([sk["q"], sk["chunk_q"]]))
stock_clean = bool(sk["steered_flag"] == [False, False])
res = {"steered_deterministic": det, "steered_vs_stock_differ": differ,
       "stock_carries_no_steer_flag": stock_clean}
json.dump(res, open(f"{out}/smoke_gate.json", "w"), indent=2)
print("[smoke-gate]", json.dumps(res))
assert det, "two identical steered requests did NOT return identical actions"
assert differ, "steered and stock actions are IDENTICAL - init_noise not reaching the sampler"
assert stock_clean, "stock response carries steered flag"
PY
log "SMOKE GATE PASS"

# ---- 4. Run A — STEERING ON (fresh sidecar => clean per-run counters) ----------
log "STEP4 RUN A: steering ON, N=$EVAL_N (plans seed 7777)"
start_sidecar ON "$WORK/conditioner_A.log" || fail "conditioner-runA-not-up"
mkdir -p "$OUTDIR/eval_A"
PLANS="$PLANS" OUT="$OUTDIR/eval_A" PORT="$AGENT_PORT" AGENTS=groot N="$EVAL_N" \
  N_ACTION_STEPS="$N_ACTION_STEPS" MAX_TICKS="$MAX_TICKS" \
  PUPPETEER_CORE="$PUP" CHROME="$CHROME" DISPLAY="$DISPLAY" \
  "$NODE" eval_browser_groot.js > "$WORK/eval_A_node.log" 2>&1
rc=$?
tail -20 "$WORK/eval_A_node.log"
kill_chrome_pidfile "$OUTDIR/eval_A/eval_pid.txt"
[ "$rc" -eq 0 ] || fail "eval-A(rc=$rc)"
[ -f "$OUTDIR/eval_A/eval_groot.json" ] || fail "no-eval-A-json"
curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health > "$OUTDIR/cond_health_A.json" || true
curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health > "$OUTDIR/bridge_health_postA.json" || true
log "RUN A done: $(cat "$OUTDIR/bridge_health_postA.json" 2>/dev/null || echo '?')"

# ---- 5. Run B — STOCK (steering OFF) -------------------------------------------
log "STEP5 RUN B: steering OFF, N=$EVAL_N (same plans)"
start_sidecar OFF "$WORK/conditioner_B.log" || fail "conditioner-runB-not-up"
mkdir -p "$OUTDIR/eval_B"
PLANS="$PLANS" OUT="$OUTDIR/eval_B" PORT="$AGENT_PORT" AGENTS=groot N="$EVAL_N" \
  N_ACTION_STEPS="$N_ACTION_STEPS" MAX_TICKS="$MAX_TICKS" \
  PUPPETEER_CORE="$PUP" CHROME="$CHROME" DISPLAY="$DISPLAY" \
  "$NODE" eval_browser_groot.js > "$WORK/eval_B_node.log" 2>&1
rc=$?
tail -20 "$WORK/eval_B_node.log"
kill_chrome_pidfile "$OUTDIR/eval_B/eval_pid.txt"
[ "$rc" -eq 0 ] || fail "eval-B(rc=$rc)"
[ -f "$OUTDIR/eval_B/eval_groot.json" ] || fail "no-eval-B-json"
curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health > "$OUTDIR/bridge_health_postB.json" || true
log "RUN B done: $(cat "$OUTDIR/bridge_health_postB.json" 2>/dev/null || echo '?')"

# ---- 6. verdict ----------------------------------------------------------------
log "STEP6 analyse + write $RESULT"
"$PYUR5E" - "$OUTDIR" "$RESULT" "$EVAL_N" "${SMOKE_ATTACH:-0}" <<'PY' || fail "analyse"
import json, sys
import statistics as stats
out, result, eval_n = sys.argv[1], sys.argv[2], int(sys.argv[3])
smoke_attach = int(sys.argv[4])

def L(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d

A = L(f"{out}/eval_A/eval_groot.json", {})
B = L(f"{out}/eval_B/eval_groot.json", {})
gate = L(f"{out}/smoke_gate.json", {})
condA = L(f"{out}/cond_health_A.json", {})
brA = L(f"{out}/bridge_health_postA.json", {})
brB = L(f"{out}/bridge_health_postB.json", {})

def pads_cm(ev):
    per = [(None if r.get("min_pad_to_vial") is None else round(r["min_pad_to_vial"] * 100.0, 2))
           for r in ev.get("results", [])]
    vals = [p for p in per if p is not None]
    return {"mean": round(stats.mean(vals), 2) if vals else None,
            "median": round(stats.median(vals), 2) if vals else None,
            "per_episode": per}

def run_block(ev):
    return {"success": ev.get("success"), "n_lifted": ev.get("n_lifted"),
            "n_seated": ev.get("n_seated"), "min_pad_cm": pads_cm(ev),
            "total_seconds": ev.get("total_seconds"), "results": ev.get("results")}

padA, padB = pads_cm(A), pads_cm(B)
nA, nB = A.get("n_success", 0), B.get("n_success", 0)
liftA = A.get("n_lifted", 0)
pad_improve = (None if (padA["median"] is None or padB["median"] is None)
               else round(padB["median"] - padA["median"], 2))
if nA > nB:
    verdict = "STEERED_SUCCESS"
elif (pad_improve is not None and pad_improve > 2.0) or (nA == 0 and liftA > 0):
    verdict = "STEERED_PROGRESS_NO_SUCCESS"
else:
    verdict = "NO_EFFECT"

steer = (condA.get("steering") or {})
ticks = steer.get("phase_ticks") or {}
tot = sum(ticks.values()) or 1
frac = {k: round(v / tot, 4) for k, v in ticks.items()}
attA = brA.get("init_noise_attached")
attB = brB.get("init_noise_attached")
qA = sum(r.get("queries", 0) for r in A.get("results", []))

res = {
  "smoke": {
    "steered_vs_stock_differ": gate.get("steered_vs_stock_differ"),
    "steered_deterministic": gate.get("steered_deterministic"),
    "noise_attached_count": smoke_attach,
  },
  "run_A_steered": run_block(A),
  "run_B_stock": run_block(B),
  "baseline_history": f"0/{eval_n} (all prior variants: render-gap browser GR00T, "
                      "DAgger, observer-conditioned ckpt-12000, nominal-target ablation; "
                      "headless reference 2/20)",
  "phase_inference": {
    "method": "grip-command + FK'd gr_pinch xy vs open->close locked xy state machine "
              "(A3 bucket semantics; GT-replay agreement 88.9% on 12 training episodes, "
              "mismatch dominated by the post-place home2 tail); GT classifier does not "
              "exist for phases, so option (a) heuristic was used",
    "per_phase_tick_frac": frac,
  },
  "audit": {
    "bridge_init_noise_attached_postA": attA,
    "bridge_init_noise_attached_postB": attB,
    "run_B_added_zero_attachments": (None if (attA is None or attB is None) else bool(attA == attB)),
    "run_A_queries": qA,
    "sidecar_attached_run_A": steer.get("attached"),
  },
  "verdict": verdict,
  "notes": "",
}
notes = []
notes.append(f"steered {A.get('success')} vs stock {B.get('success')} on the same seed-7777 poses; "
             f"min_pad median steered {padA['median']}cm vs stock {padB['median']}cm "
             f"(improvement {pad_improve}cm).")
if attA is not None and attB is not None and attA != attB:
    notes.append(f"AUDIT FAIL: bridge attach count moved during stock run B ({attA}->{attB}).")
if steer.get("attached") is not None and qA and steer.get("attached") != qA:
    notes.append(f"sidecar attached {steer.get('attached')} noises vs {qA} run-A queries "
                 "(mismatch = smoke/off-run requests or dropped queries).")
res["notes"] = " ".join(notes)
json.dump(res, open(result, "w"), indent=2)
print("[stageb] VERDICT:", verdict)
print("[stageb]", json.dumps({k: res[k] for k in ("smoke", "audit")}, indent=2))
PY

cleanup
log "DONE — results in $RESULT"
echo "STAGEB_AB DONE"
