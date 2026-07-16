#!/usr/bin/env bash
# ============================================================================
# run_probe_phase1.sh — SELF-DRIVING Phase-1 inference-input integrity probe
# (PLAN_MULTIAGENT.md Phase 1). Launched DETACHED; writes
# ~/probe_phase1_result.json + a PROBE_PHASE1 DONE/FAILED marker to $LOG.
#
# WHY: Gate 0 proved the eval ACTION path is faithful (expert 5/5 through the
# zero-order-hold), so the in-browser 0/15 must be policy-side — but Gate 0
# replayed RECORDED actions, so it could not see the observation/inference path.
# This probe instruments exactly that: what GR00T actually receives per tick,
# what it returns, whether the returned actions are at the right SCALE, and
# which inputs (target/proprio/images/instruction) the policy responds to.
#
# CHAIN:
#  1. VM: serve the obscond checkpoint (ZMQ :5558 + 10-D bridge :5596), tunnel.
#  2. Local: Observer conditioner (:5604) + OWN agent-service (:8032).
#  3. probe_browser_groot.js — 3 instrumented live episodes (same seed-7777 poses
#     as the 0/15 runs) + two raw frame/state dumps.
#  4. probe_inference_path.py — expert stats + controlled payload battery.
#  5. VM native-vs-bridge diff (vm_probe_native_diff.py).
#  6. Analyzer verdict -> ~/probe_phase1_result.json.
#
# ISOLATION: same conventions as run_observer_cond.sh — own ports (8032/5604/
# 5596/5558), own VM tmux (groot_obscond_*), own Chrome by PID, NEVER pkill
# chrome; ccproxy/gateway/groot_browser_* untouched. No training: serving only.
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

WORK=$HOME/probe_phase1
LOG=$HOME/probe_phase1.log
RESULT=$HOME/probe_phase1_result.json
OUTDIR=$WORK/probe_out
# Fresh OUTDIR every run: stale frame dumps / ep jsonls from an aborted attempt
# must never leak into a later verdict.
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

AGENT_PORT=8032
ZMQ_PORT=5558
BRIDGE_PORT=5596
COND_PORT=5604
CKPT=/home/ubuntu/ckpt/ur5e_drugsort_obscond/full/checkpoint-12000
OBS_WEIGHTS=$BC/percep_weights_browser
DATASET=$BC/dataset_browser_cond
PLANS=$HOME/observer_cond/plans_eval.json
PROBE_N=${PROBE_N:-3}
K=${K:-5}

export DISPLAY=${DISPLAY:-:1}
cd "$BC"
exec >>"$LOG" 2>&1
echo "" ; echo "#################################################################"
log(){ echo "=== [probe1] $* $(date -u +%FT%TZ) ==="; }

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
    if tr '\0' ' ' < "/proc/$cp/cmdline" 2>/dev/null | grep -q "chrome-udd-probe"; then
      echo "  [probe1] kill Chrome PID $cp"; kill "$cp" 2>/dev/null || true
    fi
  fi
  rm -f "$pf" 2>/dev/null || true
}
fail(){
  trap - TERM INT HUP
  log "FAILED at: $*"
  printf '{"phase":"1-inference-input-integrity-probe","status":"FAILED","failed_at":"%s"}\n' "$*" > "$RESULT"
  kill_chrome_pidfile "$OUTDIR/probe_pid.txt"
  cleanup
  echo "PROBE_PHASE1 FAILED: $*"
  exit 1
}
# A signal to the detached driver must still leave a RESULT + marker and tear the
# stack down (fail() kills Chrome by pidfile, runs cleanup, emits the marker).
trap 'fail "signal"' TERM INT HUP

log "START pid=$$ ckpt=$CKPT n=$PROBE_N k=$K"
echo "PROBE_PHASE1 PID=$$"

# ---- 0. sanity ---------------------------------------------------------------
[ -f "$PLANS" ] || fail "no-plans($PLANS)"
[ -f "$DATASET/meta/info.json" ] || fail "no-dataset($DATASET)"
[ -f "$OBS_WEIGHTS/observer_head.pt" ] || fail "no-observer-weights"
$SSH "$VM" "echo ok >/dev/null" || fail "vm-unreachable"
$SSH "$VM" "[ -d $CKPT ]" || fail "no-vm-ckpt($CKPT)"
for port in $AGENT_PORT $COND_PORT; do
  ss -tln 2>/dev/null | grep -q ":$port " && fail "port-busy($port)"
done

# ---- 1. VM serving + tunnel ---------------------------------------------------
log "STEP1 deploy obscond ckpt on VM (ZMQ :$ZMQ_PORT + bridge :$BRIDGE_PORT)"
FREE=$($SSH "$VM" "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" | head -1 | tr -d ' ')
[ -n "${FREE:-}" ] && [ "$FREE" -gt 20000 ] || fail "gpu-not-free(${FREE:-?}MiB)"
$SSH "$VM" "bash /home/ubuntu/vm_deploy_obscond.sh $CKPT $ZMQ_PORT $BRIDGE_PORT" || fail "vm-deploy"
pkill -f "127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT" 2>/dev/null || true; sleep 2
$SSH -f -N -o ExitOnForwardFailure=yes -L 127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT "$VM" || fail "tunnel"
for i in $(seq 1 30); do curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health | grep -q '"ok": *true' && break; sleep 5; done
curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health | grep -q '"ok": *true' || fail "bridge-health"
log "bridge tunnel green"

# ---- 2. conditioner + agent-service ------------------------------------------
log "STEP2 start Observer conditioner :$COND_PORT"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OBSERVER_WEIGHTS="$OBS_WEIGHTS" \
  CONDITION_BRIDGE_URL="http://127.0.0.1:$BRIDGE_PORT" OBSERVER_DEVICE=cuda PYTHONPATH="$ODY/src" \
  setsid "$OBSPY" "$ODY/scripts/serve_observer_conditioning.py" --port "$COND_PORT" \
  >"$WORK/conditioner.log" 2>&1 &
echo $! > "$WORK/cond_pid.txt"
for i in $(seq 1 40); do curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health 2>/dev/null | grep -q '"ok": *true' && break; sleep 3; done
CH=$(curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health || true)
{ echo "$CH" | grep -q '"observer_ready": *true' && echo "$CH" | grep -q '"ok": *true'; } || fail "conditioner-not-up($CH)"

log "STEP2 launch OWN agent-service :$AGENT_PORT -> conditioner :$COND_PORT"
cd "$AGENT_DIR"
env -u PYTHONPATH ENVIRONMENT=development \
    DATABASE_URL="sqlite+aiosqlite:///./probe1_agent${AGENT_PORT}.db" \
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

# ---- 3. instrumented live episodes --------------------------------------------
log "STEP3 probe_browser_groot.js (N=$PROBE_N instrumented episodes)"
PLANS="$PLANS" OUT="$OUTDIR" PORT="$AGENT_PORT" N="$PROBE_N" \
  N_ACTION_STEPS=8 MAX_TICKS=900 \
  PUPPETEER_CORE="$PUP" CHROME="$CHROME" DISPLAY="$DISPLAY" \
  "$NODE" probe_browser_groot.js > "$WORK/probe_browser.log" 2>&1
rc=$?
tail -25 "$WORK/probe_browser.log"
kill_chrome_pidfile "$OUTDIR/probe_pid.txt"
[ "$rc" -eq 0 ] || fail "probe-browser(rc=$rc)"
[ -f "$OUTDIR/probe_episodes.json" ] || fail "no-probe-episodes"
[ -f "$OUTDIR/frame_q0_ext.b64" ] || fail "no-frame-dumps"

# ---- 4. VM native-vs-bridge diff ----------------------------------------------
log "STEP4 VM native-ZMQ vs HTTP-bridge diff"
TGT=$("$PYUR5E" -c "
import json
r=[json.loads(l) for l in open('$OUTDIR/probe_ep0.jsonl') if l.strip()]
t=[x['grasp_target'] for x in r if x.get('grasp_target')]
print(','.join(f'{v:.5f}' for v in t[0]) if t else '-0.425,-0.015,0.272')")
# NON-FATAL: the analyzer tolerates a missing vm_diff (bridge_matches_native ->
# null) — a transient ZMQ hiccup here must not throw away the whole run.
# --target= equals form is REQUIRED (argparse misparses a bare leading-dash list).
VM_DIFF_OK=0
if scp -q -o StrictHostKeyChecking=no vm_probe_native_diff.py \
  "$OUTDIR/frame_q0_ext.b64" "$OUTDIR/frame_q0_wrist.b64" "$OUTDIR/frame_q0_state.json" \
  "$VM:/home/ubuntu/"; then
  if $SSH "$VM" "cd /home/ubuntu && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    timeout 900 ~/Isaac-GR00T/.venv/bin/python vm_probe_native_diff.py \
    --ext frame_q0_ext.b64 --wrist frame_q0_wrist.b64 --state frame_q0_state.json \
    --target='$TGT' --zmq-port $ZMQ_PORT --bridge http://127.0.0.1:$BRIDGE_PORT \
    --k $K --out vm_diff.json"; then
    scp -q -o StrictHostKeyChecking=no "$VM:/home/ubuntu/vm_diff.json" "$OUTDIR/vm_diff.json" && VM_DIFF_OK=1
  fi
fi
[ "$VM_DIFF_OK" = "1" ] || log "vm-diff FAILED (non-fatal — verdict proceeds without the bridge/native comparison)"

# ---- 5. battery + verdict -----------------------------------------------------
log "STEP5 controlled payload battery + analyzer verdict"
HOMEQ=$("$PYUR5E" -c "
import json
p=json.load(open('$PLANS'))['plans'][0]
print(','.join(f'{v:.5f}' for v in p['home_q']))")
env -u PYTHONPATH "$PYUR5E" probe_inference_path.py \
  --out "$OUTDIR" --dataset "$DATASET" \
  --agent-url "http://127.0.0.1:$AGENT_PORT" --bridge-url "http://127.0.0.1:$BRIDGE_PORT" \
  --vm-diff "$OUTDIR/vm_diff.json" --k "$K" --home-q="$HOMEQ" \
  --result "$RESULT" || fail "battery"

log "STEP6 verdict written: $RESULT"
"$PYUR5E" -c "import json;d=json.load(open('$RESULT'));print('[probe1] VERDICT:',json.dumps(d['verdict']))"
cleanup
log "DONE — results in $RESULT"
echo "PROBE_PHASE1 DONE"
