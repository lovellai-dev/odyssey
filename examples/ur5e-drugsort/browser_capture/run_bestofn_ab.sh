#!/usr/bin/env bash
# ============================================================================
# run_bestofn_ab.sh — R2 best-of-N browser eval (selection-first pivot).
#
# Serves the best-of-N stack (in-process GR00T + CBF filter + CLF rank around
# the PROVEN v0.2 steering mean) and runs 15 browser episodes on the seed-7777
# eval poses. Compared against the BANKED baselines measured on the same poses
# and protocol: steered-only v0.2 = 0/15 (median pad 4.4cm, 2 mistimed closes),
# stock = 0/15 (median 8.1cm). Writes ~/bestofn_ab_result.json + marker
# BESTOFN_AB DONE|FAILED.
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
XML=/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/aseptipack_description/aseptipack.xml
PUP=/home/daniel/.npm/_npx/23232c69e5d221f3/node_modules/puppeteer-core
CHROME=/usr/bin/google-chrome-stable
NODE=/home/daniel/.nvm/versions/node/v22.22.0/bin/node

WORK=$HOME/bestofn_ab
LOG=$HOME/bestofn_ab.log
RESULT=$HOME/bestofn_ab_result.json
rm -rf "$WORK/out"; mkdir -p "$WORK/out"

AGENT_PORT=8032
HTTP_PORT=5596
FK_PORT=5560
COND_PORT=5604
CKPT=/home/ubuntu/ckpt/ur5e_drugsort_obscond/full/checkpoint-12000
STEER_SRC=${STEER_SRC:-/home/ubuntu/steering_net_v02.npz}
OBS_WEIGHTS=$BC/percep_weights_browser
PLANS=$HOME/observer_cond/plans_eval.json
EVAL_N=${EVAL_N:-15}
BON_K=${BON_K:-16}
BON_SIGMA=${BON_SIGMA:-0.25}

export DISPLAY=${DISPLAY:-:1}
cd "$BC"
exec >>"$LOG" 2>&1
echo "" ; echo "#################################################################"
log(){ echo "=== [bestofn] $* $(date -u +%FT%TZ) ==="; }

cleanup(){
  log "cleanup (own agent/conditioner/tunnel/VM bestofn sessions; leave everything else)"
  for pf in "$WORK/agent_pid.txt" "$WORK/cond_pid.txt"; do
    [ -f "$pf" ] || continue
    p=$(cat "$pf" 2>/dev/null || true); [ -n "${p:-}" ] && kill "$p" 2>/dev/null || true
  done
  ss -tlnp 2>/dev/null | grep -E ":($AGENT_PORT|$COND_PORT) " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | xargs -r kill 2>/dev/null || true
  pkill -f "127.0.0.1:$HTTP_PORT:127.0.0.1:$HTTP_PORT" 2>/dev/null || true
  $SSH "$VM" "tmux kill-session -t groot_bestofn_svc 2>/dev/null; tmux kill-session -t groot_bestofn_fk 2>/dev/null" 2>/dev/null || true
}
kill_chrome_pidfile(){
  local pf="$1"; [ -f "$pf" ] || return 0
  local cp; cp=$(cat "$pf" 2>/dev/null || true)
  if [ -n "${cp:-}" ] && kill -0 "$cp" 2>/dev/null; then
    if tr '\0' ' ' < "/proc/$cp/cmdline" 2>/dev/null | grep -q "chrome-udd"; then
      echo "  [bestofn] kill Chrome PID $cp"; kill "$cp" 2>/dev/null || true
    fi
  fi
  rm -f "$pf" 2>/dev/null || true
}
fail(){
  trap - TERM INT HUP
  log "FAILED at: $*"
  printf '{"eval":"bestofn-r2","status":"FAILED","failed_at":"%s"}\n' "$*" > "$RESULT"
  kill_chrome_pidfile "$WORK/out/eval_pid.txt"
  cleanup
  echo "BESTOFN_AB FAILED: $*"
  exit 1
}
trap 'fail "signal"' TERM INT HUP

log "START pid=$$ k=$BON_K sigma=$BON_SIGMA steer=$STEER_SRC"
echo "BESTOFN_AB PID=$$"

# ---- 0. sanity ----------------------------------------------------------------
[ -f "$PLANS" ] || fail "no-plans"
[ -f "$OBS_WEIGHTS/observer_head.pt" ] || fail "no-observer-weights"
$SSH "$VM" "echo ok >/dev/null" || fail "vm-unreachable"
$SSH "$VM" "[ -d $CKPT ] && [ -f $STEER_SRC ]" || fail "vm-assets"
FREE=$($SSH "$VM" "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" | head -1 | tr -d ' ')
[ -n "${FREE:-}" ] && [ "$FREE" -gt 20000 ] || fail "gpu-not-free(${FREE:-?}MiB)"
for port in $AGENT_PORT $COND_PORT; do
  ss -tln 2>/dev/null | grep -q ":$port " && fail "port-busy($port)"
done
ss -tln 2>/dev/null | grep -q ":$HTTP_PORT " && { pkill -f "127.0.0.1:$HTTP_PORT:127.0.0.1:$HTTP_PORT" 2>/dev/null || true; sleep 2; }

# ---- 1. stage + deploy the bestofn stack on the VM -----------------------------
log "STEP1 scp bestofn stack + deploy (svc :$HTTP_PORT, fk :$FK_PORT)"
scp -q -o StrictHostKeyChecking=no "$ODY/scripts/serve_groot_bestofn.py" \
  "$ODY/scripts/serve_fk_ur5e.py" "$ODY/scripts/bestofn_select.py" \
  "$ODY/scripts/clf_reward_ur5e.py" "$ODY/scripts/cbf_constraints_ur5e.py" \
  "$ODY/scripts/probe_flow_inversion_groot.py" \
  vm_deploy_bestofn.sh "$VM:/home/ubuntu/" || fail "scp"
$SSH "$VM" "bash /home/ubuntu/vm_deploy_bestofn.sh $CKPT $HTTP_PORT $FK_PORT" || fail "vm-deploy"
$SSH -f -N -o ExitOnForwardFailure=yes -L 127.0.0.1:$HTTP_PORT:127.0.0.1:$HTTP_PORT "$VM" || fail "tunnel"
for i in $(seq 1 30); do curl -s --max-time 5 http://127.0.0.1:$HTTP_PORT/health | grep -q '"ok": *true' && break; sleep 5; done
curl -s --max-time 5 http://127.0.0.1:$HTTP_PORT/health | grep -q '"bestofn": *true' || fail "bestofn-health"

# ---- 2. sidecar (steering mean + bestofn cfg) + agent-service ------------------
log "STEP2 sidecar :$COND_PORT (v0.2 mean, K=$BON_K sigma=$BON_SIGMA)"
scp -q -o StrictHostKeyChecking=no "$VM:$STEER_SRC" "$WORK/steering_net.npz" || fail "fetch-steer"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OBSERVER_WEIGHTS="$OBS_WEIGHTS" \
  CONDITION_BRIDGE_URL="http://127.0.0.1:$HTTP_PORT" OBSERVER_DEVICE=cuda PYTHONPATH="$ODY/src" \
  STEERING_WEIGHTS="$WORK/steering_net.npz" STEERING_FK_XML="$XML" \
  STEER_BESTOFN_K="$BON_K" STEER_BESTOFN_SIGMA="$BON_SIGMA" \
  setsid "$OBSPY" "$ODY/scripts/serve_observer_conditioning.py" --port "$COND_PORT" \
  >"$WORK/conditioner.log" 2>&1 &
echo $! > "$WORK/cond_pid.txt"
for i in $(seq 1 40); do curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health 2>/dev/null | grep -q '"ok": *true' && break; sleep 3; done
CH=$(curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health || true)
{ echo "$CH" | grep -q '"observer_ready": *true' && echo "$CH" | grep -q '"ok": *true'; } || fail "conditioner($CH)"

log "STEP2 agent-service :$AGENT_PORT"
cd "$AGENT_DIR"
env -u PYTHONPATH ENVIRONMENT=development \
    DATABASE_URL="sqlite+aiosqlite:///./bestofn_agent${AGENT_PORT}.db" \
    GROOT_BRIDGE_URL="http://127.0.0.1:$COND_PORT" \
    GROOT_STATE_CONDITIONER_URL="http://127.0.0.1:$COND_PORT" \
    GROOT_OBSERVER_URL="http://127.0.0.1:$COND_PORT" \
    DISPLAY="$DISPLAY" "$AGENT_PY" -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$AGENT_PORT" --loop asyncio > "$WORK/agent.log" 2>&1 &
echo $! > "$WORK/agent_pid.txt"
cd "$BC"
for i in $(seq 1 20); do
  curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null | grep -q 200 && break
  sleep 3
done
curl -s --max-time 8 "http://127.0.0.1:$AGENT_PORT/api/groot/health" | grep -q '"ok": *true' || fail "agent-health"

# ---- 3. smoke: one steered+bestofn POST must carry a selection report ----------
log "STEP3 smoke (bestofn report present, feasible>0)"
"$PYUR5E" - "$AGENT_PORT" "$WORK/smoke.json" <<'PY' || fail "smoke"
import base64, io, json, sys, time, urllib.request
port, outp = sys.argv[1], sys.argv[2]
from PIL import Image
buf = io.BytesIO(); Image.new("RGB", (256, 256), (127, 127, 127)).save(buf, format="PNG")
img = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
body = json.dumps({"image_b64": img, "image_b64_wrist": img,
                   "state": [-1.57, -1.56, 1.58, -1.57, -1.57, 0.0, 0.05],
                   "instruction": "pick up the vial and place it in the rack",
                   "sid": f"smoke-{time.time_ns()}"}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/api/groot/get_action", data=body,
                             headers={"content-type": "application/json"}, method="POST")
res = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
bon = res.get("bestofn") or {}
ok = (len(res.get("q") or []) == 6 and bon.get("k", 0) > 1)
json.dump({"ok": ok, "bestofn": bon}, open(outp, "w"), indent=2)
print("[smoke]", json.dumps({"ok": ok, "k": bon.get("k"), "n_feasible": bon.get("n_feasible"),
                             "chosen": bon.get("chosen"), "hold": bon.get("fallback_hold")}))
assert ok, res
PY

# ---- 4. eval: 15 episodes ------------------------------------------------------
log "STEP4 eval N=$EVAL_N (seed-7777 poses; baselines banked: v0.2 steered 0/15 pad 4.4, stock 0/15 pad 8.1)"
PLANS="$PLANS" OUT="$WORK/out" PORT="$AGENT_PORT" AGENTS=groot N="$EVAL_N" \
  N_ACTION_STEPS=8 MAX_TICKS=900 \
  PUPPETEER_CORE="$PUP" CHROME="$CHROME" DISPLAY="$DISPLAY" \
  "$NODE" eval_browser_groot.js > "$WORK/eval_node.log" 2>&1
rc=$?
tail -20 "$WORK/eval_node.log"
kill_chrome_pidfile "$WORK/out/eval_pid.txt"
[ "$rc" -eq 0 ] || fail "eval(rc=$rc)"
[ -f "$WORK/out/eval_groot.json" ] || fail "no-eval-json"

# ---- 5. aggregate --------------------------------------------------------------
log "STEP5 aggregate (+ service-side selection stats)"
BH=$(curl -s --max-time 5 http://127.0.0.1:$HTTP_PORT/health || echo '{}')
"$PYUR5E" - "$WORK/out/eval_groot.json" "$RESULT" "$BON_K" "$BON_SIGMA" "$BH" <<'PY' || fail "aggregate"
import json, sys
ev = json.load(open(sys.argv[1]))
out = {
    "eval": "bestofn-r2", "k": int(sys.argv[3]), "sigma": float(sys.argv[4]),
    "success": ev.get("success"), "n_lifted": ev.get("n_lifted"),
    "n_seated": ev.get("n_seated"),
    "results": ev.get("results"),
    "service_health": json.loads(sys.argv[5] or "{}"),
    "baselines_same_protocol": {
        "steered_v02": "0/15, pad median 4.4cm, grips [2 full closes mistimed]",
        "stock": "0/15, pad median 8.1cm, 3 freezes",
    },
    "status": "DONE",
}
json.dump(out, open(sys.argv[2], "w"), indent=2)
pads = [r.get("min_pad_to_vial") for r in ev.get("results") or []]
print("[bestofn] headline:", json.dumps({
    "success": out["success"], "lifted": out["n_lifted"],
    "pads_cm": [None if p is None else round(p * 100, 1) for p in pads],
    "grips": [round(r.get("gripMax", 0), 2) for r in ev.get("results") or []]}))
PY
cleanup
log "DONE — results in $RESULT"
echo "BESTOFN_AB DONE"
