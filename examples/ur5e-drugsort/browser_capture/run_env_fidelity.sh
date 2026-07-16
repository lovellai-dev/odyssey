#!/usr/bin/env bash
# run_env_fidelity.sh — DETACHED driver for the in-browser eval action-path fidelity
# diagnostic. Brings up an OWN agent-service (static Playground page only — no GR00T
# bridge, no GPU), runs env_fidelity.js (A1 expert-native vs A2 expert-through-eval-path
# + a hold/interp sweep), writes ~/env_fidelity_result.json, and appends
# "ENV_FIDELITY DONE" to ~/env_fidelity.log.
#
# ISOLATION: OWN agent-service on :$AGENT_PORT (killed by PID), OWN Chrome (unique
# --user-data-dir, killed by PID captured in env_fidelity_out/fidelity_pid.txt). NEVER
# pkill chrome. Touches NO other port / tmux / VM. set -uo pipefail + explicit fail().
set -uo pipefail

BC=/home/daniel/LovellAI/odyssey-ur5e/examples/ur5e-drugsort/browser_capture
AGENT_DIR=/home/daniel/LovellAI/lai-agent-multiagent/agent_service
AGENT_PY=/home/daniel/LovellAI/lai-agent/agent_service/.venv/bin/python
NODE=/home/daniel/.nvm/versions/node/v22.22.0/bin/node
PUP=/home/daniel/.npm/_npx/23232c69e5d221f3/node_modules/puppeteer-core
CHROME=/usr/bin/google-chrome-stable
AGENT_PORT=${AGENT_PORT:-8043}
OUT=$BC/env_fidelity_out
LOG=$HOME/env_fidelity.log
RESULT=$HOME/env_fidelity_result.json
export DISPLAY=${DISPLAY:-:1}
mkdir -p "$OUT"
cd "$BC"
exec >>"$LOG" 2>&1

log(){ echo "=== [env_fidelity] $* $(date -u +%FT%TZ) ==="; }
AGENT_PID=""
cleanup(){
  # Kill the OWN agent-service by PID (and its uvicorn worker holding the port).
  if [ -n "${AGENT_PID:-}" ]; then kill "$AGENT_PID" 2>/dev/null || true; fi
  local p; p=$(ss -ltnp 2>/dev/null | grep ":$AGENT_PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
  if [ -n "${p:-}" ]; then kill "$p" 2>/dev/null || true; fi
  # Kill the OWN Chrome by the PID env_fidelity.js recorded (NEVER pkill chrome).
  if [ -f "$OUT/fidelity_pid.txt" ]; then
    local cpid; cpid=$(cat "$OUT/fidelity_pid.txt" 2>/dev/null || true)
    if [ -n "${cpid:-}" ] && ps -p "$cpid" -o comm= 2>/dev/null | grep -qi chrome; then kill "$cpid" 2>/dev/null || true; fi
  fi
}
fail(){ log "FAILED: $*"; echo "ENV_FIDELITY FAILED: $*"; cleanup; echo "ENV_FIDELITY DONE"; exit 1; }
trap cleanup EXIT

log "START pid=$$ agent_port=$AGENT_PORT display=$DISPLAY"

# ---- 1. bring up OWN agent-service (static page only) -----------------------
STALE=$(ss -tlnp 2>/dev/null | grep ":$AGENT_PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
[ -n "${STALE:-}" ] && fail "port $AGENT_PORT already in use (pid $STALE) — refusing to disturb it"
cd "$AGENT_DIR"
env -u PYTHONPATH ENVIRONMENT=development \
    DATABASE_URL="sqlite+aiosqlite:///./env_fidelity_agent${AGENT_PORT}.db" \
    DISPLAY="$DISPLAY" "$AGENT_PY" -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$AGENT_PORT" --loop asyncio > "$OUT/agent${AGENT_PORT}.log" 2>&1 &
AGENT_PID=$!
cd "$BC"
log "launched agent-service pid=$AGENT_PID"
sleep 10
ok=""
for i in $(seq 1 20); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null)
  [ "$code" = "200" ] && { ok=1; break; }
  sleep 3
done
[ -n "$ok" ] || fail "agent-service not serving on $AGENT_PORT"
log "agent-service serving (pid=$AGENT_PID)"

# ---- 2. run the fidelity experiment -----------------------------------------
log "STEP2 env_fidelity.js (A1 vs A2 + sweep) N=${N:-5}"
PLANS="$BC/plans_eval.json" OUT="$OUT" PORT="$AGENT_PORT" N="${N:-5}" \
  CHROME="$CHROME" PUPPETEER_CORE="$PUP" DISPLAY="$DISPLAY" \
  "$NODE" "$BC/env_fidelity.js" || fail "env_fidelity.js crashed"

# ---- 3. finalize ------------------------------------------------------------
if [ -f "$RESULT" ]; then
  log "result written: $RESULT"
  "$AGENT_PY" - "$RESULT" <<'PY' 2>/dev/null || true
import json,sys
d=json.load(open(sys.argv[1]))
print("VERDICT:", d.get("verdict"))
print("A1", d["A1_expert_native"]["success"], "A2", d["A2_expert_through_eval_path"]["success"])
PY
else
  fail "no result JSON produced"
fi
cleanup
log "DONE"
echo "ENV_FIDELITY DONE"
