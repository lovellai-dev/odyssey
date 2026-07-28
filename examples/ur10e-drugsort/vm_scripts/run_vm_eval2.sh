#!/usr/bin/env bash
# VM-native eval, parametrized: MODE=bare|v4 (default v4). bare = raw GR00T policy
# (no steering head, K=1) to test whether GR00T ITSELF reaches on the playground.
set -uo pipefail
export NVM_DIR=$HOME/.nvm; . "$NVM_DIR/nvm.sh" >/dev/null 2>&1; nvm use 22 >/dev/null 2>&1
MODE=${MODE:-v4}; N=${N:-4}; TAG=${TAG:-$MODE}
ODY=$HOME/odyssey-ur5e; BC=$ODY/examples/ur5e-drugsort/browser_capture
AGENT_DIR=$HOME/lai-agent-multiagent/agent_service
GROOT_PY=$HOME/Isaac-GR00T/.venv/bin/python; EVAL_PY=$HOME/odyssey-eval-venv/bin/python
AGENT_PY=$AGENT_DIR/.venv/bin/python; NODE=$(command -v node)
XML=${EVAL_XML:-$HOME/lai-agent-multiagent/src/embodiments/urdf/aseptipack_description/aseptipack.xml}
OBS_WEIGHTS=${OBS_WEIGHTS_OVERRIDE:-$BC/percep_weights_browser}
CKPT=${CKPT_OVERRIDE:-$HOME/ckpt/ur5e_drugsort_obscond/full/checkpoint-12000}; V4NET=$HOME/steering_net_v4.npz
PUP=$HOME/vm_eval/node_modules/puppeteer-core; CHROME=/usr/bin/google-chrome-stable
HTTP_PORT=5596; FK_PORT=5560; COND_PORT=5604; AGENT_PORT=8032
WORK=$HOME/vm_eval_$TAG; rm -rf "$WORK"; mkdir -p "$WORK"
LOG=$HOME/vm_eval_$TAG.log; : > "$LOG"; exec >>"$LOG" 2>&1
log(){ echo "=== [vm-eval:$MODE] $* $(date -u +%FT%TZ) ==="; }
cleanup(){ for f in "$WORK"/cond_pid.txt "$WORK"/agent_pid.txt "$WORK"/out/eval_pid.txt; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; done; }
fail(){ log "FAILED at: $*"; echo "VM_EVAL FAILED: $*"; cleanup; exit 1; }
trap cleanup EXIT
log "START pid=$$ MODE=$MODE N=$N"

if [ "$MODE" = "bare" ]; then DEPLOY_V4=""; BON_K=1; SIDE=(STEERING_WEIGHTS=); else DEPLOY_V4="$V4NET"; BON_K=16; SIDE=(STEERING_WEIGHTS= STEER_SERVER_HEAD=1); fi

log "STEP1 deploy FK + bestofn (v4net='$DEPLOY_V4')"
STEER_SERVO=${STEER_SERVO:-0} STEER_HANDOFF=${STEER_HANDOFF:-0} STEER_DIAG=0 STEER_HANDOFF_SMOOTH=${STEER_HANDOFF_SMOOTH:-15} STEER_HANDOFF_COVMAX=${STEER_HANDOFF_COVMAX:-2.0} STEER_HANDOFF_NEARWIN=${STEER_HANDOFF_NEARWIN:-1} STEER_HANDOFF_LOG=${STEER_HANDOFF_LOG:-0} STEER_HANDOFF_HOLD=${STEER_HANDOFF_HOLD:-2} STEER_HANDOFF_VERIFY=${STEER_HANDOFF_VERIFY:-3} STEER_HANDOFF_ATTEMPTS=${STEER_HANDOFF_ATTEMPTS:-4} STEER_HANDOFF_RETRACT=${STEER_HANDOFF_RETRACT:-0.06} STEER_HANDOFF_JITTER=${STEER_HANDOFF_JITTER:-0.012} bash "$BC/vm_deploy_bestofn.sh" "$CKPT" "$HTTP_PORT" "$FK_PORT" "$DEPLOY_V4" || fail "deploy"
# Engagement guard: env knobs must be IN the service process env, not just ours
# (block-2 lesson: STEER_PLACE_* set here never reached the tmux-launched service).
# The tmux bash wrapper matches the same pgrep pattern but lacks the inline
# env (only the python child has it) — pick the pid that actually carries the
# knobs; fall back to any match for the error message.
SVCP=""
for P in $(pgrep -f "serve_groot_bestofn.py --model-path"); do
  if tr "\0" "\n" < "/proc/$P/environ" 2>/dev/null | grep -q "^STEER_HANDOFF="; then SVCP=$P; break; fi
done
[ -z "$SVCP" ] && SVCP=$(pgrep -f "serve_groot_bestofn.py --model-path" | head -1)
for V in STEER_PLACE_Z_HI STEER_HANDOFF_DESCEND_STEP STEER_HANDOFF_CLOSE_RAMP STEER_HANDOFF_APPROACH_BUDGET STEER_HANDOFF_SEAT STEER_HANDOFF_RISE_STEP STEER_HANDOFF_DESCEND_LATCH STEER_HANDOFF_COMMIT_THROUGH STEER_HANDOFF_TRANSIT_STEP STEER_HANDOFF_DROP_RETRY; do
  WANT=$(eval echo "\${$V:-}")
  if [ -n "$WANT" ]; then
    tr "\0" "\n" < "/proc/$SVCP/environ" | grep -q "^$V=$WANT$" || fail "$V not in service env"
  fi
done

log "STEP2 conditioner (K=$BON_K)"
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OBSERVER_WEIGHTS="$OBS_WEIGHTS" HANDOFF_CFG="${STEER_HANDOFF:-0}" GRASP_OK_HYST="${GRASP_OK_HYST:-2}" \
  CONDITION_BRIDGE_URL="http://127.0.0.1:$HTTP_PORT" OBSERVER_DEVICE=cuda PYTHONPATH="$ODY/src" \
  "${SIDE[@]}" STEERING_FK_XML="$XML" STEER_BESTOFN_K=$BON_K STEER_BESTOFN_K_GRASP=$BON_K \
  STEER_BESTOFN_SIGMA=0.25 STEER_CLF_CENTERING=off STEER_CEM_ROUNDS=$([ "$MODE" = bare ] && echo 0 || echo 2) \
  setsid "$GROOT_PY" "$ODY/scripts/serve_observer_conditioning.py" --port "$COND_PORT" >"$WORK/conditioner.log" 2>&1 &
echo $! > "$WORK/cond_pid.txt"
for i in $(seq 1 70); do curl -s --max-time 5 "http://127.0.0.1:$COND_PORT/health" 2>/dev/null | grep -q '"observer_ready": *true' && break; sleep 3; done
curl -s --max-time 5 "http://127.0.0.1:$COND_PORT/health" | grep -q '"observer_ready": *true' || fail "conditioner"

log "STEP3 agent"
cd "$AGENT_DIR"
env -u PYTHONPATH ENVIRONMENT=development DATABASE_URL="sqlite+aiosqlite:///$WORK/agent.db" \
  GROOT_BRIDGE_URL="http://127.0.0.1:$COND_PORT" GROOT_STATE_CONDITIONER_URL="http://127.0.0.1:$COND_PORT" \
  GROOT_OBSERVER_URL="http://127.0.0.1:$COND_PORT" \
  setsid "$AGENT_PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$AGENT_PORT" --loop asyncio >"$WORK/agent.log" 2>&1 &
echo $! > "$WORK/agent_pid.txt"
for i in $(seq 1 40); do curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null | grep -q 200 && break; sleep 3; done
curl -s --max-time 8 "http://127.0.0.1:$AGENT_PORT/api/groot/health" | grep -q '"ok": *true' || fail "agent-health"

log "STEP4 browser eval N=$N"
cd "$BC"; cp -f "${PLANS_OVERRIDE:-$BC/plans_eval.json}" "$WORK/plans.json"
PLANS="$WORK/plans.json" OUT="$WORK/out" PORT="$AGENT_PORT" AGENTS=groot N="$N" N_ACTION_STEPS=8 MAX_TICKS=${MAX_TICKS:-900} \
  PUPPETEER_CORE="$PUP" CHROME="$CHROME" RAW_DUMP=${RAW_DUMP:-0} "$NODE" "$BC/eval_browser_groot.js" >"$WORK/eval.log" 2>&1
tail -8 "$WORK/eval.log"; echo "VM_EVAL DONE MODE=$MODE"
