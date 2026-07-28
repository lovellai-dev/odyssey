#!/usr/bin/env bash
# CUSTOMER DEMO stack: FK + GR00T-bestofn (handoff + brain telemetry) +
# Observer conditioner (Obs-C2 + uncertainty guard) + agent service (:8032).
# No eval driver — the mission runs inside the presenter-facing browser via
# demo_mission.js. Idempotent: safe to re-run; kills only its own children.
set -uo pipefail
ODY=$HOME/odyssey-ur5e; BC=$ODY/examples/ur5e-drugsort/browser_capture
AGENT_DIR=$HOME/lai-agent-multiagent/agent_service
GROOT_PY=$HOME/Isaac-GR00T/.venv/bin/python
AGENT_PY=$AGENT_DIR/.venv/bin/python
XML=$HOME/lai-agent-multiagent/src/embodiments/urdf/aseptipack_ur10e_description/aseptipack.xml
CKPT=${UR10E_CKPT:-$HOME/ckpt/ur10e_drugsort_condaug/checkpoint-12000}
OBS=${UR10E_OBS:-$HOME/ur10e_percep_weights}   # UR10e v1 observer
[ -d "$CKPT" ] || { echo "UR10E_DEMO_FAIL no-ckpt $CKPT"; exit 1; }
[ -d "$OBS" ] || { echo "UR10E_DEMO_FAIL no-observer $OBS"; exit 1; }
HTTP_PORT=5596; FK_PORT=5560; COND_PORT=5604; AGENT_PORT=8032
WORK=$HOME/demo_stack_ur10e; mkdir -p "$WORK"
LOG=$HOME/demo_stack_ur10e.log; : > "$LOG"; exec >>"$LOG" 2>&1
log(){ echo "=== [demo] $* $(date -u +%FT%TZ) ==="; }
log "START"

log "STEP1 FK + bestofn (handoff + telemetry)"
STEER_SERVO=0 STEER_DIAG=0 STEER_HANDOFF=1 \
STEER_HANDOFF_ZONE=0.30 STEER_HANDOFF_DESCEND=0.05 STEER_HANDOFF_GRASPTOL=0.015 \
STEER_HANDOFF_MOVECAP=0.8 STEER_HANDOFF_XY=observer STEER_HANDOFF_GRASPZ=0.21 \
STEER_HANDOFF_SMOOTH=15 STEER_HANDOFF_HOLD=2 STEER_HANDOFF_VERIFY=4 \
STEER_HANDOFF_ATTEMPTS=8 STEER_HANDOFF_NEARWIN=0 STEER_HANDOFF_COVMAX=999 \
STEER_HANDOFF_LOG=1 FK_XML_OVERRIDE=$XML \
bash "$BC/vm_deploy_bestofn.sh" "$CKPT" "$HTTP_PORT" "$FK_PORT" "" || { echo DEMO_FAIL_bestofn; exit 1; }

log "STEP2 conditioner (Obs-C2)"
pkill -f "serve_observer_conditioning.py --port $COND_PORT" 2>/dev/null; sleep 2
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OBSERVER_WEIGHTS="$OBS" \
  HANDOFF_CFG=1 GRASP_OK_HYST=2 \
  CONDITION_BRIDGE_URL="http://127.0.0.1:$HTTP_PORT" OBSERVER_DEVICE=cuda PYTHONPATH="$ODY/src" \
  STEERING_FK_XML="$XML" STEER_BESTOFN_K=1 STEER_CLF_CENTERING=off STEER_CEM_ROUNDS=0 \
  setsid "$GROOT_PY" "$ODY/scripts/serve_observer_conditioning.py" --port "$COND_PORT" \
  >"$WORK/conditioner.log" 2>&1 &
echo $! > "$WORK/cond_pid.txt"
for i in $(seq 1 70); do curl -s --max-time 5 "http://127.0.0.1:$COND_PORT/health" 2>/dev/null | grep -q "\"observer_ready\": *true" && break; sleep 3; done
curl -s --max-time 5 "http://127.0.0.1:$COND_PORT/health" | grep -q "\"observer_ready\": *true" || { echo DEMO_FAIL_conditioner; exit 1; }

log "STEP3 agent service"
pkill -f "uvicorn app.main:app --host 127.0.0.1 --port $AGENT_PORT" 2>/dev/null; sleep 2
cd "$AGENT_DIR"
env -u PYTHONPATH ENVIRONMENT=development DATABASE_URL="sqlite+aiosqlite:///$WORK/agent.db" \
  GROOT_BRIDGE_URL="http://127.0.0.1:$COND_PORT" GROOT_STATE_CONDITIONER_URL="http://127.0.0.1:$COND_PORT" \
  GROOT_OBSERVER_URL="http://127.0.0.1:$COND_PORT" \
  setsid "$AGENT_PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$AGENT_PORT" --loop asyncio \
  >"$WORK/agent.log" 2>&1 &
echo $! > "$WORK/agent_pid.txt"
for i in $(seq 1 40); do curl -s -o /dev/null -w "%{http_code}" --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting&arm=ur10e" 2>/dev/null | grep -q 200 && break; sleep 3; done
curl -s --max-time 8 "http://127.0.0.1:$AGENT_PORT/api/groot/health" | grep -q "\"ok\": *true" || { echo DEMO_FAIL_agent; exit 1; }
log "READY — playground :$AGENT_PORT | brain :$HTTP_PORT/brain/state"
echo "UR10E_DEMO_STACK_READY"
