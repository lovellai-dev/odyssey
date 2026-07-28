#!/usr/bin/env bash
# UR10e fold evals (7777 then 8383), n=15 each — r3-exact env + hybrid carry,
# UR10e checkpoint/observer/scene. Runs after the UR10e pipeline completes;
# tears down the demo stack first (port/GPU overlap), restores it after.
# Markers: UR10E_EVAL_FOLD <seed> <EVAL_SUMMARY...>, UR10E_EVAL_DONE, UR10E_EVAL_FAIL <why>.
set -uo pipefail
UR10E_XML=$HOME/lai-agent-multiagent/src/embodiments/urdf/aseptipack_ur10e_description/aseptipack.xml
CKPT=$HOME/ckpt/ur10e_drugsort_condaug/checkpoint-12000
OBS=$HOME/ur10e_percep_weights
LOG=$HOME/ur10e_fold_eval.log
: > "$LOG"; exec >>"$LOG" 2>&1
log(){ echo "=== [ur10e-eval] $* $(date -u +%FT%TZ) ==="; }
fail(){ echo "UR10E_EVAL_FAIL $*"; exit 1; }

log "START"
[ -d "$CKPT" ] || fail "no-ckpt ($CKPT)"
[ -d "$OBS" ] || fail "no-observer ($OBS)"
[ -f "$HOME/plans_ur10e_pow7777.json" ] || fail no-fold-plans

log "teardown demo-stack pieces (port/GPU overlap with eval serving)"
pkill -f "uvicorn app.main:app --host 127.0.0.1 --port 8032" 2>/dev/null
pkill -f "serve_observer_conditioning.py" 2>/dev/null
sleep 5

for SEED in 7777 8383; do
  log "fold $SEED start (n=15, ~50 min)"
  env MODE=bare N=15 MAX_TICKS=1000 ARM=ur10e \
    CKPT_OVERRIDE="$CKPT" \
    OBS_WEIGHTS_OVERRIDE="$OBS" \
    PLANS_OVERRIDE="$HOME/plans_ur10e_pow$SEED.json" \
    EVAL_XML="$UR10E_XML" FK_XML_OVERRIDE="$UR10E_XML" \
    STEER_HANDOFF=1 STEER_HANDOFF_ZONE=0.3 STEER_HANDOFF_MOVECAP=0.8 STEER_HANDOFF_XY=obs \
    STEER_HANDOFF_GRASPZ=0.21 STEER_HANDOFF_VERIFY=4 STEER_HANDOFF_ATTEMPTS=6 \
    STEER_PLACE_Z_HI=0.3 STEER_PLACE_TOL=0.03 STEER_HANDOFF_LOG=1 \
    STEER_HANDOFF_CARRY=pilot \
    TAG=ur10e_$SEED bash "$HOME/run_vm_eval2.sh" || true
  S=$(grep -o "EVAL_SUMMARY.*" "$HOME/vm_eval_ur10e_$SEED.log" 2>/dev/null | tail -1)
  [ -n "$S" ] || { grep -o "VM_EVAL FAILED.*" "$HOME/vm_eval_ur10e_$SEED.log" 2>/dev/null | tail -1; fail "fold-$SEED-no-summary"; }
  echo "UR10E_EVAL_FOLD $SEED $S"
done

log "bring up the UR10e demo stack (playground + brain telemetry)"
if bash "$HOME/run_demo_stack_ur10e.sh"; then
  echo "UR10E_DEMO_READY http://127.0.0.1:8032/robot-playground.html?demo=drugsorting&arm=ur10e (brain :5596/brain/state)"
else
  log "UR10e demo stack failed — restoring the UR5e demo stack instead"
  bash "$HOME/run_demo_stack.sh" || log "UR5e demo restore failed too (non-fatal)"
  echo "UR10E_DEMO_FALLBACK_UR5E"
fi
echo "UR10E_EVAL_DONE"
