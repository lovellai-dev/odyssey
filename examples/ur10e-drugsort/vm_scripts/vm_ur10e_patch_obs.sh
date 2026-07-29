#!/usr/bin/env bash
# UR10e patch-token observer retrain chain — replicates the UR5e 2.19cm lever.
# Waits for the pilot-only eval to release the serving stack, then:
#   STEP1 capture plans (fresh seed 5150 — frozen eval folds stay unleaked)
#   STEP2 RAW_DUMP policy-frame capture (hybrid serving, current v1 observer)
#   STEP3 label policy frames via UR10e-cell FK (perception_from_browser)
#   STEP4 merge expert (11.2k frames) + policy frames -> ur10e_percep_mixed
#   STEP5 frame-head retrain on mixed, val = policy episodes
#   STEP6 uncertainty head (CERTLOCK input; non-fatal if it fails)
#   STEP7 patch-token head (train_patch_observer_ur10e.py)
#   STEP8 fold evals 7777/8383 with the new observer (TAG=ur10e_<seed>_patchobs)
#   STEP9 UR10e demo redeploy serving the patch observer
set -uo pipefail
ODY=$HOME/odyssey-ur5e; BC=$ODY/examples/ur5e-drugsort/browser_capture
EVAL_PY=$HOME/odyssey-eval-venv/bin/python
GROOT_PY=$HOME/Isaac-GR00T/.venv/bin/python
UR10E_XML=$HOME/lai-agent-multiagent/src/embodiments/urdf/aseptipack_ur10e_description/aseptipack.xml
CKPT=$HOME/ckpt/ur10e_drugsort_condaug/checkpoint-12000
EXP=$HOME/ur10e_percep_weights            # expert arrays + v1 heads
POL=$HOME/ur10e_percep_policy             # labeled policy frames
MIX=$HOME/ur10e_percep_mixed              # merged observer training set
OBS_OUT=$HOME/ur10e_percep_weights_patch  # final observer weights dir
LOG=$HOME/ur10e_patch_obs.log; : > "$LOG"; exec >>"$LOG" 2>&1
log(){ echo "=== [patch-obs] $* $(date -u +%FT%TZ) ==="; }
fail(){ log "FAILED: $*"; echo "PATCHOBS FAILED: $*"; exit 1; }
log "START pid=$$"

log "STEP0 wait for pilot-only eval to release the serving stack"
while tmux has-session -t ur10e_pilotonly 2>/dev/null; do sleep 60; done
log "pilot-only finished"

log "STEP1 capture plans seed 5150"
cd "$BC" || fail cd-bc
if [ ! -f "$HOME/plans_ur10e_pow5150.json" ]; then
  env -u PYTHONPATH ASEPTIPACK_XML="$UR10E_XML" "$GROOT_PY" precompute_plans.py \
    --num-episodes 15 --seed 5150 --out "$HOME/plans_ur10e_pow5150.json" || fail plans
fi

log "STEP2 RAW_DUMP policy capture (n=15, hybrid, v1 observer)"
pkill -f "uvicorn app.main:app --host 127.0.0.1 --port 8032" 2>/dev/null
pkill -f "serve_observer_conditioning.py" 2>/dev/null
sleep 5
env MODE=bare N=15 MAX_TICKS=1000 ARM=ur10e RAW_DUMP=1 \
  CKPT_OVERRIDE="$CKPT" OBS_WEIGHTS_OVERRIDE="$EXP" \
  PLANS_OVERRIDE="$HOME/plans_ur10e_pow5150.json" \
  EVAL_XML="$UR10E_XML" FK_XML_OVERRIDE="$UR10E_XML" \
  STEER_HANDOFF=1 STEER_HANDOFF_ZONE=0.3 STEER_HANDOFF_MOVECAP=0.8 STEER_HANDOFF_XY=obs \
  STEER_HANDOFF_GRASPZ=0.21 STEER_HANDOFF_VERIFY=4 STEER_HANDOFF_ATTEMPTS=6 \
  STEER_PLACE_Z_HI=0.3 STEER_PLACE_TOL=0.03 STEER_HANDOFF_LOG=1 \
  STEER_HANDOFF_CARRY=pilot \
  TAG=ur10e_policycap bash "$HOME/run_vm_eval2.sh" || fail capture
[ -d "$HOME/vm_eval_ur10e_policycap/out/raw" ] || fail no-raw-dump

log "STEP3 label policy frames (UR10e-cell FK)"
rm -rf "$POL"
env -u PYTHONPATH MUJOCO_GL=egl "$EVAL_PY" "$BC/perception_from_browser.py" \
  --raw "$HOME/vm_eval_ur10e_policycap/out/raw" --out "$POL" --xml "$UR10E_XML" \
  --keep-every 1 --max-frames-per-ep 150 --max-total-frames 20000 || fail label
NP=$("$EVAL_PY" -c "import numpy as np;print(len(np.load('$POL/episode.npy')))" 2>/dev/null || echo 0)
log "labeled policy frames: $NP"
[ "$NP" -ge 200 ] || fail "too-few-policy($NP)"

log "STEP4 merge expert + policy -> $MIX"
rm -rf "$MIX"; mkdir -p "$MIX"
"$EVAL_PY" - <<PY || fail merge
import numpy as np
exp="$EXP"; pol="$POL"; out="$MIX"
for k in ["ext","wrist","grasp_target_base","labels","episode","success"]:
    a=np.load(f"{exp}/{k}.npy"); b=np.load(f"{pol}/{k}.npy")
    if k=="episode": b=b+1000
    np.save(f"{out}/{k}.npy", np.concatenate([a,b],axis=0))
print("merged n=", len(np.load(f"{out}/episode.npy")))
PY
echo "1001,1006,1011" > "$MIX/val_episodes.txt"

log "STEP5 frame head retrain on mixed (val = policy eps 1001,1006,1011)"
rm -rf "$OBS_OUT"; mkdir -p "$OBS_OUT"
cd "$ODY" || fail cd-ody
env -u PYTHONPATH "$GROOT_PY" scripts/train_ur5e_perception.py \
  --data "$MIX" --out "$OBS_OUT" --device cuda \
  --skip-classifier --obs-epochs 100 --val-episodes 1001,1006,1011 || fail frame-head
[ -f "$OBS_OUT/observer_head.pt" ] || fail frame-head-missing
cp -f "$EXP/success_cnn.pt" "$OBS_OUT/" || fail success-cnn-copy

log "STEP6 uncertainty head (CERTLOCK)"
env -u PYTHONPATH UNC_DATA="$MIX" UNC_WEIGHTS="$OBS_OUT" \
  "$GROOT_PY" "$HOME/train_obs_uncertainty.py" \
  || log "uncertainty head failed (non-fatal; serving auto-detects its absence)"

log "STEP7 patch-token head"
env -u PYTHONPATH "$GROOT_PY" "$HOME/train_patch_observer_ur10e.py" || fail patch-head
[ -f "$OBS_OUT/patch_head.pt" ] || fail patch-head-missing

log "STEP8 fold evals with the patch observer"
for SEED in 7777 8383; do
  log "fold $SEED (patch observer, n=15)"
  env MODE=bare N=15 MAX_TICKS=1000 ARM=ur10e \
    CKPT_OVERRIDE="$CKPT" OBS_WEIGHTS_OVERRIDE="$OBS_OUT" \
    PLANS_OVERRIDE="$HOME/plans_ur10e_pow$SEED.json" \
    EVAL_XML="$UR10E_XML" FK_XML_OVERRIDE="$UR10E_XML" \
    STEER_HANDOFF=1 STEER_HANDOFF_ZONE=0.3 STEER_HANDOFF_MOVECAP=0.8 STEER_HANDOFF_XY=obs \
    STEER_HANDOFF_GRASPZ=0.21 STEER_HANDOFF_VERIFY=4 STEER_HANDOFF_ATTEMPTS=6 \
    STEER_PLACE_Z_HI=0.3 STEER_PLACE_TOL=0.03 STEER_HANDOFF_LOG=1 \
    STEER_HANDOFF_CARRY=pilot \
    TAG=ur10e_${SEED}_patchobs bash "$HOME/run_vm_eval2.sh" || true
  S=$(grep -o "EVAL_SUMMARY.*" "$HOME/vm_eval_ur10e_${SEED}_patchobs.log" 2>/dev/null | tail -1)
  [ -n "$S" ] || { grep -o "VM_EVAL FAILED.*" "$HOME/vm_eval_ur10e_${SEED}_patchobs.log" 2>/dev/null | tail -1; fail "fold-$SEED-no-summary"; }
  echo "PATCHOBS_FOLD $SEED $S"
done

log "STEP9 UR10e demo redeploy (patch observer)"
if UR10E_OBS="$OBS_OUT" bash "$HOME/run_demo_stack_ur10e.sh"; then
  echo "PATCHOBS_DEMO_READY http://127.0.0.1:8032/robot-playground.html?demo=drugsorting&arm=ur10e"
else
  echo "PATCHOBS_DEMO_FAILED"
fi
echo "PATCHOBS_DONE"
