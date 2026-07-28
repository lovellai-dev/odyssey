#!/usr/bin/env bash
# UR10e drug-sort: FULL data-collection + training pipeline (one shot, tmux).
# Mirrors the UR5e champion path + the augmentation stage:
#   plans (DR train set + frozen eval folds) -> 160-ep browser capture (deployment
#   renderer) -> LeRobot assembly (success-only) -> observer-conditioned 10-D twin
#   -> temporally-coherent photometric augmentation -> merge -> GPU gate ->
#   GR00T-N1.7-3B 12k finetune (obscond config + online color jitter) ->
#   observer training -> demo-stack restore.
# Markers: "UR10E_STEP <n> <name>", "UR10E_PIPELINE_DONE", "UR10E_PIPELINE_FAIL <why>".
set -uo pipefail
VMHOME=$HOME
BC=$HOME/odyssey-ur5e/examples/ur5e-drugsort/browser_capture
UR10E_XML=$HOME/lai-agent-multiagent/src/embodiments/urdf/aseptipack_ur10e_description/aseptipack.xml
GROOT_PY=$HOME/Isaac-GR00T/.venv/bin/python
NODE=$HOME/.nvm/versions/node/v22.23.1/bin/node
PUP=$HOME/vm_eval/node_modules/puppeteer-core
CHROME=/usr/bin/google-chrome-stable
RAWOUT=$HOME/ur10e_capture_out
DS_FULL=$HOME/ur10e_dataset_browser_full
DS_COND=$HOME/ur10e_dataset_browser_cond
DS_AUGO=$HOME/ur10e_dataset_browser_cond_augonly
DS_AUG=$HOME/ur10e_dataset_browser_cond_aug
CKPT_OUT=$HOME/ckpt/ur10e_drugsort_condaug
OBS_OUT=$HOME/ur10e_percep_weights
N_TRAIN=160
LOG=$HOME/ur10e_pipeline.log
: > "$LOG"; exec >>"$LOG" 2>&1
log(){ echo "=== [ur10e] $* $(date -u +%FT%TZ) ==="; }
fail(){ echo "UR10E_PIPELINE_FAIL $*"; exit 1; }

log "START"
[ -e "$RAWOUT" ] && fail "raw-out-exists ($RAWOUT) — refusing to clobber"
[ -e "$DS_FULL" ] && fail "dataset-exists ($DS_FULL) — refusing to clobber"

echo "UR10E_STEP 0 plans"
cd "$BC" || fail cd-bc
env -u PYTHONPATH ASEPTIPACK_XML="$UR10E_XML" "$GROOT_PY" precompute_plans.py \
  --num-episodes $N_TRAIN --seed 1000 --randomize --out "$HOME/plans_ur10e_train.json" \
  || fail plans-train
for SEED in 7777 8383; do
  env -u PYTHONPATH ASEPTIPACK_XML="$UR10E_XML" "$GROOT_PY" precompute_plans.py \
    --num-episodes 15 --seed $SEED --out "$HOME/plans_ur10e_pow$SEED.json" \
    || fail plans-fold-$SEED
done
log "plans done"

echo "UR10E_STEP 1 capture ($N_TRAIN eps, ~11h)"
curl -s -o /dev/null -w '%{http_code}' -m8 "http://127.0.0.1:8032/robot-playground.html?demo=drugsorting&arm=ur10e" | grep -q 200 || {
  log "playground down — restoring demo stack for the static server"
  bash "$HOME/run_demo_stack.sh" || fail demo-stack-for-capture
}
cd "$BC"
setsid env ARM=ur10e PLANS="$HOME/plans_ur10e_train.json" OUT="$RAWOUT" PORT=8032 \
  PUPPETEER_CORE="$PUP" CHROME="$CHROME" "$NODE" browser_harness.js \
  > "$HOME/ur10e_capture.log" 2>&1 &
HPID=$!
# stall watchdog: no new frames for 20 min while the harness lives -> kill + fail
LASTN=0; STALL=0
while kill -0 "$HPID" 2>/dev/null; do
  sleep 120
  N=$(find "$RAWOUT/raw" -name '*.png' 2>/dev/null | wc -l)
  if [ "$N" -gt "$LASTN" ]; then LASTN=$N; STALL=0; else STALL=$((STALL+1)); fi
  if [ "$STALL" -ge 10 ]; then
    log "capture stalled at $N frames"
    pkill -g "$(ps -o pgid= -p $HPID | tr -d ' ')" 2>/dev/null
    echo "CAPTURE_STALL frames=$N"
    fail capture-stall
  fi
done
wait "$HPID"; HRC=$?
NSUC=$(grep -c "SUCCESS" "$HOME/ur10e_capture.log" || true)
log "capture exit rc=$HRC successes=$NSUC"
grep -q "HARNESS_DONE" "$HOME/ur10e_capture.log" || fail capture-incomplete
[ "$NSUC" -ge 100 ] || fail "capture-too-few-successes ($NSUC)"

echo "UR10E_STEP 2 assemble"
env -u PYTHONPATH "$GROOT_PY" assemble_lerobot.py --raw "$RAWOUT/raw" --out "$DS_FULL" \
  || fail assemble
log "assembled $(ls "$DS_FULL/data/chunk-000" 2>/dev/null | wc -l) episodes"

echo "UR10E_STEP 3 conditioned twin"
env -u PYTHONPATH "$GROOT_PY" augment_state_grasp_target.py \
  --src "$DS_FULL" --out "$DS_COND" --xml "$UR10E_XML" || fail cond

echo "UR10E_STEP 4 photometric augmentation"
env -u PYTHONPATH "$GROOT_PY" "$HOME/dagger_augment_dataset.py" \
  --src "$DS_COND" --out "$DS_AUGO" --seed 77 || fail augment

echo "UR10E_STEP 5 merge"
env -u PYTHONPATH "$GROOT_PY" dagger_relabel_assemble.py merge \
  --base "$DS_COND" --add "$DS_AUGO" --out "$DS_AUG" || fail merge

echo "UR10E_STEP 6 gpu gate + finetune (12k, ~4h)"
tmux kill-session -t groot_bestofn_svc 2>/dev/null
tmux kill-session -t groot_bestofn_fk 2>/dev/null
tmux kill-session -t groot_dagger_server 2>/dev/null
tmux kill-session -t groot_dagger_bridge 2>/dev/null
pkill -f "serve_observer_conditioning.py" 2>/dev/null
for i in $(seq 1 60); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$FREE" -gt 45000 ] && break; sleep 10
done
[ "$FREE" -gt 45000 ] || fail gpu-gate
MODALITY_CONFIG=examples/UR10e_DrugSort/ur10e_config_obscond.py \
EXTRA_ARGS="--color-jitter-params brightness 0.3 contrast 0.3 saturation 0.3 hue 0.08" \
  bash "$HOME/vm_train_dagger.sh" "$DS_AUG" "$CKPT_OUT" 12000 3000 16 || true
grep -q "DAGGER_TRAIN_RC=0" "$HOME/train_dagger.log" || fail train
echo "UR10E_TRAIN_DONE $CKPT_OUT"

echo "UR10E_STEP 7 observer training"
env -u PYTHONPATH "$GROOT_PY" perception_from_browser.py \
  --raw "$RAWOUT/raw" --out "$OBS_OUT" --xml "$UR10E_XML" || fail observer
echo "UR10E_OBSERVER_DONE $OBS_OUT"

echo "UR10E_STEP 8 demo stack restore"
bash "$HOME/run_demo_stack.sh" || log "demo stack restore failed (non-fatal)"

echo "UR10E_PIPELINE_DONE ckpt=$CKPT_OUT observer=$OBS_OUT folds=plans_ur10e_pow7777/8383"
