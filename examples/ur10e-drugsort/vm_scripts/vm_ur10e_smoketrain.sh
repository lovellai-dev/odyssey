#!/usr/bin/env bash
# UR10e PILOT v0: partial dataset (episodes captured so far) through the FULL
# quality chain (success filter -> 10-D conditioning -> photometric aug -> 2x
# merge, gated at >=15k frames) -> 8k-step GR00T finetune -> v0 observer -> n=2
# mini-eval -> deploy the UR10e demo stack serving THIS checkpoint so inference
# runs live on the playground. Runs ALONGSIDE the 160-ep capture (hardlink
# snapshot of completed episodes only; only the bestofn pair is borrowed).
# Markers: "SMK_STEP <n> <name>", "SMK_DATASET_FRAMES", "SMK_TRAIN_DONE",
# "SMK_EVAL", "UR10E_V0_DEMO_READY", "UR10E_SMOKETRAIN_DONE"/"_FAIL <why>".
set -uo pipefail
BC=$HOME/odyssey-ur5e/examples/ur5e-drugsort/browser_capture
UR10E_XML=$HOME/lai-agent-multiagent/src/embodiments/urdf/aseptipack_ur10e_description/aseptipack.xml
GROOT_PY=$HOME/Isaac-GR00T/.venv/bin/python
RAW=$HOME/ur10e_partial_raw
DS=$HOME/ur10e_partial_full
DSC=$HOME/ur10e_partial_cond
DSA=$HOME/ur10e_partial_cond_augonly
DSB=$HOME/ur10e_partial_cond_augonly_b
DST=$HOME/ur10e_partial_cond_aug_tmp
DSM=$HOME/ur10e_partial_cond_aug
CKPT=$HOME/ckpt/ur10e_smoketrain
OBS=$HOME/ur10e_percep_weights_v0
LOG=$HOME/ur10e_smoketrain.log
: > "$LOG"; exec >>"$LOG" 2>&1
log(){ echo "=== [smk] $* $(date -u +%FT%TZ) ==="; }
fail(){ echo "UR10E_SMOKETRAIN_FAIL $*"; exit 1; }
log START

echo "SMK_STEP 1 snapshot completed episodes"
rm -rf "$RAW" "$DS" "$DSC" "$DSA" "$DSB" "$DST" "$DSM"
mkdir -p "$RAW"
N=0
for d in "$HOME"/ur10e_capture_out/raw/ep*/; do
  if [ -f "$d/meta.json" ]; then cp -al "$d" "$RAW/"; N=$((N+1)); fi
done
log "snapshot $N completed episodes"
[ "$N" -ge 10 ] || fail "too-few-episodes ($N)"

echo "SMK_STEP 2 assemble + quality filter"
cd "$BC"
env -u PYTHONPATH "$GROOT_PY" assemble_lerobot.py --raw "$RAW" --out "$DS" || fail assemble
sed -i 's/"robot_type": "ur5e_robotiq_2f85"/"robot_type": "ur10e_robotiq_2f85"/' "$DS/meta/info.json"
log "assembled $(ls "$DS/data/chunk-000" | wc -l) episodes (success-only)"

echo "SMK_STEP 3 conditioned 10-D twin"
env -u PYTHONPATH "$GROOT_PY" augment_state_grasp_target.py \
  --src "$DS" --out "$DSC" --xml "$UR10E_XML" || fail cond

echo "SMK_STEP 4 photometric augmentation x2 seeds -> ~50k-sample aggregate"
env -u PYTHONPATH "$GROOT_PY" "$HOME/dagger_augment_dataset.py" \
  --src "$DSC" --out "$DSA" --seed 77 || fail augment-a
env -u PYTHONPATH "$GROOT_PY" "$HOME/dagger_augment_dataset.py" \
  --src "$DSC" --out "$DSB" --seed 101 --limit 12 || fail augment-b
env -u PYTHONPATH "$GROOT_PY" dagger_relabel_assemble.py merge \
  --base "$DSC" --add "$DSA" --out "$DST" || fail merge-a
env -u PYTHONPATH "$GROOT_PY" dagger_relabel_assemble.py merge \
  --base "$DST" --add "$DSB" --out "$DSM" || fail merge-b
rm -rf "$DST"
FRAMES=$("$GROOT_PY" -c "import json; print(json.load(open('$DSM/meta/info.json'))['total_frames'])")
echo "SMK_DATASET_FRAMES total=$FRAMES eps=$(ls "$DSM/data/chunk-000" | wc -l) (target ~50000)"
[ "$FRAMES" -ge 45000 ] || fail "dataset-below-45k-frames ($FRAMES)"

echo "SMK_STEP 5 gpu gate (borrow serving; capture + playground unaffected)"
tmux kill-session -t groot_bestofn_svc 2>/dev/null
tmux kill-session -t groot_bestofn_fk 2>/dev/null
pkill -f "serve_observer_conditioning.py" 2>/dev/null
FREE=0
for i in $(seq 1 60); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$FREE" -gt 45000 ] && break; sleep 10
done
[ "$FREE" -gt 45000 ] || fail gpu-gate

echo "SMK_STEP 6 observer v0 training"
env -u PYTHONPATH "$GROOT_PY" perception_from_browser.py \
  --raw "$RAW" --out "$OBS" --xml "$UR10E_XML" || fail observer

echo "SMK_STEP 7 GR00T finetune (8k steps, ~2.6h)"
MODALITY_CONFIG=examples/UR10e_DrugSort/ur10e_config_obscond.py \
EXTRA_ARGS="--color-jitter-params brightness 0.3 contrast 0.3 saturation 0.3 hue 0.08" \
  bash "$HOME/vm_train_dagger.sh" "$DSM" "$CKPT" 8000 2000 16 || true
grep -q "DAGGER_TRAIN_RC=0" "$HOME/train_dagger.log" || fail train
echo "SMK_TRAIN_DONE $CKPT/checkpoint-8000"

echo "SMK_STEP 8 mini-eval n=2 (pilot drives the UR10e)"
pkill -f "uvicorn app.main:app --host 127.0.0.1 --port 8032" 2>/dev/null; sleep 3
env MODE=bare N=2 MAX_TICKS=1000 ARM=ur10e \
  CKPT_OVERRIDE="$CKPT/checkpoint-8000" \
  OBS_WEIGHTS_OVERRIDE="$OBS" \
  PLANS_OVERRIDE="$HOME/plans_ur10e_pow7777.json" \
  EVAL_XML="$UR10E_XML" FK_XML_OVERRIDE="$UR10E_XML" \
  STEER_HANDOFF=1 STEER_HANDOFF_ZONE=0.3 STEER_HANDOFF_MOVECAP=0.8 STEER_HANDOFF_XY=obs \
  STEER_HANDOFF_GRASPZ=0.21 STEER_HANDOFF_VERIFY=4 STEER_HANDOFF_ATTEMPTS=6 \
  STEER_PLACE_Z_HI=0.3 STEER_PLACE_TOL=0.03 STEER_HANDOFF_LOG=1 \
  STEER_HANDOFF_CARRY=pilot \
  TAG=ur10e_smoke3k bash "$HOME/run_vm_eval2.sh" || true
S=$(grep -o "EVAL_SUMMARY.*" "$HOME/vm_eval_ur10e_smoke3k.log" 2>/dev/null | tail -1)
echo "SMK_EVAL ${S:-no-summary}"
grep -E "^ATT " "$HOME/vm_eval_ur10e_smoke3k/eval.log" 2>/dev/null | tail -4

echo "SMK_STEP 9 deploy UR10e v0 demo stack (live playground inference) + capture health check"
if UR10E_CKPT="$CKPT/checkpoint-8000" UR10E_OBS="$OBS" bash "$HOME/run_demo_stack_ur10e.sh"; then
  echo "UR10E_V0_DEMO_READY http://localhost:8032/robot-playground.html?demo=drugsorting&arm=ur10e"
else
  log "UR10e v0 demo stack failed — restoring UR5e demo stack"
  bash "$HOME/run_demo_stack.sh" || log "demo restore failed (non-fatal)"
  echo "UR10E_V0_DEMO_FALLBACK_UR5E"
fi
CAP1=$(find "$HOME/ur10e_capture_out/raw" -name '*.png' | wc -l); sleep 60
CAP2=$(find "$HOME/ur10e_capture_out/raw" -name '*.png' | wc -l)
log "capture frames: $CAP1 -> $CAP2 (must be advancing)"
[ "$CAP2" -gt "$CAP1" ] || echo "SMK_WARN capture-not-advancing"
echo "UR10E_SMOKETRAIN_DONE"
