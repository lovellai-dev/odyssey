#!/usr/bin/env bash
# Runs ON the H100 VM. GR00T-N1.7-3B fine-tune on the DAgger-AGGREGATED browser
# (Three.js) dataset — one DAgger-on-browser iteration. Classic DAgger: retrain on
# the growing aggregate D <- D u {relabelled visited states}. ISOLATED from v4 AND
# from the render-gap train: OWN tmux session (groot_dagger_train), OWN log
# (train_dagger.log), OWN marker (DAGGER_TRAIN_RC). Does NOT touch v4's
# ckpt/:5555/:5599 or the render-gap :5556/:5598 deploy. Same proven recipe
# (from base, LR 1e-4, batch 16) so only the training DATA differs.
set -uo pipefail
DATASET=${1:?dataset dir}
CKPT=${2:?output ckpt dir}
MAX_STEPS=${3:-12000}
SAVE_STEPS=${4:-3000}
BATCH=${5:-16}
BASE_MODEL=${6:-nvidia/GR00T-N1.7-3B}
EXTRA_ARGS=${EXTRA_ARGS:-}
MODALITY_CONFIG=${MODALITY_CONFIG:-examples/UR5e_DrugSort/ur5e_config.py}
GROOT=/home/ubuntu/Isaac-GR00T
GROOT_PY=$GROOT/.venv/bin/python
LOG=/home/ubuntu/train_dagger.log
: > "$LOG"
cd "$GROOT"
echo "=== [vm-train-dagger] START $(date -u +%FT%TZ) dataset=$DATASET ckpt=$CKPT steps=$MAX_STEPS batch=$BATCH base=$BASE_MODEL ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env -u PYTHONPATH "$GROOT_PY" \
  -m gr00t.experiment.launch_finetune \
  --base-model-path "$BASE_MODEL" \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path $MODALITY_CONFIG \
  --output-dir "$CKPT" \
  --num-gpus 1 --global-batch-size "$BATCH" --learning-rate 0.0001 \
  --max-steps "$MAX_STEPS" --save-steps "$SAVE_STEPS" --dataloader-num-workers 8 $EXTRA_ARGS >> "$LOG" 2>&1
RC=$?
echo "=== [vm-train-dagger] DONE rc=$RC $(date -u +%FT%TZ) ===" | tee -a "$LOG"
echo "DAGGER_TRAIN_RC=$RC" >> "$LOG"
