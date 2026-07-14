#!/usr/bin/env bash
# Runs ON the H100 VM. GR00T-N1.7-3B fine-tune on the BROWSER (Three.js) dataset —
# the render-gap close. ISOLATED from v4: own output dir + its own tmux; does NOT
# touch v4's ckpt/dataset/:5555/:5599. Same proven v3/dagger recipe (from base, LR
# 1e-4, batch 16) so only the training PIXELS differ. Writes BROWSER_TRAIN_RC=<n>
# to the log so the orchestrator can detect completion.
set -uo pipefail
DATASET=${1:-/home/ubuntu/ur5e_drugsort_browser}
CKPT=${2:-/home/ubuntu/ckpt/ur5e_drugsort_browser}
MAX_STEPS=${3:-12000}
SAVE_STEPS=${4:-3000}
BATCH=${5:-16}
GROOT=/home/ubuntu/Isaac-GR00T
GROOT_PY=$GROOT/.venv/bin/python
LOG=/home/ubuntu/train_browser.log
: > "$LOG"
cd "$GROOT"
echo "=== [vm-train] START $(date -u +%FT%TZ) dataset=$DATASET ckpt=$CKPT steps=$MAX_STEPS batch=$BATCH ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env -u PYTHONPATH "$GROOT_PY" \
  -m gr00t.experiment.launch_finetune \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UR5e_DrugSort/ur5e_config.py \
  --output-dir "$CKPT" \
  --num-gpus 1 --global-batch-size "$BATCH" --learning-rate 0.0001 \
  --max-steps "$MAX_STEPS" --save-steps "$SAVE_STEPS" --dataloader-num-workers 8 >> "$LOG" 2>&1
RC=$?
echo "=== [vm-train] DONE rc=$RC $(date -u +%FT%TZ) ===" | tee -a "$LOG"
echo "BROWSER_TRAIN_RC=$RC" >> "$LOG"
