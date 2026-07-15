#!/usr/bin/env bash
# Runs ON the H100 VM. GR00T-N1.7-3B fine-tune on the OBSERVER-CONDITIONED browser
# dataset (state 7 -> 10: proprio + 3D vial-cap grasp target) — roadmap Step 2.
# ISOLATED from v4 / render-gap / dagger: own output dir + own tmux + own log +
# own 10-D modality config; does NOT touch their ckpts/datasets/ports. Same proven
# recipe (from base, LR 1e-4, batch 16) so only the STATE (target channel) differs.
# Writes OBSCOND_TRAIN_RC=<n> to the log so the orchestrator can detect completion.
set -uo pipefail
DATASET=${1:?dataset dir}
CKPT=${2:?ckpt dir}
MAX_STEPS=${3:-12000}
SAVE_STEPS=${4:-3000}
BATCH=${5:-16}
MODCFG=${6:-examples/UR5e_DrugSort/ur5e_config_obscond.py}   # 10-D state modality config
GROOT=/home/ubuntu/Isaac-GR00T
GROOT_PY=$GROOT/.venv/bin/python
LOG=/home/ubuntu/train_obscond.log
: > "$LOG"
cd "$GROOT"
echo "=== [vm-train-obscond] START $(date -u +%FT%TZ) dataset=$DATASET ckpt=$CKPT steps=$MAX_STEPS batch=$BATCH cfg=$MODCFG ===" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env -u PYTHONPATH "$GROOT_PY" \
  -m gr00t.experiment.launch_finetune \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$DATASET" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODCFG" \
  --output-dir "$CKPT" \
  --num-gpus 1 --global-batch-size "$BATCH" --learning-rate 0.0001 \
  --max-steps "$MAX_STEPS" --save-steps "$SAVE_STEPS" --dataloader-num-workers 8 >> "$LOG" 2>&1
RC=$?
echo "=== [vm-train-obscond] DONE rc=$RC $(date -u +%FT%TZ) ===" | tee -a "$LOG"
echo "OBSCOND_TRAIN_RC=$RC" >> "$LOG"
