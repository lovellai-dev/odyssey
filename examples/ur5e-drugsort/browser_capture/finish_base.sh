#!/usr/bin/env bash
# ============================================================================
# finish_base.sh — the autonomous loop's L0 driver: FINISH the interrupted
# base-fix assembly and SFT a FRESH competent base checkpoint.
#
# The expensive part (5h of browser rollout generation) is DONE and preserved
# in ~/base_fix/out_dr/raw_all (~83 episodes). This driver resumes from there:
#   1. assemble LeRobot (SUCCESS-FILTERED) from raw_all           -> dataset_dr
#   2. augment observation.state 7 -> 10 (Observer grasp target)  -> dataset_dr_cond
#   3. rsync -> VM:/home/ubuntu/ur5e_drugsort_dr
#   4. GPU-gate (>45GB free); SFT a FRESH ckpt ur5e_drugsort_dr (15000 steps)
#      via vm_train_basefix.sh (tmux groot_basefix_train). Does NOT overwrite the
#      obscond ckpt — the v4 image-conditioned head depends on it.
#
# IDEMPOTENT: every sub-step skips if its output already exists (assembled
# dataset, augmented dataset, VM checkpoint-15000, or an in-flight training
# tmux is adopted rather than relaunched) — a killed/resumed run never redoes
# finished work. ISOLATION: own VM tmux (groot_basefix_train) only; the
# protected sessions (ccproxy, gateway, groot_browser_*) and every other
# ckpt/port are untouched. NEVER pkill chrome (this driver spawns no browser).
#
# Markers: FINISH_BASE DONE | FINISH_BASE FAILED: <stage>  in ~/finish_base.log.
# Result:  ~/finish_base_result.json  {"checkpoint": "...", "status": "DONE"}.
# ============================================================================
set -uo pipefail

# ---- paths / config ---------------------------------------------------------
BC=/home/daniel/LovellAI/odyssey-ur5e/examples/ur5e-drugsort/browser_capture
ODY=/home/daniel/LovellAI/odyssey-ur5e
PYUR5E=$ODY/.venv-ur5e/bin/python
VM=ubuntu@192.222.52.169
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
XML=/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/aseptipack_description/aseptipack.xml

WORK=$HOME/base_fix
RAW_ALL=$WORK/out_dr/raw_all
LOG=$HOME/finish_base.log
RESULT=$HOME/finish_base_result.json

VM_DATASET=/home/ubuntu/ur5e_drugsort_dr
VM_CKPT_ROOT=/home/ubuntu/ckpt/ur5e_drugsort_dr
VM_CFG=examples/UR5e_DrugSort/ur5e_config_obscond.py     # 10-D modality config (already on VM)

MAX_STEPS=${MAX_STEPS:-15000}
SAVE_STEPS=${SAVE_STEPS:-7500}
BATCH=${BATCH:-16}
GPU_FREE_MIN=${GPU_FREE_MIN:-45000}
MIN_EPISODES=${MIN_EPISODES:-40}

export DISPLAY=${DISPLAY:-:1}
cd "$BC"
exec >>"$LOG" 2>&1
echo ""; echo "#################################################################"
log(){ echo "=== [finish_base] $* $(date -u +%FT%TZ) ==="; }

write_result(){   # $1 status ; $2 checkpoint ; $3 stage
  "$PYUR5E" - "$RESULT" "$1" "${2:-}" "${3:-}" <<'PY' || true
import json, sys
result, status, ckpt, stage = sys.argv[1:5]
json.dump({"driver": "finish_base", "status": status,
           "checkpoint": ckpt or None, "stage": stage or None,
           "max_steps": __import__("os").environ.get("MAX_STEPS", "15000")},
          open(result, "w"), indent=2)
PY
}
fail(){
  log "FAILED at: $*"
  write_result FAILED "" "$*"
  echo "FINISH_BASE FAILED: $*"
  exit 1
}
gpu_gate(){
  local i FREE
  log "gate on VM GPU free (> ${GPU_FREE_MIN} MiB)"
  for i in $(seq 1 360); do
    FREE=$($SSH "$VM" "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" 2>/dev/null | head -1 | tr -d ' ')
    { [ -n "${FREE:-}" ] && [ "$FREE" -gt "$GPU_FREE_MIN" ]; } && { log "GPU free=${FREE}MiB — go"; return 0; }
    echo "  [gpu-gate] free=${FREE:-?}MiB waiting"; sleep 60
  done
  return 1
}

# ============================================================================
log "START pid=$$ raw_all=$RAW_ALL steps=$MAX_STEPS save=$SAVE_STEPS batch=$BATCH"
echo "FINISH_BASE PID=$$"
write_result RUNNING "" "start"

# ---- 0. sanity -------------------------------------------------------------
[ -f "$XML" ] || fail "no-xml($XML)"
[ -d "$RAW_ALL" ] || fail "no-raw-all($RAW_ALL)"
NRAW=$(ls -d "$RAW_ALL"/ep*/meta.json 2>/dev/null | wc -l)
log "raw episodes available: $NRAW"
[ "$NRAW" -ge "$MIN_EPISODES" ] || fail "too-few-raw-episodes($NRAW<$MIN_EPISODES)"
$SSH "$VM" "echo ok >/dev/null" || fail "vm-unreachable"
# GUARD: never point training at the obscond ckpt root.
case "$VM_CKPT_ROOT" in *obscond*) fail "refusing-to-write-obscond($VM_CKPT_ROOT)";; esac

# ---- 1. assemble (SUCCESS-FILTERED) ----------------------------------------
if [ -f "$WORK/dataset_dr/meta/info.json" ]; then
  log "STEP1 dataset_dr already assembled — skip"
else
  log "STEP1 assemble LeRobot (success-filtered) from $NRAW raw episodes"
  env -u PYTHONPATH MUJOCO_GL=egl "$PYUR5E" assemble_lerobot.py \
    --raw "$RAW_ALL" --out "$WORK/dataset_dr" 2>&1 | tail -10 || fail "assemble"
  [ -f "$WORK/dataset_dr/meta/info.json" ] || fail "assemble-no-info-json"
fi
FRATE=$("$PYUR5E" -c "import json;d=json.load(open('$WORK/dataset_dr/assemble_summary.json'));print(d['kept'],d['raw_episodes'],round(d['success_filter_rate']*100,1))" 2>/dev/null || echo "? ? ?")
log "STEP1 success-filter (kept raw pct): $FRATE"

# ---- 2. augment -> 10-D grasp_target ---------------------------------------
STATE_DIM=$("$PYUR5E" -c "import json;print(json.load(open('$WORK/dataset_dr_cond/meta/info.json'))['features']['observation.state']['shape'][0])" 2>/dev/null || echo "")
if [ "$STATE_DIM" = "10" ]; then
  log "STEP2 dataset_dr_cond already 10-D — skip"
else
  log "STEP2 augment -> 10-D grasp_target (dataset_dr_cond)"
  env -u PYTHONPATH MUJOCO_GL=egl "$PYUR5E" augment_state_grasp_target.py \
    --src "$WORK/dataset_dr" --out "$WORK/dataset_dr_cond" --xml "$XML" --verify 2>&1 | tail -8 || fail "augment"
  STATE_DIM=$("$PYUR5E" -c "import json;print(json.load(open('$WORK/dataset_dr_cond/meta/info.json'))['features']['observation.state']['shape'][0])" 2>/dev/null)
fi
[ "$STATE_DIM" = "10" ] || fail "augment-bad-state-dim($STATE_DIM)"
log "STEP2 dataset_dr_cond state dim = $STATE_DIM (OK)"

# ---- 3. rsync -> VM; ensure 10-D config + train wrapper are current --------
log "STEP3 rsync dataset_dr_cond -> VM:$VM_DATASET"
$SSH "$VM" "mkdir -p $VM_DATASET $VM_CKPT_ROOT" >/dev/null 2>&1 || true
rsync -a --delete -e "$SSH" "$WORK/dataset_dr_cond/" "$VM:$VM_DATASET/" || fail "rsync-dataset"
$SSH "$VM" "[ -f /home/ubuntu/Isaac-GR00T/$VM_CFG ]" || fail "no-vm-10d-config"
scp -q -o StrictHostKeyChecking=no "$ODY/examples/ur5e-drugsort/ur5e_config.py" "$VM:/home/ubuntu/Isaac-GR00T/$VM_CFG" || fail "scp-config"
scp -q -o StrictHostKeyChecking=no vm_train_basefix.sh "$VM:/home/ubuntu/" || fail "scp-train-wrapper"

# ---- 4. SFT a FRESH dr checkpoint (idempotent) -----------------------------
FULL_CKPT=$VM_CKPT_ROOT/checkpoint-$MAX_STEPS
if $SSH "$VM" "[ -d $FULL_CKPT ]"; then
  log "STEP4 checkpoint already present ($FULL_CKPT) — skip SFT"
elif $SSH "$VM" "tmux has-session -t groot_basefix_train 2>/dev/null"; then
  log "STEP4 an SFT tmux (groot_basefix_train) is already running — adopting it (no relaunch)"
else
  gpu_gate || fail "gpu-gate-timeout"
  log "STEP4 launch SFT (tmux groot_basefix_train; $MAX_STEPS steps, save $SAVE_STEPS, batch $BATCH)"
  $SSH "$VM" "tmux kill-session -t groot_basefix_train 2>/dev/null; tmux new-session -d -s groot_basefix_train \
    'bash /home/ubuntu/vm_train_basefix.sh $VM_DATASET $VM_CKPT_ROOT $MAX_STEPS $SAVE_STEPS $BATCH $VM_CFG'" || fail "launch-train"
  sleep 25
  $SSH "$VM" "tmux has-session -t groot_basefix_train 2>/dev/null" || \
    $SSH "$VM" "grep -q BASEFIX_TRAIN_RC /home/ubuntu/train_basefix.log" 2>/dev/null || fail "train-session-missing"
fi

# wait for the SFT to finish (BASEFIX_TRAIN_RC marker), unless the ckpt is already there
if ! $SSH "$VM" "[ -d $FULL_CKPT ]"; then
  log "STEP4 wait for SFT (BASEFIX_TRAIN_RC marker)"
  while true; do
    $SSH "$VM" "grep -q BASEFIX_TRAIN_RC /home/ubuntu/train_basefix.log" 2>/dev/null && break
    if ! $SSH "$VM" "tmux has-session -t groot_basefix_train 2>/dev/null"; then
      sleep 10
      $SSH "$VM" "grep -q BASEFIX_TRAIN_RC /home/ubuntu/train_basefix.log" 2>/dev/null && break || fail "train-died-no-rc"
    fi
    STEP=$($SSH "$VM" "grep -oE \"'?loss'?: *[0-9.]+|[0-9]+/$MAX_STEPS\" /home/ubuntu/train_basefix.log 2>/dev/null | tail -1" 2>/dev/null || true)
    echo "  [train-wait] $STEP $(date -u +%FT%TZ)"
    sleep 120
  done
  TRC=$($SSH "$VM" "grep BASEFIX_TRAIN_RC /home/ubuntu/train_basefix.log | tail -1 | sed 's/.*=//'" 2>/dev/null | tr -d ' ')
  log "STEP4 SFT exited rc=$TRC"
  [ "${TRC:-1}" = "0" ] || fail "train-rc=${TRC:-?}"
fi

# resolve the newest checkpoint (checkpoint-$MAX_STEPS, else the latest)
$SSH "$VM" "[ -d $FULL_CKPT ]" || FULL_CKPT=$($SSH "$VM" "ls -d $VM_CKPT_ROOT/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1")
{ [ -n "${FULL_CKPT:-}" ] && $SSH "$VM" "[ -d $FULL_CKPT ]"; } || fail "no-checkpoint"
log "STEP4 dr checkpoint = $FULL_CKPT"

# ---- 5. done ---------------------------------------------------------------
write_result DONE "$FULL_CKPT" ""
log "DONE — checkpoint in $RESULT"
echo "FINISH_BASE DONE"
