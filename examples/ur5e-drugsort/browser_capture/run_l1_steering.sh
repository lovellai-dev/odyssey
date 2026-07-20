#!/usr/bin/env bash
# ============================================================================
# run_l1_steering.sh — L1 lever: build the image-conditioned (v4) steering head
# for the CURRENT base checkpoint (default: the diverse-data dr base). The loop
# then serves it via best-of-N (V4 eval) and gates the resulting funnel.
#
# Steering is CHECKPOINT-SPECIFIC: this inverts the base's OWN expert chunks
# through the base's OWN flow field, encodes the pooled backbone features the
# noise targets depend on, and trains the head — the exact pipeline that broke
# the corrective-fit floor on obscond (0.087 -> 0.041). Writes the head to the
# path the V4 eval reads (/home/ubuntu/steering_net_v4.npz = "the head for the
# current base"). Marker: L1_STEERING DONE|FAILED in ~/l1_steering.log.
#
# Env: CKPT_OVERRIDE (base ckpt; default dr) ; L1_DATASET (default dr dataset)
# ============================================================================
set -uo pipefail
ODY=/home/daniel/LovellAI/odyssey-ur5e
PYUR5E=$ODY/.venv-ur5e/bin/python
VM=ubuntu@192.222.52.169
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30"
GPY=/home/ubuntu/Isaac-GR00T/.venv/bin/python
EVALPY=/home/ubuntu/odyssey-eval-venv/bin/python
LOG=$HOME/l1_steering.log
RESULT=$HOME/l1_steering_result.json

CKPT=${CKPT_OVERRIDE:-/home/ubuntu/ckpt/ur5e_drugsort_dr/checkpoint-15000}
DATASET=${L1_DATASET:-/home/ubuntu/ur5e_drugsort_dr}
# per-base tags so re-running for a different base never mixes artifacts
TAG=$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")
TARGETS=/home/ubuntu/steering_targets_$TAG
FEATS=/home/ubuntu/steering_feats_$TAG
HEAD_OUT=/home/ubuntu/steering_net_v4.npz   # the path V4 eval reads

exec >>"$LOG" 2>&1
echo "" ; echo "#################################################################"
log(){ echo "=== [l1] $* $(date -u +%FT%TZ) ==="; }
fail(){ log "FAILED at: $*"; printf '{"lever":"L1_steering","status":"FAILED","failed_at":"%s"}\n' "$*" > "$RESULT"; echo "L1_STEERING FAILED: $*"; exit 1; }
trap 'fail "signal"' TERM INT HUP
log "START pid=$$ ckpt=$CKPT dataset=$DATASET"
echo "L1_STEERING PID=$$"

# ---- 0. sanity + GPU gate ------------------------------------------------------
$SSH "$VM" "[ -d $CKPT ] && [ -d $DATASET ]" || fail "vm-assets($CKPT/$DATASET)"
$SSH "$VM" "grep -q init_noise ~/Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py" || fail "init-noise-patch-missing"
FREE=$($SSH "$VM" "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" | head -1 | tr -d ' ')
[ -n "${FREE:-}" ] && [ "$FREE" -gt 20000 ] || fail "gpu-not-free(${FREE:-?})"

# ---- 1. stage current scripts to the VM ----------------------------------------
log "STEP0 scp current inverter/trainer/gate"
scp -q -o StrictHostKeyChecking=no "$ODY/scripts/flow_inverter_groot.py" \
  "$ODY/scripts/probe_flow_inversion_groot.py" "$ODY/scripts/train_steering_ur5e.py" \
  "$ODY/scripts/flowdagger_offline_gate_ur5e.py" "$VM:/home/ubuntu/" || fail "scp"

# ---- 2. invert the base's expert chunks -> steering targets ---------------------
log "STEP1 invert-dataset ($DATASET through $CKPT) -> $TARGETS"
$SSH "$VM" "[ -f $TARGETS/manifest.json ]" && log "targets exist — skip invert" || \
$SSH "$VM" "cd /home/ubuntu && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 14400 \
  $GPY flow_inverter_groot.py invert-dataset --dataset $DATASET --model-path $CKPT \
  --out-dir $TARGETS --stride 8 --save-full-w --max-gpu-hours 3.0 \
  --out /home/ubuntu/l1_invert_$TAG.json" || fail "invert"
scp -q -o StrictHostKeyChecking=no "$VM:/home/ubuntu/l1_invert_$TAG.json" "$HOME/l1_invert.json" 2>/dev/null || true
log "invert: $("$PYUR5E" -c "import json;d=json.load(open('$HOME/l1_invert.json'));print({k:d.get(k) for k in ('n_chunks','drop_rate','gpu_hours')})" 2>/dev/null || echo '?')"

# ---- 3. encode pooled backbone features ----------------------------------------
log "STEP2 encode-features -> $FEATS"
$SSH "$VM" "[ -d $FEATS ] && ls $FEATS/feat_shard_*.npz >/dev/null 2>&1" && log "feats exist — skip" || \
$SSH "$VM" "cd /home/ubuntu && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 7200 \
  $GPY flow_inverter_groot.py encode-features --shards $TARGETS --dataset $DATASET \
  --model-path $CKPT --out-dir $FEATS --mini-batch 16 \
  --out /home/ubuntu/l1_feat_$TAG.json" || fail "encode-features"

# ---- 4. train the v4 image-conditioned head ------------------------------------
log "STEP3 train v4 head (features v4, hidden 512) -> $HEAD_OUT"
$SSH "$VM" "cd /home/ubuntu && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 3600 \
  $GPY train_steering_ur5e.py train --shards $TARGETS --feat-shards $FEATS \
  --output-design full5280 --features v4 --hidden 512 --dataset $DATASET \
  --source-boost=1.0 --weight-floor 0.2 --weight-ref 0.05 \
  --model-out $HEAD_OUT --out /home/ubuntu/l1_train_$TAG.json" || fail "train"
scp -q -o StrictHostKeyChecking=no "$VM:/home/ubuntu/l1_train_$TAG.json" "$HOME/l1_train.json" 2>/dev/null || true
log "train: $("$PYUR5E" -c "import json;d=json.load(open('$HOME/l1_train.json'));print({k:d.get(k) for k in ('val_mse','val_mse_by_source')})" 2>/dev/null || echo '?')"

# ---- 5. base-fidelity gate (informational; the loop's browser eval is the gate) -
log "STEP4 base gate (decode + fk)"
$SSH "$VM" "cd /home/ubuntu && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 5400 \
  $GPY flowdagger_offline_gate_ur5e.py decode --steering-net $HEAD_OUT --model-path $CKPT \
  --dataset $DATASET --decoded-out /home/ubuntu/l1_base_$TAG.npz \
  --out /home/ubuntu/l1_base_decode_$TAG.json" || log "base-decode failed (non-fatal)"
$SSH "$VM" "cd /home/ubuntu && timeout 1800 $EVALPY flowdagger_offline_gate_ur5e.py fk-gate \
  --decoded-npz /home/ubuntu/l1_base_$TAG.npz --out /home/ubuntu/l1_base_gate_$TAG.json" 2>/dev/null || true
scp -q -o StrictHostKeyChecking=no "$VM:/home/ubuntu/l1_base_gate_$TAG.json" "$HOME/l1_base_gate.json" 2>/dev/null || true

# ---- 6. result ------------------------------------------------------------------
"$PYUR5E" - "$RESULT" "$HEAD_OUT" "$CKPT" <<'PY' || true
import json, sys
def L(p):
    try: return json.load(open(p))
    except Exception: return {}
out = {"lever": "L1_steering", "status": "DONE", "steering_head": sys.argv[2],
       "base_ckpt": sys.argv[3],
       "invert": L("/home/daniel/l1_invert.json"),
       "train": L("/home/daniel/l1_train.json"),
       "base_gate": L("/home/daniel/l1_base_gate.json")}
json.dump(out, open(sys.argv[1], "w"), indent=2)
print("[l1] result:", json.dumps({"val_mse": out["train"].get("val_mse"),
      "base_ee": (out["base_gate"].get("ee_dist_cm") or {}).get("steered_mean"),
      "base_pass": out["base_gate"].get("pass")}))
PY
log "DONE — head at $HEAD_OUT (for $CKPT)"
echo "L1_STEERING DONE"
