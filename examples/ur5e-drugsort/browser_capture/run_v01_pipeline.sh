#!/usr/bin/env bash
# ============================================================================
# run_v01_pipeline.sh — SELF-DRIVING steering v0.1 retrain + gate + browser A/B.
#
# v0.1 fixes the Stage-B failure modes at their measured root cause:
#   * 68% of expert chunks are near-static (time-indexed expert) -> a time-free
#     steering net collapses to the "stay" majority -> absorbing fixed point at
#     home (13/15 steered home-freezes). FIX: t_norm input (features v2) +
#     motion-weighted loss.
#   * grasp deadlock (steered arm reached 0.2-0.7cm but never closed): phase
#     advanced only on the policy's own grip command. FIX: dwell promotion
#     REACH->GRASP on geometry + hover.
#   * per-session state leaked across episodes (no sid from the eval).
#     FIX: per-episode sid in eval_browser_groot.js.
#
# CHAIN: scp v0.1 trainer/gate -> VM retrain (features v2, motion weights)
#   -> VM stratified offline gate -> pull weights -> local browser A/B via
#   run_stageb_ab.sh (RUN_TAG=v01) -> aggregate ~/v01_result.json.
# Marker: V01_PIPELINE DONE | FAILED: <stage> in ~/v01_pipeline.log.
# ============================================================================
set -uo pipefail

BC=/home/daniel/LovellAI/odyssey-ur5e/examples/ur5e-drugsort/browser_capture
ODY=/home/daniel/LovellAI/odyssey-ur5e
PYUR5E=$ODY/.venv-ur5e/bin/python
VM=ubuntu@192.222.52.169
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30"
GPY=/home/ubuntu/Isaac-GR00T/.venv/bin/python

LOG=$HOME/v01_pipeline.log
RESULT=$HOME/v01_result.json
VM_NET=/home/ubuntu/steering_net_v01.npz
VM_TRAIN_JSON=/home/ubuntu/steering_v01_train.json
VM_GATE_JSON=/home/ubuntu/steering_v01_gate.json

cd "$BC"
exec >>"$LOG" 2>&1
echo "" ; echo "#################################################################"
log(){ echo "=== [v01] $* $(date -u +%FT%TZ) ==="; }
fail(){
  trap - TERM INT HUP
  log "FAILED at: $*"
  printf '{"pipeline":"steering-v0.1","status":"FAILED","failed_at":"%s"}\n' "$*" > "$RESULT"
  echo "V01_PIPELINE FAILED: $*"
  exit 1
}
trap 'fail "signal"' TERM INT HUP

log "START pid=$$"
echo "V01_PIPELINE PID=$$"

# ---- 0. sanity -----------------------------------------------------------------
$SSH "$VM" "echo ok >/dev/null" || fail "vm-unreachable"
FREE=$($SSH "$VM" "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" | head -1 | tr -d ' ')
[ -n "${FREE:-}" ] && [ "$FREE" -gt 20000 ] || fail "gpu-not-free(${FREE:-?}MiB)"
$SSH "$VM" "[ -d /home/ubuntu/steering_targets ] && [ -d /home/ubuntu/ur5e_drugsort_obscond ]" || fail "vm-assets"
$SSH "$VM" "$GPY -c 'import pyarrow'" || fail "vm-pyarrow"

# ---- 1. stage v0.1 scripts to VM -------------------------------------------------
log "STEP1 scp v0.1 trainer + stratified gate to VM (with the gate's import closure)"
scp -q -o StrictHostKeyChecking=no "$ODY/scripts/train_steering_ur5e.py" \
  "$ODY/scripts/flowdagger_offline_gate_ur5e.py" \
  "$ODY/scripts/flow_inverter_groot.py" \
  "$ODY/scripts/probe_flow_inversion_groot.py" "$VM:/home/ubuntu/" || fail "scp-scripts"

# ---- 2. retrain (features v2 + motion-weighted loss) -----------------------------
log "STEP2 VM retrain: features v2 (t_norm), motion-weighted (floor 0.2 ref 0.05), full5280"
$SSH "$VM" "cd /home/ubuntu && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 3600 \
  $GPY train_steering_ur5e.py train --shards /home/ubuntu/steering_targets \
  --output-design full5280 --features v2 --dataset /home/ubuntu/ur5e_drugsort_obscond \
  --weight-floor 0.2 --weight-ref 0.05 \
  --model-out $VM_NET --out $VM_TRAIN_JSON" || fail "vm-retrain"
$SSH "$VM" "[ -f $VM_NET ]" || fail "no-v01-net"
scp -q -o StrictHostKeyChecking=no "$VM:$VM_TRAIN_JSON" "$HOME/steering_v01_train.json" || true
log "retrain done: $("$PYUR5E" -c "import json;d=json.load(open('$HOME/steering_v01_train.json'));print({k:d.get(k) for k in ('val_mse','val_mse_moving','val_mse_static','p99_pred','best_epoch')})" 2>/dev/null || echo '?')"

# ---- 3. offline gate (same two-stage protocol as A5, v2-aware) --------------------
# decode = GR00T venv (GPU); fk-gate = mujoco venv (EE-distance verdict).
EVALPY=/home/ubuntu/odyssey-eval-venv/bin/python
log "STEP3a VM gate decode (GR00T venv, held-out states)"
$SSH "$VM" "cd /home/ubuntu && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 5400 \
  $GPY flowdagger_offline_gate_ur5e.py decode --steering-net $VM_NET \
  --decoded-out /home/ubuntu/v01_gate_decoded.npz --out /home/ubuntu/v01_decode.json" || fail "vm-gate-decode"
log "STEP3b VM fk-gate (mujoco venv)"
$SSH "$VM" "cd /home/ubuntu && timeout 1200 \
  $EVALPY flowdagger_offline_gate_ur5e.py fk-gate \
  --decoded-npz /home/ubuntu/v01_gate_decoded.npz --out $VM_GATE_JSON" || fail "vm-fk-gate"
scp -q -o StrictHostKeyChecking=no "$VM:$VM_GATE_JSON" "$HOME/steering_v01_gate.json" || fail "scp-gate"
GATEPASS=$("$PYUR5E" -c "import json;print(1 if json.load(open('$HOME/steering_v01_gate.json')).get('pass') else 0)" 2>/dev/null)
log "offline gate: $("$PYUR5E" -c "import json;d=json.load(open('$HOME/steering_v01_gate.json'));print({k:d.get(k) for k in ('steered_beats_stock_frac','improvement_toward_oracle','pass')})" 2>/dev/null || echo '?')"
[ "$GATEPASS" = "1" ] || fail "offline-gate-failed"

# ---- 4. browser A/B with the v0.1 net --------------------------------------------
log "STEP4 browser A/B via run_stageb_ab.sh (RUN_TAG=v01)"
RUN_TAG=v01 STEER_NPZ_SRC=$VM_NET bash "$BC/run_stageb_ab.sh"
grep -q "STAGEB_AB DONE" "$HOME/stageb_ab_v01.log" || fail "ab-run"
[ -f "$HOME/stageb_ab_v01_result.json" ] || fail "no-ab-result"

# ---- 5. aggregate -----------------------------------------------------------------
log "STEP5 aggregate"
"$PYUR5E" - <<'PY' || fail "aggregate"
import json
out = {
    "pipeline": "steering-v0.1",
    "train": json.load(open("/home/daniel/steering_v01_train.json")),
    "offline_gate": json.load(open("/home/daniel/steering_v01_gate.json")),
    "browser_ab": json.load(open("/home/daniel/stageb_ab_v01_result.json")),
    "v0_reference": {
        "steered": "0/15 (13 home-freezes, 2 sub-cm approaches 0.2/0.7cm)",
        "stock": "0/15 (4 home-freezes, 11 approaches 4.4-11.6cm)",
    },
    "status": "DONE",
}
json.dump(out, open("/home/daniel/v01_result.json", "w"), indent=2)
print("[v01] headline:", json.dumps({
    "steered": out["browser_ab"]["run_A_steered"]["success"],
    "stock": out["browser_ab"]["run_B_stock"]["success"],
    "verdict": out["browser_ab"]["verdict"],
}))
PY
log "DONE — results in $RESULT"
echo "V01_PIPELINE DONE"
