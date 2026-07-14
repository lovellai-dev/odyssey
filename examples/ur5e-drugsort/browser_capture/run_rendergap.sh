#!/usr/bin/env bash
# ============================================================================
# run_rendergap.sh — SELF-DRIVING render-gap close. Launched detached; needs ZERO
# further involvement. Waits on the already-running parallel browser data-gen, then
# runs the whole chain and writes ~/rendergap_result.json + a RENDERGAP DONE/FAILED
# marker to $LOG.
#
# Chain: (1) block until browser data-gen hits ~TARGET eps -> (2) harvest (stop OUR
# Chrome instances by PID) + merge -> (3) assemble LeRobot -> (4) transfer + GR00T
# fine-tune on the H100 (isolated ckpt/ports, gated on GPU free) ‖ (5) Observer
# convert + retrain locally on the Three.js frames -> (6) deploy new ckpt (VM ZMQ
# :5556 + bridge :5598; re-point LOCAL :5599 tunnel) -> (7) in-browser eval N=15
# each for baseline / new-GR00T / new-GR00T+new-Observer with IDENTICAL seeded poses
# -> (8) rendergap_result.json.
#
# ISOLATION: v4's :5555/:5599 (VM side), ccproxy, gateway, :8000, :8010 are NEVER
# touched. Every Chrome is our own with a unique --user-data-dir, killed by PID.
# NEVER pkill chrome.
# ============================================================================
set -uo pipefail
BC=/home/daniel/LovellAI/odyssey-ur5e/examples/ur5e-drugsort/browser_capture
ODY=/home/daniel/LovellAI/odyssey-ur5e
VM=ubuntu@192.222.52.169
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30"
PYUR5E=$ODY/.venv-ur5e/bin/python
OBSPY=/home/daniel/Isaac-GR00T/.venv/bin/python
XML=/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/aseptipack_description/aseptipack.xml
PUP=/home/daniel/.npm/_npx/23232c69e5d221f3/node_modules/puppeteer-core
CHROME=/usr/bin/google-chrome-stable
LOG=$HOME/rendergap.log
RESULT=$HOME/rendergap_result.json
TARGET=${TARGET:-150}          # harvest at this many completed episodes
MINEPS=${MINEPS:-60}           # ... but never below this even at the deadline
GEN_DEADLINE=$(( $(date +%s) + ${GEN_MAXWAIT:-14400} ))   # 4h safety cap
MAX_STEPS=${MAX_STEPS:-12000}
SAVE_STEPS=${SAVE_STEPS:-3000}
EVAL_N=${EVAL_N:-15}
ZMQ_PORT=5556; BRIDGE_PORT=5598
cd "$BC"
exec >>"$LOG" 2>&1
echo "" ; echo "#################################################################"
log(){ echo "=== [rendergap] $* $(date -u +%FT%TZ) ==="; }
fail(){ echo "=== [rendergap] FAILED at: $* $(date -u +%FT%TZ) ==="; echo "RENDERGAP FAILED: $*"; exit 1; }
log "START pid=$$ target=$TARGET eval_n=$EVAL_N max_steps=$MAX_STEPS"

# ---- 1. Block until data-gen reaches TARGET episodes (or deadline) -----------
log "STEP1 waiting for browser data-gen to reach $TARGET episodes"
while true; do
  NEP=$(ls -d out_full/inst_*/raw/ep*/meta.json 2>/dev/null | wc -l)
  ALIVE=$(pgrep -f 'node browser_harness.js' | wc -l)
  echo "  [wait] completed=$NEP alive=$ALIVE $(date -u +%FT%TZ)"
  [ "$NEP" -ge "$TARGET" ] && { log "reached target ($NEP eps)"; break; }
  [ "$(date +%s)" -ge "$GEN_DEADLINE" ] && { log "gen deadline hit ($NEP eps)"; break; }
  [ "$ALIVE" -eq 0 ] && { log "all data-gen procs exited ($NEP eps)"; break; }
  sleep 60
done
NEP=$(ls -d out_full/inst_*/raw/ep*/meta.json 2>/dev/null | wc -l)
[ "$NEP" -ge "$MINEPS" ] || fail "too-few-episodes($NEP<$MINEPS)"

# ---- 2. Harvest: stop OUR Chrome instances by PID, then merge ---------------
log "STEP2 harvest $NEP episodes (stop our gen instances by PID)"
pkill -f 'gen_parallel.sh' 2>/dev/null || true          # stop the launcher (its trap won't matter)
for pf in out_full/inst_*/harness_pid.txt; do            # kill each Chrome by recorded PID, verified
  [ -f "$pf" ] || continue
  cp=$(cat "$pf" 2>/dev/null || true)
  if [ -n "${cp:-}" ] && kill -0 "$cp" 2>/dev/null && \
     tr '\0' ' ' < "/proc/$cp/cmdline" 2>/dev/null | grep -q "out_full/inst"; then
    echo "  [harvest] kill Chrome PID $cp"; kill "$cp" 2>/dev/null || true
  fi
done
for np in $(pgrep -f 'node browser_harness.js'); do echo "  [harvest] kill node PID $np"; kill "$np" 2>/dev/null || true; done
sleep 5
RAW_ALL=out_full/raw_all
rm -rf "$RAW_ALL"; mkdir -p "$RAW_ALL"
merged=0
for ep in out_full/inst_*/raw/ep*; do
  [ -d "$ep" ] && [ -f "$ep/meta.json" ] || continue
  cp -r "$ep" "$RAW_ALL/" && merged=$((merged+1))
done
log "merged $merged completed episodes -> $RAW_ALL"
[ "$merged" -ge "$MINEPS" ] || fail "too-few-merged($merged)"

# ---- 3. Assemble the LeRobot v2.1 dataset (unchanged schema) + GT sidecar ----
log "STEP3 assemble dataset_browser_full"
$PYUR5E assemble_lerobot.py --raw "$RAW_ALL" --out dataset_browser_full || fail "assemble"

# ---- 4. Transfer + GR00T fine-tune on the H100 (isolated; gate on GPU free) --
log "STEP4 rsync dataset -> VM"
rsync -a --delete -e "$SSH" dataset_browser_full/ "$VM:/home/ubuntu/ur5e_drugsort_browser/" || fail "rsync-dataset"
log "STEP4 gate on GPU free (>45GB; do not fight v4)"
for i in $(seq 1 360); do
  FREE=$($SSH "$VM" "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" 2>/dev/null | head -1 | tr -d ' ')
  { [ -n "$FREE" ] && [ "$FREE" -gt 45000 ]; } && { log "GPU free=${FREE}MiB — go"; break; }
  echo "  [gpu-gate] free=${FREE:-?}MiB waiting"; sleep 60
done
log "STEP4 launch GR00T fine-tune (tmux groot_browser_train)"
scp -q -o StrictHostKeyChecking=no vm_train_browser.sh "$VM:/home/ubuntu/vm_train_browser.sh" || fail "scp-train"
$SSH "$VM" "tmux kill-session -t groot_browser_train 2>/dev/null; tmux new-session -d -s groot_browser_train \
  'bash /home/ubuntu/vm_train_browser.sh /home/ubuntu/ur5e_drugsort_browser /home/ubuntu/ckpt/ur5e_drugsort_browser $MAX_STEPS $SAVE_STEPS 16'" || fail "launch-train"
sleep 20
$SSH "$VM" "tmux has-session -t groot_browser_train 2>/dev/null" || fail "train-session-missing"

# ---- 5. Observer: convert Three.js frames -> .npy + retrain (local GPU) ------
# Runs CONCURRENTLY with the VM GR00T fine-tune (VM GPU vs local RTX 3070).
log "STEP5 convert Three.js frames -> perception .npy"
MUJOCO_GL=egl $PYUR5E perception_from_browser.py --raw "$RAW_ALL" --out percep_browser \
  --max-frames-per-ep 50 --max-total-frames 16000 || fail "percep-convert"
log "STEP5 retrain Observer (DINOv2 grasp-target head + success classifier) on Three.js frames"
( cd "$ODY" && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $OBSPY scripts/train_ur5e_perception.py \
    --data "$BC/percep_browser" --out "$BC/percep_weights_browser" \
    --obs-epochs 60 --clf-epochs 25 --device cuda ) 2>&1 | tee observer_train.log || fail "observer-train"
OBS_CM=$($PYUR5E -c "import json;print(json.load(open('$BC/percep_weights_browser/metrics.json'))['observer']['median_cm'])" 2>/dev/null || echo "?")
log "Observer retrained: grasp-target median error on Three.js frames = ${OBS_CM} cm"

# ---- 6. Wait for the GR00T fine-tune, pick the checkpoint -------------------
log "STEP6 wait for GR00T fine-tune to finish"
while true; do
  $SSH "$VM" "grep -q BROWSER_TRAIN_RC /home/ubuntu/train_browser.log" 2>/dev/null && break
  if ! $SSH "$VM" "tmux has-session -t groot_browser_train 2>/dev/null"; then
    sleep 10
    $SSH "$VM" "grep -q BROWSER_TRAIN_RC /home/ubuntu/train_browser.log" 2>/dev/null && break || fail "train-died-no-rc"
  fi
  sleep 120
done
TRC=$($SSH "$VM" "grep BROWSER_TRAIN_RC /home/ubuntu/train_browser.log | tail -1 | sed 's/.*=//'")
log "GR00T fine-tune exited rc=$TRC"
[ "$TRC" = "0" ] || fail "train-rc=$TRC"
CKPT=/home/ubuntu/ckpt/ur5e_drugsort_browser/checkpoint-$MAX_STEPS
$SSH "$VM" "[ -d $CKPT ]" || CKPT=$($SSH "$VM" "ls -d /home/ubuntu/ckpt/ur5e_drugsort_browser/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1")
{ [ -n "$CKPT" ] && $SSH "$VM" "[ -d $CKPT ]"; } || fail "no-checkpoint"
log "using checkpoint $CKPT"

# ---- 7a. Baseline in-browser eval (current :5599 -> dagger3), BEFORE re-point -
log "STEP7 precompute identical eval poses (seed 7777) + baseline eval"
$PYUR5E precompute_plans.py --num-episodes "$EVAL_N" --seed 7777 --out plans_eval.json || fail "eval-plans"
if ! curl -s --max-time 5 http://127.0.0.1:5599/health | grep -q ok; then
  $SSH -f -N -o ExitOnForwardFailure=yes -L 127.0.0.1:5599:127.0.0.1:5599 "$VM" 2>/dev/null || true; sleep 5
fi
PLANS=plans_eval.json OUT=eval_baseline PORT=8021 AGENTS=groot N=$EVAL_N MAX_TICKS=900 \
  PUPPETEER_CORE=$PUP CHROME=$CHROME node eval_browser_groot.js 2>&1 | grep -vE '^\s*\[page\]' | tail -25 || true

# ---- 7b. Deploy new ckpt on isolated ports + re-point local :5599 -----------
log "STEP7 deploy new ckpt (VM ZMQ :$ZMQ_PORT + bridge :$BRIDGE_PORT)"
scp -q -o StrictHostKeyChecking=no vm_deploy_browser.sh "$VM:/home/ubuntu/vm_deploy_browser.sh" || fail "scp-deploy"
$SSH "$VM" "bash /home/ubuntu/vm_deploy_browser.sh $CKPT $ZMQ_PORT $BRIDGE_PORT" || fail "vm-deploy"
pkill -f '127.0.0.1:5598:127.0.0.1:5598' 2>/dev/null || true; sleep 2
$SSH -f -N -o ExitOnForwardFailure=yes -L 127.0.0.1:5598:127.0.0.1:5598 "$VM" 2>/dev/null || true
for i in $(seq 1 30); do curl -s --max-time 5 http://127.0.0.1:5598/health | grep -q ok && break; sleep 5; done
curl -s --max-time 5 http://127.0.0.1:5598/health | grep -q ok || fail "new-bridge-unhealthy"
log "re-point local :5599 -> my bridge (:5598) so :8020/:8021 serve the new ckpt"
pkill -f '127.0.0.1:5599:127.0.0.1:5599' 2>/dev/null || true; sleep 2
$SSH -f -N -o ExitOnForwardFailure=yes -L 127.0.0.1:5599:127.0.0.1:5598 "$VM" 2>/dev/null || true
for i in $(seq 1 30); do curl -s --max-time 5 http://127.0.0.1:5599/health | grep -q ok && break; sleep 5; done

# ---- 7c. New-checkpoint in-browser eval -------------------------------------
log "STEP7 NEW-checkpoint in-browser eval (?agents=groot)"
PLANS=plans_eval.json OUT=eval_new PORT=8021 AGENTS=groot N=$EVAL_N MAX_TICKS=900 \
  PUPPETEER_CORE=$PUP CHROME=$CHROME node eval_browser_groot.js 2>&1 | grep -vE '^\s*\[page\]' | tail -25 || true

# ---- 7d. Restart Observer sidecar on browser weights + eval groot-observer --
log "STEP7 restart Observer sidecar on Three.js-matched weights + eval ?agents=groot-observer"
pkill -f 'serve_observer_correction.py' 2>/dev/null || true; sleep 2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OBSERVER_WEIGHTS="$BC/percep_weights_browser" \
  ASEPTIPACK_XML="$XML" GROOT_BRIDGE_URL=http://127.0.0.1:5599 OBSERVER_DEVICE=cuda PYTHONPATH="$ODY/src" \
  setsid $OBSPY "$ODY/scripts/serve_observer_correction.py" --port 5601 >observer_sidecar.log 2>&1 &
for i in $(seq 1 24); do curl -s --max-time 5 http://127.0.0.1:8021/api/groot/observer_health | grep -qi 'true\|enabled\|ok' && break; sleep 5; done
PLANS=plans_eval.json OUT=eval_new_observer PORT=8021 AGENTS=groot-observer N=$EVAL_N MAX_TICKS=900 \
  PUPPETEER_CORE=$PUP CHROME=$CHROME node eval_browser_groot.js 2>&1 | grep -vE '^\s*\[page\]' | tail -25 || true

# ---- 8. Collect results -----------------------------------------------------
log "STEP8 write $RESULT"
$PYUR5E - "$BC" "$RESULT" "$CKPT" "$merged" "$MAX_STEPS" <<'PY' || fail "collect"
import json, sys
bc, result, ckpt, neps, steps = sys.argv[1:6]
def L(p, d=None):
    try: return json.load(open(p))
    except Exception: return d
base = L(f"{bc}/eval_baseline/eval_groot.json", {})
new  = L(f"{bc}/eval_new/eval_groot.json", {})
obs  = L(f"{bc}/eval_new_observer/eval_groot-observer.json", {})
om   = L(f"{bc}/percep_weights_browser/metrics.json", {}) or {}
out = {
  "headline": {
    "in_browser_baseline_rendergap_dagger3": base.get("success"),
    "in_browser_new_groot_browser": new.get("success"),
    "in_browser_new_groot_plus_new_observer": obs.get("success"),
    "headless_reference": "2/20",
    "observer_grasp_target_median_cm_threejs": (om.get("observer") or {}).get("median_cm"),
  },
  "train": {"browser_episodes": int(neps), "max_steps": int(steps), "checkpoint": ckpt},
  "baseline_eval": base, "new_eval": new, "observer_eval": obs,
  "observer_localization": om.get("observer"), "observer_classifier": om.get("classifier"),
}
json.dump(out, open(result, "w"), indent=2)
print("[result] " + json.dumps(out["headline"]))
PY

# ---- 9. Restore the baseline tunnel (courtesy) ------------------------------
log "STEP9 restore local :5599 -> VM :5599 (dagger3 baseline)"
pkill -f '127.0.0.1:5599:127.0.0.1:5599' 2>/dev/null || true; sleep 2
$SSH -f -N -o ExitOnForwardFailure=yes -L 127.0.0.1:5599:127.0.0.1:5599 "$VM" 2>/dev/null || true

log "DONE — results in $RESULT"
echo "RENDERGAP DONE"
