#!/usr/bin/env bash
# ============================================================================
# run_base_fix.sh — SELF-DRIVING base-policy fix + the DECISIVE bare-base kill
# test. Launched DETACHED (setsid nohup); needs ZERO further involvement. Writes
# ~/base_fix_result.json + a BASE_FIX DONE|FAILED marker to ~/base_fix.log.
#
# HYPOTHESIS: the GR00T base is weak (~2/20 headless, 0/15 browser) because its
# ~300 demos share ONE trajectory shape -> the policy shortcuts to
# wrist-pixels->action and ignores the grasp target (attention collapse). FIX =
# trajectory-shape domain randomization (precompute_plans.py --randomize). This
# driver generates ~600 diverse browser-rendered demos, SFTs a FRESH 10-D base,
# and measures it BARE (no steering, no best-of-N, no CBF/CEM) in-browser.
#
# CHAIN:
#  0. bring up a gen static server (agent-service :8021, COOP/COEP for MuJoCo-WASM)
#  1. SMOKE-GATE the DR jitters: 12-ep DR gen -> browser-physics success rate; if
#     < 70% reduce the jitters (--dr-clearance 0.03 --dr-approach-xy 0.02) so we
#     never ship 600 bad demos.
#  2. precompute 600 DR plans -> parallel browser data-gen (6 shards) -> raw frames
#  3. assemble LeRobot (SUCCESS-FILTERED) -> augment to 10-D (grasp_target) -> verify
#  4. rsync -> VM; GPU-gate; SFT a FRESH ckpt ur5e_drugsort_dr (15000 steps, 10-D cfg)
#  5. deploy the dr ckpt (ZMQ :5562 + 10-D bridge :5590, groot_basefix_* tmux)
#  6. KILL TEST — bare base:
#       HEADLESS: eval_ur5e_drugsort_groot.py --grasp-target-from-gt N=20 (direct ZMQ)
#       BROWSER : bare Observer conditioner (STEERING off) + agent-service :8034 ->
#                 eval_browser_groot.js AGENTS=groot N=15 (seed-7777 poses)
#  7. write ~/base_fix_result.json + verdict.
#
# ISOLATION: OWN ports (8021/8034/5606/5562/5590), OWN VM tmux (groot_basefix_*),
# OWN Chrome (gen_parallel + eval harness: unique --user-data-dir, killed by PID).
# NEVER pkill chrome. Protected VM tmux (ccproxy, gateway, groot_browser_*) and the
# obscond ckpt (v4 head depends on it) are NEVER touched. Training GPU-gated (>45GB).
#
# set -uo pipefail + explicit fail() (NOT set -e) so a stray non-zero from a
# best-effort curl/grep cannot abort the multi-hour detached run.
# ============================================================================
set -uo pipefail

# ---- paths / config ---------------------------------------------------------
BC=/home/daniel/LovellAI/odyssey-ur5e/examples/ur5e-drugsort/browser_capture
ODY=/home/daniel/LovellAI/odyssey-ur5e
AGENT_DIR=/home/daniel/LovellAI/lai-agent-multiagent/agent_service
AGENT_PY=/home/daniel/LovellAI/lai-agent/agent_service/.venv/bin/python
PYUR5E=$ODY/.venv-ur5e/bin/python
OBSPY=/home/daniel/Isaac-GR00T/.venv/bin/python
VM=ubuntu@192.222.52.169
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
XML=/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/aseptipack_description/aseptipack.xml
VM_XML=/home/ubuntu/aseptipack_description/aseptipack.xml
PUP=/home/daniel/.npm/_npx/23232c69e5d221f3/node_modules/puppeteer-core
CHROME=/usr/bin/google-chrome-stable
NODE=/home/daniel/.nvm/versions/node/v22.22.0/bin/node
EVAL_VENV=/home/ubuntu/odyssey-eval-venv/bin/python
GROOT_PY=/home/ubuntu/Isaac-GR00T/.venv/bin/python

WORK=$HOME/base_fix
LOG=$HOME/base_fix.log
RESULT=$HOME/base_fix_result.json
mkdir -p "$WORK"

# Isolated ports / sessions / dirs
GEN_PORT=${GEN_PORT:-8021}
AGENT_PORT=${AGENT_PORT:-8034}
COND_PORT=${COND_PORT:-5606}
ZMQ_PORT=${ZMQ_PORT:-5562}
BRIDGE_PORT=${BRIDGE_PORT:-5590}
VM_DATASET=/home/ubuntu/ur5e_drugsort_dr
VM_CKPT_ROOT=/home/ubuntu/ckpt/ur5e_drugsort_dr
VM_CFG=examples/UR5e_DrugSort/ur5e_config_obscond.py   # 10-D modality config (already on VM)
OBS_WEIGHTS=$BC/percep_weights_browser
EVAL_PLANS=$HOME/observer_cond/plans_eval.json         # seed-7777 poses (== the old obscond eval)

# Knobs
NUM_DEMOS=${NUM_DEMOS:-600}
GEN_SEED=${GEN_SEED:-909}
SHARDS=${SHARDS:-6}
SMOKE_DEMOS=${SMOKE_DEMOS:-12}
SMOKE_SHARDS=${SMOKE_SHARDS:-2}
SMOKE_MIN_RATE=${SMOKE_MIN_RATE:-70}     # percent; below this reduce the DR jitters
AREA_X=${AREA_X:-0.08}; AREA_Y=${AREA_Y:-0.10}; YAW=${YAW:-0.25}
MAX_STEPS=${MAX_STEPS:-15000}
SAVE_STEPS=${SAVE_STEPS:-7500}           # 2 saves (~72G): VM is 92% full, ~36G/ckpt
BATCH=${BATCH:-16}
EVAL_N=${EVAL_N:-15}
HL_N=${HL_N:-20}
N_ACTION_STEPS=${N_ACTION_STEPS:-8}
MAX_TICKS=${MAX_TICKS:-900}
GPU_FREE_MIN=${GPU_FREE_MIN:-45000}
GEN_DEADLINE_H=${GEN_DEADLINE_H:-6}

export DISPLAY=${DISPLAY:-:1}
cd "$BC"
exec >>"$LOG" 2>&1
echo ""; echo "#################################################################"
log(){ echo "=== [basefix] $* $(date -u +%FT%TZ) ==="; }

# ---- result collection (robust to partial completion) -----------------------
collect_result(){   # $1 status ; $2 stage
  local status="$1" stage="${2:-}"
  "$PYUR5E" - "$WORK" "$RESULT" "$status" "$stage" <<'PY' || true
import json, sys, os
work, result, status, stage = sys.argv[1:5]
def L(p, d=None):
    try: return json.load(open(p))
    except Exception: return d
asum = L(os.path.join(work, "dataset_dr", "assemble_summary.json"), {}) or {}
hl   = L(os.path.join(work, "basefix_headless.json"), {}) or {}
br   = L(os.path.join(work, "eval_browser", "eval_groot.json"), {}) or {}
tr   = L(os.path.join(work, "train.json"), {}) or {}
smoke = L(os.path.join(work, "smoke.json"), {}) or {}
def frac(k, n):
    if k is None or not n: return None
    return f"{k}/{n}"
hl_n = hl.get("n_episodes"); hl_s = hl.get("n_success"); hl_l = hl.get("n_lifted")
br_n = br.get("n_attempts"); br_s = br.get("n_success"); br_l = br.get("n_lifted")
browser_bare = frac(br_s, br_n)
verdict = None
if br_s is not None and br_n:
    rate = br_s / br_n
    if rate >= 0.30: verdict = "BASE_FIXED"
    elif rate >= 0.10: verdict = "BASE_IMPROVED"
    else: verdict = "BASE_CEILING"
out = {
  "n_demos_generated": asum.get("raw_episodes"),
  "success_filter_rate": asum.get("success_filter_rate"),
  "demos_kept": asum.get("kept"),
  "smoke_dr_success_rate": smoke.get("rate"),
  "dr_jitters": smoke.get("jitters"),
  "sft_steps": tr.get("max_steps"),
  "sft_checkpoint": tr.get("checkpoint"),
  "headless_bare": frac(hl_s, hl_n),
  "headless_bare_lifted": frac(hl_l, hl_n),
  "browser_bare": browser_bare,
  "browser_bare_lifted": frac(br_l, br_n),
  "browser_bare_seated": frac(br.get("n_seated"), br_n),
  "vs_old_base": {"headless": "~2/20", "browser": "0/15",
                  "note": "old = obscond single-shape 10-D base (151 demos)"},
  "verdict": verdict,
  "status": status, "stage": stage or None,
}
json.dump(out, open(result, "w"), indent=2)
print("[result] " + json.dumps({k: out[k] for k in
      ("n_demos_generated","success_filter_rate","sft_steps","headless_bare","browser_bare","verdict")}))
PY
}

cleanup(){
  log "cleanup (own gen/eval agent-service + conditioner + tunnel + VM basefix tmux; leave everything else)"
  for pf in "$WORK/gen_agent_pid.txt" "$WORK/agent_pid.txt" "$WORK/cond_pid.txt"; do
    [ -f "$pf" ] || continue
    p=$(cat "$pf" 2>/dev/null || true); [ -n "${p:-}" ] && kill "$p" 2>/dev/null || true
  done
  ss -tlnp 2>/dev/null | grep -E ":($GEN_PORT|$AGENT_PORT|$COND_PORT) " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | xargs -r kill 2>/dev/null || true
  pkill -f "127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT" 2>/dev/null || true
  $SSH "$VM" "tmux kill-session -t groot_basefix_server 2>/dev/null; tmux kill-session -t groot_basefix_bridge 2>/dev/null; tmux kill-session -t groot_basefix_train 2>/dev/null; tmux kill-session -t groot_basefix_hleval 2>/dev/null" 2>/dev/null || true
}
fail(){
  echo "=== [basefix] FAILED at: $* $(date -u +%FT%TZ) ==="
  collect_result FAILED "$*"
  cleanup
  echo "BASE_FIX FAILED: $*"
  exit 1
}
kill_chrome_pidfile(){   # $1 = pidfile ; $2 = cmdline substring to verify ownership
  local pf="$1" needle="${2:-chrome-udd}"; [ -f "$pf" ] || return 0
  local cp; cp=$(cat "$pf" 2>/dev/null || true)
  if [ -n "${cp:-}" ] && kill -0 "$cp" 2>/dev/null; then
    tr '\0' ' ' < "/proc/$cp/cmdline" 2>/dev/null | grep -q "$needle" && { echo "  [basefix] kill Chrome PID $cp"; kill "$cp" 2>/dev/null || true; }
  fi
  rm -f "$pf" 2>/dev/null || true
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
start_gen_server(){   # bare agent-service serving the Playground (COOP/COEP) for data-gen
  log "start gen static server (agent-service :$GEN_PORT, COOP/COEP for MuJoCo-WASM)"
  local STALE
  STALE=$(ss -tlnp 2>/dev/null | grep ":$GEN_PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
  [ -n "${STALE:-}" ] && { kill "$STALE" 2>/dev/null || true; sleep 2; }
  ( cd "$AGENT_DIR" && env -u PYTHONPATH ENVIRONMENT=development \
      DATABASE_URL="sqlite+aiosqlite:///./basefix_gen.db" \
      GROOT_BRIDGE_URL="http://127.0.0.1:1" DISPLAY="$DISPLAY" \
      "$AGENT_PY" -m uvicorn app.main:app --host 127.0.0.1 --port "$GEN_PORT" --loop asyncio \
      > "$WORK/gen_agent.log" 2>&1 & echo $! > "$WORK/gen_agent_pid.txt" )
  cd "$BC"
  local i
  for i in $(seq 1 40); do
    curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$GEN_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null | grep -q 200 && return 0
    sleep 3
  done
  return 1
}

# ============================================================================
log "START pid=$$ demos=$NUM_DEMOS shards=$SHARDS steps=$MAX_STEPS save=$SAVE_STEPS"
echo "BASE_FIX PID=$$"
collect_result RUNNING "start"

# ---- 0. sanity -------------------------------------------------------------
[ -f "$XML" ] || fail "no-xml($XML)"
[ -f "$OBS_WEIGHTS/observer_head.pt" ] || fail "no-observer-weights($OBS_WEIGHTS)"
[ -f "$EVAL_PLANS" ] || fail "no-eval-plans($EVAL_PLANS)"
$SSH "$VM" "echo ok >/dev/null" || fail "vm-unreachable"
$SSH "$VM" "[ -d $VM_CKPT_ROOT ] && echo EXISTS || mkdir -p $VM_CKPT_ROOT" >/dev/null 2>&1
$SSH "$VM" "[ -f /home/ubuntu/Isaac-GR00T/$VM_CFG ]" || fail "no-vm-10d-config"
start_gen_server || fail "gen-server-not-serving"
log "gen server up on :$GEN_PORT (pid=$(cat "$WORK/gen_agent_pid.txt" 2>/dev/null))"

# ---- 1. SMOKE-GATE the DR jitters ------------------------------------------
DR_EXTRA=""
JIT_DESC="clearance=0.04 approach_xy=0.03 (default)"
log "STEP1 DR-jitter smoke ($SMOKE_DEMOS eps) to check browser-physics success"
rm -rf "$WORK/out_smoke" "$WORK/plans_smoke.json" "$WORK/dataset_smoke"
env -u PYTHONPATH MUJOCO_GL=egl "$PYUR5E" precompute_plans.py --xml "$XML" \
  --num-episodes "$SMOKE_DEMOS" --seed "$GEN_SEED" --randomize \
  --area-x "$AREA_X" --area-y "$AREA_Y" --yaw-jitter "$YAW" \
  --out "$WORK/plans_smoke.json" || fail "smoke-precompute"
PLANS_FULL="$WORK/plans_smoke.json" SHARDS="$SMOKE_SHARDS" OUTBASE="$WORK/out_smoke" \
  PORTS="$GEN_PORT" PYBIN="$PYUR5E" CHROME="$CHROME" PUPPETEER_CORE="$PUP" \
  bash gen_parallel.sh || true
# wait for the smoke gen marker (short; PER_EP_TIMEOUT default 6h but 12 eps finish fast)
for i in $(seq 1 120); do
  grep -q "GEN_PARALLEL DONE" "$WORK/out_smoke/gen.log" 2>/dev/null && break
  sleep 15
done
env -u PYTHONPATH MUJOCO_GL=egl "$PYUR5E" assemble_lerobot.py \
  --raw "$WORK/out_smoke/raw_all" --out "$WORK/dataset_smoke" 2>&1 | tail -6 || true
SMOKE_RATE=$("$PYUR5E" -c "import json;print(round(json.load(open('$WORK/dataset_smoke/assemble_summary.json'))['success_filter_rate']*100))" 2>/dev/null || echo "0")
log "STEP1 smoke DR browser-physics success = ${SMOKE_RATE}%"
if [ "${SMOKE_RATE:-0}" -lt "$SMOKE_MIN_RATE" ]; then
  DR_EXTRA="--dr-clearance 0.03 --dr-approach-xy 0.02"
  JIT_DESC="clearance=0.03 approach_xy=0.02 (REDUCED: smoke ${SMOKE_RATE}% < ${SMOKE_MIN_RATE}%)"
  log "STEP1 DR jitters too aggressive (${SMOKE_RATE}%) -> reduce: $DR_EXTRA"
fi
"$PYUR5E" - "$WORK/smoke.json" "$SMOKE_RATE" "$JIT_DESC" <<'PY' || true
import json, sys
open(sys.argv[1], "w").write(json.dumps({"rate": f"{sys.argv[2]}%", "jitters": sys.argv[3]}) + "\n")
PY
rm -rf "$WORK/dataset_smoke"

# ---- 2. precompute 600 DR plans + parallel browser data-gen ----------------
log "STEP2 precompute $NUM_DEMOS DR plans (seed $GEN_SEED; $JIT_DESC)"
env -u PYTHONPATH MUJOCO_GL=egl "$PYUR5E" precompute_plans.py --xml "$XML" \
  --num-episodes "$NUM_DEMOS" --seed "$GEN_SEED" --randomize \
  --area-x "$AREA_X" --area-y "$AREA_Y" --yaw-jitter "$YAW" $DR_EXTRA \
  --out "$WORK/plans_dr.json" || fail "precompute-full"
IKCONV=$("$PYUR5E" -c "import json;p=json.load(open('$WORK/plans_dr.json'))['plans'];print(sum(1 for x in p if x['ik_converged']),len(p))" 2>/dev/null || echo "?")
log "STEP2 DR plans IK-converged: $IKCONV"

log "STEP2 launch parallel browser data-gen ($SHARDS shards) -> $WORK/out_dr"
rm -rf "$WORK/out_dr"
PLANS_FULL="$WORK/plans_dr.json" SHARDS="$SHARDS" OUTBASE="$WORK/out_dr" \
  PORTS="$GEN_PORT" PYBIN="$PYUR5E" CHROME="$CHROME" PUPPETEER_CORE="$PUP" \
  setsid bash gen_parallel.sh >/dev/null 2>&1 &
GEN_LAUNCH_PID=$!
log "STEP2 gen launcher pid=$GEN_LAUNCH_PID; polling for GEN_PARALLEL DONE (deadline ${GEN_DEADLINE_H}h)"
GEN_END=$(( $(date +%s) + GEN_DEADLINE_H*3600 ))
while true; do
  if grep -q "GEN_PARALLEL DONE" "$WORK/out_dr/gen.log" 2>/dev/null; then
    log "STEP2 GEN_PARALLEL DONE"; break
  fi
  NEP=$(ls -d "$WORK"/out_dr/inst_*/raw/ep*/meta.json 2>/dev/null | wc -l)
  ALIVE=$(pgrep -f "browser_harness.js" | wc -l)
  echo "  [gen-wait] completed=$NEP alive_harness=$ALIVE $(date -u +%FT%TZ)"
  if [ "$(date +%s)" -ge "$GEN_END" ]; then
    log "STEP2 gen deadline hit ($NEP eps) — harvesting what we have"
    # stop our own gen (launcher + harness nodes + our Chrome by recorded PID)
    kill "$GEN_LAUNCH_PID" 2>/dev/null || true
    for pf in "$WORK"/out_dr/inst_*/harness_pid.txt; do
      [ -f "$pf" ] || continue; cp=$(cat "$pf" 2>/dev/null || true)
      [ -n "${cp:-}" ] && kill -0 "$cp" 2>/dev/null && tr '\0' ' ' < "/proc/$cp/cmdline" 2>/dev/null | grep -q "$WORK/out_dr" && kill "$cp" 2>/dev/null || true
    done
    for np in $(pgrep -f "browser_harness.js"); do
      tr '\0' ' ' < "/proc/$np/cmdline" 2>/dev/null | grep -q "browser_harness" && kill "$np" 2>/dev/null || true
    done
    sleep 5; break
  fi
  [ "$ALIVE" -eq 0 ] && [ "$NEP" -gt 0 ] && { log "STEP2 all harness procs exited ($NEP eps)"; sleep 5; break; }
  sleep 120
done

# merge any completed episodes (gen_parallel already builds raw_all on clean exit;
# rebuild it here to also cover the deadline-harvest path).
RAW_ALL="$WORK/out_dr/raw_all"
mkdir -p "$RAW_ALL"
for ep in "$WORK"/out_dr/inst_*/raw/ep*; do
  [ -d "$ep" ] && [ -f "$ep/meta.json" ] || continue
  b=$(basename "$ep"); [ -e "$RAW_ALL/$b" ] || cp -r "$ep" "$RAW_ALL/" 2>/dev/null || true
done
NRAW=$(ls -d "$RAW_ALL"/ep*/meta.json 2>/dev/null | wc -l)
log "STEP2 merged $NRAW completed episodes -> $RAW_ALL"
[ "$NRAW" -ge 60 ] || fail "too-few-episodes($NRAW)"

# ---- 3. assemble (SUCCESS-FILTERED) -> augment to 10-D ----------------------
log "STEP3 assemble LeRobot (success-filtered) -> dataset_dr"
rm -rf "$WORK/dataset_dr" "$WORK/dataset_dr_cond"
env -u PYTHONPATH MUJOCO_GL=egl "$PYUR5E" assemble_lerobot.py \
  --raw "$RAW_ALL" --out "$WORK/dataset_dr" 2>&1 | tail -8 || fail "assemble"
[ -f "$WORK/dataset_dr/meta/info.json" ] || fail "assemble-no-dataset"
FRATE=$("$PYUR5E" -c "import json;d=json.load(open('$WORK/dataset_dr/assemble_summary.json'));print(d['kept'],d['raw_episodes'],round(d['success_filter_rate']*100,1))" 2>/dev/null || echo "? ? ?")
log "STEP3 success-filter: $FRATE (kept raw pct)"

log "STEP3 augment -> 10-D grasp_target (dataset_dr_cond)"
env -u PYTHONPATH MUJOCO_GL=egl "$PYUR5E" augment_state_grasp_target.py \
  --src "$WORK/dataset_dr" --out "$WORK/dataset_dr_cond" --xml "$XML" --verify 2>&1 | tail -8 || fail "augment"
STATE_DIM=$("$PYUR5E" -c "import json;print(json.load(open('$WORK/dataset_dr_cond/meta/info.json'))['features']['observation.state']['shape'][0])" 2>/dev/null)
[ "$STATE_DIM" = "10" ] || fail "augment-bad-state-dim($STATE_DIM)"
log "STEP3 dataset_dr_cond state dim = $STATE_DIM (OK)"
# gen server no longer needed
p=$(cat "$WORK/gen_agent_pid.txt" 2>/dev/null || true); [ -n "${p:-}" ] && kill "$p" 2>/dev/null || true

# ---- 4. rsync -> VM; GPU-gate; SFT a FRESH dr checkpoint --------------------
log "STEP4 rsync dataset_dr_cond -> VM:$VM_DATASET"
rsync -a --delete -e "$SSH" "$WORK/dataset_dr_cond/" "$VM:$VM_DATASET/" || fail "rsync-dataset"
# ensure the 10-D config + train wrapper are current on the VM
scp -q -o StrictHostKeyChecking=no "$ODY/examples/ur5e-drugsort/ur5e_config.py" "$VM:/home/ubuntu/Isaac-GR00T/$VM_CFG" || fail "scp-config"
scp -q -o StrictHostKeyChecking=no vm_train_basefix.sh vm_deploy_basefix.sh "$VM:/home/ubuntu/" || fail "scp-vm-scripts"
scp -q -o StrictHostKeyChecking=no "$ODY/scripts/serve_groot_http_bridge.py" "$VM:/home/ubuntu/serve_groot_http_bridge_obscond.py" || fail "scp-bridge"
scp -q -o StrictHostKeyChecking=no "$ODY/scripts/eval_ur5e_drugsort_groot.py" "$VM:/home/ubuntu/odyssey-ur5e/scripts/eval_ur5e_drugsort_groot.py" || fail "scp-headless-eval"

gpu_gate || fail "gpu-gate-timeout"
log "STEP4 launch SFT (tmux groot_basefix_train; $MAX_STEPS steps, save $SAVE_STEPS)"
$SSH "$VM" "tmux kill-session -t groot_basefix_train 2>/dev/null; tmux new-session -d -s groot_basefix_train \
  'bash /home/ubuntu/vm_train_basefix.sh $VM_DATASET $VM_CKPT_ROOT $MAX_STEPS $SAVE_STEPS $BATCH $VM_CFG'" || fail "launch-train"
sleep 25
$SSH "$VM" "tmux has-session -t groot_basefix_train 2>/dev/null" || \
  $SSH "$VM" "grep -q BASEFIX_TRAIN_RC /home/ubuntu/train_basefix.log" 2>/dev/null || fail "train-session-missing"
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
[ "$TRC" = "0" ] || fail "train-rc=$TRC"
FULL_CKPT=/home/ubuntu/ckpt/ur5e_drugsort_dr/checkpoint-$MAX_STEPS
$SSH "$VM" "[ -d $FULL_CKPT ]" || FULL_CKPT=$($SSH "$VM" "ls -d /home/ubuntu/ckpt/ur5e_drugsort_dr/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1")
{ [ -n "${FULL_CKPT:-}" ] && $SSH "$VM" "[ -d $FULL_CKPT ]"; } || fail "no-checkpoint"
printf '{"checkpoint":"%s","max_steps":%s,"save_steps":%s,"dataset":"%s","state_dim":10}\n' \
  "$FULL_CKPT" "$MAX_STEPS" "$SAVE_STEPS" "$VM_DATASET" > "$WORK/train.json"
log "STEP4 dr checkpoint = $FULL_CKPT"
collect_result RUNNING "trained"

# ---- 5. deploy the dr ckpt (ZMQ + 10-D bridge, groot_basefix_*) -------------
log "STEP5 deploy dr ckpt (ZMQ :$ZMQ_PORT + 10-D bridge :$BRIDGE_PORT)"
$SSH "$VM" "bash /home/ubuntu/vm_deploy_basefix.sh $FULL_CKPT $ZMQ_PORT $BRIDGE_PORT" || fail "vm-deploy"
pkill -f "127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT" 2>/dev/null || true; sleep 2
$SSH -f -N -o ExitOnForwardFailure=yes -L 127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT "$VM" 2>/dev/null || true
for i in $(seq 1 30); do curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health | grep -q ok && break; sleep 5; done
curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health | grep -q ok || fail "bridge-unhealthy"

# ---- 6a. KILL TEST — HEADLESS bare base (direct ZMQ, GT grasp_target) -------
log "STEP6a HEADLESS bare-base eval (N=$HL_N, direct ZMQ, --grasp-target-from-gt)"
$SSH "$VM" "rm -f /home/ubuntu/basefix_headless.json /home/ubuntu/basefix_headless.log; \
  tmux kill-session -t groot_basefix_hleval 2>/dev/null; \
  tmux new-session -d -s groot_basefix_hleval \
  'cd /home/ubuntu/odyssey-ur5e && MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env -u PYTHONPATH $EVAL_VENV scripts/eval_ur5e_drugsort_groot.py \
     --xml $VM_XML --host 127.0.0.1 --port $ZMQ_PORT --num-episodes $HL_N --n-action-steps $N_ACTION_STEPS \
     --max-ticks 1000 --grasp-target-from-gt --out-json /home/ubuntu/basefix_headless.json \
     > /home/ubuntu/basefix_headless.log 2>&1; echo HLEVAL_DONE rc=\$? >> /home/ubuntu/basefix_headless.log'" || fail "launch-headless"
for i in $(seq 1 120); do   # up to ~60 min
  $SSH "$VM" "grep -q HLEVAL_DONE /home/ubuntu/basefix_headless.log" 2>/dev/null && break
  $SSH "$VM" "tmux has-session -t groot_basefix_hleval 2>/dev/null" || { sleep 8; $SSH "$VM" "grep -q HLEVAL_DONE /home/ubuntu/basefix_headless.log" 2>/dev/null && break; }
  echo "  [hleval-wait] $($SSH "$VM" "grep -c SUCCESS /home/ubuntu/basefix_headless.log 2>/dev/null" 2>/dev/null || echo 0) succ-lines $(date -u +%FT%TZ)"
  sleep 30
done
scp -q -o StrictHostKeyChecking=no "$VM:/home/ubuntu/basefix_headless.json" "$WORK/basefix_headless.json" 2>/dev/null || log "WARN no headless json"
$SSH "$VM" "tail -4 /home/ubuntu/basefix_headless.log" 2>/dev/null || true
HL=$("$PYUR5E" -c "import json;d=json.load(open('$WORK/basefix_headless.json'));print(d['n_success'],'/',d['n_episodes'],'lifted',d['n_lifted'])" 2>/dev/null || echo "?")
log "STEP6a HEADLESS bare-base = $HL"
collect_result RUNNING "headless-done"

# ---- 6b. KILL TEST — BROWSER bare base (Observer target, NO steering) -------
log "STEP6b start BARE Observer conditioner :$COND_PORT (target inject only; NO steering/bestofn)"
STALE=$(ss -tlnp 2>/dev/null | grep ":$COND_PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
[ -n "${STALE:-}" ] && { kill "$STALE" 2>/dev/null || true; sleep 2; }
# NOTE: STEERING_WEIGHTS unset + STEER_BESTOFN_K unset => bare (conditioner attaches
# no init-noise steering and no best-of-N; it only injects the Observer grasp target
# to fill the 10-D state channel the policy was trained with).
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OBSERVER_WEIGHTS="$OBS_WEIGHTS" \
  CONDITION_BRIDGE_URL="http://127.0.0.1:$BRIDGE_PORT" OBSERVER_DEVICE=cuda PYTHONPATH="$ODY/src" \
  setsid "$OBSPY" "$ODY/scripts/serve_observer_conditioning.py" --port "$COND_PORT" \
  >"$WORK/conditioner.log" 2>&1 &
echo $! > "$WORK/cond_pid.txt"
for i in $(seq 1 40); do curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health | grep -q observer_ready && break; sleep 3; done
curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health | grep -q observer_ready || fail "conditioner-not-up"
# guard: conditioner must be BARE (no steering)
CH=$(curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health || echo '{}')
echo "  [cond-health] $CH"

log "STEP6b launch eval agent-service :$AGENT_PORT (GROOT_BRIDGE_URL -> conditioner)"
STALE=$(ss -tlnp 2>/dev/null | grep ":$AGENT_PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
[ -n "${STALE:-}" ] && { kill "$STALE" 2>/dev/null || true; sleep 2; }
( cd "$AGENT_DIR" && env -u PYTHONPATH ENVIRONMENT=development \
    DATABASE_URL="sqlite+aiosqlite:///./basefix_agent.db" \
    GROOT_BRIDGE_URL="http://127.0.0.1:$COND_PORT" \
    GROOT_STATE_CONDITIONER_URL="http://127.0.0.1:$COND_PORT" \
    GROOT_OBSERVER_URL="http://127.0.0.1:$COND_PORT" \
    DISPLAY="$DISPLAY" "$AGENT_PY" -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$AGENT_PORT" --loop asyncio > "$WORK/agent.log" 2>&1 & echo $! > "$WORK/agent_pid.txt" )
cd "$BC"
for i in $(seq 1 24); do
  curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null | grep -q 200 && break
  sleep 3
done
curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null | grep -q 200 || fail "agent-not-serving"
for i in $(seq 1 24); do curl -s --max-time 6 "http://127.0.0.1:$AGENT_PORT/api/groot/health" | grep -qE '"ok": *true|bridge_reachable" *: *true|"bridge_health"' && break; sleep 5; done

log "STEP6b BROWSER bare-base eval (?agents=groot, N=$EVAL_N, seed-7777 poses)"
rm -rf "$WORK/eval_browser"
PLANS="$EVAL_PLANS" OUT="$WORK/eval_browser" PORT="$AGENT_PORT" AGENTS=groot N="$EVAL_N" \
  N_ACTION_STEPS="$N_ACTION_STEPS" MAX_TICKS="$MAX_TICKS" \
  PUPPETEER_CORE="$PUP" CHROME="$CHROME" DISPLAY="$DISPLAY" \
  "$NODE" eval_browser_groot.js > "$WORK/eval_browser.log" 2>&1
kill_chrome_pidfile "$WORK/eval_browser/eval_pid.txt" "chrome-udd"
grep -E "^ATT |EVAL_SUMMARY|HEALTH " "$WORK/eval_browser.log" 2>/dev/null | tail -20
[ -f "$WORK/eval_browser/eval_groot.json" ] || fail "no-browser-eval-json"
BR=$("$PYUR5E" -c "import json;d=json.load(open('$WORK/eval_browser/eval_groot.json'));print(d.get('success'),'lifted',d.get('n_lifted'),'seated',d.get('n_seated'))" 2>/dev/null || echo "?")
log "STEP6b BROWSER bare-base = $BR"

# ---- 7. results ------------------------------------------------------------
log "STEP7 write $RESULT"
collect_result DONE ""
cleanup
log "DONE — results in $RESULT"
echo "BASE_FIX DONE"
