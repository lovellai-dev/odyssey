#!/usr/bin/env bash
# ============================================================================
# run_observer_cond.sh — SELF-DRIVING Observer-conditioned GR00T (roadmap Step 2).
# Launched DETACHED; needs ZERO further involvement. Writes ~/observer_cond_result.json
# + an OBSERVER_COND DONE/FAILED marker to $LOG.
#
# WHY: render-gap + DAgger both scored 0/15 in-browser despite the Observer being
# sub-cm — the pixel-only pilot cannot infer DESCENT DEPTH from Three.js frames. This
# retrains GR00T with the Observer's 3D grasp target as an EXPLICIT state input
# (state 7 -> 10) and, at deploy, injects the live Observer target so the policy
# descends to a KNOWN vial position.
#
# CHAIN:
#  1. AUGMENT dataset_browser_full -> dataset_browser_cond (append cap_base 7->10 from
#     the GT sidecar, identical transform to the Observer's labels).
#  2. STAGE to VM: rsync dataset, scp 10-D modality config + 10-D-aware bridge + vm
#     train/deploy scripts.
#  3. Bring up the OWN serving stack: local Observer-conditioning sidecar (:5604,
#     RTX 3070) + OWN agent-service (:8032, GROOT_BRIDGE_URL -> the sidecar).
#  4. DEPLOY-VALIDATE end-to-end on a tiny SMOKE ckpt (200 steps): health green + a
#     real /get_action carrying the 10-D state -> a 6-D action. Fails fast if the
#     10-D path is wrong, BEFORE spending an hour on the full train.
#  5. FULL fine-tune (~12k steps, GPU-gated) on the conditioned dataset.
#  6. DEPLOY the full ckpt; IN-BROWSER eval (N>=15, ?agents=groot, seed 7777 — the
#     SAME poses as the 0/15 render-gap baseline) -> the honest success rate.
#  7. Write ~/observer_cond_result.json.
#
# ISOLATION: v4 (:5555/:5599/:8020-1), render-gap (:5556/:5598/:5601), dagger
# (:5557/:5597/:8031/:5603), ccproxy, gateway, :8000/:8010 are NEVER touched. OWN VM
# ckpt (ur5e_drugsort_obscond), OWN VM tmux (groot_obscond_*), OWN agent-service
# (:8032), OWN sidecar (:5604), OWN tunnel (:5596), OWN Chrome (unique --user-data-dir,
# killed by PID). NEVER pkill chrome. Every launch_finetune gated on GPU free > 45 GB.
#
# set -uo pipefail + explicit fail() (NOT bare set -e) so a stray non-zero from a
# best-effort curl/grep cannot abort the multi-hour detached run; only fail()
# checkpoints stop it, and each writes a PARTIAL result JSON first.
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
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30"
XML=/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/aseptipack_description/aseptipack.xml
PUP=/home/daniel/.npm/_npx/23232c69e5d221f3/node_modules/puppeteer-core
CHROME=/usr/bin/google-chrome-stable
NODE=/home/daniel/.nvm/versions/node/v22.22.0/bin/node

WORK=$HOME/observer_cond
LOG=$HOME/observer_cond.log
RESULT=$HOME/observer_cond_result.json
mkdir -p "$WORK/run"

# Isolated ports / sessions / dirs (all DISTINCT from v4 / render-gap / dagger)
AGENT_PORT=${AGENT_PORT:-8032}
ZMQ_PORT=${ZMQ_PORT:-5558}
BRIDGE_PORT=${BRIDGE_PORT:-5596}
COND_PORT=${COND_PORT:-5604}
VM_DATASET=/home/ubuntu/ur5e_drugsort_obscond
VM_CKPT_ROOT=/home/ubuntu/ckpt/ur5e_drugsort_obscond
VM_CFG=examples/UR5e_DrugSort/ur5e_config_obscond.py
OBS_WEIGHTS=$BC/percep_weights_browser
SRC_DATASET=$BC/dataset_browser_full
COND_DATASET=$BC/dataset_browser_cond

# Knobs
EVAL_N=${EVAL_N:-15}
EVAL_MAX_TICKS=${EVAL_MAX_TICKS:-900}
SMOKE_STEPS=${SMOKE_STEPS:-200}
MAX_STEPS=${MAX_STEPS:-12000}
SAVE_STEPS=${SAVE_STEPS:-3000}
BATCH=${BATCH:-16}
N_ACTION_STEPS=${N_ACTION_STEPS:-8}
GPU_FREE_MIN=${GPU_FREE_MIN:-45000}
ABLATION=${ABLATION:-1}          # also eval with the Observer frozen to a nominal target (isolates the target signal)

export DISPLAY=${DISPLAY:-:1}
cd "$BC"
exec >>"$LOG" 2>&1
echo "" ; echo "#################################################################"
log(){ echo "=== [obscond] $* $(date -u +%FT%TZ) ==="; }

# ---- result collection (robust to partial completion) -----------------------
collect_result(){   # $1 = status (RUNNING|DONE|FAILED) ; $2 = stage (optional)
  local status="$1" stage="${2:-}"
  "$PYUR5E" - "$WORK" "$RESULT" "$status" "$stage" "$EVAL_N" "$OBS_WEIGHTS/metrics.json" <<'PY' || true
import json, sys, os
work, result, status, stage, eval_n, metrics_path = sys.argv[1:7]
def L(p, d=None):
    try: return json.load(open(p))
    except Exception: return d
ev   = L(os.path.join(work, "eval_groot.json"), {}) or {}
abl  = L(os.path.join(work, "eval_nominal.json"), {}) or {}
dep  = L(os.path.join(work, "deploy_validation.json"), {}) or {}
om   = L(metrics_path, {}) or {}
tr   = L(os.path.join(work, "train.json"), {}) or {}
out = {
  "headline": {
    "in_browser_observer_conditioned_groot": ev.get("success"),
    "in_browser_observer_conditioned_rate": ev.get("success_rate"),
    "in_browser_baseline_rendergap_browser_groot": f"0/{eval_n}",
    "in_browser_baseline_dagger_best": f"0/{eval_n}",
    "headless_reference": "2/20",
    "observer_grasp_target_median_cm_threejs": (om.get("observer") or {}).get("median_cm"),
    "ablation_nominal_target": abl.get("success"),
    "n_lifted": ev.get("n_lifted"), "n_seated": ev.get("n_seated"),
  },
  "state_dim": "7 -> 10 (proprio 7 + grasp_target xyz 3, robot base frame)",
  "deploy_validation": dep,
  "train": tr,
  "eval_observer_conditioned": ev,
  "eval_ablation_nominal_target": abl if abl else None,
  "status": status,
  "stage": stage or None,
}
json.dump(out, open(result, "w"), indent=2)
print("[result] " + json.dumps(out["headline"]))
PY
}
fail(){
  echo "=== [obscond] FAILED at: $* $(date -u +%FT%TZ) ==="
  collect_result FAILED "$*"
  cleanup
  echo "OBSERVER_COND FAILED: $*"
  exit 1
}

# ---- cleanup (own agent-service + sidecar + tunnel + VM sessions; NEVER pkill chrome) ----
cleanup(){
  log "cleanup (own agent-service + conditioner + tunnel + VM obscond sessions; leave everything else)"
  for pf in "$WORK/run/agent_pid.txt" "$WORK/run/cond_pid.txt"; do
    [ -f "$pf" ] || continue
    p=$(cat "$pf" 2>/dev/null || true); [ -n "${p:-}" ] && kill "$p" 2>/dev/null || true
  done
  ss -tlnp 2>/dev/null | grep -E ":($AGENT_PORT|$COND_PORT) " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | xargs -r kill 2>/dev/null || true
  pkill -f "127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT" 2>/dev/null || true
  $SSH "$VM" "tmux kill-session -t groot_obscond_server 2>/dev/null; tmux kill-session -t groot_obscond_bridge 2>/dev/null; tmux kill-session -t groot_obscond_train 2>/dev/null" 2>/dev/null || true
}
kill_chrome_pidfile(){   # $1 = pidfile written by the eval harness
  local pf="$1"; [ -f "$pf" ] || return 0
  local cp; cp=$(cat "$pf" 2>/dev/null || true)
  if [ -n "${cp:-}" ] && kill -0 "$cp" 2>/dev/null; then
    if tr '\0' ' ' < "/proc/$cp/cmdline" 2>/dev/null | grep -q "chrome-udd-eval"; then
      echo "  [harness] kill Chrome PID $cp"; kill "$cp" 2>/dev/null || true
    fi
  fi
  rm -f "$pf" 2>/dev/null || true
}
gpu_gate(){
  local i
  log "gate on GPU free (> ${GPU_FREE_MIN} MiB; do not fight other GPU users)"
  for i in $(seq 1 360); do
    FREE=$($SSH "$VM" "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" 2>/dev/null | head -1 | tr -d ' ')
    { [ -n "${FREE:-}" ] && [ "$FREE" -gt "$GPU_FREE_MIN" ]; } && { log "GPU free=${FREE}MiB — go"; return 0; }
    echo "  [gpu-gate] free=${FREE:-?}MiB waiting"; sleep 60
  done
  return 1
}
train_wait(){   # $1 = ckpt dir ; $2 = max_steps ; $3 = save_steps ; $4 = tag(for log)
  local ckpt="$1" steps="$2" save="$3" tag="$4" i
  gpu_gate || return 1
  log "$tag launch GR00T fine-tune -> $ckpt ($steps steps, save $save, batch $BATCH)"
  $SSH "$VM" "tmux kill-session -t groot_obscond_train 2>/dev/null; tmux new-session -d -s groot_obscond_train \
    'bash /home/ubuntu/vm_train_obscond.sh $VM_DATASET $ckpt $steps $save $BATCH $VM_CFG'" || return 1
  sleep 20
  $SSH "$VM" "tmux has-session -t groot_obscond_train 2>/dev/null" || \
    $SSH "$VM" "grep -q OBSCOND_TRAIN_RC /home/ubuntu/train_obscond.log" 2>/dev/null || return 1
  log "$tag wait for fine-tune (OBSCOND_TRAIN_RC marker)"
  while true; do
    $SSH "$VM" "grep -q OBSCOND_TRAIN_RC /home/ubuntu/train_obscond.log" 2>/dev/null && break
    if ! $SSH "$VM" "tmux has-session -t groot_obscond_train 2>/dev/null"; then
      sleep 10
      $SSH "$VM" "grep -q OBSCOND_TRAIN_RC /home/ubuntu/train_obscond.log" 2>/dev/null && break || return 2
    fi
    sleep 120
  done
  local rc; rc=$($SSH "$VM" "grep OBSCOND_TRAIN_RC /home/ubuntu/train_obscond.log | tail -1 | sed 's/.*=//'" 2>/dev/null | tr -d ' ')
  log "$tag fine-tune exited rc=$rc"
  [ "$rc" = "0" ] || return 3
  return 0
}
deploy_and_tunnel(){   # $1 = VM checkpoint dir
  local ckpt="$1" i
  log "deploy $ckpt on VM (ZMQ :$ZMQ_PORT + 10-D bridge :$BRIDGE_PORT)"
  $SSH "$VM" "bash /home/ubuntu/vm_deploy_obscond.sh $ckpt $ZMQ_PORT $BRIDGE_PORT" || return 1
  pkill -f "127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT" 2>/dev/null || true; sleep 2
  $SSH -f -N -o ExitOnForwardFailure=yes -L 127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT "$VM" 2>/dev/null || true
  for i in $(seq 1 30); do curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health | grep -q ok && break; sleep 5; done
  curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health | grep -q ok || return 1
  # conditioner (:$COND_PORT) forwards to the tunnel; its /health mirrors the bridge.
  for i in $(seq 1 24); do curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health | grep -q '"ok": *true' && break; sleep 5; done
  curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health | grep -q '"ok": *true' || return 1
  # agent-service /api/groot/health (routes through the conditioner) must report up.
  for i in $(seq 1 24); do curl -s --max-time 5 http://127.0.0.1:$AGENT_PORT/api/groot/health | grep -qE '"bridge_health"|bridge_reachable" *: *true' && break; sleep 5; done
  curl -s --max-time 5 http://127.0.0.1:$AGENT_PORT/api/groot/health | grep -qE '"bridge_health"|bridge_reachable" *: *true' || return 1
  return 0
}
stop_vm_serving(){ log "stop VM obscond server/bridge (free GPU for training)"; \
  $SSH "$VM" "tmux kill-session -t groot_obscond_server 2>/dev/null; tmux kill-session -t groot_obscond_bridge 2>/dev/null" 2>/dev/null || true; sleep 3; }

# ============================================================================
log "START pid=$$ eval_n=$EVAL_N smoke_steps=$SMOKE_STEPS max_steps=$MAX_STEPS"
echo "OBSERVER_COND PID=$$"

# ---- 0. sanity -------------------------------------------------------------
[ -f "$SRC_DATASET/meta/info.json" ] || fail "no-src-dataset($SRC_DATASET)"
[ -f "$OBS_WEIGHTS/observer_head.pt" ] || fail "no-observer-weights($OBS_WEIGHTS)"
$SSH "$VM" "echo ok >/dev/null" || fail "vm-unreachable"
$SSH "$VM" "mkdir -p $VM_CKPT_ROOT" || fail "vm-mkdir"

# ---- 1. augment the dataset (7 -> 10) --------------------------------------
log "STEP1 augment $SRC_DATASET -> $COND_DATASET (append grasp_target from GT sidecar)"
MUJOCO_GL=egl "$PYUR5E" augment_state_grasp_target.py --src "$SRC_DATASET" --out "$COND_DATASET" --verify || fail "augment"
STATE_DIM=$("$PYUR5E" -c "import json;print(json.load(open('$COND_DATASET/meta/info.json'))['features']['observation.state']['shape'][0])" 2>/dev/null)
[ "$STATE_DIM" = "10" ] || fail "augment-bad-state-dim($STATE_DIM)"

# ---- 2. stage to VM: dataset, 10-D config, 10-D bridge, vm scripts ----------
log "STEP2 rsync conditioned dataset -> VM:$VM_DATASET"
rsync -a --delete -e "$SSH" "$COND_DATASET/" "$VM:$VM_DATASET/" || fail "rsync-dataset"
log "STEP2 scp 10-D modality config -> VM Isaac-GR00T/$VM_CFG"
$SSH "$VM" "mkdir -p /home/ubuntu/Isaac-GR00T/examples/UR5e_DrugSort" || true
scp -q -o StrictHostKeyChecking=no "$ODY/examples/ur5e-drugsort/ur5e_config.py" "$VM:/home/ubuntu/Isaac-GR00T/$VM_CFG" || fail "scp-config"
scp -q -o StrictHostKeyChecking=no "$ODY/scripts/serve_groot_http_bridge.py" "$VM:/home/ubuntu/serve_groot_http_bridge_obscond.py" || fail "scp-bridge"
scp -q -o StrictHostKeyChecking=no vm_train_obscond.sh vm_deploy_obscond.sh "$VM:/home/ubuntu/" || fail "scp-vm-scripts"

# ---- 3. bring up the OWN serving stack (conditioner + agent-service) --------
# 3a. local Observer-conditioning sidecar (:$COND_PORT) -> tunnel :$BRIDGE_PORT
log "STEP3 start local Observer-conditioning sidecar :$COND_PORT (RTX 3070)"
STALE=$(ss -tlnp 2>/dev/null | grep ":$COND_PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
[ -n "${STALE:-}" ] && { kill "$STALE" 2>/dev/null || true; sleep 2; }
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OBSERVER_WEIGHTS="$OBS_WEIGHTS" \
  CONDITION_BRIDGE_URL="http://127.0.0.1:$BRIDGE_PORT" OBSERVER_DEVICE=cuda PYTHONPATH="$ODY/src" \
  setsid "$OBSPY" "$ODY/scripts/serve_observer_conditioning.py" --port "$COND_PORT" \
  >"$WORK/run/conditioner.log" 2>&1 &
echo $! > "$WORK/run/cond_pid.txt"
for i in $(seq 1 40); do curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health >/dev/null 2>&1 && break; sleep 3; done
curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health | grep -q observer_ready || fail "conditioner-not-up"
log "conditioner up (pid=$(cat "$WORK/run/cond_pid.txt"))"

# 3b. OWN agent-service (:$AGENT_PORT) -> conditioner (:$COND_PORT)
log "STEP3 launch OWN agent-service :$AGENT_PORT (GROOT_BRIDGE_URL + GROOT_STATE_CONDITIONER_URL -> conditioner :$COND_PORT)"
STALE=$(ss -tlnp 2>/dev/null | grep ":$AGENT_PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
[ -n "${STALE:-}" ] && { kill "$STALE" 2>/dev/null || true; sleep 2; }
cd "$AGENT_DIR"
env -u PYTHONPATH ENVIRONMENT=development \
    DATABASE_URL="sqlite+aiosqlite:///./obscond_agent${AGENT_PORT}.db" \
    GROOT_BRIDGE_URL="http://127.0.0.1:$COND_PORT" \
    GROOT_STATE_CONDITIONER_URL="http://127.0.0.1:$COND_PORT" \
    GROOT_OBSERVER_URL="http://127.0.0.1:$COND_PORT" \
    DISPLAY="$DISPLAY" "$AGENT_PY" -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$AGENT_PORT" --loop asyncio > "$WORK/run/agent${AGENT_PORT}.log" 2>&1 &
echo $! > "$WORK/run/agent_pid.txt"
cd "$BC"
sleep 12
for i in $(seq 1 20); do
  curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null | grep -q 200 && break
  sleep 3
done
curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null | grep -q 200 || fail "agent-service-not-serving"
log "agent-service :$AGENT_PORT serving (pid=$(cat "$WORK/run/agent_pid.txt"))"

# ---- 4. DEPLOY-VALIDATION on a tiny SMOKE ckpt (before the full run) --------
log "STEP4 smoke fine-tune ($SMOKE_STEPS steps) to validate the 10-D deploy path"
train_wait "$VM_CKPT_ROOT/smoke" "$SMOKE_STEPS" "$SMOKE_STEPS" "smoke"
rc=$?; [ $rc -eq 0 ] || fail "smoke-train(rc=$rc)"
SMOKE_CKPT="$VM_CKPT_ROOT/smoke/checkpoint-$SMOKE_STEPS"
$SSH "$VM" "[ -d $SMOKE_CKPT ]" || SMOKE_CKPT=$($SSH "$VM" "ls -d $VM_CKPT_ROOT/smoke/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1")
{ [ -n "${SMOKE_CKPT:-}" ] && $SSH "$VM" "[ -d $SMOKE_CKPT ]"; } || fail "no-smoke-ckpt"
stop_vm_serving
deploy_and_tunnel "$SMOKE_CKPT" || fail "smoke-deploy"

log "STEP4 real /get_action with a 10-D-conditioned state through agent :$AGENT_PORT"
"$PYUR5E" - "$AGENT_PORT" "$WORK/deploy_validation.json" <<'PY' || fail "deploy-validation-request"
import base64, io, json, sys, urllib.request
port, outp = sys.argv[1], sys.argv[2]
from PIL import Image
buf = io.BytesIO(); Image.new("RGB", (256, 256), (127, 127, 127)).save(buf, format="PNG")
img = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
body = json.dumps({
    "image_b64": img, "image_b64_wrist": img,
    "state": [-1.57, -1.56, 1.58, -1.57, -1.57, 0.0, 0.05],   # 7-D proprio; conditioner appends the target
    "instruction": "pick up the vial and place it in the rack", "sid": "deploy-validate",
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/api/groot/get_action", data=body,
                             headers={"content-type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=180) as r:
    res = json.loads(r.read().decode())
ok = isinstance(res.get("q"), list) and len(res["q"]) == 6 and isinstance(res.get("grip"), (int, float))
tgt = res.get("grasp_target"); cond = res.get("conditioner")
val = {"ok": bool(ok), "action_q_len": len(res.get("q") or []), "grip": res.get("grip"),
       "observer_target_injected": tgt, "conditioner": cond}
json.dump(val, open(outp, "w"), indent=2)
print("[deploy-validate] " + json.dumps(val))
if not ok: raise SystemExit("bad /get_action response: " + json.dumps(res)[:300])
if not (isinstance(tgt, list) and len(tgt) == 3): raise SystemExit("no 3-D observer target injected")
PY
collect_result RUNNING "deploy-validated"
log "STEP4 OBSCOND_DEPLOY_VALIDATED — 10-D conditioned deploy path is live"
echo "OBSCOND_DEPLOY_VALIDATED"

# ---- 5. FULL fine-tune on the conditioned dataset --------------------------
log "STEP5 full fine-tune ($MAX_STEPS steps) on $VM_DATASET"
stop_vm_serving
train_wait "$VM_CKPT_ROOT/full" "$MAX_STEPS" "$SAVE_STEPS" "full"
rc=$?; [ $rc -eq 0 ] || fail "full-train(rc=$rc)"
FULL_CKPT="$VM_CKPT_ROOT/full/checkpoint-$MAX_STEPS"
$SSH "$VM" "[ -d $FULL_CKPT ]" || FULL_CKPT=$($SSH "$VM" "ls -d $VM_CKPT_ROOT/full/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1")
{ [ -n "${FULL_CKPT:-}" ] && $SSH "$VM" "[ -d $FULL_CKPT ]"; } || fail "no-full-ckpt"
printf '{"checkpoint":"%s","max_steps":%s,"dataset":"%s","state_dim":10,"conditioned_eps":151}\n' \
  "$FULL_CKPT" "$MAX_STEPS" "$VM_DATASET" > "$WORK/train.json"
log "full checkpoint = $FULL_CKPT"

# ---- 6. deploy full ckpt + IN-BROWSER eval (?agents=groot, seed 7777) -------
stop_vm_serving
deploy_and_tunnel "$FULL_CKPT" || fail "full-deploy"
log "STEP6 precompute identical eval poses (seed 7777, N=$EVAL_N)"
"$PYUR5E" precompute_plans.py --xml "$XML" --num-episodes "$EVAL_N" --seed 7777 --out "$WORK/plans_eval.json" || fail "precompute-eval"

log "STEP6 IN-BROWSER eval — Observer-conditioned GR00T (?agents=groot, N=$EVAL_N)"
PLANS="$WORK/plans_eval.json" OUT="$WORK/eval_out" PORT="$AGENT_PORT" AGENTS=groot N="$EVAL_N" \
  N_ACTION_STEPS="$N_ACTION_STEPS" MAX_TICKS="$EVAL_MAX_TICKS" \
  PUPPETEER_CORE="$PUP" CHROME="$CHROME" DISPLAY="$DISPLAY" \
  "$NODE" eval_browser_groot.js 2>&1 | grep -vE '^\s*\[page\]' | tail -40
kill_chrome_pidfile "$WORK/eval_out/eval_pid.txt"
cp -f "$WORK/eval_out/eval_groot.json" "$WORK/eval_groot.json" 2>/dev/null || fail "no-eval-json"
RATE=$("$PYUR5E" -c "import json;d=json.load(open('$WORK/eval_groot.json'));print(d.get('success'),d.get('success_rate'),'lifted',d.get('n_lifted'),'seated',d.get('n_seated'))" 2>/dev/null || echo "?")
log "IN-BROWSER OBSERVER-CONDITIONED SUCCESS = $RATE"
collect_result RUNNING "eval-observer-conditioned-done"

# ---- 6b. ABLATION: freeze the Observer to a nominal target (isolate the signal) ----
if [ "$ABLATION" = "1" ]; then
  log "STEP6b ablation: restart conditioner with FREEZE_TARGET=1 (nominal target every tick)"
  ap=$(cat "$WORK/run/cond_pid.txt" 2>/dev/null || true); [ -n "${ap:-}" ] && kill "$ap" 2>/dev/null || true; sleep 2
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OBSERVER_WEIGHTS="$OBS_WEIGHTS" \
    CONDITION_BRIDGE_URL="http://127.0.0.1:$BRIDGE_PORT" OBSERVER_DEVICE=cuda PYTHONPATH="$ODY/src" \
    GRASP_TARGET_FALLBACK="-0.425,-0.015,0.272" CONDITIONER_FREEZE_NOMINAL=1 \
    setsid "$OBSPY" "$ODY/scripts/serve_observer_conditioning.py" --port "$COND_PORT" \
    >"$WORK/run/conditioner_nominal.log" 2>&1 &
  echo $! > "$WORK/run/cond_pid.txt"
  for i in $(seq 1 40); do curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health | grep -q observer_ready && break; sleep 3; done
  if curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health | grep -q observer_ready; then
    PLANS="$WORK/plans_eval.json" OUT="$WORK/eval_out_nominal" PORT="$AGENT_PORT" AGENTS=groot N="$EVAL_N" \
      N_ACTION_STEPS="$N_ACTION_STEPS" MAX_TICKS="$EVAL_MAX_TICKS" \
      PUPPETEER_CORE="$PUP" CHROME="$CHROME" DISPLAY="$DISPLAY" \
      "$NODE" eval_browser_groot.js 2>&1 | grep -vE '^\s*\[page\]' | tail -20 || true
    kill_chrome_pidfile "$WORK/eval_out_nominal/eval_pid.txt"
    cp -f "$WORK/eval_out_nominal/eval_groot.json" "$WORK/eval_nominal.json" 2>/dev/null || true
    ARATE=$("$PYUR5E" -c "import json;d=json.load(open('$WORK/eval_nominal.json'));print(d.get('success'))" 2>/dev/null || echo "?")
    log "ablation (nominal target) SUCCESS = $ARATE"
  else
    log "ablation conditioner did not come up — skipping (non-fatal)"
  fi
fi

# ---- 7. results ------------------------------------------------------------
log "STEP7 write $RESULT"
collect_result DONE ""
cleanup
log "DONE — results in $RESULT"
echo "OBSERVER_COND DONE"
