#!/usr/bin/env bash
# ============================================================================
# run_dagger_browser.sh — SELF-DRIVING DAgger-on-browser. Launched DETACHED; needs
# ZERO further involvement. Takes the render-gap browser-BC GR00T (0/15 in-browser)
# and runs 3 corrective-expert DAgger iterations ON THE DEPLOYMENT (Three.js)
# RENDERER, then writes ~/dagger_browser_result.json + a DAGGER_BROWSER DONE/FAILED
# marker to $LOG.
#
# Per iteration i (each rolls out the PREVIOUS DAgger-browser checkpoint):
#   1. deploy current ckpt on the VM (ISOLATED ZMQ :5557 + bridge :5597); tunnel
#      LOCAL :5597 -> VM :5597; the dedicated agent-service on :8031
#      (GROOT_BRIDGE_URL=:5597) serves the ?agents=groot playground.
#   2. roll the current ckpt out IN-BROWSER over ~30 randomized vial poses, capturing
#      the visited-state Three.js frames + proprio + GT (dagger_rollout_browser.js).
#   3. IK-expert relabel each visited state -> LeRobot episodes; AGGREGATE onto the
#      running browser dataset (dagger_relabel_assemble.py relabel + merge).
#   4. GR00T fine-tune on the H100 (~12k steps, GPU-gated, from base on the aggregate).
#   5. deploy the new ckpt; IN-BROWSER eval (N>=15, identical seeded poses) -> rate.
#
# ISOLATION: v4's :5555/:5599 + the render-gap :5556/:5598 (VM side), ccproxy,
# gateway, :8000/:8010/:8021 (shared agent-services) are NEVER touched. OWN VM ckpt
# dir (ur5e_dagger_browser), OWN VM tmux (groot_dagger_*), OWN agent-service :8031,
# OWN tunnel :5597, OWN Chrome (unique --user-data-dir, killed by PID). NEVER
# pkill chrome. Every launch_finetune is gated on GPU free > 45 GB (do not fight v4).
#
# NOTE: uses `set -uo pipefail` + explicit fail() (NOT bare `set -e`) exactly like
# run_rendergap.sh, so a stray non-zero from a best-effort curl/grep cannot abort a
# multi-hour detached run; only the fail() checkpoints stop it, and even then a
# PARTIAL result JSON is written first.
# ============================================================================
set -uo pipefail

# ---- paths / config ---------------------------------------------------------
BC=/home/daniel/LovellAI/odyssey-ur5e/examples/ur5e-drugsort/browser_capture
ODY=/home/daniel/LovellAI/odyssey-ur5e
AGENT_DIR=/home/daniel/LovellAI/lai-agent-multiagent/agent_service
AGENT_PY=/home/daniel/LovellAI/lai-agent/agent_service/.venv/bin/python
PYUR5E=$ODY/.venv-ur5e/bin/python
VM=ubuntu@192.222.52.169
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30"
XML=/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/aseptipack_description/aseptipack.xml
PUP=/home/daniel/.npm/_npx/23232c69e5d221f3/node_modules/puppeteer-core
CHROME=/usr/bin/google-chrome-stable
NODE=/home/daniel/.nvm/versions/node/v22.22.0/bin/node

WORK=$HOME/dagger_browser
LOG=$HOME/dagger_browser.log
RESULT=$HOME/dagger_browser_result.json
mkdir -p "$WORK/run"

# Isolated ports / sessions / dirs
AGENT_PORT=${AGENT_PORT:-8031}
ZMQ_PORT=${ZMQ_PORT:-5557}
BRIDGE_PORT=${BRIDGE_PORT:-5597}
VM_DATASET=/home/ubuntu/ur5e_dagger_browser
VM_CKPT_ROOT=/home/ubuntu/ckpt/ur5e_dagger_browser
START_CKPT=/home/ubuntu/ckpt/ur5e_drugsort_browser/checkpoint-12000   # render-gap browser BC (0/15)
BASE_DATASET=$BC/dataset_browser_full                                 # 151 browser eps

# Knobs
ITERS=${ITERS:-3}
N_ROLL=${N_ROLL:-30}
MAX_TICKS=${MAX_TICKS:-400}
EVAL_N=${EVAL_N:-15}
EVAL_MAX_TICKS=${EVAL_MAX_TICKS:-900}
MAX_STEPS=${MAX_STEPS:-12000}
SAVE_STEPS=${SAVE_STEPS:-3000}
BATCH=${BATCH:-16}
N_ACTION_STEPS=${N_ACTION_STEPS:-8}
GPU_FREE_MIN=${GPU_FREE_MIN:-45000}

export DISPLAY=${DISPLAY:-:1}
cd "$BC"
exec >>"$LOG" 2>&1
echo "" ; echo "#################################################################"
log(){ echo "=== [dagger-browser] $* $(date -u +%FT%TZ) ==="; }

# ---- result collection (robust to partial completion) -----------------------
collect_result(){   # $1 = status (DONE|FAILED) ; $2 = failed_at (optional)
  local status="$1" failed_at="${2:-}"
  "$PYUR5E" - "$WORK" "$RESULT" "$status" "$failed_at" "$EVAL_N" "$START_CKPT" <<'PY' || true
import json, sys, glob, os
work, result, status, failed_at, eval_n, start_ckpt = sys.argv[1:7]
def L(p, d=None):
    try: return json.load(open(p))
    except Exception: return d
iters = []
for ep in sorted(glob.glob(os.path.join(work, "iter*_eval.json"))):
    i = int(os.path.basename(ep).split("_")[0].replace("iter", ""))
    ev = L(ep, {}) or {}
    roll = L(os.path.join(work, f"iter{i}_rollout.json"), {}) or {}
    meta = L(os.path.join(work, f"iter{i}_meta.json"), {}) or {}
    iters.append({
        "iter": i,
        "in_browser_success": ev.get("success"),
        "in_browser_rate": ev.get("success_rate"),
        "n_lifted": ev.get("n_lifted"), "n_seated": ev.get("n_seated"),
        "n_attempts": ev.get("n_attempts"),
        "rollout_success_while_gathering": f"{roll.get('rollout_success')}/{roll.get('num_episodes')}"
            if roll else None,
        "rollout_lifted_while_gathering": roll.get("rollout_lifted"),
        "aggregate_dataset_episodes": meta.get("aggregate_eps"),
        "checkpoint": meta.get("new_ckpt"),
    })
iters.sort(key=lambda r: r["iter"])
rated = [r for r in iters if isinstance(r.get("in_browser_rate"), (int, float))]
best = max(rated, key=lambda r: r["in_browser_rate"], default=None)
out = {
  "headline": {
    "in_browser_baseline_rendergap_bc": f"0/{eval_n}",
    "headless_reference": "2/20",
    "per_iteration": [
        {"iter": r["iter"], "in_browser_success": r["in_browser_success"],
         "in_browser_rate": r["in_browser_rate"], "n_lifted": r["n_lifted"],
         "n_seated": r["n_seated"],
         "rollout_success_while_gathering": r["rollout_success_while_gathering"]}
        for r in iters
    ],
    "best_in_browser": ({"iter": best["iter"], "success": best["in_browser_success"],
                         "rate": best["in_browser_rate"]} if best else None),
  },
  "start_checkpoint": start_ckpt,
  "iterations": iters,
  "status": status,
  "failed_at": failed_at or None,
}
json.dump(out, open(result, "w"), indent=2)
print("[result] " + json.dumps(out["headline"]))
PY
}
fail(){
  echo "=== [dagger-browser] FAILED at: $* $(date -u +%FT%TZ) ==="
  collect_result FAILED "$*"
  cleanup
  echo "DAGGER_BROWSER FAILED: $*"
  exit 1
}

# ---- cleanup (own agent-service + tunnel by PID/spec; VM sessions; NEVER pkill chrome) ----
cleanup(){
  log "cleanup (own agent-service + tunnel + VM dagger sessions; leave everything else)"
  if [ -f "$WORK/run/agent_pid.txt" ]; then
    ap=$(cat "$WORK/run/agent_pid.txt" 2>/dev/null || true)
    [ -n "${ap:-}" ] && kill "$ap" 2>/dev/null || true
  fi
  # belt-and-braces: also kill whatever still listens on our own agent port
  ss -tlnp 2>/dev/null | grep ":$AGENT_PORT " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | xargs -r kill 2>/dev/null || true
  pkill -f "127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT" 2>/dev/null || true
  $SSH "$VM" "tmux kill-session -t groot_dagger_server 2>/dev/null; tmux kill-session -t groot_dagger_bridge 2>/dev/null; tmux kill-session -t groot_dagger_train 2>/dev/null" 2>/dev/null || true
}
kill_chrome_pidfile(){   # $1 = pidfile written by the rollout/eval harness
  local pf="$1"
  [ -f "$pf" ] || return 0
  local cp; cp=$(cat "$pf" 2>/dev/null || true)
  if [ -n "${cp:-}" ] && kill -0 "$cp" 2>/dev/null; then
    if tr '\0' ' ' < "/proc/$cp/cmdline" 2>/dev/null | grep -q "chrome-udd-dagger\|chrome-udd-eval"; then
      echo "  [harness] kill Chrome PID $cp"; kill "$cp" 2>/dev/null || true
    fi
  fi
  rm -f "$pf" 2>/dev/null || true
}

gpu_gate(){
  local i   # keep the outer per-iteration loop counter from being clobbered
  log "gate on GPU free (> ${GPU_FREE_MIN} MiB; do not fight v4)"
  for i in $(seq 1 360); do
    FREE=$($SSH "$VM" "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" 2>/dev/null | head -1 | tr -d ' ')
    { [ -n "${FREE:-}" ] && [ "$FREE" -gt "$GPU_FREE_MIN" ]; } && { log "GPU free=${FREE}MiB — go"; return 0; }
    echo "  [gpu-gate] free=${FREE:-?}MiB waiting"; sleep 60
  done
  return 1
}

deploy_and_tunnel(){   # $1 = VM checkpoint dir
  local ckpt="$1"
  local i   # keep the outer per-iteration loop counter from being clobbered by the retry loops
  log "deploy $ckpt on VM (ZMQ :$ZMQ_PORT + bridge :$BRIDGE_PORT)"
  scp -q -o StrictHostKeyChecking=no vm_deploy_dagger.sh "$VM:/home/ubuntu/vm_deploy_dagger.sh" || return 1
  $SSH "$VM" "bash /home/ubuntu/vm_deploy_dagger.sh $ckpt $ZMQ_PORT $BRIDGE_PORT" || return 1
  pkill -f "127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT" 2>/dev/null || true; sleep 2
  $SSH -f -N -o ExitOnForwardFailure=yes -L 127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT "$VM" 2>/dev/null || true
  for i in $(seq 1 30); do curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health | grep -q ok && break; sleep 5; done
  curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health | grep -q ok || return 1
  # agent-service /api/groot/health reports EITHER {"bridge_health":{"ok":true}} when
  # the bridge is reachable, OR {"bridge_reachable":false,...} when it is not — there is
  # no "bridge_reachable":true. Treat "bridge_health" (or an explicit reachable:true) as up.
  for i in $(seq 1 24); do curl -s --max-time 5 http://127.0.0.1:$AGENT_PORT/api/groot/health | grep -qE '"bridge_health"|bridge_reachable" *: *true' && break; sleep 5; done
  curl -s --max-time 5 http://127.0.0.1:$AGENT_PORT/api/groot/health | grep -qE '"bridge_health"|bridge_reachable" *: *true' || return 1
  return 0
}

stop_vm_serving(){ log "stop VM dagger server/bridge (free GPU for training)"; \
  $SSH "$VM" "tmux kill-session -t groot_dagger_server 2>/dev/null; tmux kill-session -t groot_dagger_bridge 2>/dev/null" 2>/dev/null || true; sleep 3; }

# ============================================================================
log "START pid=$$ iters=$ITERS n_roll=$N_ROLL eval_n=$EVAL_N max_steps=$MAX_STEPS"
echo "DAGGER_BROWSER PID=$$"

# ---- 0a. Sanity: base dataset + start ckpt ----------------------------------
[ -f "$BASE_DATASET/meta/info.json" ] || fail "no-base-dataset($BASE_DATASET)"
$SSH "$VM" "[ -d $START_CKPT ]" || fail "no-start-ckpt($START_CKPT)"
$SSH "$VM" "mkdir -p $VM_CKPT_ROOT" || fail "vm-mkdir"

# ---- 0b. Precompute the FIXED eval poses (seed 7777) once -------------------
log "precompute eval poses (seed 7777, N=$EVAL_N)"
"$PYUR5E" precompute_plans.py --xml "$XML" --num-episodes "$EVAL_N" --seed 7777 \
  --out "$WORK/plans_eval.json" || fail "precompute-eval-plans"

# ---- 0c. Launch OWN agent-service on :$AGENT_PORT (GROOT_BRIDGE_URL=:$BRIDGE_PORT) ----
log "launch dedicated agent-service :$AGENT_PORT -> bridge :$BRIDGE_PORT"
# kill any stale listener on our port (non-shared port; safe)
STALE=$(ss -tlnp 2>/dev/null | grep ":$AGENT_PORT " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
[ -n "${STALE:-}" ] && { echo "  [agent] killing stale :$AGENT_PORT pid $STALE"; kill "$STALE" 2>/dev/null || true; sleep 2; }
cd "$AGENT_DIR"
env -u PYTHONPATH ENVIRONMENT=development \
    DATABASE_URL="sqlite+aiosqlite:///./dagger_agent${AGENT_PORT}.db" \
    GROOT_BRIDGE_URL="http://127.0.0.1:$BRIDGE_PORT" GROOT_OBSERVER_URL="http://127.0.0.1:5603" \
    DISPLAY="$DISPLAY" "$AGENT_PY" -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$AGENT_PORT" --loop asyncio > "$WORK/run/agent${AGENT_PORT}.log" 2>&1 &
echo $! > "$WORK/run/agent_pid.txt"   # real uvicorn python pid (env exec's into it)
cd "$BC"
sleep 12
for i in $(seq 1 20); do
  curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null | grep -q 200 && break
  sleep 3
done
curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null | grep -q 200 || fail "agent-service-not-serving"
log "agent-service :$AGENT_PORT serving (pid=$(cat "$WORK/run/agent_pid.txt" 2>/dev/null))"

# ---- Iterate ----------------------------------------------------------------
CUR_CKPT="$START_CKPT"
PREV_AGG="$BASE_DATASET"
for i in $(seq 1 "$ITERS"); do
  log "########## DAgger-browser ITERATION $i / $ITERS (rollout ckpt=$CUR_CKPT) ##########"
  ITDIR="$WORK/iter$i"; mkdir -p "$ITDIR"
  ROLL_OUT="$ITDIR/roll_out"; RELABEL_DS="$ITDIR/relabel"; AGG_DS="$WORK/dataset_dagger_iter$i"; EVAL_OUT="$ITDIR/eval"

  # 1. deploy current ckpt + tunnel + agent health
  deploy_and_tunnel "$CUR_CKPT" || fail "iter$i-deploy-current"

  # 2. precompute rollout poses (distinct seed per iter) + in-browser rollout+capture
  log "iter$i precompute rollout poses (N=$N_ROLL, seed=$((3000 + i * 100)))"
  "$PYUR5E" precompute_plans.py --xml "$XML" --num-episodes "$N_ROLL" --seed $((3000 + i * 100)) \
    --out "$ITDIR/plans_roll.json" || fail "iter$i-precompute-roll"
  log "iter$i in-browser rollout+capture (current ckpt) over $N_ROLL poses"
  PLANS="$ITDIR/plans_roll.json" OUT="$ROLL_OUT" PORT="$AGENT_PORT" N="$N_ROLL" \
    N_ACTION_STEPS="$N_ACTION_STEPS" MAX_TICKS="$MAX_TICKS" \
    PUPPETEER_CORE="$PUP" CHROME="$CHROME" DISPLAY="$DISPLAY" \
    "$NODE" dagger_rollout_browser.js 2>&1 | grep -vE '^\s*\[page\]' | tail -60
  kill_chrome_pidfile "$ROLL_OUT/rollout_pid.txt"
  cp -f "$ROLL_OUT/rollout_summary.json" "$WORK/iter${i}_rollout.json" 2>/dev/null || true
  NEP=$(ls -d "$ROLL_OUT"/raw/ep*/meta.json 2>/dev/null | wc -l)
  [ "$NEP" -ge 5 ] || fail "iter$i-too-few-rollout-eps($NEP)"
  log "iter$i captured $NEP rollout episodes"

  # 3. free GPU, relabel with IK expert, aggregate onto running browser dataset
  stop_vm_serving
  log "iter$i IK-expert relabel visited states -> $RELABEL_DS"
  "$PYUR5E" dagger_relabel_assemble.py relabel --xml "$XML" --raw "$ROLL_OUT/raw" --out "$RELABEL_DS" || fail "iter$i-relabel"
  log "iter$i aggregate: $PREV_AGG + relabel -> $AGG_DS"
  "$PYUR5E" dagger_relabel_assemble.py merge --base "$PREV_AGG" --add "$RELABEL_DS" --out "$AGG_DS" || fail "iter$i-merge"
  AGG_EPS=$("$PYUR5E" -c "import json;print(json.load(open('$AGG_DS/meta/info.json'))['total_episodes'])" 2>/dev/null || echo "?")
  log "iter$i aggregate dataset has $AGG_EPS episodes"

  # 4. rsync aggregate to VM + GPU-gated GR00T fine-tune
  log "iter$i rsync aggregate -> VM:$VM_DATASET"
  rsync -a --delete -e "$SSH" "$AGG_DS/" "$VM:$VM_DATASET/" || fail "iter$i-rsync"
  gpu_gate || fail "iter$i-gpu-gate-timeout"
  IT_CKPT="$VM_CKPT_ROOT/iter$i"
  log "iter$i launch GR00T fine-tune -> $IT_CKPT ($MAX_STEPS steps, batch $BATCH)"
  scp -q -o StrictHostKeyChecking=no vm_train_dagger.sh "$VM:/home/ubuntu/vm_train_dagger.sh" || fail "iter$i-scp-train"
  $SSH "$VM" "tmux kill-session -t groot_dagger_train 2>/dev/null; tmux new-session -d -s groot_dagger_train \
    'bash /home/ubuntu/vm_train_dagger.sh $VM_DATASET $IT_CKPT $MAX_STEPS $SAVE_STEPS $BATCH'" || fail "iter$i-launch-train"
  sleep 20
  $SSH "$VM" "tmux has-session -t groot_dagger_train 2>/dev/null" || \
    $SSH "$VM" "grep -q DAGGER_TRAIN_RC /home/ubuntu/train_dagger.log" 2>/dev/null || fail "iter$i-train-session-missing"
  log "iter$i wait for fine-tune (DAGGER_TRAIN_RC marker)"
  while true; do
    $SSH "$VM" "grep -q DAGGER_TRAIN_RC /home/ubuntu/train_dagger.log" 2>/dev/null && break
    if ! $SSH "$VM" "tmux has-session -t groot_dagger_train 2>/dev/null"; then
      sleep 10
      $SSH "$VM" "grep -q DAGGER_TRAIN_RC /home/ubuntu/train_dagger.log" 2>/dev/null && break || fail "iter$i-train-died-no-rc"
    fi
    sleep 120
  done
  TRC=$($SSH "$VM" "grep DAGGER_TRAIN_RC /home/ubuntu/train_dagger.log | tail -1 | sed 's/.*=//'" 2>/dev/null | tr -d ' ')
  log "iter$i fine-tune exited rc=$TRC"
  [ "$TRC" = "0" ] || fail "iter$i-train-rc=$TRC"
  NEW_CKPT="$IT_CKPT/checkpoint-$MAX_STEPS"
  $SSH "$VM" "[ -d $NEW_CKPT ]" || NEW_CKPT=$($SSH "$VM" "ls -d $IT_CKPT/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1")
  { [ -n "${NEW_CKPT:-}" ] && $SSH "$VM" "[ -d $NEW_CKPT ]"; } || fail "iter$i-no-checkpoint"
  log "iter$i new checkpoint = $NEW_CKPT"

  # 5. deploy new ckpt + in-browser eval (identical seeded poses)
  deploy_and_tunnel "$NEW_CKPT" || fail "iter$i-deploy-new"
  log "iter$i in-browser eval (?agents=groot, N=$EVAL_N, seed 7777)"
  PLANS="$WORK/plans_eval.json" OUT="$EVAL_OUT" PORT="$AGENT_PORT" AGENTS=groot N="$EVAL_N" \
    N_ACTION_STEPS="$N_ACTION_STEPS" MAX_TICKS="$EVAL_MAX_TICKS" \
    PUPPETEER_CORE="$PUP" CHROME="$CHROME" DISPLAY="$DISPLAY" \
    "$NODE" eval_browser_groot.js 2>&1 | grep -vE '^\s*\[page\]' | tail -40
  kill_chrome_pidfile "$EVAL_OUT/eval_pid.txt"
  cp -f "$EVAL_OUT/eval_groot.json" "$WORK/iter${i}_eval.json" 2>/dev/null || fail "iter$i-no-eval-json"
  "$PYUR5E" -c "import json;print('[iter$i] {\"iter\":$i,\"new_ckpt\":\"$NEW_CKPT\",\"aggregate_eps\":\"$AGG_EPS\"}')" >/dev/null 2>&1 || true
  printf '{"iter":%d,"new_ckpt":"%s","aggregate_eps":"%s"}\n' "$i" "$NEW_CKPT" "$AGG_EPS" > "$WORK/iter${i}_meta.json"
  RATE=$("$PYUR5E" -c "import json;d=json.load(open('$WORK/iter${i}_eval.json'));print(d.get('success'),d.get('success_rate'),'lifted',d.get('n_lifted'),'seated',d.get('n_seated'))" 2>/dev/null || echo "?")
  log "iter$i IN-BROWSER SUCCESS = $RATE"
  collect_result RUNNING "iter$i"    # refresh partial result after every iteration

  CUR_CKPT="$NEW_CKPT"
  PREV_AGG="$AGG_DS"
done

# ---- Done -------------------------------------------------------------------
log "write $RESULT"
collect_result DONE ""
cleanup
log "DONE — results in $RESULT"
echo "DAGGER_BROWSER DONE"
