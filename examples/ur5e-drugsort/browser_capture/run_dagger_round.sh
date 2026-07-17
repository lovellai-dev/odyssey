#!/usr/bin/env bash
# ============================================================================
# run_dagger_round.sh — SELF-DRIVING Stage-C noise-space DAgger round.
#
# The v0.2 steered policy reliably reaches a 1.6-5.8cm hover band above the
# vial but can't finish (0/15, median pad 4.4cm, 76% of ticks in promoted-GRASP
# hover) — a state distribution only the steered policy visits. This round:
#   1. collects N steered rollouts on FRESH poses (seed 4242; eval seed 7777
#      stays untouched) through the full deploy stack, recording every visited
#      state at 20 Hz (frames + proprio + GT),
#   2. relabels each visited state with the IK expert -> LeRobot dataset
#      (+ meta/gt sidecar) -> 10-D augment (grasp target),
#   3. rsyncs to the VM and inverts the corrective chunks -> a SECOND shard
#      dir (steering_targets_dagger<R>),
#   4. retrains multi-source (base 17k + dagger, --source-boost), offline gate,
#   5. browser A/B via run_stageb_ab.sh (RUN_TAG=dagger<R>).
#
# MINI=1 runs the 3-episode end-to-end smoke (no browser A/B; stops after the
# offline gate) — MANDATORY before the first full round.
# Markers: DAGGER<R> DONE | FAILED: <stage> in ~/dagger<R>.log.
# ============================================================================
set -uo pipefail

BC=/home/daniel/LovellAI/odyssey-ur5e/examples/ur5e-drugsort/browser_capture
ODY=/home/daniel/LovellAI/odyssey-ur5e
AGENT_DIR=/home/daniel/LovellAI/lai-agent-multiagent/agent_service
AGENT_PY=/home/daniel/LovellAI/lai-agent/agent_service/.venv/bin/python
PYUR5E=$ODY/.venv-ur5e/bin/python
OBSPY=/home/daniel/Isaac-GR00T/.venv/bin/python
VM=ubuntu@192.222.52.169
SSH="ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30"
GPY=/home/ubuntu/Isaac-GR00T/.venv/bin/python
XML=/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/aseptipack_description/aseptipack.xml
PUP=/home/daniel/.npm/_npx/23232c69e5d221f3/node_modules/puppeteer-core
CHROME=/usr/bin/google-chrome-stable
NODE=/home/daniel/.nvm/versions/node/v22.22.0/bin/node

ROUND=${ROUND:-1}
MINI=${MINI:-0}
TAG=dagger${ROUND}${MINI:+}; [ "$MINI" = "1" ] && TAG=dagger${ROUND}mini
WORK=$HOME/$TAG
LOG=$HOME/$TAG.log
RESULT=$HOME/${TAG}_result.json
N_ROLL=${N_ROLL:-20}; [ "$MINI" = "1" ] && N_ROLL=3
ROLL_SEED=${ROLL_SEED:-$((4242 + (ROUND - 1) * 100))}
STEER_SRC=${STEER_SRC:-/home/ubuntu/steering_net_v02.npz}
BOOST=${BOOST:-1.0,4.0}
VM_SHARDS_BASE=/home/ubuntu/steering_targets
VM_SHARDS_DAG=/home/ubuntu/steering_targets_dagger${ROUND}
VM_DS_DAG=/home/ubuntu/ur5e_dagger${ROUND}_cond
VM_NET=/home/ubuntu/steering_net_dagger${ROUND}.npz
MAX_TICKS=${MAX_TICKS:-900}

AGENT_PORT=8032
ZMQ_PORT=5558
BRIDGE_PORT=5596
COND_PORT=5604
CKPT=/home/ubuntu/ckpt/ur5e_drugsort_obscond/full/checkpoint-12000
OBS_WEIGHTS=$BC/percep_weights_browser

export DISPLAY=${DISPLAY:-:1}
# RESUME=1 keeps already-collected rollout episodes (per-episode meta.json is
# the completion marker) — a Chrome crash never costs more than one episode.
[ "${RESUME:-0}" = "1" ] || rm -rf "$WORK"
mkdir -p "$WORK"
cd "$BC"
exec >>"$LOG" 2>&1
echo "" ; echo "#################################################################"
log(){ echo "=== [$TAG] $* $(date -u +%FT%TZ) ==="; }

cleanup(){
  log "cleanup (own agent/conditioner/tunnel/VM obscond serving; leave everything else)"
  for pf in "$WORK/agent_pid.txt" "$WORK/cond_pid.txt"; do
    [ -f "$pf" ] || continue
    p=$(cat "$pf" 2>/dev/null || true); [ -n "${p:-}" ] && kill "$p" 2>/dev/null || true
  done
  ss -tlnp 2>/dev/null | grep -E ":($AGENT_PORT|$COND_PORT) " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | xargs -r kill 2>/dev/null || true
  pkill -f "127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT" 2>/dev/null || true
  $SSH "$VM" "tmux kill-session -t groot_obscond_server 2>/dev/null; tmux kill-session -t groot_obscond_bridge 2>/dev/null" 2>/dev/null || true
}
kill_chrome_pidfile(){
  local pf="$1"; [ -f "$pf" ] || return 0
  local cp; cp=$(cat "$pf" 2>/dev/null || true)
  if [ -n "${cp:-}" ] && kill -0 "$cp" 2>/dev/null; then
    if tr '\0' ' ' < "/proc/$cp/cmdline" 2>/dev/null | grep -q "chrome-udd"; then
      echo "  [$TAG] kill Chrome PID $cp"; kill "$cp" 2>/dev/null || true
    fi
  fi
  rm -f "$pf" 2>/dev/null || true
}
fail(){
  trap - TERM INT HUP
  log "FAILED at: $*"
  printf '{"round":%s,"mini":%s,"status":"FAILED","failed_at":"%s"}\n' "$ROUND" "$MINI" "$*" > "$RESULT"
  kill_chrome_pidfile "$WORK/roll_out/rollout_pid.txt"
  cleanup
  echo "${TAG^^} FAILED: $*"
  exit 1
}
trap 'fail "signal"' TERM INT HUP

log "START pid=$$ round=$ROUND mini=$MINI n_roll=$N_ROLL seed=$ROLL_SEED steer=$STEER_SRC boost=$BOOST"
echo "${TAG^^} PID=$$"

# ---- 0. sanity -----------------------------------------------------------------
[ -f "$OBS_WEIGHTS/observer_head.pt" ] || fail "no-observer-weights"
[ -f "$XML" ] || fail "no-xml"
$SSH "$VM" "echo ok >/dev/null" || fail "vm-unreachable"
$SSH "$VM" "[ -d $CKPT ] && [ -f $STEER_SRC ] && [ -d $VM_SHARDS_BASE ]" || fail "vm-assets"
FREE=$($SSH "$VM" "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" | head -1 | tr -d ' ')
[ -n "${FREE:-}" ] && [ "$FREE" -gt 20000 ] || fail "gpu-not-free(${FREE:-?}MiB)"
for port in $AGENT_PORT $COND_PORT; do
  ss -tln 2>/dev/null | grep -q ":$port " && fail "port-busy($port)"
done
ss -tln 2>/dev/null | grep -q ":$BRIDGE_PORT " && { pkill -f "127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT" 2>/dev/null || true; sleep 2; }

# ---- 1. stage current scripts + VM serving + tunnel --------------------------------
log "STEP1 scp current trainer/inverter/gate to VM (Stage-C multi-source trainer)"
scp -q -o StrictHostKeyChecking=no "$ODY/scripts/train_steering_ur5e.py" \
  "$ODY/scripts/flow_inverter_groot.py" "$ODY/scripts/probe_flow_inversion_groot.py" \
  "$ODY/scripts/flowdagger_offline_gate_ur5e.py" "$VM:/home/ubuntu/" || fail "scp-scripts"
log "STEP1 deploy obscond ckpt (ZMQ :$ZMQ_PORT + bridge :$BRIDGE_PORT) + tunnel"
$SSH "$VM" "bash /home/ubuntu/vm_deploy_obscond.sh $CKPT $ZMQ_PORT $BRIDGE_PORT" || fail "vm-deploy"
$SSH -f -N -o ExitOnForwardFailure=yes -L 127.0.0.1:$BRIDGE_PORT:127.0.0.1:$BRIDGE_PORT "$VM" || fail "tunnel"
for i in $(seq 1 30); do curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health | grep -q '"ok": *true' && break; sleep 5; done
curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health | grep -q '"ok": *true' || fail "bridge-health"

# ---- 2. steered sidecar + agent-service -------------------------------------------
log "STEP2 conditioner+steering sidecar :$COND_PORT (weights <- VM $STEER_SRC)"
scp -q -o StrictHostKeyChecking=no "$VM:$STEER_SRC" "$WORK/steering_net.npz" || fail "fetch-steer-weights"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OBSERVER_WEIGHTS="$OBS_WEIGHTS" \
  CONDITION_BRIDGE_URL="http://127.0.0.1:$BRIDGE_PORT" OBSERVER_DEVICE=cuda PYTHONPATH="$ODY/src" \
  STEERING_WEIGHTS="$WORK/steering_net.npz" STEERING_FK_XML="$XML" \
  setsid "$OBSPY" "$ODY/scripts/serve_observer_conditioning.py" --port "$COND_PORT" \
  >"$WORK/conditioner.log" 2>&1 &
echo $! > "$WORK/cond_pid.txt"
for i in $(seq 1 40); do curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health 2>/dev/null | grep -q '"ok": *true' && break; sleep 3; done
CH=$(curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health || true)
{ echo "$CH" | grep -q '"observer_ready": *true' && echo "$CH" | grep -q '"ok": *true'; } || fail "conditioner-not-up($CH)"

log "STEP2 agent-service :$AGENT_PORT"
cd "$AGENT_DIR"
env -u PYTHONPATH ENVIRONMENT=development \
    DATABASE_URL="sqlite+aiosqlite:///./dagger_agent${AGENT_PORT}.db" \
    GROOT_BRIDGE_URL="http://127.0.0.1:$COND_PORT" \
    GROOT_STATE_CONDITIONER_URL="http://127.0.0.1:$COND_PORT" \
    GROOT_OBSERVER_URL="http://127.0.0.1:$COND_PORT" \
    DISPLAY="$DISPLAY" "$AGENT_PY" -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$AGENT_PORT" --loop asyncio > "$WORK/agent${AGENT_PORT}.log" 2>&1 &
echo $! > "$WORK/agent_pid.txt"
cd "$BC"
for i in $(seq 1 20); do
  curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null | grep -q 200 && break
  sleep 3
done
curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null | grep -q 200 || fail "agent-not-serving"
curl -s --max-time 8 "http://127.0.0.1:$AGENT_PORT/api/groot/health" | grep -q '"ok": *true' || fail "agent-groot-health"
log "stack green (STEERED)"

# ---- 3. steered rollout collection -------------------------------------------------
log "STEP3 precompute rollout poses (N=$N_ROLL seed=$ROLL_SEED) + steered rollouts (per-episode Chrome)"
[ -f "$WORK/plans_roll.json" ] || "$PYUR5E" precompute_plans.py --xml "$XML" \
  --num-episodes "$N_ROLL" --seed "$ROLL_SEED" --out "$WORK/plans_roll.json" || fail "precompute-roll"
# One Chrome session PER EPISODE: the first full-round attempt died at ep003
# when a single long session accumulated memory (Target closed). Per-episode
# sessions bound the blast radius to one episode, enable resume, and reset
# renderer memory every ~7 min. Retry each episode once with a fresh Chrome.
ep_i=0
while [ "$ep_i" -lt "$N_ROLL" ]; do
  epd=$(printf "ep%03d" "$ep_i")
  if [ -f "$WORK/roll_out/raw/$epd/meta.json" ]; then
    log "$epd already collected — skip"
    ep_i=$((ep_i + 1)); continue
  fi
  "$PYUR5E" -c "import json;d=json.load(open('$WORK/plans_roll.json'));json.dump({'plans':[d['plans'][$ep_i]]},open('$WORK/plan_one.json','w'))" || fail "slice-plan-$epd"
  ok=0
  for attempt in 1 2; do
    PLANS="$WORK/plan_one.json" OUT="$WORK/roll_out" PORT="$AGENT_PORT" N=1 \
      N_ACTION_STEPS=8 MAX_TICKS="$MAX_TICKS" KEEP_RAW=1 \
      PUPPETEER_CORE="$PUP" CHROME="$CHROME" DISPLAY="$DISPLAY" \
      "$NODE" dagger_rollout_browser.js >> "$WORK/rollout_node.log" 2>&1
    rc=$?
    kill_chrome_pidfile "$WORK/roll_out/rollout_pid.txt"
    if [ "$rc" -eq 0 ] && [ -f "$WORK/roll_out/raw/$epd/meta.json" ]; then ok=1; break; fi
    log "$epd attempt $attempt failed (rc=$rc) — fresh Chrome retry"
    sleep 5
  done
  [ "$ok" = "1" ] || fail "rollout-$epd"
  log "$epd collected ($(grep -cE '^ROLL' "$WORK/rollout_node.log") total)"
  ep_i=$((ep_i + 1))
done
NEPS=$(ls -d "$WORK/roll_out/raw"/ep*/meta.json 2>/dev/null | wc -l)
[ "$NEPS" -ge "$N_ROLL" ] || fail "incomplete-collection($NEPS/$N_ROLL)"
log "collected $NEPS rollout episodes"
# rollouts done -> the serving stack can come down (frees GPU for inversion)
cleanup

# ---- 4. relabel + 10-D augment ------------------------------------------------------
log "STEP4 IK-expert relabel -> LeRobot -> 10-D augment"
MUJOCO_GL=egl "$PYUR5E" dagger_relabel_assemble.py relabel --xml "$XML" \
  --raw "$WORK/roll_out/raw" --out "$WORK/relabel7" || fail "relabel"
MUJOCO_GL=egl "$PYUR5E" augment_state_grasp_target.py --src "$WORK/relabel7" \
  --out "$WORK/relabel_cond" --xml "$XML" --verify || fail "augment"
SD=$("$PYUR5E" -c "import json;print(json.load(open('$WORK/relabel_cond/meta/info.json'))['features']['observation.state']['shape'][0])")
[ "$SD" = "10" ] || fail "augment-bad-state-dim($SD)"

# ---- 5. rsync + invert on VM ---------------------------------------------------------
log "STEP5 rsync relabel_cond -> VM:$VM_DS_DAG + invert -> $VM_SHARDS_DAG"
rsync -a --delete -e "$SSH" "$WORK/relabel_cond/" "$VM:$VM_DS_DAG/" || fail "rsync-dagger-ds"
# Fresh shard dir: rollouts are stochastic, so a re-run's episode NNN differs
# from a prior run's — the inverter's resume manifest would otherwise keep
# STALE targets for same-numbered episodes.
$SSH "$VM" "rm -rf $VM_SHARDS_DAG" || true
MAXH=2.0; [ "$MINI" = "1" ] && MAXH=0.3
$SSH "$VM" "cd /home/ubuntu && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 10800 \
  $GPY flow_inverter_groot.py invert-dataset --dataset $VM_DS_DAG \
  --out-dir $VM_SHARDS_DAG --stride 8 --save-full-w --max-gpu-hours $MAXH \
  --out /home/ubuntu/${TAG}_invert.json" || fail "vm-invert"
scp -q -o StrictHostKeyChecking=no "$VM:/home/ubuntu/${TAG}_invert.json" "$WORK/invert.json" || true
log "inversion: $("$PYUR5E" -c "import json;d=json.load(open('$WORK/invert.json'));print({k:d.get(k) for k in ('n_chunks','drop_rate','gpu_hours')})" 2>/dev/null || echo '?')"

# ---- 6. multi-source retrain + offline gate --------------------------------------------
log "STEP6 retrain (base+dagger, boost=$BOOST) -> $VM_NET"
$SSH "$VM" "cd /home/ubuntu && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 3600 \
  $GPY train_steering_ur5e.py train --shards $VM_SHARDS_BASE,$VM_SHARDS_DAG \
  --output-design full5280 --features v2 \
  --dataset /home/ubuntu/ur5e_drugsort_obscond,$VM_DS_DAG \
  --source-boost=$BOOST --weight-floor 0.2 --weight-ref 0.05 \
  --model-out $VM_NET --out /home/ubuntu/${TAG}_train.json" || fail "vm-retrain"
scp -q -o StrictHostKeyChecking=no "$VM:/home/ubuntu/${TAG}_train.json" "$WORK/train.json" || true
log "train: $("$PYUR5E" -c "import json;d=json.load(open('$WORK/train.json'));print({k:d.get(k) for k in ('val_mse','val_mse_moving','val_mse_by_source')})" 2>/dev/null || echo '?')"

log "STEP6 offline gate (decode + fk-gate)"
EVALPY=/home/ubuntu/odyssey-eval-venv/bin/python
$SSH "$VM" "cd /home/ubuntu && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 5400 \
  $GPY flowdagger_offline_gate_ur5e.py decode --steering-net $VM_NET \
  --decoded-out /home/ubuntu/${TAG}_gate_decoded.npz --out /home/ubuntu/${TAG}_decode.json" || fail "gate-decode"
$SSH "$VM" "cd /home/ubuntu && timeout 1200 \
  $EVALPY flowdagger_offline_gate_ur5e.py fk-gate \
  --decoded-npz /home/ubuntu/${TAG}_gate_decoded.npz --out /home/ubuntu/${TAG}_gate.json" || fail "fk-gate"
scp -q -o StrictHostKeyChecking=no "$VM:/home/ubuntu/${TAG}_gate.json" "$WORK/gate.json" || fail "scp-gate"
GATEPASS=$("$PYUR5E" -c "import json;print(1 if json.load(open('$WORK/gate.json')).get('pass') else 0)" 2>/dev/null)
log "offline gate: $("$PYUR5E" -c "import json;d=json.load(open('$WORK/gate.json'));print({k:d.get(k) for k in ('steered_beats_stock_frac','improvement_toward_oracle','pass')})" 2>/dev/null || echo '?')"
[ "$GATEPASS" = "1" ] || fail "offline-gate-failed"

# ---- 7. mini stops here; full round runs the browser A/B --------------------------------
if [ "$MINI" = "1" ]; then
  "$PYUR5E" - "$WORK" "$RESULT" "$ROUND" <<'PY' || fail "mini-aggregate"
import json, sys
work, result, rnd = sys.argv[1], sys.argv[2], int(sys.argv[3])
def L(p):
    try: return json.load(open(f"{work}/{p}"))
    except Exception: return None
json.dump({"round": rnd, "mini": True, "status": "DONE",
           "invert": L("invert.json"), "train": L("train.json"), "gate": L("gate.json")},
          open(result, "w"), indent=2)
print("[mini] chain OK end-to-end")
PY
  log "MINI ROUND DONE — full chain validated"
  echo "${TAG^^} DONE"
  exit 0
fi

log "STEP7 browser A/B (RUN_TAG=$TAG, steered=dagger net vs stock)"
RUN_TAG=$TAG STEER_NPZ_SRC=$VM_NET bash "$BC/run_stageb_ab.sh"
grep -q "STAGEB_AB DONE" "$HOME/stageb_ab_${TAG}.log" || fail "ab-run"
[ -f "$HOME/stageb_ab_${TAG}_result.json" ] || fail "no-ab-result"

# ---- 8. aggregate --------------------------------------------------------------------
log "STEP8 aggregate"
"$PYUR5E" - "$WORK" "$RESULT" "$ROUND" "$HOME/stageb_ab_${TAG}_result.json" <<'PY' || fail "aggregate"
import json, sys
work, result, rnd, abp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
def L(p):
    try: return json.load(open(p))
    except Exception: return None
ab = L(abp)
out = {"round": rnd, "mini": False,
       "invert": L(f"{work}/invert.json"), "train": L(f"{work}/train.json"),
       "gate": L(f"{work}/gate.json"), "browser_ab": ab,
       "v02_reference": "steered 0/15, pad median 4.4cm, grips: 2 full closes (mistimed)",
       "status": "DONE"}
json.dump(out, open(result, "w"), indent=2)
A = ab["run_A_steered"]
print("[dagger] headline:", json.dumps({
    "steered": A["success"], "lifted": A.get("n_lifted"),
    "pad_median": A["min_pad_cm"]["median"],
    "verdict": ab["verdict"]}))
PY
log "DONE — results in $RESULT"
echo "${TAG^^} DONE"
