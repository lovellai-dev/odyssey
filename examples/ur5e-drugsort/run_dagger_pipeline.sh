#!/usr/bin/env bash
# ============================================================================
# run_dagger_pipeline.sh — autonomous DAgger (expert-in-the-loop) pipeline to
# fix the UR5e drug-sort GR00T grasp. Self-running on the VM; no babysitting.
#
# The closed-loop policy (v3) approaches + fires the gripper but closes ~7 cm
# short on the descent (eval 0/20, every episode lifted=False). That is an
# OFF-DISTRIBUTION failure: BC only ever saw the expert's clean descent. DAgger
# folds the states the policy ACTUALLY visits, each relabelled with the IK
# expert's correct absolute joint target, back into training.
#
# Each iteration (start from ckpt/ur5e_drugsort_v3/checkpoint-12000):
#   1. serve the CURRENT checkpoint (ZMQ), roll it out over many randomised vial
#      poses, relabel every visited state -> expert absolute joint target
#      (scripts/dagger_ur5e_drugsort.py rollout)
#   2. aggregate the relabels onto the running dataset (merge)
#   3. fine-tune GR00T-N1.7-3B on the aggregation, 12k steps (the mission path)
#   4. closed-loop eval N=20 -> record the number in dagger_results.jsonl
# Repeat ITERS times, each rolling out the previous DAgger checkpoint.
#
# GPU COORDINATION with the parallel v4 pipeline: DAgger does NO GPU-heavy work
# until v4 reaches a terminal state (PIPELINE DONE/FAILED in v4_pipeline.log),
# and every launch_finetune is additionally gated on nvidia-smi being free +
# no other launch_finetune running — so it never fights v4 for the H100. DAgger
# uses its own ZMQ port + tmux session (dagger_server) so it never touches v4's
# groot_server/groot_bridge.
#
# Writes per-iteration results to ~/dagger_results.jsonl, the best iteration to
# ~/eval_result_dagger.json (vs the 0/20 baseline), and PIPELINE DONE / FAILED
# to ~/dagger_pipeline.log. Fail-fast: any step's nonzero rc -> PIPELINE FAILED.
# ============================================================================
set -uo pipefail

LOG=/home/ubuntu/dagger_pipeline.log
exec >>"$LOG" 2>&1

# --- knobs ------------------------------------------------------------------
ITERS=${ITERS:-3}                 # DAgger iterations (spec: 3-5)
N_ROLLOUT=${N_ROLLOUT:-30}        # policy rollouts relabelled per iteration
TRAIN_STEPS=${TRAIN_STEPS:-12000} # fine-tune steps per iteration (the v3 path)
SAVE_STEPS=${SAVE_STEPS:-4000}
DPORT=${DPORT:-5560}              # DAgger ZMQ port (distinct from v4's 5555)
SERVER_FREE_MB=${SERVER_FREE_MB:-15000}
TRAIN_FREE_MB=${TRAIN_FREE_MB:-55000}

# --- paths ------------------------------------------------------------------
BASE_DS=/home/ubuntu/ur5e_drugsort_v3
BASE_CKPT=/home/ubuntu/ckpt/ur5e_drugsort_v3/checkpoint-12000
WORK=/home/ubuntu/dagger
XML=/home/ubuntu/aseptipack_description/aseptipack.xml
SRC=/home/ubuntu/odyssey-ur5e
RESULTS=/home/ubuntu/dagger_results.jsonl
FINAL=/home/ubuntu/eval_result_dagger.json
EVAL_VENV=/home/ubuntu/odyssey-eval-venv/bin/python
GROOT=/home/ubuntu/Isaac-GR00T
GROOT_PY=$GROOT/.venv/bin/python
V4LOG=/home/ubuntu/v4_pipeline.log

log(){ echo "=== [dagger] $* $(date -u +%FT%TZ) ==="; }
fail(){ echo "=== [dagger] PIPELINE FAILED: $1 (rc=${2:-?}) $(date -u +%FT%TZ) ==="; echo "PIPELINE FAILED"; exit 1; }

mkdir -p "$WORK"
: > "$RESULTS"
log "START (DAgger: ITERS=$ITERS N_ROLLOUT=$N_ROLLOUT TRAIN_STEPS=$TRAIN_STEPS port=$DPORT)"

# --- GPU coordination -------------------------------------------------------
wait_for_v4(){
  if [ ! -f "$V4LOG" ]; then log "no v4_pipeline.log found — assuming GPU free"; return 0; fi
  if grep -qE "PIPELINE DONE|PIPELINE FAILED" "$V4LOG"; then
    log "v4 already terminal: $(grep -oE 'PIPELINE DONE|PIPELINE FAILED' "$V4LOG" | tail -1)"; return 0
  fi
  log "v4 pipeline still running — DAgger waits for it to finish (not fighting for the GPU)"
  while ! grep -qE "PIPELINE DONE|PIPELINE FAILED" "$V4LOG"; do sleep 120; done
  log "v4 pipeline terminal: $(grep -oE 'PIPELINE DONE|PIPELINE FAILED' "$V4LOG" | tail -1)"
}

wait_for_gpu(){
  local need=$1 ft free
  while true; do
    ft=$(pgrep -fc "launch_finetune" 2>/dev/null || echo 0)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "${ft:-0}" -eq 0 ] && [ "${free:-0}" -ge "$need" ]; then return 0; fi
    log "GPU busy (foreign launch_finetune=$ft free=${free}MB need=${need}MB) — wait 60s"
    sleep 60
  done
}

serve_ckpt(){
  local ckpt=$1 logf=$2
  tmux kill-session -t dagger_server 2>/dev/null || true
  sleep 3
  tmux new-session -d -s dagger_server \
    "cd $GROOT && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $GROOT_PY -m gr00t.eval.run_gr00t_server --model-path $ckpt --embodiment-tag new_embodiment --port $DPORT --device cuda 2>&1 | tee $logf"
  log "serving $ckpt on :$DPORT — waiting for ready (ping)"
  local i
  for i in $(seq 1 80); do
    if $EVAL_VENV - <<PYPING 2>/dev/null
import zmq, msgpack, sys
import msgpack_numpy as mnp
ctx = zmq.Context.instance()
s = ctx.socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, 4000); s.setsockopt(zmq.SNDTIMEO, 4000)
s.connect("tcp://127.0.0.1:$DPORT")
try:
    s.send(msgpack.packb({"endpoint": "ping"}, default=mnp.encode)); s.recv(); sys.exit(0)
except Exception:
    sys.exit(1)
PYPING
    then log "server ready after ~$((i*15))s"; return 0; fi
    sleep 15
  done
  return 1
}

kill_server(){ tmux kill-session -t dagger_server 2>/dev/null || true; sleep 6; }

run_train(){  # $1=dataset  $2=output-ckpt  $3=global-batch
  cd "$GROOT"
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env -u PYTHONPATH "$GROOT_PY" \
    -m gr00t.experiment.launch_finetune \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path "$1" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/UR5e_DrugSort/ur5e_config.py \
    --output-dir "$2" \
    --num-gpus 1 --global-batch-size "$3" --learning-rate 0.0001 \
    --max-steps "$TRAIN_STEPS" --save-steps "$SAVE_STEPS" --dataloader-num-workers 8
}

# --- 0. Yield the GPU to v4 until it is done --------------------------------
wait_for_v4

# --- DAgger iterations ------------------------------------------------------
CUR_CKPT="$BASE_CKPT"
AGG_PREV="$BASE_DS"
BEST_RATE=-1; BEST_ITER=0

for i in $(seq 1 "$ITERS"); do
  log "================= DAgger ITERATION $i / $ITERS ================="
  RELABEL="$WORK/relabel_iter$i"
  AGG="$WORK/agg_iter$i"
  CKPT="/home/ubuntu/ckpt/ur5e_drugsort_dagger$i"
  CKPT_DIR="$CKPT/checkpoint-$TRAIN_STEPS"
  RENDERS="$WORK/rollouts_iter$i"

  # 1. rollout the current checkpoint + relabel every visited state -----------
  log "ITER $i STEP 1 rollout+relabel ($N_ROLLOUT eps) of $CUR_CKPT"
  wait_for_gpu "$SERVER_FREE_MB"
  serve_ckpt "$CUR_CKPT" "$WORK/server_iter$i.log" || fail "serve-rollout-$i" 1
  rm -rf "$RELABEL"
  cd "$SRC"
  MUJOCO_GL=egl PYTHONUNBUFFERED=1 $EVAL_VENV scripts/dagger_ur5e_drugsort.py rollout \
    --xml "$XML" --host 127.0.0.1 --port "$DPORT" \
    --out "$RELABEL" --num-episodes "$N_ROLLOUT" --n-action-steps 8 \
    --max-ticks 1000 --seed $((100 + i)) > "$WORK/rollout_iter$i.log" 2>&1
  RRC=$?
  kill_server
  [ $RRC -eq 0 ] || { tail -5 "$WORK/rollout_iter$i.log"; fail "rollout-$i" "$RRC"; }
  grep -E "RELABEL wrote|rollout policy success" "$WORK/rollout_iter$i.log" | tail -2

  # 2. aggregate onto the running dataset -------------------------------------
  log "ITER $i STEP 2 merge $AGG_PREV + $RELABEL -> $AGG"
  rm -rf "$AGG"
  $EVAL_VENV scripts/dagger_ur5e_drugsort.py merge \
    --base "$AGG_PREV" --add "$RELABEL" --out "$AGG" > "$WORK/merge_iter$i.log" 2>&1 \
    || { tail -5 "$WORK/merge_iter$i.log"; fail "merge-$i" $?; }
  cat "$WORK/merge_iter$i.log"
  # free the previous aggregation (keep the base v3 demos)
  [ "$AGG_PREV" != "$BASE_DS" ] && rm -rf "$AGG_PREV"

  # 3. fine-tune on the aggregation (gated on the GPU being free) -------------
  log "ITER $i STEP 3 fine-tune $TRAIN_STEPS steps -> $CKPT"
  wait_for_gpu "$TRAIN_FREE_MB"
  rm -rf "$CKPT"
  run_train "$AGG" "$CKPT" 16 > "$WORK/train_iter$i.log" 2>&1
  TRC=$?
  if [ $TRC -ne 0 ] && grep -qi "out of memory\|CUDA out of memory" "$WORK/train_iter$i.log"; then
    log "ITER $i train OOM at batch 16 -> retry batch 8"
    rm -rf "$CKPT"; wait_for_gpu "$TRAIN_FREE_MB"
    run_train "$AGG" "$CKPT" 8 >> "$WORK/train_iter$i.log" 2>&1; TRC=$?
  fi
  [ $TRC -eq 0 ] || { tail -5 "$WORK/train_iter$i.log"; fail "train-$i" "$TRC"; }
  [ -d "$CKPT_DIR" ] || fail "no-checkpoint-$i" 1
  tail -2 "$WORK/train_iter$i.log"

  # 4. closed-loop eval N=20 --------------------------------------------------
  log "ITER $i STEP 4 closed-loop eval N=20 of $CKPT_DIR"
  wait_for_gpu "$SERVER_FREE_MB"
  serve_ckpt "$CKPT_DIR" "$WORK/server_eval_iter$i.log" || fail "serve-eval-$i" 1
  rm -rf "$RENDERS"
  cd "$SRC/scripts"
  MUJOCO_GL=egl $EVAL_VENV eval_ur5e_drugsort_groot.py \
    --xml "$XML" --host 127.0.0.1 --port "$DPORT" \
    --num-episodes 20 --n-action-steps 8 --max-ticks 1100 \
    --video-dir "$RENDERS" --save-videos 3 > "$WORK/eval_iter$i.log" 2>&1
  ERC=$?
  kill_server
  [ $ERC -eq 0 ] || { tail -5 "$WORK/eval_iter$i.log"; fail "eval-$i" "$ERC"; }
  grep -E "CLOSED-LOOP GR00T SUCCESS|lifted|seated" "$WORK/eval_iter$i.log" | tail -3

  # 5. record this iteration's number -----------------------------------------
  ITER=$i RENDERS="$RENDERS" RELABEL="$RELABEL" CUR_CKPT="$CUR_CKPT" CKPT_DIR="$CKPT_DIR" \
  RESULTS="$RESULTS" $EVAL_VENV - <<'PY' || fail "record-$i" $?
import json, os
summ = json.load(open(os.path.join(os.environ["RENDERS"], "eval_summary.json")))
res = summ.get("results", [])
grip_fired = sum(1 for r in res if r.get("grip_max", 0) >= 0.9)
roll = {}
rp = os.path.join(os.environ["RELABEL"], "dagger_rollout.json")
if os.path.exists(rp):
    roll = json.load(open(rp))
row = {
    "iteration": int(os.environ["ITER"]),
    "rollout_ckpt": os.environ["CUR_CKPT"],
    "trained_ckpt": os.environ["CKPT_DIR"],
    "success_rate": summ["success_rate"],
    "success": f'{summ["n_success"]}/{summ["n_episodes"]}',
    "n_success": summ["n_success"],
    "n_episodes": summ["n_episodes"],
    "n_lifted_grasped": summ["n_lifted"],
    "n_seated_placed": summ["n_seated"],
    "grip_fired": grip_fired,
    "relabel_episodes": roll.get("relabel_episodes"),
    "relabel_frames": roll.get("relabel_frames"),
    "rollout_success_rate": roll.get("rollout_success_rate"),
}
with open(os.environ["RESULTS"], "a") as fh:
    fh.write(json.dumps(row) + "\n")
print("[result]", json.dumps(row))
PY
  RATE=$(tail -1 "$RESULTS" | $EVAL_VENV -c "import json,sys;print(json.loads(sys.stdin.read())['success_rate'])")
  log "ITER $i success_rate=$RATE"
  awk_ok=$($EVAL_VENV -c "print(1 if $RATE > $BEST_RATE else 0)")
  if [ "$awk_ok" = "1" ]; then BEST_RATE=$RATE; BEST_ITER=$i; fi

  # advance: next iteration rolls out THIS checkpoint on the grown dataset
  CUR_CKPT="$CKPT_DIR"
  AGG_PREV="$AGG"
done

# --- final: best iteration vs the 0/20 baseline -----------------------------
log "writing $FINAL (best iteration vs baseline 0/20)"
RESULTS="$RESULTS" FINAL="$FINAL" BEST_ITER="$BEST_ITER" $EVAL_VENV - <<'PY' || fail "write-final" $?
import json, os
rows = [json.loads(l) for l in open(os.environ["RESULTS"]) if l.strip()]
best = max(rows, key=lambda r: r["success_rate"]) if rows else None
out = {
    "task": "ur5e_drugsort DAgger (expert-in-the-loop grasp fix)",
    "baseline": {"checkpoint": "ur5e_drugsort_v3", "success": "0/20", "success_rate": 0.0},
    "best_iteration": best["iteration"] if best else None,
    "best_success": best["success"] if best else None,
    "best_success_rate": best["success_rate"] if best else None,
    "improved_over_baseline": bool(best and best["success_rate"] > 0.0),
    "iterations": rows,
}
json.dump(out, open(os.environ["FINAL"], "w"), indent=2)
print("[final]", json.dumps(out))
PY

log "PIPELINE DONE (best iteration $BEST_ITER, rate $BEST_RATE)"
echo "PIPELINE DONE"
