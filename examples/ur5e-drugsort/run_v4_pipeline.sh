#!/usr/bin/env bash
# ============================================================================
# run_v4_pipeline.sh — autonomous iteration-4 "scale-up" GR00T UR5e drug-sort
# pipeline. Self-running on the VM; no babysitting.
#
# The v4 lever = MANY more demos + a LONGER train (vs v3's 320 eps / 12k steps):
#   1. block until gen v4 (tmux gen_v4) finishes  (GEN_DONE_RC in datagen_v4.log)
#   2. validate the v4 dataset (exterior+wrist video keys, per-episode mp4s)
#   3. free the GPU (stop the v3 groot_server/groot_bridge)
#   4. fine-tune GR00T-N1.7-3B on ur5e_drugsort_v4, 30k steps -> ckpt/ur5e_drugsort_v4
#   5. re-serve the new checkpoint (ZMQ :5555)
#   6. closed-loop eval, N=20, absolute rep -> groot_rollouts_v4/
#   7. write ~/eval_result_v4.json (success rate + lifted/seated/grip-fired, vs_v3)
#   8. start the browser bridge (:5599) on the v4 checkpoint
#   9. append "PIPELINE DONE" to the log
# Fail-fast: any step's nonzero rc writes "PIPELINE FAILED: <step>" and exits.
# ============================================================================
set -uo pipefail

LOG=/home/ubuntu/v4_pipeline.log
exec >>"$LOG" 2>&1

DS=/home/ubuntu/ur5e_drugsort_v4
CKPT=/home/ubuntu/ckpt/ur5e_drugsort_v4
CKPT_DIR=$CKPT/checkpoint-30000
XML=/home/ubuntu/aseptipack_description/aseptipack.xml
RENDERS=/home/ubuntu/groot_rollouts_v4
RESULT=/home/ubuntu/eval_result_v4.json
EVAL_VENV=/home/ubuntu/odyssey-eval-venv/bin/python
GROOT=/home/ubuntu/Isaac-GR00T
GROOT_PY=$GROOT/.venv/bin/python

log(){ echo "=== [pipeline] $* $(date -u +%FT%TZ) ==="; }
fail(){ echo "=== [pipeline] PIPELINE FAILED: $1 (rc=${2:-?}) $(date -u +%FT%TZ) ==="; echo "PIPELINE FAILED"; exit 1; }

log "START (v4 scale-up: 2500 eps / 30k steps)"

# --- 1. Block until gen v4 finishes ----------------------------------------
log "STEP 1 waiting for gen v4 (GEN_DONE_RC in datagen_v4.log)"
while ! grep -q GEN_DONE_RC /home/ubuntu/datagen_v4.log 2>/dev/null; do sleep 30; done
GEN_RC=$(grep GEN_DONE_RC /home/ubuntu/datagen_v4.log | tail -1 | sed 's/.*=//')
log "gen finished GEN_DONE_RC=$GEN_RC"
[ "$GEN_RC" = "0" ] || fail "data-gen" "$GEN_RC"
grep -E "grasp.place success|wrote .* episodes|max joint std" /home/ubuntu/datagen_v4.log | tail -4

# --- 2. Validate dataset ----------------------------------------------------
log "STEP 2 validating dataset"
$EVAL_VENV - <<'PY' || fail "validate" $?
import json, glob
root = "/home/ubuntu/ur5e_drugsort_v4"
info = json.load(open(f"{root}/meta/info.json"))
mod  = json.load(open(f"{root}/meta/modality.json"))
vids = set(mod["video"])
assert vids == {"exterior", "wrist"}, f"video keys wrong: {vids}"
feats = info["features"]
for k in ("observation.images.exterior", "observation.images.wrist"):
    assert k in feats and feats[k]["dtype"] == "video", f"missing feature {k}"
    assert feats[k]["shape"] == [256, 256, 3], feats[k]["shape"]
ne = info["total_episodes"]
assert ne >= 2000, f"too few episodes: {ne}"
for k in ("exterior", "wrist"):
    n = len(glob.glob(f"{root}/videos/chunk-*/observation.images.{k}/episode_*.mp4"))
    assert n == ne, f"{k}: {n} mp4 vs {ne} episodes"
np = len(glob.glob(f"{root}/data/chunk-*/episode_*.parquet"))
assert np == ne, f"parquet {np} vs {ne}"
print(f"[validate] OK episodes={ne} frames={info['total_frames']} "
      f"videos={info['total_videos']} keys={sorted(vids)}")
PY
log "dataset valid"

# --- 3. Free the GPU: stop the v3 server + bridge ---------------------------
log "STEP 3 freeing GPU (stop v3 groot_server/groot_bridge)"
tmux kill-session -t groot_server 2>/dev/null || true
tmux kill-session -t groot_bridge 2>/dev/null || true
sleep 8
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true

# --- 4. Fine-tune (30k steps). One OOM-fallback retry at batch 8. -----------
run_train(){
  local bs=$1
  cd "$GROOT"
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 env -u PYTHONPATH "$GROOT_PY" \
    -m gr00t.experiment.launch_finetune \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path "$DS" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/UR5e_DrugSort/ur5e_config.py \
    --output-dir "$CKPT" \
    --num-gpus 1 --global-batch-size "$bs" --learning-rate 0.0001 \
    --max-steps 30000 --save-steps 5000 --dataloader-num-workers 8
}
log "STEP 4 training 30k steps (global-batch 16)"
run_train 16 > /home/ubuntu/train_v4.log 2>&1
TRC=$?
if [ $TRC -ne 0 ]; then
  if grep -qi "out of memory\|CUDA out of memory" /home/ubuntu/train_v4.log; then
    log "train OOM at batch 16 -> retry batch 8"
    rm -rf "$CKPT"
    run_train 8 >> /home/ubuntu/train_v4.log 2>&1
    TRC=$?
  fi
fi
log "train exited rc=$TRC"
[ $TRC -eq 0 ] || fail "train" "$TRC"
[ -d "$CKPT_DIR" ] || fail "no-checkpoint-30000" 1
tail -3 /home/ubuntu/train_v4.log

# --- 5. Re-serve the new checkpoint (ZMQ :5555) -----------------------------
log "STEP 5 serving $CKPT_DIR on :5555"
tmux kill-session -t groot_server 2>/dev/null || true
tmux new-session -d -s groot_server \
  "cd $GROOT && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $GROOT_PY -m gr00t.eval.run_gr00t_server --model-path $CKPT_DIR --embodiment-tag new_embodiment --port 5555 --device cuda 2>&1 | tee /home/ubuntu/groot_server_v4.log"
log "waiting for server ready (ping :5555)"
READY=0
for i in $(seq 1 80); do
  if $EVAL_VENV - <<'PYPING' 2>/dev/null
import zmq, msgpack, sys
import msgpack_numpy as mnp
ctx = zmq.Context.instance()
s = ctx.socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, 4000); s.setsockopt(zmq.SNDTIMEO, 4000)
s.connect("tcp://127.0.0.1:5555")
try:
    s.send(msgpack.packb({"endpoint": "ping"}, default=mnp.encode)); s.recv(); sys.exit(0)
except Exception:
    sys.exit(1)
PYPING
  then READY=1; break; fi
  sleep 15
done
[ $READY -eq 1 ] || fail "server-not-ready" 1
log "server ready after ~$((i*15))s"

# --- 6. Closed-loop eval (N=20, absolute rep, both cameras) -----------------
log "STEP 6 closed-loop eval N=20"
cd /home/ubuntu/odyssey-ur5e/scripts
MUJOCO_GL=egl $EVAL_VENV eval_ur5e_drugsort_groot.py \
  --xml "$XML" --host 127.0.0.1 --port 5555 \
  --num-episodes 20 --n-action-steps 8 --max-ticks 1100 \
  --video-dir "$RENDERS" --save-videos 4 > /home/ubuntu/eval_v4.log 2>&1
ERC=$?
log "eval exited rc=$ERC"
[ $ERC -eq 0 ] || fail "eval" "$ERC"
grep -E "CLOSED-LOOP GR00T SUCCESS|lifted|seated" /home/ubuntu/eval_v4.log | tail -4

# --- 7. Write eval_result_v4.json -------------------------------------------
log "STEP 7 writing $RESULT"
$EVAL_VENV - <<'PY' || fail "write-result" $?
import json
summ = json.load(open("/home/ubuntu/groot_rollouts_v4/eval_summary.json"))
res = summ.get("results", [])
grip_fired = sum(1 for r in res if r.get("grip_max", 0) >= 0.9)
out = {
    "iteration": 4,
    "success_rate": summ["success_rate"],
    "success": f'{summ["n_success"]}/{summ["n_episodes"]}',
    "n_success": summ["n_success"],
    "n_episodes": summ["n_episodes"],
    "n_lifted_grasped": summ["n_lifted"],
    "n_seated_placed": summ["n_seated"],
    "grip_fired": grip_fired,
    "n_action_steps": summ.get("n_action_steps"),
    "max_ticks": summ.get("max_ticks"),
    "vs_v3": "0/20",
    "changes": ["scaled data-gen 320 -> 2500 episodes (8x more demos)",
                "longer fine-tune 12k -> 30k steps",
                "same adaptive-IK expert + wrist+exterior + visual DR + grasp densification"],
}
json.dump(out, open("/home/ubuntu/eval_result_v4.json", "w"), indent=2)
print("[result]", json.dumps(out))
PY

# --- 8. Start the browser bridge (:5599) on the v4 checkpoint ----------------
log "STEP 8 starting browser bridge :5599 (v4)"
tmux kill-session -t groot_bridge 2>/dev/null || true
tmux new-session -d -s groot_bridge \
  "cd /home/ubuntu/odyssey-ur5e && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $GROOT_PY scripts/serve_groot_http_bridge.py --isaac-gr00t $GROOT --zmq-port 5555 --http-host 127.0.0.1 --http-port 5599 2>&1 | tee /home/ubuntu/groot_bridge_v4.log"
sleep 6
curl -s http://127.0.0.1:5599/health || true

log "PIPELINE DONE"
echo "PIPELINE DONE"
