#!/usr/bin/env bash
# Runs ON the H100 VM. Serves the R2 best-of-N stack on isolated ports:
#   FK sidecar (mujoco venv)  :5560   gr_pinch batch FK
#   best-of-N policy service  :5596   in-process GR00T + CBF/CLF selection
# Replaces the ZMQ-server+bridge pair (same tunnel port :5596 -> local stack
# unchanged). Idempotent; kills only its own groot_bestofn_* tmux sessions.
set -uo pipefail
CKPT=${1:?checkpoint dir}
HTTP_PORT=${2:-5596}
FK_PORT=${3:-5560}
STEER_NET=${4:-}   # optional: v4 image-conditioned head .npz (server-side mean)
GROOT_PY=/home/ubuntu/Isaac-GR00T/.venv/bin/python
EVAL_PY=/home/ubuntu/odyssey-eval-venv/bin/python
XML=/home/ubuntu/aseptipack_description/aseptipack.xml
echo "=== [vm-bestofn] serving $CKPT (bestofn :$HTTP_PORT, fk :$FK_PORT) ==="
tmux kill-session -t groot_bestofn_fk 2>/dev/null || true
tmux kill-session -t groot_bestofn_svc 2>/dev/null || true
sleep 2
tmux new-session -d -s groot_bestofn_fk \
  "cd /home/ubuntu && MUJOCO_GL=egl $EVAL_PY serve_fk_ur5e.py --xml $XML --port $FK_PORT \
   2>&1 | tee /home/ubuntu/bestofn_fk.log"
for i in $(seq 1 20); do
  curl -s --max-time 3 http://127.0.0.1:$FK_PORT/health | grep -q '"ok": *true' && break
  sleep 3
done
curl -s --max-time 3 http://127.0.0.1:$FK_PORT/health | grep -q '"ok": *true' || { echo "FK_FAIL"; exit 1; }
STEER_ARG=""
[ -n "$STEER_NET" ] && STEER_ARG="--steering-net $STEER_NET"
tmux new-session -d -s groot_bestofn_svc \
  "cd /home/ubuntu && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $GROOT_PY serve_groot_bestofn.py \
   --model-path $CKPT --http-host 127.0.0.1 --http-port $HTTP_PORT \
   --fk-url http://127.0.0.1:$FK_PORT $STEER_ARG 2>&1 | tee /home/ubuntu/bestofn_svc.log"
for i in $(seq 1 60); do
  curl -s --max-time 5 http://127.0.0.1:$HTTP_PORT/health | grep -q '"ok": *true' && { echo " BESTOFN_OK"; exit 0; }
  sleep 10
done
echo "BESTOFN_FAIL"; exit 1
