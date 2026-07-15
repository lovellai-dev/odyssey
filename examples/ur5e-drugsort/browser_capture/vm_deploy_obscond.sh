#!/usr/bin/env bash
# Runs ON the H100 VM. Serves the OBSERVER-CONDITIONED GR00T checkpoint (10-D state)
# on ISOLATED ports (ZMQ :5558 server + HTTP bridge :5596) so v4's :5555/:5599, the
# render-gap :5556/:5598 and the dagger :5557/:5597 all stay up + untouched. Uses the
# 10-D-aware bridge scp'd to /home/ubuntu/serve_groot_http_bridge_obscond.py (passes
# the state's grasp_target slot 7:10 through to the model). Idempotent: kills only MY
# own groot_obscond_* tmux sessions.
set -uo pipefail
CKPT=${1:?checkpoint dir}
ZMQ_PORT=${2:-5558}
BRIDGE_PORT=${3:-5596}
GROOT=/home/ubuntu/Isaac-GR00T
GROOT_PY=$GROOT/.venv/bin/python
BRIDGE=/home/ubuntu/serve_groot_http_bridge_obscond.py
echo "=== [vm-deploy-obscond] serving $CKPT on ZMQ :$ZMQ_PORT + bridge :$BRIDGE_PORT ==="
tmux kill-session -t groot_obscond_server 2>/dev/null || true
tmux kill-session -t groot_obscond_bridge 2>/dev/null || true
sleep 3
tmux new-session -d -s groot_obscond_server \
  "cd $GROOT && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $GROOT_PY -m gr00t.eval.run_gr00t_server \
   --model-path $CKPT --embodiment-tag new_embodiment --port $ZMQ_PORT --device cuda \
   2>&1 | tee /home/ubuntu/groot_server_obscond.log"
# wait for the ZMQ server to answer a ping
READY=0
for i in $(seq 1 60); do
  if $GROOT_PY - "$ZMQ_PORT" <<'PYPING' 2>/dev/null
import zmq, msgpack, sys, msgpack_numpy as mnp
ctx = zmq.Context.instance(); s = ctx.socket(zmq.REQ)
s.setsockopt(zmq.RCVTIMEO, 4000); s.setsockopt(zmq.SNDTIMEO, 4000)
s.connect("tcp://127.0.0.1:%s" % sys.argv[1])
try: s.send(msgpack.packb({"endpoint":"ping"}, default=mnp.encode)); s.recv(); sys.exit(0)
except Exception: sys.exit(1)
PYPING
  then READY=1; break; fi
  sleep 10
done
[ $READY -eq 1 ] || { echo "SERVER_NOT_READY"; exit 1; }
echo "=== [vm-deploy-obscond] server ready; starting 10-D-aware bridge ==="
tmux new-session -d -s groot_obscond_bridge \
  "cd /home/ubuntu && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $GROOT_PY $BRIDGE \
   --isaac-gr00t $GROOT --zmq-port $ZMQ_PORT --http-host 127.0.0.1 --http-port $BRIDGE_PORT \
   2>&1 | tee /home/ubuntu/groot_bridge_obscond.log"
sleep 6
curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health | grep -q ok && echo " BRIDGE_OK" || { echo "BRIDGE_FAIL"; exit 1; }
