#!/usr/bin/env bash
# Runs ON the H100 VM. Serves MY browser-matched GR00T checkpoint on ISOLATED ports
# (ZMQ :5556 server + HTTP bridge :5598) so v4's :5555/:5599 (dagger3 baseline) stay
# up + untouched. The orchestrator re-points the LOCAL :5599 tunnel to :5598 for the
# in-browser eval. Idempotent: kills only MY own groot_browser_* tmux sessions.
set -uo pipefail
CKPT=${1:?checkpoint dir}
ZMQ_PORT=${2:-5556}
BRIDGE_PORT=${3:-5598}
GROOT=/home/ubuntu/Isaac-GR00T
GROOT_PY=$GROOT/.venv/bin/python
echo "=== [vm-deploy] serving $CKPT on ZMQ :$ZMQ_PORT + bridge :$BRIDGE_PORT ==="
tmux kill-session -t groot_browser_server 2>/dev/null || true
tmux kill-session -t groot_browser_bridge 2>/dev/null || true
sleep 3
tmux new-session -d -s groot_browser_server \
  "cd $GROOT && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $GROOT_PY -m gr00t.eval.run_gr00t_server \
   --model-path $CKPT --embodiment-tag new_embodiment --port $ZMQ_PORT --device cuda \
   2>&1 | tee /home/ubuntu/groot_server_browser.log"
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
echo "=== [vm-deploy] server ready; starting bridge ==="
tmux new-session -d -s groot_browser_bridge \
  "cd /home/ubuntu/odyssey-ur5e && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $GROOT_PY scripts/serve_groot_http_bridge.py \
   --isaac-gr00t $GROOT --zmq-port $ZMQ_PORT --http-host 127.0.0.1 --http-port $BRIDGE_PORT \
   2>&1 | tee /home/ubuntu/groot_bridge_browser.log"
sleep 6
curl -s --max-time 5 http://127.0.0.1:$BRIDGE_PORT/health && echo " BRIDGE_OK" || { echo "BRIDGE_FAIL"; exit 1; }
