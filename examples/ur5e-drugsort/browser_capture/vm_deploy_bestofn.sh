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
# L5 lever: STEER_SERVO=1 (from the eval env) makes the service compose the
# bounded final-cm visual servo onto its selected chunk. Default 0 -> off, the
# served response is byte-identical to the current V4 path.
STEER_SERVO=${STEER_SERVO:-0}
STEER_HANDOFF_CARRY=${STEER_HANDOFF_CARRY:-owned}
# Render-gap diagnostic: STEER_DIAG=1 makes the service dump the head-mean's
# decoded pinch-vs-target + the received browser frames for the first N queries,
# so an offline pass can replay the same states on dataset frames. Off by default.
STEER_DIAG=${STEER_DIAG:-0}
STEER_DIAG_MAX=${STEER_DIAG_MAX:-40}
# Observer->IK handoff (full-authority final-cm capture) + its tuning knobs.
STEER_HANDOFF=${STEER_HANDOFF:-0}
STEER_HANDOFF_ZONE=${STEER_HANDOFF_ZONE:-0.10}
STEER_HANDOFF_DESCEND=${STEER_HANDOFF_DESCEND:-0.05}
STEER_HANDOFF_GRASPTOL=${STEER_HANDOFF_GRASPTOL:-0.015}
STEER_HANDOFF_MOVECAP=${STEER_HANDOFF_MOVECAP:-0.4}
STEER_HANDOFF_XY=${STEER_HANDOFF_XY:-groot}
STEER_HANDOFF_GRASPZ=${STEER_HANDOFF_GRASPZ:-}
# Grasp verify-and-retry: closing commits against the Observer estimate, so a
# failed weld must withdraw and re-descend against a fresh (closer) read.
STEER_HANDOFF_SMOOTH=${STEER_HANDOFF_SMOOTH:-15}
STEER_HANDOFF_COVMAX=${STEER_HANDOFF_COVMAX:-2.0}
STEER_HANDOFF_NEARWIN=${STEER_HANDOFF_NEARWIN:-1}
STEER_HANDOFF_LOG=${STEER_HANDOFF_LOG:-0}
STEER_HANDOFF_HOLD=${STEER_HANDOFF_HOLD:-2}
STEER_HANDOFF_VERIFY=${STEER_HANDOFF_VERIFY:-3}
STEER_HANDOFF_ATTEMPTS=${STEER_HANDOFF_ATTEMPTS:-4}
STEER_HANDOFF_RETRACT=${STEER_HANDOFF_RETRACT:-0.06}
STEER_HANDOFF_JITTER=${STEER_HANDOFF_JITTER:-0.012}
# Owned carry+place: transit/seat heights over the STATIC pocket. These MUST be
# forwarded into the tmux env list below or the service silently keeps its code
# defaults (block-2 lesson: env set in the eval driver never reached the service).
STEER_PLACE_Z_HI=${STEER_PLACE_Z_HI:-0.34}
STEER_PLACE_Z_LO=${STEER_PLACE_Z_LO:-0.28}
STEER_PLACE_TOL=${STEER_PLACE_TOL:-0.02}
STEER_HANDOFF_DESCEND_STEP=${STEER_HANDOFF_DESCEND_STEP:-}
STEER_HANDOFF_CLOSE_RAMP=${STEER_HANDOFF_CLOSE_RAMP:-0}
STEER_HANDOFF_APPROACH_BUDGET=${STEER_HANDOFF_APPROACH_BUDGET:-}
STEER_HANDOFF_SEAT=${STEER_HANDOFF_SEAT:-0}
STEER_HANDOFF_RISE_STEP=${STEER_HANDOFF_RISE_STEP:-}
STEER_HANDOFF_TRANSIT_STEP=${STEER_HANDOFF_TRANSIT_STEP:-}
STEER_HANDOFF_DROP_RETRY=${STEER_HANDOFF_DROP_RETRY:-0}
STEER_HANDOFF_DESCEND_LATCH=${STEER_HANDOFF_DESCEND_LATCH:-0}
STEER_HANDOFF_COMMIT_THROUGH=${STEER_HANDOFF_COMMIT_THROUGH:-0}
GROOT_PY=/home/ubuntu/Isaac-GR00T/.venv/bin/python
EVAL_PY=/home/ubuntu/odyssey-eval-venv/bin/python
XML=${FK_XML_OVERRIDE:-/home/ubuntu/aseptipack_description/aseptipack.xml}
echo "=== [vm-bestofn] serving $CKPT (bestofn :$HTTP_PORT, fk :$FK_PORT, servo=$STEER_SERVO) ==="
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
  "cd /home/ubuntu && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 STEER_SERVO=$STEER_SERVO STEER_DIAG=$STEER_DIAG STEER_DIAG_MAX=$STEER_DIAG_MAX STEER_HANDOFF=$STEER_HANDOFF STEER_HANDOFF_ZONE=$STEER_HANDOFF_ZONE STEER_HANDOFF_DESCEND=$STEER_HANDOFF_DESCEND STEER_HANDOFF_GRASPTOL=$STEER_HANDOFF_GRASPTOL STEER_HANDOFF_MOVECAP=$STEER_HANDOFF_MOVECAP STEER_HANDOFF_XY=$STEER_HANDOFF_XY STEER_HANDOFF_GRASPZ=$STEER_HANDOFF_GRASPZ STEER_HANDOFF_SMOOTH=$STEER_HANDOFF_SMOOTH STEER_HANDOFF_COVMAX=$STEER_HANDOFF_COVMAX STEER_HANDOFF_NEARWIN=$STEER_HANDOFF_NEARWIN STEER_HANDOFF_LOG=$STEER_HANDOFF_LOG STEER_HANDOFF_HOLD=$STEER_HANDOFF_HOLD STEER_HANDOFF_VERIFY=$STEER_HANDOFF_VERIFY STEER_HANDOFF_ATTEMPTS=$STEER_HANDOFF_ATTEMPTS STEER_HANDOFF_RETRACT=$STEER_HANDOFF_RETRACT STEER_HANDOFF_JITTER=$STEER_HANDOFF_JITTER STEER_PLACE_Z_HI=$STEER_PLACE_Z_HI STEER_PLACE_Z_LO=$STEER_PLACE_Z_LO STEER_PLACE_TOL=$STEER_PLACE_TOL STEER_HANDOFF_DESCEND_STEP=$STEER_HANDOFF_DESCEND_STEP STEER_HANDOFF_CLOSE_RAMP=$STEER_HANDOFF_CLOSE_RAMP STEER_HANDOFF_APPROACH_BUDGET=$STEER_HANDOFF_APPROACH_BUDGET STEER_HANDOFF_SEAT=$STEER_HANDOFF_SEAT STEER_HANDOFF_RISE_STEP=$STEER_HANDOFF_RISE_STEP STEER_HANDOFF_TRANSIT_STEP=$STEER_HANDOFF_TRANSIT_STEP STEER_HANDOFF_DROP_RETRY=$STEER_HANDOFF_DROP_RETRY STEER_HANDOFF_DESCEND_LATCH=$STEER_HANDOFF_DESCEND_LATCH STEER_HANDOFF_CARRY=$STEER_HANDOFF_CARRY STEER_HANDOFF_COMMIT_THROUGH=$STEER_HANDOFF_COMMIT_THROUGH $GROOT_PY serve_groot_bestofn.py \
   --model-path $CKPT --http-host 127.0.0.1 --http-port $HTTP_PORT \
   --fk-url http://127.0.0.1:$FK_PORT $STEER_ARG 2>&1 | tee /home/ubuntu/bestofn_svc.log"
for i in $(seq 1 60); do
  curl -s --max-time 5 http://127.0.0.1:$HTTP_PORT/health | grep -q '"ok": *true' && { echo " BESTOFN_OK"; exit 0; }
  sleep 10
done
echo "BESTOFN_FAIL"; exit 1
