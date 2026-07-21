#!/usr/bin/env bash
# ============================================================================
# run_powered_eval.sh — the 45-episode POWERED evaluation (roadmap promotion
# protocol): 3 x 15 fixed-seed blocks (7777 / 8888 / 9999), Wilson 95% CIs.
#
# Configuration PINNED to the best-evidenced selection setup (run-1 shaping,
# which produced the first lift + 13/15 closures) + the CEM-at-grasp search
# upgrade (orthogonal to shaping):
#   STEER_CLF_CENTERING=off  K=16 sigma=0.25  CEM_ROUNDS=2  (v4-tapered CBF)
# Writes ~/powered_eval_result.json + marker POWERED_EVAL DONE|FAILED.
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
XML=/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/aseptipack_description/aseptipack.xml
PUP=/home/daniel/.npm/_npx/23232c69e5d221f3/node_modules/puppeteer-core
CHROME=/usr/bin/google-chrome-stable
NODE=/home/daniel/.nvm/versions/node/v22.22.0/bin/node

# V4=1 -> serve the image-conditioned steering head server-side (sidecar ships
# its low15; the service computes the mean from pooled backbone features).
V4=${V4:-0}
# BARE=1 -> genuinely bare pilot through the SAME 45-ep powered protocol: no
# steering head, single GR00T sample (K=1 => the sidecar sends no best-of-N
# payload, so the bridge returns one raw action), no CEM. The autonomous loop
# uses this to measure the bare-base funnel. Default 0 keeps the pinned config
# byte-for-byte (backward compatible).
BARE=${BARE:-0}
# STEER_SERVO=1 -> the best-of-N SERVICE composes the L5 bounded final-cm visual
# servo onto its selected chunk (arm dims only). Default 0 keeps the served
# response byte-identical to the current V4 path (backward compatible).
STEER_SERVO=${STEER_SERVO:-0}
# SUF distinguishes candidate configs so their blocks never cross-contaminate.
# Critical: servo runs MUST NOT reuse the v4 blocks (RESUME=1 would skip them),
# or the servo is never actually evaluated (the L5 gate would compare a config
# to itself). v4 -> _v4 ; v4+servo -> _v4_servo.
SUF=""; [ "$V4" = "1" ] && SUF="_v4"
[ "$STEER_SERVO" = "1" ] && SUF="${SUF}_servo"
WORK=$HOME/powered_eval$SUF
LOG=$HOME/powered_eval$SUF.log
RESULT=$HOME/powered_eval${SUF}_result.json
# RESUME=1 keeps completed blocks (out_<seed>/eval_groot.json = block marker) —
# a mid-run infrastructure failure (e.g. a half-dead SSH tunnel) never costs
# finished blocks.
[ "${RESUME:-0}" = "1" ] || rm -rf "$WORK"
mkdir -p "$WORK"

AGENT_PORT=8032
HTTP_PORT=5596
FK_PORT=5560
COND_PORT=5604
CKPT=${CKPT_OVERRIDE:-/home/ubuntu/ckpt/ur5e_drugsort_obscond/full/checkpoint-12000}
STEER_SRC=/home/ubuntu/steering_net_v02.npz
OBS_WEIGHTS=$BC/percep_weights_browser
SEEDS=(7777 8888 9999)

export DISPLAY=${DISPLAY:-:1}
cd "$BC"
exec >>"$LOG" 2>&1
echo "" ; echo "#################################################################"
log(){ echo "=== [powered] $* $(date -u +%FT%TZ) ==="; }
cleanup(){
  log "cleanup"
  for pf in "$WORK/agent_pid.txt" "$WORK/cond_pid.txt"; do
    [ -f "$pf" ] || continue
    p=$(cat "$pf" 2>/dev/null || true); [ -n "${p:-}" ] && kill "$p" 2>/dev/null || true
  done
  ss -tlnp 2>/dev/null | grep -E ":($AGENT_PORT|$COND_PORT) " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | xargs -r kill 2>/dev/null || true
  pkill -f "127.0.0.1:$HTTP_PORT:127.0.0.1:$HTTP_PORT" 2>/dev/null || true
  $SSH "$VM" "tmux kill-session -t groot_bestofn_svc 2>/dev/null; tmux kill-session -t groot_bestofn_fk 2>/dev/null" 2>/dev/null || true
}
kill_chrome_pidfile(){
  local pf="$1"; [ -f "$pf" ] || return 0
  local cp; cp=$(cat "$pf" 2>/dev/null || true)
  if [ -n "${cp:-}" ] && kill -0 "$cp" 2>/dev/null; then
    tr '\0' ' ' < "/proc/$cp/cmdline" 2>/dev/null | grep -q "chrome-udd" && kill "$cp" 2>/dev/null || true
  fi
  rm -f "$pf" 2>/dev/null || true
}
fail(){
  trap - TERM INT HUP
  log "FAILED at: $*"
  printf '{"eval":"powered-45","status":"FAILED","failed_at":"%s"}\n' "$*" > "$RESULT"
  for s in "${SEEDS[@]}"; do kill_chrome_pidfile "$WORK/out_$s/eval_pid.txt"; done
  cleanup
  echo "POWERED_EVAL FAILED: $*"
  exit 1
}
trap 'fail "signal"' TERM INT HUP

log "START pid=$$ (3x15 blocks; config: centering=off v4-tapered-CBF K=16 sigma=0.25 cem=2; servo=$STEER_SERVO)"
echo "POWERED_EVAL PID=$$"

# wait (<=1h) for the stateless-dagger gates to release the GPU. SKIP_GPU_WAIT=1
# (set by the autonomous loop, which GPU-gates itself) or a missing dagger1s.log
# bypasses this so the loop's eval isn't stalled up to an hour on a stale gate.
if [ "${SKIP_GPU_WAIT:-0}" != "1" ] && [ -f "$HOME/dagger1s.log" ]; then
  for i in $(seq 1 120); do
    last=$(tail -1 "$HOME/dagger1s.log" 2>/dev/null || true)
    echo "$last" | grep -qE "DAGGER1S (DONE|FAILED)" && break
    sleep 30
  done
  log "dagger1s released the GPU ($(tail -1 "$HOME/dagger1s.log" 2>/dev/null))"
else
  log "GPU wait skipped (SKIP_GPU_WAIT=${SKIP_GPU_WAIT:-0}, dagger1s.log $([ -f "$HOME/dagger1s.log" ] && echo present || echo absent))"
fi

# ---- sanity + stack (once, shared across blocks) --------------------------------
[ -f "$OBS_WEIGHTS/observer_head.pt" ] || fail "no-observer-weights"
$SSH "$VM" "echo ok >/dev/null" || fail "vm-unreachable"
for port in $AGENT_PORT $COND_PORT; do
  ss -tln 2>/dev/null | grep -q ":$port " && fail "port-busy($port)"
done
ss -tln 2>/dev/null | grep -q ":$HTTP_PORT " && { pkill -f "127.0.0.1:$HTTP_PORT:127.0.0.1:$HTTP_PORT" 2>/dev/null || true; sleep 2; }
log "STEP1 deploy bestofn stack"
scp -q -o StrictHostKeyChecking=no "$ODY/scripts/serve_groot_bestofn.py" \
  "$ODY/scripts/serve_fk_ur5e.py" "$ODY/scripts/bestofn_select.py" \
  "$ODY/scripts/clf_reward_ur5e.py" "$ODY/scripts/cbf_constraints_ur5e.py" \
  "$ODY/scripts/servo_ur5e.py" \
  "$ODY/scripts/probe_flow_inversion_groot.py" \
  vm_deploy_bestofn.sh "$VM:/home/ubuntu/" || fail "scp"
V4NET=""; [ "$V4" = "1" ] && V4NET=/home/ubuntu/steering_net_v4.npz
# Forward STEER_SERVO so the VM-side service launches with the L5 servo on/off.
$SSH "$VM" "STEER_SERVO=$STEER_SERVO bash /home/ubuntu/vm_deploy_bestofn.sh $CKPT $HTTP_PORT $FK_PORT $V4NET" || fail "vm-deploy"
$SSH -f -N -o ExitOnForwardFailure=yes -L 127.0.0.1:$HTTP_PORT:127.0.0.1:$HTTP_PORT "$VM" || fail "tunnel"
for i in $(seq 1 30); do curl -s --max-time 5 http://127.0.0.1:$HTTP_PORT/health | grep -q '"ok": *true' && break; sleep 5; done
curl -s --max-time 5 http://127.0.0.1:$HTTP_PORT/health | grep -q '"bestofn": *true' || fail "bestofn-health"

log "STEP2 sidecar ($([ "$BARE" = "1" ] && echo BARE-single-sample || echo PINNED config)) + agent"
# best-of-N knobs (PINNED defaults); BARE overrides them to a single raw sample.
BON_K=16; BON_KG=16; BON_SIGMA=0.25; BON_CEM=2; CENTERING=off
if [ "$BARE" = "1" ]; then
  # no steering head + K=1 => the sidecar attaches no best-of-N payload, so the
  # bridge returns a single raw GR00T action. The Observer still injects the
  # grasp target to fill the 10-D state channel the policy was trained with.
  SIDE_STEER=(STEERING_WEIGHTS=)
  BON_K=1; BON_KG=1; BON_CEM=0
elif [ "$V4" = "1" ]; then
  SIDE_STEER=(STEERING_WEIGHTS= STEER_SERVER_HEAD=1)
else
  scp -q -o StrictHostKeyChecking=no "$VM:$STEER_SRC" "$WORK/steering_net.npz" || fail "fetch-steer"
  SIDE_STEER=(STEERING_WEIGHTS="$WORK/steering_net.npz")
fi
env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OBSERVER_WEIGHTS="$OBS_WEIGHTS" \
  CONDITION_BRIDGE_URL="http://127.0.0.1:$HTTP_PORT" OBSERVER_DEVICE=cuda PYTHONPATH="$ODY/src" \
  "${SIDE_STEER[@]}" STEERING_FK_XML="$XML" \
  STEER_BESTOFN_K=$BON_K STEER_BESTOFN_K_GRASP=$BON_KG STEER_BESTOFN_SIGMA=$BON_SIGMA \
  STEER_CLF_CENTERING=$CENTERING STEER_CEM_ROUNDS=$BON_CEM \
  setsid "$OBSPY" "$ODY/scripts/serve_observer_conditioning.py" --port "$COND_PORT" \
  >"$WORK/conditioner.log" 2>&1 &
echo $! > "$WORK/cond_pid.txt"
for i in $(seq 1 40); do curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health 2>/dev/null | grep -q '"ok": *true' && break; sleep 3; done
CH=$(curl -s --max-time 5 http://127.0.0.1:$COND_PORT/health || true)
{ echo "$CH" | grep -q '"observer_ready": *true' && echo "$CH" | grep -q '"ok": *true'; } || fail "conditioner"
cd "$AGENT_DIR"
env -u PYTHONPATH ENVIRONMENT=development \
    DATABASE_URL="sqlite+aiosqlite:///./powered_agent.db" \
    GROOT_BRIDGE_URL="http://127.0.0.1:$COND_PORT" \
    GROOT_STATE_CONDITIONER_URL="http://127.0.0.1:$COND_PORT" \
    GROOT_OBSERVER_URL="http://127.0.0.1:$COND_PORT" \
    DISPLAY="$DISPLAY" "$AGENT_PY" -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$AGENT_PORT" --loop asyncio > "$WORK/agent.log" 2>&1 &
echo $! > "$WORK/agent_pid.txt"
cd "$BC"
for i in $(seq 1 20); do
  curl -s -o /dev/null -w '%{http_code}' --max-time 6 "http://127.0.0.1:$AGENT_PORT/robot-playground.html?demo=drugsorting" 2>/dev/null | grep -q 200 && break
  sleep 3
done
curl -s --max-time 8 "http://127.0.0.1:$AGENT_PORT/api/groot/health" | grep -q '"ok": *true' || fail "agent-health"

# ---- v4 head-verification smoke -------------------------------------------------
if [ "$V4" = "1" ]; then
  log "SMOKE: response must carry steer_head=v4"
  "$PYUR5E" - "$AGENT_PORT" <<'PY' || fail "v4-smoke"
import base64, io, json, sys, time, urllib.request
port = sys.argv[1]
from PIL import Image
buf = io.BytesIO(); Image.new("RGB", (256, 256), (127, 127, 127)).save(buf, format="PNG")
img = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
body = json.dumps({"image_b64": img, "image_b64_wrist": img,
                   "state": [-1.57, -1.56, 1.58, -1.57, -1.57, 0.0, 0.05],
                   "instruction": "pick up the vial and place it in the rack",
                   "sid": f"v4smoke-{time.time_ns()}"}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/api/groot/get_action", data=body,
                             headers={"content-type": "application/json"}, method="POST")
res = json.loads(urllib.request.urlopen(req, timeout=300).read().decode())
print("[v4smoke]", json.dumps({"steer_head": res.get("steer_head"),
                               "k": (res.get("bestofn") or {}).get("k")}))
assert res.get("steer_head") == "v4", res.get("steer_head")
PY
fi

# ---- 3 blocks ------------------------------------------------------------------
for s in "${SEEDS[@]}"; do
  if [ -f "$WORK/out_$s/eval_groot.json" ]; then
    log "BLOCK seed=$s already complete — skip"
    continue
  fi
  rm -rf "$WORK/out_$s"
  log "BLOCK seed=$s (15 episodes)"
  if [ "$s" = "7777" ] && [ -f "$HOME/observer_cond/plans_eval.json" ]; then
    cp -f "$HOME/observer_cond/plans_eval.json" "$WORK/plans_$s.json"
  else
    "$PYUR5E" precompute_plans.py --xml "$XML" --num-episodes 15 --seed "$s" \
      --out "$WORK/plans_$s.json" || fail "plans-$s"
  fi
  PLANS="$WORK/plans_$s.json" OUT="$WORK/out_$s" PORT="$AGENT_PORT" AGENTS=groot N=15 \
    N_ACTION_STEPS=8 MAX_TICKS=900 \
    PUPPETEER_CORE="$PUP" CHROME="$CHROME" DISPLAY="$DISPLAY" \
    "$NODE" eval_browser_groot.js > "$WORK/eval_$s.log" 2>&1
  rc=$?
  tail -4 "$WORK/eval_$s.log"
  kill_chrome_pidfile "$WORK/out_$s/eval_pid.txt"
  [ "$rc" -eq 0 ] || fail "block-$s(rc=$rc)"
  [ -f "$WORK/out_$s/eval_groot.json" ] || fail "no-json-$s"
done

# ---- aggregate with Wilson CIs --------------------------------------------------
log "STEP4 aggregate (Wilson 95% CI over 45)"
BH=$(curl -s --max-time 5 http://127.0.0.1:$HTTP_PORT/health || echo '{}')
"$PYUR5E" - "$WORK" "$RESULT" "$BH" <<'PY' || fail "aggregate"
import json, math, sys
work, outp, bh = sys.argv[1], sys.argv[2], sys.argv[3]
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (p, max(0.0, c-h), min(1.0, c+h))
blocks, results = {}, []
for s in (7777, 8888, 9999):
    ev = json.load(open(f"{work}/out_{s}/eval_groot.json"))
    rs = ev.get("results") or []
    results += rs
    blocks[str(s)] = {"success": ev.get("success"), "lifted": ev.get("n_lifted"),
                      "seated": ev.get("n_seated")}
n = len(results)
succ = sum(1 for r in results if r.get("success"))
lift = sum(1 for r in results if r.get("lifted"))
close = sum(1 for r in results if r.get("gripMax", 0) >= 0.5)
pads = sorted((r.get("min_pad_to_vial") or 9) * 100 for r in results)
out = {
    "eval": "powered-45", "status": "DONE",
    "config": {"clf_centering": "off", "cbf": "v4-tapered",
               "k": 16, "sigma": 0.25, "cem_rounds": 2,
               "steering": "v0.2 mean"},
    "n": n,
    "success": {"k": succ, "wilson": wilson(succ, n)},
    "lifted": {"k": lift, "wilson": wilson(lift, n)},
    "closed": {"k": close, "wilson": wilson(close, n)},
    "pad_cm": {"median": pads[n//2] if n else None, "p90": pads[int(0.9*n)] if n else None},
    "blocks": blocks, "service_health": json.loads(bh or "{}"),
    "per_episode": [{k: r.get(k) for k in ("episode","success","lifted","seated",
                     "lift_height","place_dist","gripMax","min_pad_to_vial")} for r in results],
}
json.dump(out, open(outp, "w"), indent=2)
print("[powered] HEADLINE", json.dumps({k: out[k] for k in ("n","success","lifted","closed","pad_cm")}))
PY
cleanup
log "DONE — $RESULT"
echo "POWERED_EVAL DONE"
