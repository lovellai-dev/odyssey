#!/usr/bin/env bash
# ============================================================================
# run_autonomous_loop.sh — DETACHED, RESUMABLE wrapper around the step machine.
#
# Launch it detached so it survives session/login death (the whole reason this
# exists — every prior manual driver died with its session):
#
#     setsid nohup bash scripts/run_autonomous_loop.sh </dev/null >/dev/null 2>&1 &
#
# It loops calling `run_autonomous_loop.py step` (one atomic, idempotent action
# per call) until the phase is `done` or `blocked`, or a step hard-fails. All
# state lives in ~/auto_loop/state.json; re-running this script RE-ATTACHES to
# that state and continues from the current phase. A single-instance flock stops
# two wrappers from racing on the state file.
#
# Log: ~/auto_loop.log with markers  AUTO_LOOP STEP <phase>  and
#      AUTO_LOOP DONE | AUTO_LOOP BLOCKED | AUTO_LOOP FAILED.
# ============================================================================
set -uo pipefail

ODY=/home/daniel/LovellAI/odyssey-ur5e
PY=$ODY/.venv-ur5e/bin/python
SCRIPT=$ODY/scripts/run_autonomous_loop.py
STATE_DIR=$HOME/auto_loop
LOG=$HOME/auto_loop.log
LOCK=$STATE_DIR/wrapper.lock
INTERVAL=${AUTO_LOOP_INTERVAL:-60}
MAX_STEPS_GUARD=${AUTO_LOOP_MAX_STEPS_GUARD:-100000}   # safety cap on loop iterations

export DISPLAY=${DISPLAY:-:1}
mkdir -p "$STATE_DIR"
exec >>"$LOG" 2>&1

# ---- single-instance lock (re-running is safe: a live wrapper keeps the lock) --
exec 200>"$LOCK"
if ! flock -n 200; then
  echo "=== AUTO_LOOP wrapper already running (lock held) — not starting a second $(date -u +%FT%TZ) ==="
  exit 0
fi

run_py(){ env -u PYTHONPATH "$PY" "$SCRIPT" "$@"; }

echo ""; echo "#################################################################"
echo "=== AUTO_LOOP wrapper START pid=$$ interval=${INTERVAL}s $(date -u +%FT%TZ) ==="

run_py init || { echo "AUTO_LOOP FAILED (init) $(date -u +%FT%TZ)"; exit 1; }

i=0
while :; do
  i=$((i + 1))
  if [ "$i" -gt "$MAX_STEPS_GUARD" ]; then
    echo "AUTO_LOOP FAILED (step guard $MAX_STEPS_GUARD exceeded) $(date -u +%FT%TZ)"; exit 1
  fi
  PHASE=$(run_py phase 2>/dev/null | tail -1)
  case "$PHASE" in
    done)    echo "AUTO_LOOP DONE $(date -u +%FT%TZ)";    run_py status; exit 0 ;;
    blocked) echo "AUTO_LOOP BLOCKED $(date -u +%FT%TZ)"; run_py status; exit 0 ;;
    "")      echo "AUTO_LOOP FAILED (no phase from state) $(date -u +%FT%TZ)"; exit 1 ;;
  esac
  echo "=== AUTO_LOOP STEP $PHASE (iter $i) $(date -u +%FT%TZ) ==="
  if ! run_py step; then
    echo "AUTO_LOOP FAILED (step hard-error in phase $PHASE) $(date -u +%FT%TZ)"; exit 1
  fi
  sleep "$INTERVAL"
done
