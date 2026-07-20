#!/usr/bin/env bash
# ============================================================================
# run_l5_servo.sh — L5 lever: ENABLE the bounded final-centimeter visual servo.
#
# L5 trains NOTHING — the servo is a SERVING-TIME correction composed into the
# best-of-N service (scripts/serve_groot_bestofn.py, gated by STEER_SERVO=1). So
# this driver is trivial: it records that L5 is enabled (a result JSON + the
# L5_SERVO DONE marker the autonomous loop polls) and returns. The loop's
# SUBSEQUENT candidate eval runs run_powered_eval.sh with STEER_SERVO=1 (the
# "servo" candidate mode in run_autonomous_loop._eval_env), which serves GR00T
# with the servo active and gates the resulting funnel.
#
# It stays on the SAME base checkpoint (no new ckpt) — the result carries no
# 'checkpoint', so the loop keeps the current base and the candidate eval reuses
# it with the servo turned on.
#
# Env: CKPT_OVERRIDE (base ckpt to record; informational).
# ============================================================================
set -uo pipefail
ODY=/home/daniel/LovellAI/odyssey-ur5e
PYUR5E=$ODY/.venv-ur5e/bin/python
LOG=$HOME/l5_servo.log
RESULT=$HOME/l5_servo_result.json
CKPT=${CKPT_OVERRIDE:-/home/ubuntu/ckpt/ur5e_drugsort_dr/checkpoint-15000}

exec >>"$LOG" 2>&1
echo "" ; echo "#################################################################"
log(){ echo "=== [l5] $* $(date -u +%FT%TZ) ==="; }
fail(){ log "FAILED at: $*"; printf '{"lever":"L5_servo","status":"FAILED","failed_at":"%s"}\n' "$*" > "$RESULT"; echo "L5_SERVO FAILED: $*"; exit 1; }
trap 'fail "signal"' TERM INT HUP
log "START pid=$$ base_ckpt=$CKPT"
echo "L5_SERVO PID=$$"

# Sanity: the servo module + its tests must exist and pass locally (no GPU) —
# L5 is a code lever, so the deliverable IS the servo function being correct.
[ -f "$ODY/scripts/servo_ur5e.py" ] || fail "servo-module-missing"
log "STEP1 unit-test the servo (pure numpy, no GPU)"
env -u PYTHONPATH "$PYUR5E" -m pytest "$ODY/tests/unit/test_servo_ur5e.py" -q \
  || fail "servo-unit-tests"

# Record the enablement. The presence of this marker + result is what the loop
# treats as "L5 done"; the servo itself is exercised by the candidate eval.
"$PYUR5E" - "$RESULT" "$CKPT" <<'PY' || fail "write-result"
import json, sys
out = {
    "lever": "L5_servo", "status": "DONE",
    "kind": "serving-time",
    "base_ckpt": sys.argv[2],
    "serve_flag": "STEER_SERVO=1",
    "note": ("L5 is a serving-time correction (no training). The subsequent "
             "candidate eval serves with STEER_SERVO=1 so the best-of-N service "
             "composes the bounded final-centimeter visual servo onto the "
             "selected chunk; the funnel gate promotes iff lift improves."),
}
json.dump(out, open(sys.argv[1], "w"), indent=2)
print("[l5] enabled:", json.dumps({k: out[k] for k in ("serve_flag", "base_ckpt")}))
PY
log "DONE — servo enabled for the next candidate eval (base $CKPT)"
echo "L5_SERVO DONE"
