#!/usr/bin/env bash
# Pilot-ALONE fold eval: STEER_HANDOFF=0, K=1 — does the finetuned GR00T
# complete the task without the specialist handoff? Redeploys the demo after.
set -uo pipefail
UR10E_XML=$HOME/lai-agent-multiagent/src/embodiments/urdf/aseptipack_ur10e_description/aseptipack.xml
LOG=$HOME/ur10e_pilotonly.log
: > "$LOG"; exec >>"$LOG" 2>&1
echo "PILOTONLY_START $(date -u +%FT%TZ)"
pkill -f "uvicorn app.main:app --host 127.0.0.1 --port 8032" 2>/dev/null
pkill -f "serve_observer_conditioning.py" 2>/dev/null
sleep 5
env MODE=bare N=15 MAX_TICKS=1000 ARM=ur10e \
  CKPT_OVERRIDE="$HOME/ckpt/ur10e_drugsort_condaug/checkpoint-12000" \
  OBS_WEIGHTS_OVERRIDE="$HOME/ur10e_percep_weights" \
  PLANS_OVERRIDE="$HOME/plans_ur10e_pow7777.json" \
  EVAL_XML="$UR10E_XML" FK_XML_OVERRIDE="$UR10E_XML" \
  STEER_HANDOFF=0 \
  TAG=ur10e_7777_pilotonly bash "$HOME/run_vm_eval2.sh" || true
S=$(grep -o "EVAL_SUMMARY.*" "$HOME/vm_eval_ur10e_7777_pilotonly.log" 2>/dev/null | tail -1)
echo "PILOTONLY_RESULT ${S:-no-summary}"
grep -E "^ATT " "$HOME/vm_eval_ur10e_7777_pilotonly/eval.log" 2>/dev/null | head -15
UR10E_CKPT="$HOME/ckpt/ur10e_drugsort_condaug/checkpoint-12000" UR10E_OBS="$HOME/ur10e_percep_weights" \
  bash "$HOME/run_demo_stack_ur10e.sh" && echo "PILOTONLY_DEMO_RESTORED" || echo "PILOTONLY_DEMO_FAIL"
echo "PILOTONLY_DONE"
