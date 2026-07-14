#!/usr/bin/env bash
# ============================================================================
# gen_v4.sh — iteration-4 ("v4 scale-up") GR00T UR5e drug-sort data generation.
#
# Same adaptive-IK expert / absolute action rep / wrist+exterior cameras /
# visual-DR / grasp-phase-densified demos as v3 (scripts/gen_ur5e_drugsort_demos.py),
# but scaled from 320 -> 2500 episodes to give GR00T many more examples — the
# "several examples" lever and a stronger base for the parallel RL work.
#
# Runs single-process on the VM (MuJoCo CPU + EGL offscreen render, ~15s/ep,
# so ~10h for 2500 eps). Coexists with a running GR00T server (gen barely uses
# VRAM); the v4 pipeline frees the GPU before training. Writes GEN_DONE_RC=<rc>
# as the last line of the log — the sentinel run_v4_pipeline.sh blocks on.
# ============================================================================
set -uo pipefail

LOG=/home/ubuntu/datagen_v4.log
PY=/home/ubuntu/odyssey-eval-venv/bin/python
SRC=/home/ubuntu/odyssey-ur5e
XML=/home/ubuntu/aseptipack_description/aseptipack.xml
OUT=/home/ubuntu/ur5e_drugsort_v4
N=2500

cd "$SRC"
env MUJOCO_GL=egl PYTHONUNBUFFERED=1 "$PY" scripts/gen_ur5e_drugsort_demos.py \
  --xml "$XML" \
  --out "$OUT" \
  --num-episodes "$N" \
  --area-x 0.06 --area-y 0.08 --yaw-jitter 0.12 --visual-dr 1.0 \
  --seed 0 > "$LOG" 2>&1
echo "GEN_DONE_RC=$?" >> "$LOG"
