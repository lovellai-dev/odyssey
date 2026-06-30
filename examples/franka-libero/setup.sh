#!/usr/bin/env bash
#
# Setup for the LIBERO eval examples (single- and multi-agent).
#
# It ONLY sets things up — it does NOT run a mission. It installs the LIBERO package
# and the imageio[ffmpeg] mp4 encoder into the OpenVLA env, sets the headless-render
# env vars, and prints next steps.
#
# EVAL-ONLY: the published OpenVLA-LIBERO checkpoint auto-downloads from HF on first
# `odyssey run`. You do NOT need a dataset on disk — the 10.2 GB
# `openvla/modified_libero_rlds` is only for fine-tuning, not for evaluating.
#
# Prerequisites:
#   * env_pilot built via examples/multiagent-openvla-gemma/setup.sh
#   * (multi-agent only) `source examples/multiagent-openvla-gemma/.env` for
#     ODYSSEY_SPECIALIST_PYTHON
#
# ⚠️ DEPENDENCY SPIKE: LIBERO pins its own robosuite/robomimic versions. This script
# attempts to install it into env_pilot; if pip reports conflicts with the OpenVLA
# stack, that's the integration risk flagged in the plan — capture the output and we
# resolve it (pin, or a dedicated venv). Installing LIBERO is the first real test.
#
# Usage:
#   examples/franka-libero/setup.sh
#
# Linux + NVIDIA GPU assumed (GCP L4). Re-runnable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- activate the pilot venv if present ---
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$REPO_ROOT/env_pilot/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/env_pilot/bin/activate"
fi

# --- LIBERO package (provides the env, task suites, bddl + init states) ---
if ! python -c "import libero" >/dev/null 2>&1; then
  echo "[setup] installing LIBERO (dependency spike — watch for robosuite/robomimic conflicts)…"
  pip install "libero @ git+https://github.com/Lifelong-Robot-Learning/LIBERO.git"
fi

# --- mp4 encoder for capture_video ---
if ! python -c "import imageio_ffmpeg" >/dev/null 2>&1; then
  echo "[setup] installing imageio[ffmpeg] (mp4 encoder)…"
  pip install "imageio[ffmpeg]"
fi

echo "[setup] libero  : $(python -c 'import libero,os;print(os.path.dirname(libero.__file__))' 2>/dev/null || echo 'NOT importable — resolve the dep conflict')"

cat <<'EOF'

[setup] Done — setup only, no mission run. Next steps:
  1. headless render (OSMesa = CPU, always works on compute-only drivers; use egl if
     the VM has NVIDIA EGL):
       export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  2. SINGLE-AGENT — score the published checkpoint (it auto-downloads from HF):
       odyssey validate examples/franka-libero/mission.yaml
       odyssey run      examples/franka-libero/mission.yaml
  3. MULTI-AGENT (Gemma planner) — needs the specialist venv loaded first:
       source examples/multiagent-openvla-gemma/.env     # sets ODYSSEY_SPECIALIST_PYTHON
       odyssey run examples/franka-libero/mission-multiagent.yaml
  4. find the per-episode MP4s:
       find ~/.odyssey/runs -path "*/videos/*.mp4" -exec ls -lh {} \;

Try other suites by editing benchmark_name + checkpoint + unnorm_key together, e.g.:
  libero_spatial / openvla-7b-finetuned-libero-spatial / libero_spatial
  libero_goal    / openvla-7b-finetuned-libero-goal    / libero_goal
  libero_10      / openvla-7b-finetuned-libero-10       / libero_10
EOF
