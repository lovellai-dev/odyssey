#!/usr/bin/env bash
#
# Setup for testing ROLLOUT VIDEO CAPTURE on the single-agent OpenVLA quickstart.
#
# It ONLY sets things up — it does NOT run a mission (matches the multi-agent
# setup.sh). It installs the imageio[ffmpeg] mp4 encoder and generates a
# machine-local mission file tuned for a fast end-to-end smoke: a tiny LoRA
# (max_steps=10) + eval on Robosuite Lift with `capture_video: true` (one MP4 per
# episode). The point is to exercise the video feature fast — NOT to get a good
# policy (expect grade F).
#
# The committed examples/quickstart-openvla/mission.yaml is left untouched: the
# machine-local bits (dataset path + small max_steps/save_steps) go into a patched
# copy under $TMPDIR so the example stays portable.
#
# Prerequisites (one-time):
#   * env_pilot built via examples/multiagent-openvla-gemma/setup.sh
#   * the Bridge V2 RLDS dataset on disk at <DATA_ROOT>/bridge_orig/
#
# Usage:
#   examples/quickstart-openvla/setup.sh [DATA_ROOT]
#     DATA_ROOT  parent dir of the RLDS dataset folder (default: $HOME).
#                e.g. dataset at /home/me/bridge_orig/  ->  DATA_ROOT=/home/me
#
# Linux + NVIDIA GPU assumed (GCP L4). Re-runnable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="${1:-${DATA_ROOT:-$HOME}}"

# --- activate the pilot venv if present, so the encoder installs into it ---
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$REPO_ROOT/env_pilot/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/env_pilot/bin/activate"
fi

# --- the mp4 encoder. The robosuite extra ships imageio[ffmpeg]; make sure the
#     ffmpeg plugin is importable, else the encode silently no-ops (no video). ---
if ! python -c "import imageio_ffmpeg" >/dev/null 2>&1; then
  echo "[setup] installing imageio[ffmpeg] (mp4 encoder)…"
  pip install "imageio[ffmpeg]"
fi

# --- generate a patched, machine-local mission (committed example stays clean) ---
SRC="$SCRIPT_DIR/mission.yaml"
WORK="${TMPDIR:-/tmp}/odyssey-quickstart-video-test.yaml"
sed -e "s|data_root_dir: /path/to/dataset|data_root_dir: ${DATA_ROOT}|" \
    -e "s|^      epochs: 1|      epochs: 1\n      max_steps: 10\n      save_steps: 5|" \
    "$SRC" > "$WORK"

echo "[setup] mission     : $WORK"
echo "[setup] data_root   : $DATA_ROOT  (expects ${DATA_ROOT}/bridge_orig/)"
echo "[setup] training    : max_steps=10, save_steps=5 (fast smoke)"

cat <<EOF

[setup] Done — setup only, no mission run. Next steps:
  1. load the venv + base env vars (NCCL / WANDB / MUJOCO_GL):
       source examples/multiagent-openvla-gemma/.env
  2. headless render — use OSMesa (CPU) on compute-only drivers without NVIDIA EGL:
       export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  3. (disk tight? each run stages ~15 GB) clean old runs:
       rm -rf ~/.odyssey/runs/*
  4. run the mission (training -> eval with video capture):
       odyssey run $WORK
  5. find the per-episode MP4s the eval wrote:
       find ~/.odyssey/runs -path "*/videos/*.mp4" -exec ls -lh {} \;
  6. view a clip from your laptop (the VM is headless) — copy it down with scp:
       gcloud compute scp --zone=<ZONE> <VM_NAME>:<path-to.mp4> ./rollout.mp4
       # e.g. zone us-west1-a, VM odyssey-training-west; then open ./rollout.mp4
EOF
