#!/usr/bin/env bash
#
# End-to-end smoke test for ROLLOUT VIDEO CAPTURE on the single-agent OpenVLA
# quickstart. It trains a tiny LoRA (max_steps=10) and evaluates on Robosuite
# Lift with `capture_video: true`, writing one MP4 per episode. The point is to
# exercise the video feature fast — NOT to get a good policy (expect grade F).
#
# It does NOT edit the committed mission.yaml: it generates a patched copy in a
# temp file with the machine-local bits (dataset path + small max_steps/save_steps)
# so the example stays portable.
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

# --- activate the pilot venv if we're not already inside a venv ---
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$REPO_ROOT/env_pilot/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/env_pilot/bin/activate"
fi

# --- environment ---
export NCCL_NET="${NCCL_NET:-Socket}"            # GCP single-GPU: bypass the gIB NCCL plugin
export WANDB_MODE="${WANDB_MODE:-disabled}"       # OpenVLA calls wandb.init() unconditionally
# Headless rendering for the eval. EGL uses the GPU but needs NVIDIA's EGL
# userspace; on compute-only drivers it isn't present, so default to OSMesa
# (CPU software rendering) which always works. Override by exporting MUJOCO_GL
# (e.g. =egl) before calling this script if your VM has NVIDIA EGL.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"

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
echo "[setup] render      : MUJOCO_GL=$MUJOCO_GL"
echo

# --- disk note: each run stages ~15 GB + a merged checkpoint into ~/.odyssey/runs/.
#     Clean old runs first if the disk is tight: rm -rf ~/.odyssey/runs/* ---

# --- run training -> eval (with video capture) ---
odyssey run "$WORK"

# --- locate the videos the eval just wrote ---
echo
echo "[setup] rollout videos written to <run>/<eval-task>/videos/ :"
find "$HOME/.odyssey/runs" -path "*/videos/*.mp4" -printf '%T@ %p\n' 2>/dev/null \
  | sort -rn | head -20 | cut -d' ' -f2- | while read -r f; do ls -lh "$f"; done

cat <<'EOF'

To view a clip from your laptop (the VM is headless), copy it down with scp:
  gcloud compute scp --zone=<ZONE> \
    <VM_NAME>:<path-to.mp4> ./rollout.mp4
then open ./rollout.mp4 locally. (e.g. zone us-west1-a, VM odyssey-training-west)
EOF
