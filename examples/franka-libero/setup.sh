#!/usr/bin/env bash
#
# Setup for the LIBERO eval examples (single- and multi-agent).
#
# It ONLY sets things up — it does NOT run a mission. It clones LIBERO, installs the
# subset of its deps that co-exists with the OpenVLA stack, registers the LIBERO
# source on the venv path, and installs the imageio[ffmpeg] mp4 encoder.
#
# EVAL-ONLY: the published OpenVLA-LIBERO checkpoint auto-downloads from HF on first
# `odyssey run`. You do NOT need a dataset on disk — the 10.2 GB
# `openvla/modified_libero_rlds` is only for fine-tuning, not for evaluating.
#
# ─── WHY THIS IS NOT A ONE-LINER (validated on a GCP L4, 2026-07-13) ───────────────
#  * LIBERO ships NO top-level `libero/__init__.py` — it's a PEP 420 namespace package
#    meant to run from the repo root. `pip install -e .` registers nothing
#    (its setup.py `find_packages()` finds no package) and the `git+` VCS install is
#    metadata-only → `import libero` fails. Fix: clone + put the repo root on the path
#    via a `.pth` file.
#  * LIBERO's `install_requires` is empty; its real deps live in `requirements.txt`,
#    which HARD-PINS `transformers==4.21.1` and `numpy==1.22.4`. Installing that as-is
#    DOWNGRADES the OpenVLA stack (OpenVLA needs transformers 4.40.x) and cascades a
#    scipy downgrade. The env/benchmark code we use does NOT need transformers (only
#    `libero.lifelong` does), so we install LIBERO's deps MINUS transformers & numpy.
#  * `robosuite` is downgraded 1.5.2 → 1.4.0 (LIBERO requires it). That's fine in a
#    DEDICATED venv but would break the RobosuiteRunner eval in a shared env_pilot —
#    so install into `env_pilot_libero`, not `env_pilot` (use --venv, see below).
#  * `robomimic` pulls `egl_probe`, which builds from C source and needs system
#    `cmake` + a compiler + GL/EGL headers. Install those with apt first (see below).
#
# Prerequisites:
#   * System build + render deps (needs sudo):
#       sudo apt-get update && sudo apt-get install -y cmake build-essential \
#         libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libosmesa6-dev
#   * A dedicated venv with the OpenVLA stack, e.g. env_pilot_libero:
#       examples/multiagent-openvla-gemma/setup.sh --pilot-venv "$PWD/env_pilot_libero"
#   * (multi-agent only) `source examples/multiagent-openvla-gemma/.env` for
#     ODYSSEY_SPECIALIST_PYTHON
#
# Usage:
#   source env_pilot_libero/bin/activate && examples/franka-libero/setup.sh
#   examples/franka-libero/setup.sh [--venv PATH]
#
#   --venv PATH   target venv (default: <repo>/env_pilot). Point it at a DEDICATED venv
#                 (e.g. <repo>/env_pilot_libero) so the robosuite 1.4.0 downgrade can't
#                 disturb the validated env_pilot (OpenVLA + robosuite 1.5.2 eval).
#   An already-active venv ($VIRTUAL_ENV) always wins over --venv.
#
# Env overrides: LIBERO_DIR (default: $HOME/LIBERO)
#
# Linux + NVIDIA GPU assumed (GCP L4). Re-runnable.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LIBERO_DIR="${LIBERO_DIR:-$HOME/LIBERO}"

# --- venv selection (default env_pilot; --venv overrides for isolation) ---
VENV="${VENV:-$REPO_ROOT/env_pilot}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (see --help)" >&2; exit 2 ;;
  esac
done

# --- activate the target venv if none is already active ---
if [ -z "${VIRTUAL_ENV:-}" ]; then
  if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
  else
    echo "[setup] WARNING: no venv at $VENV — installing into the current Python." >&2
    echo "        Build a dedicated one first, e.g.:" >&2
    echo "          examples/multiagent-openvla-gemma/setup.sh --pilot-venv $VENV" >&2
  fi
fi

# --- system build dep check (egl_probe → robomimic builds from C source) ---
if ! command -v cmake >/dev/null 2>&1; then
  echo "[setup] ERROR: 'cmake' not found. LIBERO's egl_probe (via robomimic) builds from" >&2
  echo "        source and needs it. Install the system build + render deps first:" >&2
  echo "          sudo apt-get update && sudo apt-get install -y cmake build-essential \\" >&2
  echo "            libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libosmesa6-dev" >&2
  exit 1
fi

# --- clone LIBERO (namespace package; registered via .pth below, NOT pip install -e) ---
if [ ! -d "$LIBERO_DIR/.git" ]; then
  echo "[setup] cloning LIBERO into $LIBERO_DIR"
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
else
  echo "[setup] LIBERO already at $LIBERO_DIR — reusing"
fi

# --- LIBERO deps MINUS transformers & numpy (protect the OpenVLA stack; see header) ---
echo "[setup] installing LIBERO deps (excluding transformers/numpy to protect OpenVLA)…"
pip install \
  robosuite==1.4.0 bddl==1.0.1 robomimic==0.2.0 hydra-core==1.2.0 easydict==1.9 \
  einops==0.4.1 gym==0.25.2 cloudpickle==2.1.0 future==0.18.2 thop==0.1.1.post2209072238 \
  opencv-python==4.6.0.66 matplotlib==3.5.3 wandb==0.13.1

# --- register the LIBERO repo root on the venv path (PEP 420 namespace package) ---
SP="$(python -c 'import site;print(site.getsitepackages()[0])')"
echo "$LIBERO_DIR" > "$SP/libero_src.pth"
echo "[setup] registered $LIBERO_DIR on the path via $SP/libero_src.pth"

# --- pre-initialize ~/.libero/config.yaml NON-interactively ---
# The first `import libero` prompts on stdin ("custom dataset path? (Y/N)") and would
# otherwise BLOCK `odyssey run`. Answer 'n' once here to write the default config.
echo "n" | python -c "import libero" >/dev/null 2>&1 || true

# --- mp4 encoder for capture_video ---
if ! python -c "import imageio_ffmpeg" >/dev/null 2>&1; then
  echo "[setup] installing imageio[ffmpeg] (mp4 encoder)…"
  pip install "imageio[ffmpeg]"
fi

# --- verify (namespace package → __file__ is None; check __path__ + a real submodule) ---
python - <<'PY'
import importlib, sys
try:
    import libero
    from libero.libero import benchmark  # noqa: F401
    import transformers, robosuite
    print(f"[setup] libero       : path={list(libero.__path__)}")
    print(f"[setup] transformers : {transformers.__version__}  (must stay 4.40.x for OpenVLA)")
    print(f"[setup] robosuite    : {robosuite.__version__}  (1.4.0 expected in this venv)")
except Exception as e:  # noqa: BLE001
    print(f"[setup] VERIFY FAILED: {e}", file=sys.stderr)
    sys.exit(1)
PY

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
