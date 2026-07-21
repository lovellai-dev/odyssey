#!/usr/bin/env bash
#
# Setup for the LIBERO eval examples (single- and multi-agent) — OpenVLA or GR00T pilot.
#
# It ONLY sets things up — it does NOT run a mission. It clones LIBERO, installs the
# subset of its deps that co-exists with the pilot stack, registers the LIBERO
# source on the venv path, and installs the imageio[ffmpeg] mp4 encoder.
#
# With --pilot gr00t it is END-TO-END (one command): it installs the system build/
# render deps, builds the pilot venv if missing, builds the GR00T policy-server venv
# (~/Isaac-GR00T/.venv via quickstart-gr00t/setup.sh — NO Isaac Sim, LIBERO = MuJoCo),
# and pre-downloads the requested LIBERO suite checkpoint.
#
# EVAL-ONLY: the published checkpoints auto-download from HF on first `odyssey run`.
# You do NOT need a training dataset on disk.
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
#    `cmake` + a compiler + GL/EGL headers + Python.h. Install those with apt first.
#
# Prerequisites (openvla pilot; --pilot gr00t installs these for you):
#   * System build + render deps (needs sudo):
#       sudo apt-get update && sudo apt-get install -y cmake build-essential python3-dev python3.10-dev \
#         libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libosmesa6-dev
#   * A dedicated venv with the pilot stack, e.g. env_pilot_libero:
#       examples/multiagent-openvla-gemma/setup.sh --pilot-venv "$PWD/env_pilot_libero"
#   * (multi-agent only) `source examples/multiagent-openvla-gemma/.env` for
#     ODYSSEY_SPECIALIST_PYTHON
#
# Usage:
#   # OpenVLA pilot (needs an active / --venv pilot venv):
#   source env_pilot_libero/bin/activate && examples/franka-libero/setup.sh
#   # GR00T pilot — END-TO-END, one command (installs deps, builds venvs, downloads):
#   examples/franka-libero/setup.sh --pilot gr00t
#
#   --venv PATH   target venv (default: <repo>/env_pilot; gr00t: <repo>/env_pilot_libero).
#                 A DEDICATED venv keeps the robosuite 1.4.0 downgrade from disturbing
#                 the validated env_pilot (OpenVLA + robosuite 1.5.2 eval).
#   --pilot NAME  openvla (default, LIBERO only) | gr00t (end-to-end: system deps +
#                 pilot venv + GR00T policy-server venv + suite checkpoint download)
#   --suite NAME  (gr00t only) LIBERO suite checkpoint to download (default: libero_object)
#   An already-active venv ($VIRTUAL_ENV) always wins over --venv.
#
# Env overrides: LIBERO_DIR (default: $HOME/LIBERO), ISAAC_GR00T_DIR (default: $HOME/Isaac-GR00T)
#
# Linux + NVIDIA GPU assumed (GCP L4). Re-runnable / idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LIBERO_DIR="${LIBERO_DIR:-$HOME/LIBERO}"
ISAAC_GR00T_DIR="${ISAAC_GR00T_DIR:-$HOME/Isaac-GR00T}"

# --- venv + pilot selection ---
VENV="${VENV:-}"
PILOT="${PILOT:-openvla}"          # openvla (LIBERO only) | gr00t (end-to-end)
SUITE="${SUITE:-libero_object}"    # gr00t: which suite checkpoint to download
while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --pilot) PILOT="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (see --help)" >&2; exit 2 ;;
  esac
done
# gr00t isolates LIBERO's robosuite 1.4.0 downgrade in its own venv by default
if [ -z "$VENV" ]; then
  [ "$PILOT" = "gr00t" ] && VENV="$REPO_ROOT/env_pilot_libero" || VENV="$REPO_ROOT/env_pilot"
fi

# --- gr00t only: system deps + build the pilot venv if missing (end-to-end) ---
if [ "$PILOT" = "gr00t" ]; then
  echo "[setup] --pilot gr00t: installing system build + render deps (sudo)…"
  sudo apt-get update && sudo apt-get install -y cmake build-essential python3-dev python3.10-dev \
    libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libosmesa6-dev
  command -v uv >/dev/null || {
    echo "[setup] ERROR: 'uv' not found — install then re-run:" >&2
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc" >&2
    exit 1
  }
  if [ ! -f "$VENV/bin/activate" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[setup] building pilot venv at $VENV…"
    "$REPO_ROOT/examples/multiagent-openvla-gemma/setup.sh" --pilot-venv "$VENV"
  fi
fi

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
  echo "          sudo apt-get update && sudo apt-get install -y cmake build-essential python3-dev python3.10-dev \\" >&2
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
# The config init prompts on stdin ("custom dataset path? (Y/N)") and would otherwise
# BLOCK `odyssey run` AND the verify below. Two subtleties, both learned the hard way:
#   * `libero` is an EMPTY PEP 420 namespace package — importing it is a no-op and does
#     NOT trigger the init. The prompt fires on `libero.libero`, so import THAT.
#   * `yes N` answers EVERY prompt (a single `echo n` isn't enough if it asks twice).
# Together they write the default config. Skip if already initialized.
if [ ! -f "$HOME/.libero/config.yaml" ]; then
  yes N | python -c "import libero.libero" >/dev/null 2>&1 || true
fi

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

# --- gr00t only: build the GR00T policy-server venv + pre-download the suite ---
if [ "$PILOT" = "gr00t" ]; then
  echo "[setup] --pilot gr00t: building the GR00T policy server + downloading '$SUITE'…"
  # The LIBERO recipe is the ZMQ CLIENT of the GR00T server — it needs the light
  # transport deps in THIS (pilot) venv; the server's own venv has its own copy.
  pip install msgpack msgpack-numpy pyzmq
  [ -d "$ISAAC_GR00T_DIR/.git" ] || \
    git clone https://github.com/NVIDIA/Isaac-GR00T.git "$ISAAC_GR00T_DIR"
  # quickstart-gr00t/setup.sh step [2/4] builds $ISAAC_GR00T_DIR/.venv (py3.12); its
  # Isaac-Lab step SKIPs itself (no ISAAC_PYTHON) — LIBERO needs no Isaac Sim.
  # ODYSSEY_VENV=.venv-core so it never clobbers a pre-existing core .venv.
  ODYSSEY_VENV=.venv-core bash "$REPO_ROOT/examples/quickstart-gr00t/setup.sh"
  # The server runs offline → pre-cache BOTH the suite checkpoint AND the backbone.
  huggingface-cli download nvidia/GR00T-N1.7-LIBERO --include "$SUITE/*"
  # Every GR00T checkpoint loads the GATED VLM backbone nvidia/Cosmos-Reason2-2B; it
  # must be in the HF cache for the (offline) server to start. Needs prior access +
  # auth. Non-fatal: warn clearly with the fix steps if it isn't accessible yet.
  echo "[setup] pre-caching the gated GR00T backbone nvidia/Cosmos-Reason2-2B…"
  if ! huggingface-cli download nvidia/Cosmos-Reason2-2B; then
    echo "[setup] WARNING: could not fetch nvidia/Cosmos-Reason2-2B (gated) — GR00T will" >&2
    echo "        NOT start until it is cached. To fix:" >&2
    echo "          1) request access: https://huggingface.co/nvidia/Cosmos-Reason2-2B" >&2
    echo "          2) huggingface-cli login" >&2
    echo "          3) huggingface-cli download nvidia/Cosmos-Reason2-2B  (or re-run this setup)" >&2
  fi
  cat <<EOF

[setup] Done (GR00T pilot). Next:
  1. In examples/franka-libero/mission-gr00t.yaml, under task config, set:
       server_python: $ISAAC_GR00T_DIR/.venv/bin/python
  2. source $VENV/bin/activate
     export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
     odyssey run examples/franka-libero/mission-gr00t.yaml
  3. server log (source of truth): tail -n 60 /tmp/gr00t_server_5555.log
EOF
  exit 0
fi

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
