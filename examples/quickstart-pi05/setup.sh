#!/usr/bin/env bash
#
# Setup for the π0.5 (openpi) LIBERO eval quickstart — GCP L4 24GB target.
#
# It ONLY sets things up — it does NOT run a mission (and, being externally-served,
# it does NOT boot the policy server for you unless you pass --serve). Two SEPARATE
# environments by design (their python/CUDA/JAX-vs-torch ABIs clash); they are wired
# by host:port, not co-installed:
#
#   1. Odyssey client env  — a dedicated venv (this repo + openpi-client + LIBERO)
#                            -> drives the mission, holds the LIBERO/MuJoCo env
#   2. openpi server env   — openpi's own uv env (JAX)   -> serves π0.5 on host:port
#
# ─── ⚠️ NOT YET VALIDATED ON HARDWARE ──────────────────────────────────────────────
#  Unlike examples/franka-libero/setup.sh (validated on a GCP L4), the openpi SERVER
#  half here is UNVERIFIED end-to-end. The odyssey client half reuses the validated
#  LIBERO install; the openpi clone/sync/serve steps follow openpi's docs but have
#  not been run through to a green LIBERO rollout. Treat the server section as a
#  best-effort scaffold — confirm the exact `serve_policy.py` flags + checkpoint id
#  against YOUR openpi version. The two things to watch on the first smoke: gripper
#  polarity (π0.5 uses no fix-up) and the `infer` -> {"actions": ...} wire shape.
#
# What the client half installs (mirrors the validated franka-libero LIBERO deps):
#  * LIBERO ships NO top-level package — it's a PEP 420 namespace package; register
#    the repo root via a .pth file (pip install -e registers nothing).
#  * robosuite is pinned 1.4.0 (LIBERO requires it) in a DEDICATED venv so it can't
#    disturb a validated OpenVLA/RobosuiteRunner env elsewhere.
#  * robomimic pulls egl_probe, which builds from C source -> needs cmake + headers.
#  * π0.5 needs NO `transformers` (FAST is reserved/off the inference path), so the
#    OpenVLA transformers pin does not apply here.
#
# Usage:
#   bash examples/quickstart-pi05/setup.sh                 # set up both envs
#   bash examples/quickstart-pi05/setup.sh --serve         # ...then exec the server (foreground)
#
#   --venv PATH        odyssey client venv (default: <repo>/env_pilot_pi05)
#   --openpi-dir PATH  openpi checkout (default: $HOME/openpi)
#   --checkpoint ID    π0.5 checkpoint the server serves
#                      (default: gs://openpi-assets/checkpoints/pi05_libero)
#   --host / --port    server bind address baked into the print-out (default 127.0.0.1:8000)
#   --serve            after setup, exec the openpi server in the foreground (blocks)
#
# Env overrides: LIBERO_DIR (default $HOME/LIBERO), HF_HOME (default $HOME/.cache/huggingface)
#
# Linux + NVIDIA GPU assumed (GCP L4, 24GB). Re-runnable / idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VENV="${VENV:-$REPO_ROOT/env_pilot_pi05}"
OPENPI_DIR="${OPENPI_DIR:-$HOME/openpi}"
LIBERO_DIR="${LIBERO_DIR:-$HOME/LIBERO}"
CHECKPOINT="gs://openpi-assets/checkpoints/pi05_libero"
HOST="127.0.0.1"
PORT="8000"
DO_SERVE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --openpi-dir) OPENPI_DIR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --serve) DO_SERVE=1; shift ;;
    -h|--help) grep '^#' "$0" | grep -v '^#!' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1 (see --help)" >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null || {
  echo "[setup] ERROR: 'uv' not found — install then re-run:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc" >&2
  exit 1
}

# ---------------------------------------------------------------------------
echo "==> [1/5] system build + render deps (sudo; egl_probe/MuJoCo need these)"
# ---------------------------------------------------------------------------
sudo apt-get update && sudo apt-get install -y \
  cmake build-essential python3-dev python3.10-dev \
  libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libosmesa6-dev

# ---------------------------------------------------------------------------
echo "==> [2/5] openpi checkout (source of both the client package and the server)"
# ---------------------------------------------------------------------------
if [ ! -d "$OPENPI_DIR/.git" ]; then
  echo "[setup] cloning Physical-Intelligence/openpi into $OPENPI_DIR"
  git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
else
  echo "[setup] openpi already at $OPENPI_DIR — reusing"
fi

# ---------------------------------------------------------------------------
echo "==> [3/5] odyssey client venv ($VENV): odyssey + openpi-client + LIBERO deps"
# ---------------------------------------------------------------------------
uv venv --python 3.10 "$VENV"
PYBIN="$VENV/bin/python"
uv pip install --python "$PYBIN" -e "$REPO_ROOT[dev,huggingface]"

# openpi's lightweight websocket client (what pi05.py imports as `openpi_client`).
# It lives in the openpi repo at packages/openpi-client — install THAT (no JAX).
if [ -d "$OPENPI_DIR/packages/openpi-client" ]; then
  uv pip install --python "$PYBIN" -e "$OPENPI_DIR/packages/openpi-client"
else
  echo "[setup] WARNING: $OPENPI_DIR/packages/openpi-client not found — trying PyPI 'openpi-client'." >&2
  uv pip install --python "$PYBIN" openpi-client \
    || echo "[setup] WARNING: could not install openpi-client; the eval will raise NotImplementedError until it is present." >&2
fi

# LIBERO client stack (pilot-agnostic; same pins the validated franka-libero uses,
# MINUS transformers/numpy — π0.5 needs neither). Installed into the pi05 venv.
uv pip install --python "$PYBIN" \
  robosuite==1.4.0 bddl==1.0.1 robomimic==0.2.0 hydra-core==1.2.0 easydict==1.9 \
  einops==0.4.1 gym==0.25.2 cloudpickle==2.1.0 future==0.18.2 thop==0.1.1.post2209072238 \
  opencv-python==4.6.0.66 matplotlib==3.5.3
uv pip install --python "$PYBIN" "imageio[ffmpeg]"   # mp4 encoder for capture_video

# LIBERO is a PEP 420 namespace package: clone + register the repo root on the path.
if [ ! -d "$LIBERO_DIR/.git" ]; then
  echo "[setup] cloning LIBERO into $LIBERO_DIR"
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
else
  echo "[setup] LIBERO already at $LIBERO_DIR — reusing"
fi
SP="$("$PYBIN" -c 'import site;print(site.getsitepackages()[0])')"
echo "$LIBERO_DIR" > "$SP/libero_src.pth"
echo "[setup] registered $LIBERO_DIR via $SP/libero_src.pth"

# Pre-init ~/.libero/config.yaml non-interactively (the init prompts on stdin and
# would otherwise block `odyssey run`). The prompt fires on `libero.libero`, and
# `yes N` answers every prompt. Skip if already initialized.
if [ ! -f "$HOME/.libero/config.yaml" ]; then
  yes N | "$PYBIN" -c "import libero.libero" >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------------------
echo "==> [4/5] verify the client env (no transformers needed for π0.5)"
# ---------------------------------------------------------------------------
"$PYBIN" - <<'PY'
import sys
try:
    import odyssey                                   # noqa: F401
    import libero
    from libero.libero import benchmark              # noqa: F401
    import robosuite, imageio_ffmpeg                 # noqa: F401
    import importlib.util as u
    has_client = bool(u.find_spec("openpi_client"))
    print(f"[setup] odyssey       : OK")
    print(f"[setup] libero        : path={list(libero.__path__)}")
    print(f"[setup] robosuite     : {robosuite.__version__}  (1.4.0 expected in this venv)")
    print(f"[setup] openpi_client : {'OK' if has_client else 'MISSING (install before running)'}")
except Exception as e:  # noqa: BLE001
    print(f"[setup] VERIFY FAILED: {e}", file=sys.stderr)
    sys.exit(1)
PY

# ---------------------------------------------------------------------------
echo "==> [5/5] openpi SERVER env (JAX, separate) — best-effort; see ⚠️ header"
# ---------------------------------------------------------------------------
# openpi manages its own env with uv. `uv sync` inside the checkout builds it.
# This is the UNVERIFIED half — it downloads a large JAX/CUDA stack; on an L4 it is
# the single-agent VRAM tenant. Non-fatal: warn + print the manual steps on failure.
if ( cd "$OPENPI_DIR" && uv sync ); then
  echo "[setup] openpi env synced under $OPENPI_DIR (.venv)"
else
  echo "[setup] WARNING: 'uv sync' in $OPENPI_DIR did not complete — finish it per openpi's" >&2
  echo "        README before serving (GIT_LFS_SKIP_SMUDGE, CUDA wheels, etc. may apply)." >&2
fi

SERVE_CMD=( uv run --project "$OPENPI_DIR" scripts/serve_policy.py
            policy:checkpoint --policy.config=pi05_libero
            "--policy.dir=$CHECKPOINT" "--host=$HOST" "--port=$PORT" )

cat <<EOF

[setup] Done. Two-terminal flow (externally-served: you start the server):

  TERMINAL 1 — serve π0.5 (openpi env; downloads the checkpoint on first run):
    ${SERVE_CMD[*]}
    # ⚠ exact flags vary by openpi version — the simplest form is often:
    #   uv run --project $OPENPI_DIR scripts/serve_policy.py --env LIBERO
    # Verify the server prints it is listening on $HOST:$PORT.

  TERMINAL 2 — run the eval (this repo's venv):
    source $VENV/bin/activate
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl     # L4 has NVIDIA EGL; osmesa = CPU fallback
    # point the mission at the server if you changed host/port:
    #   examples/quickstart-pi05/mission.yaml -> config.host/config.port
    odyssey validate examples/quickstart-pi05/mission.yaml
    odyssey run      examples/quickstart-pi05/mission.yaml

  Per-episode MP4s:
    find ~/.odyssey/runs -path "*/videos/*.mp4" -exec ls -lh {} \\;

  First-smoke watch-list (see the README + PR #81 limits):
    * gripper polarity — π0.5 applies NO fix-up; if inverted, patch pi05_transforms.py
    * infer wire shape — recipe expects {"actions": <ndarray>} (7-D)
EOF

if [ "$DO_SERVE" -eq 1 ]; then
  echo "[setup] --serve: exec'ing the openpi server (foreground; Ctrl-C to stop)…"
  exec "${SERVE_CMD[@]}"
fi
