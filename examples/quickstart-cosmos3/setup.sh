#!/usr/bin/env bash
#
# Setup for the Cosmos 3 (WAM) LIBERO eval quickstart — GPU box target (H100 for
# Nano 16B; Edge 4B fits smaller GPUs — Nano does NOT fit an L4 24GB in bf16).
#
# It ONLY sets things up — it does NOT run a mission (and, being externally-served,
# it does NOT boot the policy server for you unless you pass --serve). Two SEPARATE
# environments by design (cosmos-framework manages its own uv env with a heavy
# torch/CUDA stack); they are wired by host:port, not co-installed:
#
#   1. Odyssey client env      — a dedicated venv (this repo + LIBERO stack)
#                                -> drives the mission, holds the LIBERO/MuJoCo env.
#                                The Cosmos3 client itself is STDLIB urllib — no
#                                cosmos package is needed on this side at all.
#   2. cosmos-framework server — its own uv env  -> serves Cosmos 3 on host:port
#                                (action_policy_server_libero, HTTP /predict + /info)
#
# ─── ⚠️ NOT YET VALIDATED ON HARDWARE ──────────────────────────────────────────────
#  The odyssey client half reuses the validated LIBERO install (franka-libero /
#  quickstart-pi05 pins). The cosmos-framework SERVER half follows NVIDIA's cookbook
#  (uv sync groups, NGC container recommended) but has NOT been run through to a
#  green rollout. First-smoke watch-list: the /predict wire end-to-end, concat_view
#  ordering (third-person LEFT | wrist RIGHT assumed), rot6d axes + gripper polarity
#  (no fix-up applied), and the domain_name/image_size values.
#
#  ⚠ CHECKPOINT REALITY: only DROID policies are published (OOD on LIBERO — smoke
#  only, success not expected). A real LIBERO score needs your own SFT export
#  (training runner = follow-up PR); pass it via --checkpoint.
#
# Usage:
#   bash examples/quickstart-cosmos3/setup.sh                 # set up both halves
#   bash examples/quickstart-cosmos3/setup.sh --serve         # ...then exec the server (foreground)
#
#   --venv PATH        odyssey client venv (default: <repo>/env_pilot_cosmos3)
#   --cosmos-dir PATH  cosmos-framework checkout (default: $HOME/cosmos-framework)
#   --checkpoint ID    checkpoint the server serves — HF id or an export_model dir
#                      (default: nvidia/Cosmos3-Nano-Policy-DROID)
#   --host / --port    server bind address baked into the print-out (default 127.0.0.1:8000)
#   --cuda12           use cosmos-framework's cu128-train group (CUDA 12.x driver)
#   --serve            after setup, exec the policy server in the foreground (blocks)
#
# Env overrides: LIBERO_DIR (default $HOME/LIBERO). Published policy checkpoints are public (no HF token).
#
# Linux + NVIDIA GPU assumed. Re-runnable / idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VENV="${VENV:-$REPO_ROOT/env_pilot_cosmos3}"
COSMOS_DIR="${COSMOS_DIR:-$HOME/cosmos-framework}"
LIBERO_DIR="${LIBERO_DIR:-$HOME/LIBERO}"
CHECKPOINT="nvidia/Cosmos3-Nano-Policy-DROID"
HOST="127.0.0.1"
PORT="8000"
CUDA_GROUP="cu130-train"
DO_SERVE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --cosmos-dir) COSMOS_DIR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --cuda12) CUDA_GROUP="cu128-train"; shift ;;
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
echo "==> [2/5] odyssey client venv ($VENV): odyssey + LIBERO stack (no cosmos deps)"
# ---------------------------------------------------------------------------
uv venv --python 3.10 "$VENV"
PYBIN="$VENV/bin/python"
uv pip install --python "$PYBIN" -e "$REPO_ROOT[dev,huggingface]"

# LIBERO client stack — the same known-good pins quickstart-pi05 uses (standalone
# venv, so mujoco/numpy ARE pinned; see that script's rationale).
uv pip install --python "$PYBIN" \
  robosuite==1.4.0 mujoco==2.3.2 "numpy<2" \
  bddl==1.0.1 robomimic==0.2.0 hydra-core==1.2.0 easydict==1.9 \
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
# would otherwise block `odyssey run`).
if [ ! -f "$HOME/.libero/config.yaml" ]; then
  yes N | "$PYBIN" -c "import libero.libero" >/dev/null 2>&1 || true
fi

# ---------------------------------------------------------------------------
echo "==> [3/5] verify the client env (Cosmos3 client is stdlib — nothing extra)"
# ---------------------------------------------------------------------------
"$PYBIN" - <<'PY'
import sys
try:
    import odyssey                                        # noqa: F401
    import libero
    from libero.libero import benchmark                   # noqa: F401
    import robosuite, imageio_ffmpeg                      # noqa: F401
    from odyssey.runners.models.cosmos3 import make_cosmos3_pilot  # noqa: F401
    print("[setup] odyssey + cosmos3 pilot : OK (client is stdlib urllib)")
    print(f"[setup] libero                 : path={list(libero.__path__)}")
    print(f"[setup] robosuite              : {robosuite.__version__}  (1.4.0 expected)")
except Exception as e:  # noqa: BLE001
    print(f"[setup] VERIFY FAILED: {e}", file=sys.stderr)
    sys.exit(1)
PY

# ---------------------------------------------------------------------------
echo "==> [4/5] cosmos-framework SERVER env (separate) — best-effort; see ⚠️ header"
# ---------------------------------------------------------------------------
if [ ! -d "$COSMOS_DIR/.git" ]; then
  echo "[setup] cloning NVIDIA/cosmos-framework into $COSMOS_DIR"
  git clone https://github.com/NVIDIA/cosmos-framework.git "$COSMOS_DIR"
else
  echo "[setup] cosmos-framework already at $COSMOS_DIR — reusing"
fi
# NVIDIA recommends the NGC PyTorch container (nvcr.io/nvidia/pytorch:25.09-py3;
# :25.06-py3 for CUDA 12.8) — this bare-metal sync is the cookbook's alternative.
# Downloads a large torch/CUDA stack; non-fatal: warn + print manual steps on failure.
if ( cd "$COSMOS_DIR" && uv sync --all-extras "--group=$CUDA_GROUP" --group=policy-server ); then
  echo "[setup] cosmos-framework env synced under $COSMOS_DIR (.venv)"
else
  echo "[setup] WARNING: 'uv sync' in $COSMOS_DIR did not complete — finish it per the" >&2
  echo "        cosmos-framework setup docs (NGC container / CUDA groups) before serving." >&2
fi
# The server's --checkpoint-path expects a LOCAL DIRECTORY (validated on the
# H100 smoke: an HF id raises "Checkpoint directory does not exist"), so stage
# HF checkpoints explicitly. Published policy checkpoints are public (no token).
if [[ "$CHECKPOINT" == */* && ! -d "$CHECKPOINT" && ! -e "$CHECKPOINT" ]]; then
  CKPT_DIR="$HOME/checkpoints/$(basename "$CHECKPOINT")"
  if [ ! -d "$CKPT_DIR" ]; then
    echo "[setup] downloading $CHECKPOINT -> $CKPT_DIR (hf download)"
    uv run --project "$COSMOS_DIR" hf download "$CHECKPOINT" --local-dir "$CKPT_DIR" \
      || echo "[setup] WARNING: checkpoint download failed — stage it manually before serving." >&2
  else
    echo "[setup] checkpoint already staged at $CKPT_DIR — reusing"
  fi
  CHECKPOINT="$CKPT_DIR"
fi

# ---------------------------------------------------------------------------
echo "==> [5/5] done — two-terminal flow"
# ---------------------------------------------------------------------------
SERVE_CMD=( uv run --project "$COSMOS_DIR" python -m
            cosmos_framework.scripts.action_policy_server_libero
            "--checkpoint-path" "$CHECKPOINT" "--port" "$PORT" )

cat <<EOF

[setup] Done. Two-terminal flow (externally-served: you start the server):

  TERMINAL 1 — serve Cosmos 3 (cosmos-framework env):
    ${SERVE_CMD[*]}
    # Any family member works; --checkpoint-path must be a LOCAL directory
    # (an hf-downloaded repo or an export_model dir). If port $PORT is taken by
    # another service (check: ss -tlnp | grep $PORT), pick another and mirror it
    # in the mission's config.port. Verify OUR server answers:
    #   curl http://$HOST:$PORT/info   # shows the resolved action_chunk_size

  TERMINAL 2 — run the eval (this repo's venv):
    source $VENV/bin/activate
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    # point the mission at the server if you changed host/port:
    #   examples/quickstart-cosmos3/mission.yaml -> config.host/config.port
    odyssey validate examples/quickstart-cosmos3/mission.yaml
    odyssey run      examples/quickstart-cosmos3/mission.yaml

  Per-episode MP4s:
    find ~/.odyssey/runs -path "*/videos/*.mp4" -exec ls -lh {} \\;

  First-smoke watch-list (see the mission header + docs/migration doc):
    * /predict wire end-to-end + GET /info chunk size adoption
    * concat_view order (third-person LEFT | wrist RIGHT assumed) + 180° flip
    * rot6d axes + gripper polarity — NO fix-up applied (patch cosmos3_transforms.py)
    * DROID checkpoints are OOD on LIBERO: episodes must complete; success not expected
EOF

if [ "$DO_SERVE" -eq 1 ]; then
  echo "[setup] --serve: exec'ing the Cosmos3 policy server (foreground; Ctrl-C to stop)…"
  exec "${SERVE_CMD[@]}"
fi
