#!/usr/bin/env bash
# Set up the GR00T + Odyssey + Isaac Lab eval stack with uv.
#
# Three SEPARATE environments by design (their torch/python/CUDA ABIs clash);
# they are wired together by interpreter env vars, not co-installed:
#
#   1. Odyssey core      — uv venv, this repo            -> drives missions / training runner
#   2. GR00T server      — uv venv in $ISAAC_GR00T_DIR   -> GR00T_VENV_PYTHON  (torch 2.7.1+cu128)
#   3. Isaac Lab eval    — Isaac Sim's python            -> ISAAC_PYTHON       (isaacsim 5.1.0)
#
# Pins live in constraints/{gr00t-server,isaac-eval}-known-good.txt (same
# pattern as constraints/{openvla,specialist}-known-good.txt). Idempotent.
#
# Usage:  bash examples/quickstart-gr00t/setup.sh
# Overridable: ODYSSEY_DIR, ISAAC_GR00T_DIR, ISAACLAB_PATH, ISAAC_PYTHON, HF_HOME
set -euo pipefail

# This script lives at examples/quickstart-gr00t/setup.sh -> repo root is two up.
ODYSSEY_DIR="${ODYSSEY_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ISAAC_GR00T_DIR="${ISAAC_GR00T_DIR:-$HOME/Isaac-GR00T}"
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
ISAAC_PYTHON="${ISAAC_PYTHON:-$HOME/miniconda3/envs/isaaclab/bin/python}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
C="$ODYSSEY_DIR/constraints"
command -v uv >/dev/null || { echo "uv not found — install from https://docs.astral.sh/uv/"; exit 1; }

echo "==> [1/4] Odyssey core env (uv)"
cd "$ODYSSEY_DIR"
uv venv --python 3.10 .venv
# huggingface extra: the GR00T training runner uses odyssey's HF provider.
uv pip install --python .venv/bin/python -e ".[dev,huggingface]"
GR00T_VENV_PYTHON="${GR00T_VENV_PYTHON:-$ISAAC_GR00T_DIR/.venv/bin/python}"

echo "==> [2/4] GR00T policy-server venv (uv) — torch 2.7.1+cu128, separate ABI"
if [ -d "$ISAAC_GR00T_DIR" ]; then
  cd "$ISAAC_GR00T_DIR"
  uv venv --python 3.10 .venv
  uv pip install --python .venv/bin/python \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      -c "$C/gr00t-server-known-good.txt" -e .
  echo "    NOTE: install flash-attn (cu128 x86_64 wheel) per Isaac-GR00T's pyproject."
  "$GR00T_VENV_PYTHON" -c "import gr00t; print('    gr00t OK')" 2>/dev/null || echo "    (gr00t import deferred — finish flash-attn install)"
else
  echo "    SKIP: \$ISAAC_GR00T_DIR ($ISAAC_GR00T_DIR) not found — clone NVIDIA/Isaac-GR00T first."
fi

echo "==> [3/4] Isaac Lab eval env (client transport into Isaac's python)"
if [ -x "$ISAAC_PYTHON" ]; then
  # additive only — never clobber Isaac's own deps; odyssey installed --no-deps
  "$ISAAC_PYTHON" -m uv pip install --python "$ISAAC_PYTHON" \
      -c "$C/isaac-eval-known-good.txt" pyzmq msgpack msgpack-numpy aiosqlite nest_asyncio \
    || "$ISAAC_PYTHON" -m pip install -c "$C/isaac-eval-known-good.txt" pyzmq msgpack msgpack-numpy aiosqlite nest_asyncio
  "$ISAAC_PYTHON" -m pip install --no-deps -e "$ODYSSEY_DIR"
  echo "    isaaclab importable: $("$ISAAC_PYTHON" -c 'import importlib.util as u; print(bool(u.find_spec("isaaclab")))' 2>/dev/null)"
else
  echo "    SKIP: \$ISAAC_PYTHON ($ISAAC_PYTHON) not found — install Isaac Lab / Isaac Sim first."
fi

echo "==> [4/4] Write the interpreter map -> $ODYSSEY_DIR/odyssey-eval-env.sh"
cat > "$ODYSSEY_DIR/odyssey-eval-env.sh" <<EOF
# Source this before running a CC/Odyssey GR00T Isaac-Lab eval.
# IMPORTANT: launch with \`env -u PYTHONPATH -u VIRTUAL_ENV …\` so the ROS Jazzy
# PYTHONPATH leak / a parent venv don't override Isaac's bundled python.
export ODYSSEY_DIR="$ODYSSEY_DIR"
export ISAAC_GR00T_DIR="$ISAAC_GR00T_DIR"
export ISAACLAB_PATH="$ISAACLAB_PATH"
export GR00T_VENV_PYTHON="$GR00T_VENV_PYTHON"
export ISAAC_PYTHON="$ISAAC_PYTHON"
export HF_HOME="$HF_HOME"
EOF
echo "==> done. Source: . $ODYSSEY_DIR/odyssey-eval-env.sh ; see examples/quickstart-gr00t/README.md"
