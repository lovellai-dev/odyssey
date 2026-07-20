#!/usr/bin/env bash
#
# End-to-end setup for the GR00T PILOT on LIBERO (single-agent eval). NO Isaac Sim.
#
# GR00T+LIBERO runs in MuJoCo, not Isaac Lab, so it needs only TWO envs (the Isaac
# Sim binary is NOT required):
#
#   1. LIBERO eval venv (env_pilot_libero) — the recipe + a lightweight GR00T ZMQ
#      client; where you run `odyssey run`. Built by the existing LIBERO scripts.
#   2. GR00T policy-server venv (~/Isaac-GR00T/.venv, py3.12) — the model. Built by
#      quickstart-gr00t/setup.sh step [2/4]; its Isaac-Lab step [3/4] SKIPs itself.
#
# This script ORCHESTRATES the existing setup scripts (it does not reimplement
# them) + pre-downloads ONLY the requested LIBERO suite (the server runs offline
# by design, so the checkpoint must be cached first) + tells you how to wire
# `server_python`. Idempotent; safe to re-run.
#
# Usage:   bash examples/franka-libero/setup-gr00t.sh
# Env overrides:
#   SUITE            LIBERO suite to download (default: libero_object)
#   PILOT_VENV       LIBERO eval venv     (default: <repo>/env_pilot_libero)
#   ISAAC_GR00T_DIR  Isaac-GR00T checkout (default: $HOME/Isaac-GR00T)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUITE="${SUITE:-libero_object}"
PILOT_VENV="${PILOT_VENV:-$REPO_ROOT/env_pilot_libero}"
ISAAC_GR00T_DIR="${ISAAC_GR00T_DIR:-$HOME/Isaac-GR00T}"

echo "==> [0/5] system build + render deps (needs sudo)"
sudo apt-get update && sudo apt-get install -y cmake build-essential \
  libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libosmesa6-dev

command -v uv >/dev/null || {
  echo "ERROR: 'uv' not found. Install it, then re-run:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc" >&2
  exit 1
}

echo "==> [1/5] LIBERO eval venv -> $PILOT_VENV"
"$REPO_ROOT/examples/multiagent-openvla-gemma/setup.sh" --pilot-venv "$PILOT_VENV"
# shellcheck disable=SC1091
source "$PILOT_VENV/bin/activate"
"$REPO_ROOT/examples/franka-libero/setup.sh"

echo "==> [2/5] GR00T policy-server venv -> $ISAAC_GR00T_DIR/.venv"
[ -d "$ISAAC_GR00T_DIR/.git" ] || \
  git clone https://github.com/NVIDIA/Isaac-GR00T.git "$ISAAC_GR00T_DIR"
# ODYSSEY_VENV=.venv-core so it doesn't clobber a pre-existing core .venv; the
# Isaac-Lab step SKIPs itself (no ISAAC_PYTHON) — LIBERO does not need Isaac Sim.
ODYSSEY_VENV=.venv-core bash "$REPO_ROOT/examples/quickstart-gr00t/setup.sh"

echo "==> [3/5] pre-download ONLY the '$SUITE' checkpoint (server runs offline)"
# The full nvidia/GR00T-N1.7-LIBERO repo bundles 4 suites (~25-30GB); one is ~5-8GB.
huggingface-cli download nvidia/GR00T-N1.7-LIBERO --include "$SUITE/*"

echo "==> [4/5] wire server_python into the mission (manual — avoids dirtying git blindly)"
echo "    Edit examples/franka-libero/mission-gr00t.yaml and set under task config:"
echo "      server_python: $ISAAC_GR00T_DIR/.venv/bin/python"

cat <<EOF

==> [5/5] done. To run:
  source $PILOT_VENV/bin/activate
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
  cd $REPO_ROOT && odyssey run examples/franka-libero/mission-gr00t.yaml

  # server log = source of truth:
  tail -n 60 /tmp/gr00t_server_5555.log
EOF
