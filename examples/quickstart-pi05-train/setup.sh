#!/usr/bin/env bash
#
# Setup for the π0.5 (openpi) FINE-TUNE quickstart — single-GPU box (A100/H100 class).
#
# It ONLY sets things up — it does NOT run a mission and does NOT train.
#
# ─── ONE environment, by design (opposite of the eval quickstart) ──────────────────
#  The EVAL quickstart (examples/quickstart-pi05/) uses TWO envs wired by host:port,
#  because π0.5 is served out-of-process. TRAINING is different: the odyssey `pi05`
#  runner shells out to openpi's own scripts (compute_norm_stats.py, train.py) via
#  `sys.executable` — the SAME interpreter that runs `odyssey`. So openpi and odyssey
#  must live in ONE env. This script builds openpi's uv env, then installs odyssey
#  INTO it, and you run training via `<openpi>/.venv/bin/odyssey` (mirrors the
#  validated GR00T recipe: `~/Isaac-GR00T/.venv/bin/odyssey`).
#
# ─── ⚠️ NOT YET VALIDATED ON HARDWARE ──────────────────────────────────────────────
#  The odyssey-side wiring is complete + unit-tested on CPU, but no end-to-end π0.5
#  fine-tune has been run through this script. Treat the openpi steps as a best-effort
#  scaffold — confirm the exact `train.py` / `compute_norm_stats.py` flags and the
#  TrainConfig registration against YOUR openpi version (see README.md).
#
# ─── FAST note ─────────────────────────────────────────────────────────────────────
#  Nothing to install for FAST. π0.5 training uses a FAST-tokenized objective, but it
#  is INTERNAL to openpi's `pi05` TrainConfig/model — the runner never touches it, and
#  odyssey's pi05_fast.py (eval-side scaffolding) is not on this path.
#
# Usage:
#   bash examples/quickstart-pi05-train/setup.sh
#   bash examples/quickstart-pi05-train/setup.sh --openpi-dir ~/openpi --dataset /data/ur10e_drugsort_v0/ur10e_partial_cond_aug
#
#   --openpi-dir PATH   openpi checkout / env (default: $HOME/openpi)
#   --dataset PATH      unpacked LeRobot dataset dir (optional; sanity-checked + used
#                       to print the HF_LEROBOT_HOME / data.repo_id linkage)
#   --config-name NAME  openpi TrainConfig you will register (default: pi05_ur10e_drugsort)
#   -h | --help         show this header
#
# Linux + NVIDIA GPU assumed. Re-runnable / idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OPENPI_DIR="${OPENPI_DIR:-$HOME/openpi}"
DATASET=""
CONFIG_NAME="pi05_ur10e_drugsort"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --openpi-dir) OPENPI_DIR="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --config-name) CONFIG_NAME="$2"; shift 2 ;;
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
echo "==> [1/4] openpi checkout ($OPENPI_DIR)"
# ---------------------------------------------------------------------------
if [ ! -d "$OPENPI_DIR/.git" ]; then
  echo "[setup] cloning Physical-Intelligence/openpi into $OPENPI_DIR"
  git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
else
  echo "[setup] openpi already at $OPENPI_DIR — reusing"
fi

# ---------------------------------------------------------------------------
echo "==> [2/4] openpi training env (uv sync — JAX/torch, the FULL openpi, GPU deps)"
# ---------------------------------------------------------------------------
# This is the heavy step: it builds openpi's .venv from its lockfile (large
# JAX/CUDA download). compute_norm_stats.py + train.py run from THIS env.
if ( cd "$OPENPI_DIR" && uv sync ); then
  echo "[setup] openpi env synced under $OPENPI_DIR/.venv"
else
  echo "[setup] ERROR: 'uv sync' in $OPENPI_DIR failed — finish it per openpi's README" >&2
  echo "        (CUDA wheels, GIT_LFS_SKIP_SMUDGE, etc. may apply) before continuing." >&2
  exit 1
fi
PYBIN="$OPENPI_DIR/.venv/bin/python"

# ---------------------------------------------------------------------------
echo "==> [3/4] install odyssey INTO the openpi env (so sys.executable sees both)"
# ---------------------------------------------------------------------------
# Base install only (no [dev]) — the training runner is pure subprocess
# orchestration + the pydantic spec; it needs none of the heavy test/eval extras,
# and keeping them out avoids clashing with openpi's pinned numpy/torch.
uv pip install --python "$PYBIN" -e "$REPO_ROOT"

# ---------------------------------------------------------------------------
echo "==> [4/4] verify the combined env"
# ---------------------------------------------------------------------------
"$PYBIN" - <<'PY'
import importlib.util as u, sys
missing = [m for m in ("odyssey", "openpi") if u.find_spec(m) is None]
print(f"[setup] odyssey : {'OK' if u.find_spec('odyssey') else 'MISSING'}")
print(f"[setup] openpi  : {'OK' if u.find_spec('openpi') else 'MISSING'}")
if missing:
    print(f"[setup] VERIFY FAILED: missing {missing}", file=sys.stderr)
    sys.exit(1)
PY

# Optional dataset sanity check + the repo_id linkage the runner relies on.
DATA_HINT=""
if [ -n "$DATASET" ]; then
  if [ -f "$DATASET/meta/info.json" ]; then
    HFLR="$(dirname "$DATASET")"
    REPO_ID="$(basename "$DATASET")"
    DATA_HINT=$'\n'"[setup] dataset OK: $DATASET"$'\n'"        -> runner sets HF_LEROBOT_HOME=$HFLR ; data.repo_id must be '$REPO_ID'"
  else
    echo "[setup] WARNING: $DATASET/meta/info.json not found — is this an unpacked LeRobot v2.1 dir?" >&2
  fi
fi

cat <<EOF

[setup] Done.${DATA_HINT}

  Next — register the openpi TrainConfig (one-time), then run the fine-tune:

  1) Add a TrainConfig named '$CONFIG_NAME' to:
       $OPENPI_DIR/src/openpi/training/config.py
     Base it on the shipped 'pi05_libero' config; point its data at your LeRobot
     dataset (repo_id = the dataset folder name) and map the cameras
     (observation.images.exterior -> base_0_rgb, .wrist -> left_wrist_0_rgb).
     Drop the GR00T-only grasp_target channel. See README.md.

  2) Export the checkout path so the runner finds openpi's scripts:
       export OPENPI_REPO_PATH=$OPENPI_DIR

  3) Run the mission with the openpi-env odyssey (NOT a system odyssey):
       export OPENPI_REPO_PATH=$OPENPI_DIR
       $OPENPI_DIR/.venv/bin/odyssey validate examples/quickstart-pi05-train/mission.yaml
       $OPENPI_DIR/.venv/bin/odyssey run      examples/quickstart-pi05-train/mission.yaml

     The runner then executes, in order:
       python scripts/compute_norm_stats.py --config-name $CONFIG_NAME
       python scripts/train.py $CONFIG_NAME --exp-name <task> --overwrite [overrides]
     with cwd=<task output_dir>, so ./assets and ./checkpoints land under ~/.odyssey/runs.

  First-run watch-list (see README + the PR's known gap):
    * config_name must be registered in openpi's config.py (else train.py errors)
    * data.repo_id in mission.yaml must equal the dataset folder name
    * flow-matching model -> no FAST flags to pass; FAST is internal to openpi
EOF
