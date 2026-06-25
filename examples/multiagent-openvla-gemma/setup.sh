#!/usr/bin/env bash
#
# Setup for the multi-agent mission (OpenVLA pilot + Gemma planner).
#
# This is the FRAGILE part of multi-agent: it needs TWO Python environments with
# mutually incompatible `transformers` versions. The names say which is which:
#   * env_pilot      → OpenVLA's pinned transformers==4.40.1 (the PILOT + eval)
#   * env_specialist → a modern transformers + torchvision (the Gemma SPECIALIST)
# Mixing them is where setups break. This script builds both, idempotently.
#
# It ONLY sets things up — it does NOT run a mission. After it finishes:
#   source examples/multiagent-openvla-gemma/.env   # load the env vars
#   odyssey run examples/multiagent-openvla-gemma/mission.yaml
#
# Linux + an NVIDIA GPU (24 GB, e.g. L4) assumed — the install pulls torch,
# robosuite/mujoco, and bitsandbytes. Re-runnable: existing venvs/clones are
# reused, not clobbered.
#
# Usage:
#   examples/multiagent-openvla-gemma/setup.sh [options]
# Options:
#   --pilot-venv PATH        pilot/main venv location   (default: <repo>/env_pilot)
#   --specialist-venv PATH   specialist venv location   (default: <repo>/env_specialist)
#   --openvla-repo PATH      where to clone OpenVLA      (default: ~/openvla)
#   --skip-pilot             skip the pilot venv + OpenVLA (already set up)
#   --smoke                  run the planner smoke check at the end (downloads Gemma)
#   -h, --help               show this help

set -euo pipefail

# --- locate the repo root (this script lives in examples/multiagent-openvla-gemma/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- defaults (overridable via flags) — both venvs sit at the repo root, named
#     by role so a plain `ls` tells you which environment is which ---
PILOT_VENV="${PILOT_VENV:-$REPO_ROOT/env_pilot}"
SPECIALIST_VENV="${SPECIALIST_VENV:-$REPO_ROOT/env_specialist}"
OPENVLA_REPO="${OPENVLA_REPO_PATH:-$HOME/openvla}"
SKIP_PILOT=0
RUN_SMOKE=0

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup] WARNING:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[setup] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pilot-venv)      PILOT_VENV="$2"; shift 2 ;;
    --specialist-venv) SPECIALIST_VENV="$2"; shift 2 ;;
    --openvla-repo)    OPENVLA_REPO="$2"; shift 2 ;;
    --skip-pilot)      SKIP_PILOT=1; shift ;;
    --smoke)           RUN_SMOKE=1; shift ;;
    -h|--help)         sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                 die "unknown option: $1 (try --help)" ;;
  esac
done

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null || die "'$PYTHON' not found — install Python 3.10."

cd "$REPO_ROOT"
log "repo root: $REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. env_pilot: odyssey + OpenVLA stack (the PILOT + the robosuite eval)
# ---------------------------------------------------------------------------
if [[ "$SKIP_PILOT" -eq 1 ]]; then
  log "--skip-pilot: assuming the pilot venv at $PILOT_VENV is already set up"
  [[ -x "$PILOT_VENV/bin/odyssey" ]] || warn "no 'odyssey' binary at $PILOT_VENV/bin — is env_pilot really set up?"
else
  if [[ ! -d "$PILOT_VENV" ]]; then
    log "creating pilot venv (env_pilot) at $PILOT_VENV"
    "$PYTHON" -m venv "$PILOT_VENV"
  else
    log "env_pilot already exists at $PILOT_VENV — reusing"
  fi

  log "installing odyssey + extras into env_pilot (pinned to the known-good OpenVLA stack)"
  "$PILOT_VENV/bin/pip" install --upgrade pip >/dev/null
  # torch/vision/audio are CUDA builds (+cu121) that live on the PyTorch index,
  # NOT on PyPI — install them first from there, then the rest from PyPI under
  # the same constraints (matches constraints/openvla-known-good.txt's header).
  log "  [1/2] torch stack from the PyTorch cu121 index"
  "$PILOT_VENV/bin/pip" install -c constraints/openvla-known-good.txt \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
  log "  [2/2] odyssey + extras from PyPI (torch already satisfied)"
  "$PILOT_VENV/bin/pip" install -e ".[all]" -c constraints/openvla-known-good.txt

  # Upstream OpenVLA repo (carries draccus + finetune.py)
  if [[ ! -d "$OPENVLA_REPO/.git" ]]; then
    log "cloning OpenVLA into $OPENVLA_REPO"
    git clone https://github.com/openvla/openvla.git "$OPENVLA_REPO"
  else
    log "OpenVLA repo already at $OPENVLA_REPO — reusing"
  fi
  "$PILOT_VENV/bin/pip" install -e "$OPENVLA_REPO" -c constraints/openvla-known-good.txt
fi

# ---------------------------------------------------------------------------
# 2. env_specialist: the multimodal Gemma planner (modern transformers)
# ---------------------------------------------------------------------------
if [[ ! -d "$SPECIALIST_VENV" ]]; then
  log "creating specialist venv (env_specialist) at $SPECIALIST_VENV"
  "$PYTHON" -m venv "$SPECIALIST_VENV"
else
  log "env_specialist already exists at $SPECIALIST_VENV — reusing"
fi

log "installing the specialist (Gemma) deps into env_specialist — modern transformers + torchvision"
"$SPECIALIST_VENV/bin/pip" install --upgrade pip >/dev/null
"$SPECIALIST_VENV/bin/pip" install -e ".[specialist]" -c constraints/specialist-known-good.txt
SPECIALIST_PYTHON="$SPECIALIST_VENV/bin/python"

# ---------------------------------------------------------------------------
# 3. HuggingFace auth — non-interactive, never blocks setup
#    Gemma 4 (specialist) is ungated; only the OpenVLA-7b PILOT is gated, and it
#    downloads at RUN time. So this is a soft check: warn, don't fail.
# ---------------------------------------------------------------------------
HF_CLI="$PILOT_VENV/bin/huggingface-cli"
if [[ -x "$HF_CLI" ]]; then
  if "$HF_CLI" whoami >/dev/null 2>&1; then
    log "HuggingFace: already authenticated ($("$HF_CLI" whoami 2>/dev/null | head -1))"
  elif [[ -n "${HF_TOKEN:-}" ]]; then
    log "HuggingFace: logging in non-interactively from \$HF_TOKEN"
    "$HF_CLI" login --token "$HF_TOKEN" --add-to-git-credential=false >/dev/null 2>&1 \
      && log "HuggingFace: token accepted" \
      || warn "HuggingFace login with \$HF_TOKEN failed — check the token."
  else
    warn "Not logged in to HuggingFace and \$HF_TOKEN is unset."
    warn "The PILOT 'openvla/openvla-7b' is GATED — accept its license on HF and"
    warn "  export HF_TOKEN=hf_xxx   (or run: $HF_CLI login)"
    warn "before 'odyssey run'. Setup continues (the model downloads at run time)."
  fi
fi

# ---------------------------------------------------------------------------
# 4. Write the env file to source before running (no secrets baked in)
# ---------------------------------------------------------------------------
ENV_FILE="$SCRIPT_DIR/.env"
log "writing env file: $ENV_FILE"
cat > "$ENV_FILE" <<EOF
# Source this before running the multi-agent mission:
#   source examples/multiagent-openvla-gemma/.env
# (use 'source', NOT './' — exports must land in your current shell)
source "$PILOT_VENV/bin/activate"   # env_pilot — the OpenVLA pilot + eval venv

# --- pilot / training (OpenVLA) ---
export OPENVLA_REPO_PATH="$OPENVLA_REPO"
export NCCL_NET=Socket          # GCP single-GPU: bypass the gIB NCCL plugin
export WANDB_MODE=disabled      # OpenVLA calls wandb.init() unconditionally

# --- planner / specialist (out-of-process Gemma venv = env_specialist) ---
export ODYSSEY_SPECIALIST_PYTHON="$SPECIALIST_PYTHON"

# --- evaluation (Robosuite / MuJoCo, headless) ---
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# --- HuggingFace auth (gated PILOT openvla-7b) — set your own token, do NOT commit it ---
# export HF_TOKEN=hf_xxxxxxxx
EOF

# ---------------------------------------------------------------------------
# 5. Optional smoke check (downloads Gemma — opt-in)
# ---------------------------------------------------------------------------
if [[ "$RUN_SMOKE" -eq 1 ]]; then
  log "running the out-of-process planner smoke check (this downloads Gemma)..."
  ODYSSEY_SPECIALIST_PYTHON="$SPECIALIST_PYTHON" \
    "$PILOT_VENV/bin/python" tests/manual/smoke_remote_planner.py \
    || warn "smoke check failed — inspect the output above."
fi

# ---------------------------------------------------------------------------
log "done. env_pilot=$PILOT_VENV  env_specialist=$SPECIALIST_VENV"
cat <<EOF

Next steps:
  1. source examples/multiagent-openvla-gemma/.env
  2. export HF_TOKEN=hf_xxx            # if not already authenticated (gated pilot)
  3. odyssey validate examples/multiagent-openvla-gemma/mission.yaml
  4. odyssey run      examples/multiagent-openvla-gemma/mission.yaml

Smoke-test the planner alone (no GPU-heavy run):
  ODYSSEY_SPECIALIST_PYTHON="$SPECIALIST_PYTHON" \\
    "$PILOT_VENV/bin/python" tests/manual/smoke_remote_planner.py
EOF
