#!/usr/bin/env bash
#
# Setup for the Cosmos 3 × RoboLab eval (custom runner) — GPU box target (H100 for
# Nano 16B; Edge 4B fits smaller GPUs).
#
# It ONLY sets things up — it does NOT run a mission (and does NOT boot the policy
# server unless you pass --serve). THREE pieces, wired by host:port / paths:
#
#   1. Odyssey client env      — a plain venv with this repo (the RobolabRunner
#                                itself has no sim deps). No sim here.
#   2. cosmos-framework server — its own uv env -> serves a Policy-DROID checkpoint
#                                (action_policy_server_robolab, WebSocket)
#   3. RoboLab checkout        — NVlabs' Isaac Lab sim client (docker build);
#                                the runner launches its policies/cosmos3/run.py
#
# ─── ⚠️ NOT YET VALIDATED ON HARDWARE ──────────────────────────────────────────────
#  All three halves follow the NVIDIA cookbook (run_policy_with_cosmos_framework.md)
#  but have NOT been run through to a green rollout here. First-smoke watch-list:
#  RoboLab's actual result output (JSON files vs stdout — the bridge's parser is
#  deliberately tolerant and must be pinned; prefer --results_glob), and the exact
#  run.py flags for pointing at the server.
#
# Usage:
#   bash examples/cosmos3-robolab/setup.sh                 # set up everything
#   bash examples/cosmos3-robolab/setup.sh --serve         # ...then exec the server (foreground)
#
#   --venv PATH         odyssey venv (default: <repo>/env_cosmos3_robolab)
#   --cosmos-dir PATH   cosmos-framework checkout (default: $HOME/cosmos-framework)
#   --robolab-dir PATH  RoboLab checkout (default: $HOME/RoboLab)
#   --checkpoint ID     policy the server serves (default: nvidia/Cosmos3-Nano-Policy-DROID;
#                       Edge additionally needs --format-prompt-as-json True — see below)
#   --port PORT         server port baked into the print-out (default 8000)
#   --cuda12            use cosmos-framework's cu128-train group (CUDA 12.x driver)
#   --no-docker         skip the RoboLab docker build (bare checkout only)
#   --serve             after setup, exec the policy server in the foreground (blocks)
#
# Published policy checkpoints are public on HF (no token needed).
#
# Linux + NVIDIA GPU assumed. Re-runnable / idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VENV="${VENV:-$REPO_ROOT/env_cosmos3_robolab}"
COSMOS_DIR="${COSMOS_DIR:-$HOME/cosmos-framework}"
ROBOLAB_DIR="${ROBOLAB_DIR:-$HOME/RoboLab}"
CHECKPOINT="nvidia/Cosmos3-Nano-Policy-DROID"
PORT="8000"
CUDA_GROUP="cu130-train"
DO_DOCKER=1
DO_SERVE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --cosmos-dir) COSMOS_DIR="$2"; shift 2 ;;
    --robolab-dir) ROBOLAB_DIR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --cuda12) CUDA_GROUP="cu128-train"; shift ;;
    --no-docker) DO_DOCKER=0; shift ;;
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
echo "==> [1/4] odyssey venv ($VENV) — the bridge itself is stdlib-only"
# ---------------------------------------------------------------------------
uv venv --python 3.10 "$VENV"
PYBIN="$VENV/bin/python"
uv pip install --python "$PYBIN" -e "$REPO_ROOT[dev,huggingface]"
"$PYBIN" -c "import odyssey; print('[setup] odyssey: OK')"

# ---------------------------------------------------------------------------
echo "==> [2/4] cosmos-framework SERVER env — best-effort; see ⚠️ header"
# ---------------------------------------------------------------------------
if [ ! -d "$COSMOS_DIR/.git" ]; then
  echo "[setup] cloning NVIDIA/cosmos-framework into $COSMOS_DIR"
  git clone https://github.com/NVIDIA/cosmos-framework.git "$COSMOS_DIR"
else
  echo "[setup] cosmos-framework already at $COSMOS_DIR — reusing"
fi
if ( cd "$COSMOS_DIR" && uv sync --all-extras "--group=$CUDA_GROUP" --group=policy-server ); then
  echo "[setup] cosmos-framework env synced under $COSMOS_DIR (.venv)"
else
  echo "[setup] WARNING: 'uv sync' in $COSMOS_DIR did not complete — finish it per the" >&2
  echo "        cosmos-framework setup docs (NGC container / CUDA groups) before serving." >&2
fi
# The server's --checkpoint-path expects a LOCAL DIRECTORY (validated on the
# H100 smoke: an HF id raises "Checkpoint directory does not exist"). Published
# policy checkpoints are public (no HF token needed) — stage them explicitly.
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
echo "==> [3/4] RoboLab checkout (client sim)"
# ---------------------------------------------------------------------------
if [ ! -d "$ROBOLAB_DIR/.git" ]; then
  echo "[setup] cloning NVlabs/RoboLab into $ROBOLAB_DIR"
  git clone https://github.com/NVlabs/RoboLab.git "$ROBOLAB_DIR"
else
  echo "[setup] RoboLab already at $ROBOLAB_DIR — reusing"
fi
if [ "$DO_DOCKER" -eq 1 ]; then
  if command -v docker >/dev/null; then
    ( cd "$ROBOLAB_DIR" && ./docker/build_docker.sh latest ) \
      || echo "[setup] WARNING: RoboLab docker build failed — build it manually per its README." >&2
  else
    echo "[setup] WARNING: docker not found — skipping the RoboLab image build (--no-docker to silence)." >&2
  fi
fi

# ---------------------------------------------------------------------------
echo "==> [4/4] done — flow"
# ---------------------------------------------------------------------------
SERVE_CMD=( uv run --project "$COSMOS_DIR" python -m
            cosmos_framework.scripts.action_policy_server_robolab
            "--checkpoint-path" "$CHECKPOINT" "--port" "$PORT" )
case "$CHECKPOINT" in
  *Edge*) SERVE_CMD+=( "--format-prompt-as-json" "True" ) ;;  # Edge-DROID trains with JSON prompts
esac

cat <<EOF

[setup] Done. Flow (externally-served: you start the server):

  TERMINAL 1 — serve the DROID policy (cosmos-framework env):
    ${SERVE_CMD[*]}
    # --checkpoint-path must be a LOCAL directory; if the port is taken by
    # another service (ss -tlnp | grep $PORT), pick another one.
    # "Access denied. This repository requires approval" at startup means the
    # GATED nvidia/Cosmos-Guardrail1 (guardrails default ON): request HF access
    # + export HF_TOKEN, or disable guardrails (content moderation, not actions):
    #   sed -i 's/guardrails: bool = True/guardrails: bool = False/' \\
    #       $COSMOS_DIR/cosmos_framework/inference/common/args.py

  TERMINAL 2 — run the mission (this repo's venv; the RobolabRunner launches RoboLab):
    source $VENV/bin/activate
    # edit examples/cosmos3-robolab/mission.yaml first:
    #   config.robolab_root: $ROBOLAB_DIR
    #   config.eval_python: the Isaac Sim interpreter (RoboLab docker/venv)
    #   config.remote_host/remote_port: the policy server address
    odyssey validate examples/cosmos3-robolab/mission.yaml
    odyssey run      examples/cosmos3-robolab/mission.yaml

  Results: <robolab_root>/output/odyssey_<task-id>/episode_results.jsonl,
  scored into the mission summary and copied into the task output dir.
EOF

if [ "$DO_SERVE" -eq 1 ]; then
  echo "[setup] --serve: exec'ing the RoboLab policy server (foreground; Ctrl-C to stop)…"
  exec "${SERVE_CMD[@]}"
fi
