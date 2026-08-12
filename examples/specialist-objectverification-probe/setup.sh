#!/usr/bin/env bash
# One-command setup for the object-verification bake-off (RoboBrain 2.5 vs SAM 3.1).
#
# Captures everything the two arms need on a fresh H100 box: serve each model,
# run the probe over the drug-sort rollouts, and build the comparative
# scoreboard. Sub-commands are idempotent — re-run any of them freely.
#
#   ./setup.sh serve-robobrain     # RoboBrain 2.5 via docker vLLM on :$RB_PORT
#   ./setup.sh serve-sam           # SAM 3.1 real, in a dedicated transformers>=5.14 venv, on :$SAM_PORT
#   ./setup.sh serve-sam-fake      # SAM endpoint with NO weights (plumbing / dashboard demo)
#   ./setup.sh run-robobrain [side|wrist]   # probe RoboBrain (default: side, the good camera)
#   ./setup.sh run-sam                       # probe SAM 3.1
#   ./setup.sh scoreboard          # build the bake-off scoreboard from the result JSONs
#   ./setup.sh all-fake            # serve-sam-fake + run both + scoreboard (no gated weights, quick smoke)
#
# WHY TWO PYTHONS (learned the hard way):
#   * PROBE CLIENT ($CLIENT_PY) imports odyssey's judge -> needs pydantic + imageio
#     + pillow. The Isaac-GR00T venv has all three; the minimal ~/odyssey-eval-venv
#     does NOT (ModuleNotFoundError: pydantic).
#   * SAM SERVER ($SAM_PY) needs transformers>=5.14 for the Sam3 classes. The
#     Isaac-GR00T venv ships 4.57 (no Sam3Model), so SAM gets its OWN venv — do
#     NOT upgrade transformers in the GR00T venv, it will break the pilot.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC="$(cd "$HERE/../.." && pwd)/src"

# ---- knobs (override via env) ----------------------------------------------
VIDEOS_DIR="${VIDEOS_DIR:-$HOME/cosmos3_probe_videos_success}"
INSTRUCTION="${INSTRUCTION:-pick up the red capsule and place it in the blue tray}"
OBJECTS="${OBJECTS:-red capsule, blue tray}"
DISTRACTORS="${DISTRACTORS:-green bottle, yellow block}"
MAX_VIDEOS="${MAX_VIDEOS:-3}"

RB_MODEL="${RB_MODEL:-BAAI/RoboBrain2.5-8B-NV}"
RB_PORT="${RB_PORT:-8005}"
RB_UTIL="${RB_UTIL:-0.40}"          # 0.40*80=32GB; needs ~32GB FREE (stop other vLLMs first)
RB_MAXLEN="${RB_MAXLEN:-8192}"
RB_IMAGE="${RB_IMAGE:-vllm/vllm-omni:v0.26.0}"   # plain `vllm serve`, NOT --omni
RB_CTR="${RB_CTR:-robobrain-objverif}"

SAM_MODEL="${SAM_MODEL:-facebook/sam3.1}"         # GATED: manual approval required (see serve-sam)
SAM_PORT="${SAM_PORT:-8006}"
SAM_VENV="${SAM_VENV:-$HOME/sam3-venv}"
SAM_PY="$SAM_VENV/bin/python"

CLIENT_PY="${CLIENT_PY:-$HOME/Isaac-GR00T/.venv/bin/python}"   # has pydantic+imageio+pillow
OUT_DIR="${OUT_DIR:-/tmp}"

export PYTHONPATH="$REPO_SRC${PYTHONPATH:+:$PYTHONPATH}"

log() { printf '\033[36m[objverif]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[objverif] %s\033[0m\n' "$*" >&2; exit 1; }

# ---- RoboBrain arm ----------------------------------------------------------
serve_robobrain() {
  command -v docker >/dev/null || die "docker not found"
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  log "RoboBrain: $free MiB free (needs ~$(python3 -c "print(int($RB_UTIL*80000))") MiB; stop other vLLMs if short)"
  sudo docker rm -f "$RB_CTR" 2>/dev/null || true
  sudo docker run -d --name "$RB_CTR" --gpus all --network host \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    "$RB_IMAGE" \
    vllm serve "$RB_MODEL" --host 0.0.0.0 --port "$RB_PORT" \
      --gpu-memory-utilization "$RB_UTIL" --max-model-len "$RB_MAXLEN" >/dev/null
  log "serving $RB_MODEL on :$RB_PORT — polling /v1/models (up to ~5min)"
  for _ in $(seq 1 20); do
    curl -s -m3 "http://127.0.0.1:$RB_PORT/v1/models" 2>/dev/null | grep -q RoboBrain && { log "RoboBrain READY"; return 0; }
    sudo docker ps --filter "name=$RB_CTR" --format '{{.Names}}' | grep -q "$RB_CTR" || {
      sudo docker logs "$RB_CTR" 2>&1 | grep -iE "ValueError|out of memory|max seq" | tail -3; die "RoboBrain container died (see log above — usually VRAM)"; }
    sleep 15
  done
  die "RoboBrain did not become ready"
}

run_robobrain() {
  local view="${1:-side}"   # `side` = workspace plane (small objects need it); `wrist` = gripper close-up
  log "probing RoboBrain (view=$view) over $MAX_VIDEOS rollouts"
  "$CLIENT_PY" "$HERE/robobrain_probe.py" \
    --checkpoint "$RB_MODEL" --out-json "$OUT_DIR/rb_${view}.json" \
    --base_url "http://127.0.0.1:$RB_PORT/v1" --model "$RB_MODEL" \
    --videos_dir "$VIDEOS_DIR" --instruction "$INSTRUCTION" \
    --objects "$OBJECTS" --distractors "$DISTRACTORS" \
    --view "$view" --upscale 3 --max_videos "$MAX_VIDEOS" --max_tokens 512
  log "wrote $OUT_DIR/rb_${view}.json"
}

# ---- SAM 3.1 arm ------------------------------------------------------------
sam_venv() {
  if [ ! -x "$SAM_PY" ]; then
    log "creating SAM venv at $SAM_VENV (transformers>=5.14 for Sam3)"
    python3 -m venv "$SAM_VENV"
    "$SAM_PY" -m pip install -q --upgrade pip
    "$SAM_PY" -m pip install -q "transformers>=5.14" torch pillow numpy
  fi
  "$SAM_PY" -c "from transformers import Sam3Model" 2>/dev/null \
    || die "Sam3Model still not importable — bump transformers in $SAM_VENV"
}

sam_gated_check() {
  log "checking gated access to $SAM_MODEL"
  HF_HUB_OFFLINE=0 "$SAM_PY" - "$SAM_MODEL" <<'PY' || die "no access — request it (see message above) then re-run"
import sys
from huggingface_hub import hf_hub_download
try:
    hf_hub_download(sys.argv[1], "config.json")
    print("[objverif] gated access OK")
except Exception as e:
    print("[objverif] GATED: %s" % str(e)[:160])
    print("[objverif] ACTION: log in as the HF account that owns the token")
    print("[objverif]   (huggingface-cli login), open https://huggingface.co/%s" % sys.argv[1])
    print("[objverif]   and click 'Agree and access repository'; approval for manual-gated")
    print("[objverif]   repos can take a while. Then re-run serve-sam.")
    sys.exit(1)
PY
}

serve_sam() {
  sam_venv
  sam_gated_check
  local free; free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  log "SAM: $free MiB free (SAM3 ViT runs at 1008px — free VRAM if it OOMs)"
  log "serving REAL $SAM_MODEL on :$SAM_PORT (first run downloads weights)"
  exec "$SAM_PY" "$HERE/sam_server.py" --host 0.0.0.0 --port "$SAM_PORT" --model "$SAM_MODEL"
}

serve_sam_fake() {
  log "serving FAKE SAM (no weights) on :$SAM_PORT — plumbing/dashboard only"
  SAM_SERVER_FAKE=1 exec "$CLIENT_PY" "$HERE/sam_server.py" --host 0.0.0.0 --port "$SAM_PORT"
}

run_sam() {
  log "probing SAM 3.1 over $MAX_VIDEOS rollouts"
  "$CLIENT_PY" "$HERE/sam_probe.py" \
    --checkpoint "$SAM_MODEL" --out-json "$OUT_DIR/sam.json" \
    --base_url "http://127.0.0.1:$SAM_PORT" --model "$SAM_MODEL" \
    --videos_dir "$VIDEOS_DIR" --instruction "$INSTRUCTION" \
    --objects "$OBJECTS" --distractors "$DISTRACTORS" \
    --view wrist --upscale 3 --max_videos "$MAX_VIDEOS" --score_threshold 0.5
  log "wrote $OUT_DIR/sam.json"
}

# ---- scoreboard -------------------------------------------------------------
scoreboard() {
  local args=()
  [ -f "$OUT_DIR/rb_side.json" ]  && args+=(--result "RoboBrain·side=$OUT_DIR/rb_side.json")
  [ -f "$OUT_DIR/rb_wrist.json" ] && args+=(--result "RoboBrain·wrist=$OUT_DIR/rb_wrist.json")
  if [ -f "$OUT_DIR/sam.json" ]; then args+=(--result "SAM 3.1=$OUT_DIR/sam.json")
  else args+=(--pending "SAM 3.1"); fi
  [ ${#args[@]} -gt 0 ] || die "no result JSONs yet — run the probes first"
  "$CLIENT_PY" "$HERE/utils/scoreboard.py" "${args[@]}" \
    --title "Object verification — RoboBrain vs SAM 3.1 (drug-sort)" \
    --out "$OUT_DIR/objverif_scoreboard.html"
  log "scoreboard -> $OUT_DIR/objverif_scoreboard.html (scp it + no sidecar needed; it's static)"
}

# ---- dispatch ---------------------------------------------------------------
cmd="${1:-}"; shift || true
case "$cmd" in
  serve-robobrain) serve_robobrain ;;
  run-robobrain)   run_robobrain "${1:-side}" ;;
  serve-sam)       serve_sam ;;
  serve-sam-fake)  serve_sam_fake ;;
  run-sam)         run_sam ;;
  scoreboard)      scoreboard ;;
  all-fake)        ( serve_sam_fake & ) ; sleep 3 ; run_sam ; scoreboard ;;
  *) grep -E '^#( |$|  )' "$0" | sed 's/^# \{0,1\}//' | head -30 ;;
esac
