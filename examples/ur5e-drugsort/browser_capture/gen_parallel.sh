#!/usr/bin/env bash
# ============================================================================
# gen_parallel.sh — PARALLEL browser-frame (Three.js) data-gen for the render-gap
# close. Runs several INDEPENDENT headless-Chrome harness instances concurrently
# (near-linear speedup vs the ~90 s/episode single tab), each replaying its own
# shard of precomputed expert plans against the live Playground static server,
# capturing the REAL deployment Three.js frames.
#
# SAFE browser hygiene: each instance is its OWN google-chrome with a UNIQUE
# --user-data-dir (browser_harness.js sets it under its OUT dir) and records its
# Chrome PID to <OUT>/harness_pid.txt. Cleanup kills ONLY those recorded PIDs +
# the node PIDs we launched. NEVER pkill chrome (the user's own Chrome + the
# claude-in-chrome session must survive).
#
# Env:
#   PLANS_FULL   full plans.json                     (default plans_full.json)
#   SHARDS       number of parallel instances        (default 5)
#   OUTBASE      base output dir                      (default ./out_full)
#   PORTS        space-separated Playground ports     (default "8021 8020")
#   PUPPETEER_CORE / CHROME  passed through to harness
# Writes: $OUTBASE/gen.log progress; final line "GEN_PARALLEL DONE rc=<n>".
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"

PLANS_FULL=${PLANS_FULL:-plans_full.json}
SHARDS=${SHARDS:-5}
OUTBASE=${OUTBASE:-$(pwd)/out_full}
PORTS=(${PORTS:-8021 8020})
CHROME=${CHROME:-/usr/bin/google-chrome-stable}
PUPPETEER_CORE=${PUPPETEER_CORE:-/home/daniel/.npm/_npx/23232c69e5d221f3/node_modules/puppeteer-core}
PYBIN=${PYBIN:-/home/daniel/LovellAI/odyssey-ur5e/.venv-ur5e/bin/python}
PER_EP_TIMEOUT=${PER_EP_TIMEOUT:-6h}

mkdir -p "$OUTBASE"
LOG="$OUTBASE/gen.log"
exec >>"$LOG" 2>&1
echo "=== [gen] START $(date -u +%FT%TZ) shards=$SHARDS ports=${PORTS[*]} ==="

# 1. Split the full plan set into $SHARDS shards (preserving episode ids).
"$PYBIN" split_plans.py --plans "$PLANS_FULL" --shards "$SHARDS" --prefix "$OUTBASE/shard" || {
  echo "GEN_PARALLEL DONE rc=2"; exit 2; }

NODE_PIDS=(); CHROME_PID_FILES=()
cleanup() {
  echo "=== [gen] cleanup $(date -u +%FT%TZ) ==="
  for pf in "${CHROME_PID_FILES[@]}"; do
    [ -f "$pf" ] || continue
    cp=$(cat "$pf" 2>/dev/null || true)
    if [ -n "${cp:-}" ] && kill -0 "$cp" 2>/dev/null; then
      # verify it's OUR chrome (udd under $OUTBASE) before killing — never a stray chrome
      if tr '\0' ' ' < "/proc/$cp/cmdline" 2>/dev/null | grep -q -- "$OUTBASE"; then
        echo "  [gen] kill chrome PID $cp"; kill "$cp" 2>/dev/null || true
      fi
    fi
  done
  for np in "${NODE_PIDS[@]}"; do kill "$np" 2>/dev/null || true; done
}
trap cleanup EXIT

# 2. Launch one harness per shard, alternating ports.
i=0
for shard in "$OUTBASE"/shard_*.json; do
  port=${PORTS[$(( i % ${#PORTS[@]} ))]}
  out="$OUTBASE/inst_$i"
  mkdir -p "$out"
  echo "=== [gen] launch inst $i shard=$(basename "$shard") port=$port out=$out ==="
  PLANS="$shard" OUT="$out" PORT="$port" CHROME="$CHROME" PUPPETEER_CORE="$PUPPETEER_CORE" \
    timeout "$PER_EP_TIMEOUT" node browser_harness.js > "$out/inst.log" 2>&1 &
  NODE_PIDS+=("$!")
  CHROME_PID_FILES+=("$out/harness_pid.txt")
  i=$((i+1))
  sleep 8   # stagger startups so the Chrome/WASM boots don't thrash together
done
echo "=== [gen] launched ${#NODE_PIDS[@]} instances; PIDs ${NODE_PIDS[*]} ==="

# 3. Wait for all instances.
RC=0
for np in "${NODE_PIDS[@]}"; do
  if ! wait "$np"; then echo "=== [gen] instance PID $np exited NONZERO ==="; RC=1; fi
done
echo "=== [gen] all instances finished rc=$RC $(date -u +%FT%TZ) ==="

# 4. Merge raw episodes into one dir (episode ids are globally unique across shards).
RAW_ALL="$OUTBASE/raw_all"
rm -rf "$RAW_ALL"; mkdir -p "$RAW_ALL"
nmerged=0
for out in "$OUTBASE"/inst_*; do
  [ -d "$out/raw" ] || continue
  for ep in "$out/raw"/ep*; do
    [ -d "$ep" ] || continue
    # require a meta.json (episode completed) before merging
    if [ -f "$ep/meta.json" ]; then
      cp -r "$ep" "$RAW_ALL/" && nmerged=$((nmerged+1))
    fi
  done
done
echo "=== [gen] merged $nmerged completed episodes -> $RAW_ALL ==="

# 5. Success tally across instance summaries.
"$PYBIN" - "$OUTBASE" <<'PY'
import json, glob, sys
base = sys.argv[1]
tot=succ=0; secs=[]
for s in glob.glob(f"{base}/inst_*/harness_summary.json"):
    d=json.load(open(s)); tot+=d["num_episodes"]; succ+=d["success"]
    secs.append(d.get("avg_seconds_per_episode",0))
print(f"[gen] instance summaries: {succ}/{tot} expert-SUCCESS, avg {sum(secs)/max(1,len(secs)):.1f}s/ep")
PY

echo "GEN_PARALLEL DONE rc=$RC"
exit $RC
