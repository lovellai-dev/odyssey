#!/usr/bin/env bash
# FlowDAgger Stage A detached driver (runs on the H100 VM, tmux groot_flowdagger_stagea).
# Chains: pad-check -> A3 dataset inversion -> A4 steering-net train -> A5 offline gate.
# Writes /home/ubuntu/stagea_result.json + progress to /home/ubuntu/flowdagger_stagea.log.
# Terminal marker: "FLOWDAGGER_STAGEA DONE" or "FLOWDAGGER_STAGEA FAILED: <stage>".
set -uo pipefail

GROOT_PY=/home/ubuntu/Isaac-GR00T/.venv/bin/python
MUJOCO_PY=/home/ubuntu/odyssey-eval-venv/bin/python
SCRIPTS=/home/ubuntu/odyssey-ur5e/scripts
DATASET=/home/ubuntu/ur5e_drugsort_obscond
CKPT=/home/ubuntu/ckpt/ur5e_drugsort_obscond/full/checkpoint-12000
XML=/home/ubuntu/aseptipack_description/aseptipack.xml
OUT=/home/ubuntu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

ulimit -n 8192 2>/dev/null || true   # ffmpeg video readers hold fds; headroom

log(){ echo "[$(date +%H:%M:%S)] $*"; }
fail(){ echo "FLOWDAGGER_STAGEA FAILED: $1"; exit 1; }

cd "$SCRIPTS" || fail "cd_scripts"

# ---- GPU gate --------------------------------------------------------------
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
log "GPU free MiB: $FREE"
[ "$FREE" -lt 20000 ] && fail "gpu_gate_${FREE}MiB"

# ---- A4-pre: pad-influence check ------------------------------------------
log "=== pad-check ==="
$GROOT_PY train_steering_ur5e.py pad-check --model-path "$CKPT" --dataset "$DATASET" \
  --n 48 --stride 8 --out "$OUT/stagea_padcheck.json" 2>&1 | grep -vi "albumentations\|check_for_updates\|FutureWarning\|flash_attention\|self\." || true
[ -f "$OUT/stagea_padcheck.json" ] || fail "padcheck"
DESIGN=$($GROOT_PY -c "import json;print(json.load(open('$OUT/stagea_padcheck.json'))['output_design_decision'])")
log "pad-check design decision: $DESIGN"
SAVE_FULL=""
[ "$DESIGN" = "full5280" ] && SAVE_FULL="--save-full-w"

# ---- A3: full dataset inversion -------------------------------------------
log "=== A3 invert-dataset (design=$DESIGN save_full='$SAVE_FULL') ==="
$GROOT_PY flow_inverter_groot.py invert-dataset --model-path "$CKPT" --dataset "$DATASET" \
  --out-dir "$OUT/steering_targets" --stride 8 --fp-iters 16 --mini-batch 8 \
  --shard-size 256 --quality-thresh 1e-3 --max-gpu-hours 4.0 --bf16-n 32 $SAVE_FULL \
  --out "$OUT/stagea_a3.json" 2>&1 | grep -vi "albumentations\|check_for_updates\|FutureWarning\|flash_attention\|self\." || true
[ -f "$OUT/stagea_a3.json" ] || fail "a3_inversion"

# ---- A4: train steering net -----------------------------------------------
log "=== A4 train steering net (design=$DESIGN) ==="
$GROOT_PY train_steering_ur5e.py train --shards "$OUT/steering_targets" \
  --output-design "$DESIGN" --model-out "$OUT/steering_net_v0.npz" \
  --epochs 200 --patience 20 --batch 256 --lr 3e-4 \
  --out "$OUT/stagea_a4.json" 2>&1 | grep -vi "albumentations\|check_for_updates" || true
[ -f "$OUT/stagea_a4.json" ] || fail "a4_train"

# ---- A5: offline gate (decode in GR00T venv, FK+gate in mujoco venv) -------
log "=== A5 gate: decode ==="
$GROOT_PY flowdagger_offline_gate_ur5e.py decode --model-path "$CKPT" --dataset "$DATASET" \
  --steering-net "$OUT/steering_net_v0.npz" --n 100 --n-seeds 8 --stride 8 \
  --decoded-out "$OUT/flowdagger_gate_decoded.npz" --out "$OUT/stagea_gate_decode.json" \
  2>&1 | grep -vi "albumentations\|check_for_updates\|FutureWarning\|flash_attention\|self\." || true
[ -f "$OUT/flowdagger_gate_decoded.npz" ] || fail "a5_decode"

log "=== A5 gate: fk-gate (mujoco) ==="
$MUJOCO_PY flowdagger_offline_gate_ur5e.py fk-gate --src /home/ubuntu/odyssey-ur5e --xml "$XML" \
  --decoded-npz "$OUT/flowdagger_gate_decoded.npz" --frac-thresh 0.70 --improve-thresh 0.30 \
  --out "$OUT/stagea_gate.json" 2>&1 | tail -40
[ -f "$OUT/stagea_gate.json" ] || fail "a5_fkgate"

# ---- merge into the Stage-A result JSON -----------------------------------
log "=== merge ==="
$GROOT_PY - "$OUT" <<'PY'
import json, sys
OUT = sys.argv[1]
def L(p):
    try:
        return json.load(open(f"{OUT}/{p}"))
    except Exception:
        return {}
pad = L("stagea_padcheck.json"); a3 = L("stagea_a3.json"); a4 = L("stagea_a4.json"); gate = L("stagea_gate.json")
a1 = L("flowdagger_a1_determinism.json")
bf = a3.get("bf16_check", {})
gate_pass = bool(gate.get("pass", False))
res = {
 "a1_patch": {"determinism_pass": bool(a1.get("determinism_pass", False)),
              "wire_options_pass": bool(a1.get("wire_options_pass", False)),
              "patch_file": "scripts/patches/isaac_groot_init_noise.patch"},
 "a2_inverter": {"bf16_decode_mse": bf.get("bf16_decode_mse_vs_fp32"),
                 "bf16_decode_mse_vs_target": bf.get("bf16_decode_mse_vs_target"),
                 "bf16_ok": bool(bf.get("bf16_ok", False))},
 "a3_inversion": {"n_chunks": a3.get("n_kept"), "n_chunks_in": a3.get("n_chunks_in"),
                  "drop_rate": a3.get("drop_rate"), "gpu_hours": a3.get("gpu_hours"),
                  "stride": a3.get("stride"), "t_per_chunk_s": a3.get("t_per_chunk_s")},
 "a4_steering": {"output_design": a4.get("output_design"),
                 "pad_influence_mse": pad.get("pad_influence_mse"),
                 "train_mse": a4.get("train_mse"), "val_mse": a4.get("val_mse"),
                 "p99_pred": a4.get("p99_pred")},
 "a5_offline_gate": {"n_holdout_states": gate.get("n_holdout_states"),
                     "steered_beats_stock_frac": gate.get("steered_beats_stock_frac"),
                     "ee_dist_cm": gate.get("ee_dist_cm"),
                     "improvement_toward_oracle": gate.get("improvement_toward_oracle"),
                     "pass": gate_pass},
 "verdict": "STAGE_B_GO" if gate_pass else "STAGE_A_FAIL",
 "notes": "",
}
json.dump(res, open(f"{OUT}/stagea_result.json","w"), indent=2)
print(json.dumps(res, indent=2))
PY
[ -f "$OUT/stagea_result.json" ] || fail "merge"
log "Stage A result written to $OUT/stagea_result.json"
echo "FLOWDAGGER_STAGEA DONE"
