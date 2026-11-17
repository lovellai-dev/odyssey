# WALL-OSS on LIBERO (eval-only)

Scores an [X Square Robot WALL-OSS](https://github.com/X-Square-Robot/wall-x)
checkpoint on LIBERO with `pilot: wall_oss`. WALL-OSS is a 4B VLA: a
Qwen2.5-VL-3B backbone plus a flow-matching action expert, released under
Apache-2.0.

**Status: the wiring is done and passes CPU tests. The GPU smoke test is still
pending.**

## Why LIBERO

The paper's headline numbers come from real-robot suites in X Square's own lab,
plus some VQA and embodied-understanding benchmarks. LIBERO is the only
simulation benchmark it uses: the action-loss ablation in §5.2 reports 96.5%
average success. The `wall-x` repo ships a LIBERO eval (`scripts/run_libero.sh`),
a LIBERO fine-tune template (`workspace/example/libero.yml`) and a `--env LIBERO`
serving mode.

## How it runs

```
odyssey (LIBERO env)                                 wall-x env (GPU)
LiberoRunner ─► wall_oss_libero_eval.py ──ws msgpack──► scripts/run_serving.sh --env LIBERO
                  ChunkPilotAdapter                     WallXPolicy (flow) → predict_action (T, 26)
```

- **Request:** `state.follow2_pos` is 8 floats: eef pos, axis-angle, and two
  gripper finger values. `views` holds the face and right-wrist cameras as
  base64 JPEG, flipped 180°. `instruction` is the task text.
- **Response:** `predict_action` with shape `(T, 26)`. The first 7 columns are
  the LIBERO OSC_POSE delta (`dpos`, `drot` as ZYX euler, gripper). The
  remaining 19 columns are `action_padding` and are dropped.

## 1. Checkpoint

X Square has **not** published a LIBERO-finetuned checkpoint. Only the base
models (`x-square-robot/wall-oss-0.5`, `-flow`, `-fast`) are on Hugging Face.

> ⚠️ The community LIBERO fine-tunes on Hugging Face (`ases200q2/lerobot_libero_wallx_*`,
> `TInkybala/libero_COT_wall_x_*`) are **LeRobot-format** policies
> (`policy_preprocessor.json` / `policy_postprocessor*`). They are not native wall-x
> checkpoints (`config.yml` + `normalizer_*.pth`), so `run_serving.sh` most likely
> can't load them as-is. Check this first on the GPU box.

The reliable path is to fine-tune `wall-oss-0.5` with wall-x's own template:

```bash
python scripts/compute_norm_stats.py --train_config workspace/example/libero.yml \
    --data_root /data/libero_all --output_path /data/libero_all_norm_stats.json
torchrun --nproc_per_node=<N> wall_x/trainer/fsdp_trainer/train_fsdp.py \
    --config workspace/example/libero.yml
```

## 2. Start the wall-x server (wall-x env)

```bash
git clone https://github.com/X-Square-Robot/wall-x && cd wall-x
# Follow the README install: Python 3.10, requirements.txt, flash-attn 2.8.3, then
# MAX_JOBS=8 pip install --no-build-isolation -e .
bash scripts/run_serving.sh \
    --checkpoint-path /data/wall-oss/libero_ft/checkpoint \
    --train-config-path /data/wall-oss/libero_ft/config.yml \
    --env LIBERO --port 32195 \
    -- --model-config.norm-key libero_all
```

## 3. Run the mission (LIBERO / odyssey env)

```bash
pip install ".[wall-oss]"      # websockets + msgpack client only (no torch)
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    odyssey run examples/quickstart-wall-oss/mission.yaml
```

## Knobs (`task.config`)

| key | default | meaning |
|---|---|---|
| `host` / `port` | `127.0.0.1` / `32195` | wall-x server address |
| `n_action_steps` | `10` | chunk steps replayed before re-querying (≤ server `ACTION_HORIZON`) |
| `invert_gripper` | `false` | flip gripper polarity (`WALLX_LIBERO_INVERT_GRIPPER`) |
| `binarize_gripper` | `false` | snap gripper to ±1 (`WALLX_LIBERO_GRIPPER_BINARIZE`) |
| `jpeg_quality` | `95` | quality of the base64 views |
| `translation_only` | `false` | de-risk: zero rotation, gripper forced open |

**Sanity check:** run the same checkpoint through wall-x's own
`bash scripts/run_libero.sh <ckpt>` with `SMOKE=1`. If its success rate is far
from odyssey's, the transforms don't match: look at rotation and gripper sign
first.
