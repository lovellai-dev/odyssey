# RUNBOOK — UR5e drug-sorting GR00T fine-tune

End-to-end: generate demonstration data locally (no GPU) → fine-tune GR00T N1.7
on a GPU box → deploy the checkpoint into the Lovell AI Robot Playground.

Conventions: `$ODYSSEY` = this repo, `$ISAAC_GR00T_REPO_PATH` = your
`NVIDIA/Isaac-GR00T` checkout. Always launch Python with `env -u PYTHONPATH` so a
sourced venv or the ROS Jazzy `PYTHONPATH` leak doesn't shadow the interpreter.

---

## Step 1 — Generate the dataset locally (CPU only)

The generator loads the AseptiPack MJCF in **headless MuJoCo**, replays the
scripted pick-and-place policy with vial-pose domain randomization, renders the
`exterior` camera offscreen, and writes a GR00T LeRobot v2.1 dataset. It needs an
offscreen GL context (EGL on a headless GPU host, GLFW on a desktop) — **not** a
CUDA GPU.

```bash
cd "$ODYSSEY"
# deps: mujoco + numpy + pyarrow (system `ffmpeg` encodes the videos)
pip install -e ".[drugsort]"      # or: pip install mujoco numpy pyarrow  (+ apt install ffmpeg)

# Smoke set (proves the format; ~seconds):
MUJOCO_GL=egl python scripts/gen_ur5e_drugsort_demos.py \
    --xml /path/to/aseptipack_description/aseptipack.xml \
    --out /data/ur5e_drugsort_smoke --num-episodes 3

# Full set for a real fine-tune (episode 0 is nominal; the rest are randomized):
MUJOCO_GL=egl python scripts/gen_ur5e_drugsort_demos.py \
    --xml /path/to/aseptipack_description/aseptipack.xml \
    --out /data/ur5e_drugsort --num-episodes 100 --jitter 0.008 --seed 0
```

The MJCF ships in the `lai-agent` tree at
`src/embodiments/urdf/aseptipack_description/aseptipack.xml` (with its `assets/`
meshes). Each episode logs `SUCCESS/FAIL` with the vial lift height + placement
distance; expect the nominal episode and small-jitter episodes to succeed
(grasp + seat in the nest pocket). Useful knobs: `--num-episodes`, `--jitter`
(vial xy half-range, m), `--fps` (default 20), `--width/--height` (default 256),
`--seed`.

Output layout (GR00T-flavoured LeRobot v2.1):

```
/data/ur5e_drugsort/
  meta/{info.json, modality.json, stats.json, tasks.jsonl, episodes.jsonl}
  data/chunk-000/episode_000000.parquet           # action + observation.state (list<float32>)
  videos/chunk-000/observation.images.exterior/episode_000000.mp4
```

Verify it loads with the **upstream** GR00T loader (optional, in the Isaac-GR00T
venv):

```bash
cd "$ISAAC_GR00T_REPO_PATH"
env -u PYTHONPATH .venv/bin/python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ur5e_config", "PATH/TO/examples/ur5e-drugsort/ur5e_config.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)   # registers NEW_EMBODIMENT
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset
ds = ShardedSingleStepDataset(
    dataset_path="/data/ur5e_drugsort",
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    modality_configs=MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value],
    shard_size=256, seed=0)
print("OK — steps:", len(ds))   # builds shards from the dataset
PY
```

---

## Step 2 — Fine-tune on the GPU (H100 VM)

**Preconditions.** A CUDA-12.x GPU (**verified target: the H100 VM,
`ubuntu@192.222.52.169`**; a single ≥24 GB GPU is enough for the 3B model with a
modest batch). The GR00T model stack (`torch 2.7.1+cu128`, flash-attn,
`transformers 4.57`) lives in its **own** venv inside the Isaac-GR00T checkout —
never co-installed with odyssey. Build it once with the quickstart setup:

```bash
export ISAAC_GR00T_REPO_PATH="$HOME/Isaac-GR00T"      # your Isaac-GR00T checkout
bash "$ODYSSEY/examples/quickstart-gr00t/setup.sh"    # builds the 3 venvs with uv
```

**Register the UR5e modality config** where `launch_finetune` can import it. Two
options — either passes through the odyssey GR00T runner:

```bash
# (a) copy into the Isaac-GR00T checkout (mirrors upstream examples/SO100) and
#     reference it repo-relative (resolved against $ISAAC_GR00T_REPO_PATH):
mkdir -p "$ISAAC_GR00T_REPO_PATH/examples/UR5e_DrugSort"
cp "$ODYSSEY/examples/ur5e-drugsort/ur5e_config.py" \
   "$ISAAC_GR00T_REPO_PATH/examples/UR5e_DrugSort/ur5e_config.py"
#   -> mission.yaml's `modality_config_path: examples/UR5e_DrugSort/ur5e_config.py`

# (b) OR set modality_config_path in mission.yaml to the ABSOLUTE path of the
#     odyssey copy — launch_finetune importlib-loads it by path from anywhere.
```

**Base checkpoint.** `nvidia/GR00T-N1.7-3B` (gated HF download under `$HF_HOME`;
accept the NVIDIA license first). The odyssey GR00T runner resolves it from the
agent's `model.base` in `mission.yaml`, or set
`NVIDIA_GR00T_N1_7_3B_PATH=/path/to/local/checkpoint` to skip the download.

**Run the fine-tune** through odyssey (drives `launch_finetune` as a subprocess,
streams Trainer progress, and returns the checkpoint path):

```bash
cd "$ODYSSEY"
export ISAAC_GR00T_REPO_PATH="$HOME/Isaac-GR00T"
# point the runner at the GR00T model venv's python + your dataset:
#   - edit mission.yaml `dataset.ref` to /data/ur5e_drugsort (absolute) if needed
env -u PYTHONPATH odyssey run examples/ur5e-drugsort/mission.yaml
```

Equivalent **direct** upstream invocation (what the runner assembles) — use this
to debug or run standalone:

```bash
cd "$ISAAC_GR00T_REPO_PATH"
CUDA_VISIBLE_DEVICES=0 env -u PYTHONPATH .venv/bin/python \
    -m gr00t.experiment.launch_finetune \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path /data/ur5e_drugsort \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/UR5e_DrugSort/ur5e_config.py \
    --output-dir /data/ckpt/ur5e_drugsort \
    --num-gpus 1 --global-batch-size 16 --max-steps 2000 --save-steps 1000 \
    --dataloader-num-workers 4
```

VRAM tips if you OOM: lower `--global-batch-size`, raise
`--gradient-accumulation-steps`, keep `--num-gpus 1`. For a fast sanity pass use
`--max-steps 30 --skip-weight-loading` (architecture-only, no shard download).

---

## Step 3 — Where the checkpoint lands

- The HF Trainer writes the final model at `--output-dir` root
  (`config.json` + `model.safetensors`); intermediate saves go to
  `checkpoint-<step>/`.
- The odyssey GR00T runner returns `{"checkpoint_path": ...}` — it prefers the
  output root, else the newest `checkpoint-*` subdir.
- Open-loop sanity check (GT vs predicted actions, no sim), in the Isaac-GR00T
  venv:

  ```bash
  env -u PYTHONPATH .venv/bin/python gr00t/eval/open_loop_eval.py \
      --dataset-path /data/ur5e_drugsort --embodiment-tag NEW_EMBODIMENT \
      --model-path /data/ckpt/ur5e_drugsort/checkpoint-2000 \
      --modality-keys single_arm gripper --action-horizon 16
  ```

---

## Step 4 — Deploy (primary: the Playground; secondary: Isaac closed-loop)

**Primary — Lovell AI Robot Playground.** Serve the checkpoint as a GR00T policy
server; the agent-service inference client queries it each tick and writes the
returned action back through the cell's `setTargets(q, grip)` bridge
(`aseptipack-pickplace.js`), closing the loop in the browser:

```bash
# on the GPU box: serve the fine-tuned checkpoint
cd "$ISAAC_GR00T_REPO_PATH"
env -u PYTHONPATH .venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path /data/ckpt/ur5e_drugsort/checkpoint-2000 \
    --embodiment-tag NEW_EMBODIMENT --port 5555
```

The action the server returns is already the Playground action space — 6 joint
targets (rad) + gripper 0..1 — so it maps straight onto `setTargets`. The obs the
inference client sends is the Playground `TrueSensorRenderer` primary
(`exterior`) frame + the 6 arm qpos + gripper, matching training exactly.

**Secondary (optional) — closed-loop Isaac Lab eval.** The mission's evaluation
task auto-serves the trained checkpoint and scores it in Isaac Lab
(`serve_checkpoint: true`, `embodiment_tag: new_embodiment`). It needs an Isaac
Lab install with the Cosmos visuomotor env and the GR00T server venv — see
`examples/quickstart-gr00t/README.md`. This is a nice-to-have grade; the product
loop closes on the Playground.

---

## What runs where

| Step | Host | GPU? |
|---|---|---|
| 1 — generate dataset | any Linux (EGL/GLFW) | No |
| 2 — fine-tune | H100 VM / ≥24 GB CUDA | **Yes** |
| 3 — checkpoint / open-loop | GPU box | Yes |
| 4 — Playground deploy | GR00T server (GPU) + browser | Yes (server) |
