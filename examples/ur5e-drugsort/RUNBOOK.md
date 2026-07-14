# RUNBOOK — UR5e drug-sorting GR00T fine-tune

End-to-end: generate demonstration data locally (no GPU) → fine-tune GR00T N1.7
on a GPU box → deploy the checkpoint into the Lovell AI Robot Playground.

Conventions: `$ODYSSEY` = this repo, `$ISAAC_GR00T_REPO_PATH` = your
`NVIDIA/Isaac-GR00T` checkout. Always launch Python with `env -u PYTHONPATH` so a
sourced venv or the ROS Jazzy `PYTHONPATH` leak doesn't shadow the interpreter.

---

## Step 1 — Generate the dataset locally (CPU only)

The generator loads the AseptiPack MJCF in **headless MuJoCo**, drives the
**adaptive IK expert** (`src/odyssey/embodiments/ur5e_drugsort/ik.py`) with
vial-pose domain randomization, renders the `exterior` camera offscreen, and
writes a GR00T LeRobot v2.1 dataset. It needs an offscreen GL context (EGL on a
headless GPU host, GLFW on a desktop) — **not** a CUDA GPU.

The IK expert re-solves the arm joint targets (damped-least-squares Jacobian IK
on the `gr_pinch` site) to the **actual randomized vial / pocket pose** each
episode, so the recorded action/proprio trajectories genuinely vary with the
scene — a policy trained on them must read the camera. Per run it prints and
writes an adaptivity proof (`<out>/ik_adaptivity.json`): the per-joint std of the
first-approach joint angles across episodes (materially > 0), vs the
zero-variance `--fixed-waypoints` baseline (the old fixed-replay behaviour).
The FSM's interpolation, convergence and gripper open/close *timing* are
unchanged — only the joint *targets* adapt.

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
    --out /data/ur5e_drugsort --num-episodes 64 --seed 0
```

The vial is randomized over the reachable tray each episode (defaults
`--area-x 0.05 --area-y 0.07` m half-ranges + `--yaw-jitter 0.08` rad visual
jitter). `--noslip-iterations 20` (default) enables MuJoCo's noslip friction
refinement so the grasped vial is not ejected on lift (an in-memory solver
option — it edits no MJCF and changes no recorded obs/action). Use
`--fixed-waypoints` for the zero-variance scripted baseline, or the deprecated
single-range `--jitter` alias. A recent 64-episode run scored **64/64
grasp+place** with first-approach joint std up to ~0.06 rad (range ~0.23 rad)
and IK error < 1 mm — see the printed proof + `ik_adaptivity.json`.

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

## Step 1b — Browser-frame (Three.js) capture — closing the render gap

Step 1 renders with MuJoCo's `Renderer`, but the policy **deploys** on the Lovell
AI Robot Playground's **Three.js** renderer. That render gap degrades both GR00T
(≈2/20 headless → ≈0 in-browser) and the Observer (2.5 cm → 3.5–5.5 cm). To retrain
on the deployment appearance we first capture training frames from the Playground's
*real* renderer. The harness lives in `browser_capture/` and reuses the SAME adaptive
IK expert + LeRobot writer as Step 1 — only the *pixels* change.

Three stages (headless Chrome; needs a running Playground server + `puppeteer-core`
+ a Chrome/Chromium, plus `MUJOCO_GL=egl` for the IK solve):

```bash
cd examples/ur5e-drugsort/browser_capture

# 1) Precompute the per-episode adaptive-IK expert plans (Python mujoco; no GPU).
#    Randomizes the vial + solves DLS-IK exactly as scripts/gen_ur5e_drugsort_demos.py,
#    emitting the vial qpos + 10 adaptive joint-target waypoints to plans.json.
MUJOCO_GL=egl python precompute_plans.py --num-episodes 5 --seed 0 --out plans.json

# 2) Replay the expert IN THE BROWSER, capturing the exterior+wrist frames rendered
#    by the Playground's DEPLOYMENT sensor path (groot-pilot.js makeCapture -> the same
#    THREE.WebGLRenderer + scene the served GR00T policy receives), + proprio + expert
#    action + ground-truth vial/nest pose. SAFE: own Chrome, unique --user-data-dir,
#    PID captured + cleaned up by PID; read-only page loads (never touches the /api bridge).
PLANS=plans.json OUT=./out PORT=8021 PUPPETEER_CORE=/path/to/puppeteer-core \
    node browser_harness.js          # writes out/raw/epNNN/{exterior,wrist}/*.png + meta.json

# 3) Assemble into the SAME GR00T LeRobot v2.1 schema as Step 1 (LeRobotDatasetWriter +
#    ur5e_drugsort.embodiment, unchanged) + a per-frame GT-pose sidecar (meta/gt/) for
#    the Observer's grasp-target labels.
python assemble_lerobot.py --raw ./out/raw --out ./dataset_browser

# 4) (optional) Confirm it loads in the UPSTREAM GR00T loader (Isaac-GR00T model venv):
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$ISAAC_GR00T_REPO_PATH/.venv/bin/python" validate_groot_loader.py --dataset ./dataset_browser
```

The resulting `dataset_browser/` is byte-identical in layout to the Step-1 dataset
(same `meta/{info,modality,stats}.json`, same `observation.images.exterior` /
`observation.images.wrist` video keys) — it drops straight into `launch_finetune`
and the Observer training with no code change; only the frames are Three.js instead
of MuJoCo. GT poses are written to `meta/gt/episode_XXXXXX.json` (a sidecar OUTSIDE
the loader-read files, so the core dataset still loads unchanged).

**Why browser-drive (not a standalone headless-Three.js renderer):** fidelity. The
frames must match what the policy deploys on, so we render the Playground's own
`window.viewer.scene` through its own `window.viewer.renderer` with its PBR
"cell-upgrade" materials (glass vials, polished metal, wood worktop). A separate
Three scene would risk material/lighting drift from the real deploy target.

**Smoke-set result (5 episodes, verified):** 5/5 grasp+place SUCCESS in-browser
(lift ≈11.4 cm, place <1 cm from the nest), ~919 frames/episode, and the dataset
built 18 shards in the upstream GR00T loader. **Throughput ≈96 s/episode** (single
headless tab), i.e. **≈8 h wall-clock for a full ~300-episode dataset**. The cost is
dominated by physics stepping + the two 256×256 `readRenderTargetPixels` captures per
20 Hz step; episodes run long because the browser arm often rides the FSM's
convergence dwell. Levers if 8 h is too slow: run several headless tabs/Chrome
instances in parallel (near-linear speedup; each is independent), tighten the FSM
convergence window, or lower the capture rate.

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

> **Gated-processor gotcha.** GR00T-N1.7-3B's Qwen3VL processor pulls its
> image-processor config from the **gated** repo `nvidia/Cosmos-Reason2-2B`. If
> that repo (and the base model) are already cached but the host isn't logged in,
> `from_pretrained` still 401s while re-validating the gated repo online. Fix:
> run with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to load both from cache. (If
> not cached, `hf auth login` with a token that has accepted both licenses, then
> pre-download once.)

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
