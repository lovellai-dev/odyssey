# Multi-agent on a GCP GPU VM — OpenVLA pilot + Gemma planner (end-to-end)

This is the complete, self-contained procedure for running the **multi-agent
mission** on a **Google Cloud GPU VM**: an OpenVLA **PILOT** fine-tuned on Bridge
V2, guided at evaluation time by a vision-language **SPECIALIST** task planner
(multimodal **Gemma 4 E2B-it**), evaluated on Robosuite **Lift**.

It is GCP-specific on purpose: a few things bite you only on GCP (the NCCL `gIB`
plugin, L4 stockouts, disk sizing) and this guide front-loads them so you don't
lose a day to them.

> **Validated on:** `g2-standard-8` · NVIDIA **L4 (24 GB)** · Ubuntu. Use an
> **on-demand** L4 in **`us-west1-a`** — it has proved the most reliable; avoid
> `us-central1-a` preemptible, which is chronically stocked out of L4. The steps
> themselves are zone-independent. Other clouds (AWS/Azure) or GPUs may not need the
> NCCL workaround.

> A simpler **single-agent** path (OpenVLA only, one venv) is documented in the
> [single-agent GCP tutorial](./gcp-training-tutorial.md). This guide is the
> multi-agent superset and stands on its own.

---

## What multi-agent adds

A mission with a SPECIALIST agent runs a **plan-then-execute** loop during eval:
the SPECIALIST decomposes the instruction into sub-steps **once per episode**, and
the PILOT executes each step. Only the PILOT produces actions and only the PILOT
is trained — the **SPECIALIST is inference-only** (it runs its base checkpoint to
plan and has no training task).

The catch is dependencies. The multimodal Gemma 4 planner needs a **modern
`transformers` + `torchvision`**, which is **incompatible** with OpenVLA's pinned
**`transformers==4.40.1`**. They cannot coexist in one environment, so the planner
**must run out of process** in its own venv. The PILOT stays in the main venv; the
two talk over a JSON-lines subprocess protocol (the planner runs once per episode,
off the per-step hot loop).

Hence **two venvs, named by role**: **`env_pilot`** hosts the OpenVLA pilot and the
Robosuite eval (pinned to OpenVLA's stack), and **`env_specialist`** hosts the
Gemma 4 planner (modern `transformers` + `torchvision`). Building both correctly is
the **fragile part of multi-agent**, so this tutorial leans on the **`setup.sh`
script**, which builds both idempotently and pins each to its known-good stack.

---

## What you'll do

1. [Provision a GPU VM](#1-provision-the-vm) (with enough disk — this matters)
2. [Connect and install system deps](#2-connect--system-dependencies)
3. [Build both venvs with the setup script](#3-build-both-venvs-with-the-setup-script)
4. [Download the Bridge V2 dataset](#4-the-dataset-bridge-v2-in-rlds-format) (RLDS, ~124 GB)
5. [Load the environment + smoke-test the planner](#5-load-the-environment--smoke-test-the-planner)
6. [Run the multi-agent mission](#6-run-the-multi-agent-mission)
7. [Troubleshoot](#7-troubleshooting--debugging-playbook) when it silently dies
8. [Get your results and stop the VM](#8-wrap-up-get-your-results-and-stop-the-vm)

Plan for **about an hour**: the steps themselves are quick; the 124 GB dataset
download dominates (**~30 min** on a GCP VM, measured; longer on slower bandwidth).

> 💸 **Validate for free first.** A GPU VM costs real money the whole time it's
> running. Before you provision anything, confirm the mission spec and the whole
> orchestration flow on your laptop with the CPU mock — no GPU, no cost:
> ```bash
> odyssey run examples/multiagent-openvla-gemma/mission.yaml --use-mock-runner
> ```
> Only spin up the VM once that runs clean. And see [§8](#8-wrap-up-get-your-results-and-stop-the-vm)
> — **stop the VM when you're done** so it stops billing.

---

## 0. Prerequisites

- A **GCP project** with billing enabled and the [`gcloud` CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated.
- **GPU quota for L4.** New projects start with **zero** GPU quota — request an
  increase for `NVIDIA_L4_GPUS` (and/or `GPUS_ALL_REGIONS`) in your target region
  via *IAM & Admin → Quotas*. Approval can take minutes to a day, so **request it
  before you need it**. Hitting `Quota 'NVIDIA_L4_GPUS' exceeded` at VM-creation
  time means this step was skipped.
- *(Optional)* a **HuggingFace token.** Both default models (`openvla/openvla-7b`
  and `google/gemma-4-E2B-it`) are **ungated**, so **no token is needed**. Set
  `HF_TOKEN` (or `hf auth login`) only to avoid anonymous download rate limits, or
  to swap in a gated model (e.g. the larger Gemma 4 E4B, or gemma-2/3).
- Basic familiarity with SSH and the Linux shell.

---

## 1. Provision the VM

| Setting | Value | Why |
|---|---|---|
| Machine type | `g2-standard-8` | 8 vCPU / 32 GB RAM, pairs with one L4 |
| GPU | 1 × NVIDIA **L4 (24 GB)** | Fits the ~14 GB bf16 pilot **and** the ~5 GB int4 planner |
| OS image | Ubuntu (Deep Learning VM image works well) | CUDA drivers preinstalled |
| **Boot disk** | **≥ 300 GB** | 124 GB dataset + ~15 GB model cache + run outputs |

> ⚠️ **Size the disk up front.** The default boot disk (≈ 51 GB) is nowhere near
> enough — the Bridge V2 dataset alone is 124 GB. **Disk-full is the single most
> common cause of silent failures** in this pipeline (see
> [§7](#silent-exit1--the-decoder)). Provision **300 GB** from the start.
>
> Disk-light alternative: instead of a 300 GB disk you can **stream the dataset
> from a GCS bucket via [`gcsfuse`](https://cloud.google.com/storage/docs/gcsfuse-quickstart-mount-bucket)**
> (mount the bucket and point `data_root_dir` at the mount). Slower per-step I/O,
> but no large local disk.

> 💸 **Cost & quota.** A `g2-standard-8` + L4 runs on the order of **~$0.70–1/hour**
> (varies by region; check the [pricing page](https://cloud.google.com/compute/gpus-pricing)),
> plus a few $/month for the 300 GB disk. **You pay while the VM is running, GPU
> idle or not** — so stop it between sessions ([§8](#8-wrap-up-get-your-results-and-stop-the-vm)).
> A stopped VM still bills for its disk.

> ⚠️ **Use an on-demand L4 in `us-west1-a`.** `us-central1-a` is chronically stocked
> out of L4, and a preemptible VM gets reclaimed mid-run — we lost days to both.
> An **on-demand** L4 in **`us-west1-a`** has been reliable. If a zone is exhausted,
> try `us-west1-b/c` or `us-east1`. **Take a
> [disk snapshot](https://cloud.google.com/compute/docs/disks/create-snapshots)
> before stopping or resizing a working VM**: we once lost a VM to a resize that
> left it unbootable, then couldn't get a new L4 for hours.

### Hardware: both models share the GPU

The venv split solves the **dependency** conflict, not VRAM — both models load on
the **same 24 GB GPU**. The mission uses **Gemma 4 E2B-it** (~5 GB int4) precisely
so it fits alongside the ~14 GB bf16 pilot → ~19 GB peak, comfortable on an L4.

> The larger **E4B** (~9.3 GB int4) is too tight next to the pilot — bitsandbytes
> rejects CPU offload for 4-bit and the placement fails. E4B works *standalone*
> (e.g. the smoke test), just not alongside the pilot. The specialist is pinned to
> **GPU 0** (`device_map={"": 0}`) so layers never silently offload to CPU.

---

## 2. Connect & system dependencies

```bash
gcloud compute ssh <VM_NAME> --zone=<ZONE>
```

Confirm the GPU is visible:

```bash
nvidia-smi      # expect: NVIDIA L4, 24 GB
```

For the **evaluation** step (Robosuite / MuJoCo runs headless on the VM), install
the EGL/GL libraries — without these the simulator can't create a GL context:

```bash
sudo apt-get update
sudo apt-get install -y libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev
```

---

## 3. Build both venvs with the setup script

Clone Odyssey, then let the setup script build **both** environments. It is the
fastest, least error-prone path — it creates `env_pilot` and `env_specialist`,
pins each to its known-good stack, clones the upstream OpenVLA repo, and writes a
sourceable `.env`. It is **idempotent** (re-running reuses existing venvs/clones)
and it **only sets up — it does not run a mission.**

```bash
git clone https://github.com/lovellai-dev/odyssey.git ~/odyssey
cd ~/odyssey

# Builds env_pilot + env_specialist, clones OpenVLA, writes .env. Re-runnable.
examples/multiagent-openvla-gemma/setup.sh
```

What the script does, in order:

1. **`env_pilot`** — creates the venv, installs `odyssey` with all extras pinned to
   `constraints/openvla-known-good.txt` (torch 2.2.0 +cu121 from the PyTorch index,
   then the rest from PyPI), and clones + installs the upstream OpenVLA repo (which
   carries `draccus` + `finetune.py`).
2. **`env_specialist`** — creates the venv and installs the Gemma planner deps
   (`-e ".[specialist]"`) pinned to `constraints/specialist-known-good.txt` (modern
   transformers + torchvision).
3. **Writes `examples/multiagent-openvla-gemma/.env`** — the file you `source`
   before every run (details in [§5](#5-load-the-environment--smoke-test-the-planner)).

### Building it by hand instead

If you'd rather not use the script, the equivalent manual steps are:

```bash
# env_pilot — the OpenVLA pilot + eval
python3 -m venv env_pilot
source env_pilot/bin/activate
pip install -e ".[all,dev]" -c constraints/openvla-known-good.txt
git clone https://github.com/openvla/openvla.git ~/openvla
pip install -e ~/openvla -c constraints/openvla-known-good.txt
export OPENVLA_REPO_PATH=~/openvla

# env_specialist — the Gemma planner (separate, modern stack)
python3 -m venv env_specialist
env_specialist/bin/pip install -e ".[specialist]" -c constraints/specialist-known-good.txt
export ODYSSEY_SPECIALIST_PYTHON="$PWD/env_specialist/bin/python"
```

> The known-good stacks live in `constraints/openvla-known-good.txt` (torch 2.2.0,
> transformers 4.40.1) and `constraints/specialist-known-good.txt` (modern
> transformers + torchvision). They no longer need to be mutually compatible —
> that's the whole point of the venv split.

> 🔥 **Do not run the pilot on torch 2.6.** It caused a **silent `exit(1)`** during
> training (an inductor compile-worker fork-after-CUDA race). OpenVLA's pinned
> **`torch==2.2.0`** fixes it — the constraints file pins it for you.

### `setup.sh` flags

| Flag | Effect |
|---|---|
| `--skip-pilot` | skip `env_pilot` + OpenVLA (you already built it, e.g. for single-agent) |
| `--pilot-venv PATH` | relocate the pilot venv (default `<repo>/env_pilot`) |
| `--specialist-venv PATH` | relocate the specialist venv (default `<repo>/env_specialist`) |
| `--openvla-repo PATH` | where to clone OpenVLA (default `~/openvla`) |
| `--smoke` | run the planner smoke check at the end (downloads Gemma) |
| `-h`, `--help` | show all options |

---

## 4. The dataset: Bridge V2 in RLDS format

You'll download the ~124 GB Bridge V2 dataset and rename it to the key OpenVLA
expects (`bridge_orig`). **Do this early and read the heads-up below** — it's where
people lose time (a full disk, or a crawl that never "finishes") right at the end.

The pilot's `finetune.py` expects **RLDS / TensorFlow-Datasets format** (not
LeRobot/HuggingFace). Download it under `nohup` so it survives an SSH drop, logging
to a file:

```bash
cd ~
nohup wget -r -nH --cut-dirs=4 --reject="index.html*" \
  https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/ \
  > ~/bridge_download.log 2>&1 &
```

> ⚠️ **`wget -r` keeps crawling *past* the dataset — stop it once the shards are in.**
> The recursive download follows every link, so after the RLDS shards it starts
> pulling unrelated sibling files (we've seen a **~100 GB `demos_*.zip`**) — that's
> the "stuck at the end forever" phase, and it can fill the disk and trip a silent
> `exit(1)`. **Don't wait for `finished`** — stop it as soon as the shards +
> metadata are present (next step).

Monitor it (the per-line percentages are **per-file**, not overall — track total
progress with `du -sh`):

```bash
tail -f ~/bridge_download.log
du -sh ~/bridge_dataset                                        # heading toward ~124 GB
find ~/bridge_dataset -name "*.tfrecord-*" | wc -l             # 1024 train shards (1152 incl. 128 val)
ls ~/bridge_dataset/1.0.0/ | grep -E "dataset_info|features"   # metadata present?
```

Once all the shards + `dataset_info.json` + `features.json` are there, stop the
crawl and rename to the OXE key (match by pattern — the PID changes between runs):

```bash
pkill -f 'wget.*bridge_dataset'   # stop the recursive download
mv ~/bridge_dataset ~/bridge_orig
```

**Why the rename?** Berkeley serves the dataset as `bridge_dataset`, but OpenVLA's
OXE registry knows it by the key **`bridge_orig`**. The `mv` only renames the folder
so it matches the key OpenVLA resolves at `<data_root_dir>/bridge_orig/<version>/`;
skip it and training fails at dataset creation with "dataset not found".

> ⚠️ **`data_root_dir` is the *parent* of the dataset folder, not the folder itself.**
> If the dataset lives at `/home/<user>/bridge_orig/`, set
> `data_root_dir: /home/<user>` — **not** `/home/<user>/bridge_orig`. A wrong path
> fails right after model load + LoRA wrap.

In the mission this maps straight through (Odyssey does **not** download it):

| `mission.yaml` | becomes the flag | meaning |
|---|---|---|
| `dataset.ref: bridge_orig` | `--dataset_name bridge_orig` | the OXE registry key |
| `config.data_root_dir: <path>` | `--data_root_dir <path>` | the parent dir of the RLDS folder |

> 💡 **Keep the shuffle buffer small.** The upstream RLDS default (256k) eats all
> 32 GB RAM on `g2-standard-8` and freezes the VM — the example mission already sets
> `shuffle_buffer_size: 10000`.

Edit `config.data_root_dir` in `examples/multiagent-openvla-gemma/mission.yaml` to
your dataset's parent dir before running.

---

## 5. Load the environment & smoke-test the planner

The setup script wrote `examples/multiagent-openvla-gemma/.env`. **Source it** (do
not run it with `./`) so the exports land in your current shell:

```bash
source examples/multiagent-openvla-gemma/.env
```

That activates `env_pilot` and exports everything the mission needs:

```bash
source "<repo>/env_pilot/bin/activate"        # the OpenVLA pilot + eval venv

# --- pilot / training (OpenVLA) ---
export OPENVLA_REPO_PATH="<repo>/openvla"
export NCCL_NET=Socket          # GCP single-GPU: bypass the gIB NCCL plugin (see §7 Troubleshooting)
export WANDB_MODE=disabled      # OpenVLA calls wandb.init() unconditionally

# --- planner / specialist (out-of-process Gemma venv) ---
export ODYSSEY_SPECIALIST_PYTHON="<repo>/env_specialist/bin/python"

# --- evaluation (Robosuite / MuJoCo, headless) ---
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

> **Why `source`, not `./`:** variables `export`ed inside a script run with `./`
> live only in that script's subshell and vanish when it exits. `source .env`
> (or `. .env`) runs it in your current shell, so the vars stick. These are
> **per-shell** — re-`source` after every SSH reconnect.

> ⚠️ **`ODYSSEY_SPECIALIST_PYTHON` is required for any mission with a SPECIALIST.**
> The planner is launched in that venv (`RemotePlanner` →
> `python -m odyssey.runners.agents.planner_server`). If it is unset, multi-agent
> eval **fails fast** with a clear `RuntimeError`: the multimodal Gemma 4 planner
> cannot load in the main venv (which pins `transformers==4.40.1`).

### Smoke-test the planner alone

Before a full GPU run, confirm the out-of-process planner works on its own. This
launches the planner in `env_specialist`, feeds it a prompt, and prints a
decomposition — **no OpenVLA, no simulator needed** (it does download Gemma):

```bash
python tests/manual/smoke_remote_planner.py
```

A printed plan (a list of sub-steps) means the venv split, the specialist deps,
and `ODYSSEY_SPECIALIST_PYTHON` are all wired correctly. If this fails, fix it
**before** the full run — it isolates the planner from every pilot/sim variable.
(You can also have the setup script run this automatically with `--smoke`.)

---

## 6. Run the multi-agent mission

```bash
# Sanity-check the spec first (instant, no GPU)
odyssey validate examples/multiagent-openvla-gemma/mission.yaml

# Pilot LoRA fine-tune → plan-then-execute eval on Robosuite Lift
odyssey run examples/multiagent-openvla-gemma/mission.yaml
```

The mission has two tasks — `finetune-pilot` (training) → `eval-robosuite-lift`
(evaluation) — plus the SPECIALIST agent declared under `robot.agents`:

```yaml
robot:
  agents:
    - id: pilot
      role: PILOT
      model: { source: huggingface, base: openvla/openvla-7b }
    - id: task-planner
      role: SPECIALIST
      model:
        source: huggingface
        base: google/gemma-4-E2B-it
        quantization: int4
        modality: multimodal
```

The shipped config is tuned for a **quick smoke run** (`max_steps: 10`,
`save_steps: 5`, `num_episodes: 2`) so you can confirm the whole pipeline cheaply.
Bump those up once it runs clean.

### What success looks like

The run chains `training → evaluation` and ends with mission status **COMPLETED**.
During eval you'll see the planner launch out of process and emit a per-episode
plan, then the pilot execute it. The `eval`'s `checkpoint_path` referencing the
model the training task just produced is the proof that **train → eval chained
correctly through the engine**, and the planner grounding a plan in the scene image
is the proof that the **PILOT + SPECIALIST loop** works.

> **Expect a low score (often 0 successful lifts) — and that's fine.** The pilot is
> trained on **real-robot Bridge data** (a WidowX arm) but evaluated **in simulation
> on a Franka** (Robosuite Lift). That train→eval **domain + embodiment gap** means
> the policy rarely completes the lift, even at higher step counts — while the Gemma
> planner still emits correct per-episode plans. The goal here is a clean end-to-end
> multi-agent run; treat the lift itself as aspirational.

### See the rollout (video)

Set `capture_video: true` in the eval task's `config` to save **one MP4 per episode**
under the run's `videos/` dir (also surfaced in `result_summary.artifacts.videos`).
The VM is headless, so copy a clip to your laptop to watch it:

```bash
# on your laptop — adjust the run/task ids from the run output
gcloud compute scp --recurse --zone=<ZONE> \
  <VM_NAME>:~/.odyssey/runs/<mission-id>/<eval-task-id>/videos ./rollout-videos
```

---

## 7. Troubleshooting / debugging playbook

The recurring theme on GCP: **the pilot's training dies with `exit code 1` and no
Python traceback.** That's almost always the environment, not the code.

### Why errors are invisible

`odyssey run` launches OpenVLA through **`torchrun`, which swallows the child's
stderr** (`error_file: <N/A>`). To see the real error, run `finetune.py` directly
with plain `python`, setting the env vars torchrun would:

```bash
RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
  python ~/openvla/vla-scripts/finetune.py --vla_path openvla/openvla-7b ...
```

(When run via `torchrun`, the real Python traceback also prints **above** the
`ChildFailedError` block — scroll up past the elastic summary.)

### Silent `exit(1)` — the decoder

| Exit code | Meaning | Likely cause |
|---|---|---|
| `1` | clean / `os._exit` | **disk full** (TFDS can't write its sqlite cache → `sqlite3.OperationalError: disk I/O error`) |
| `137` | OOM-kill (SIGKILL) | GPU/RAM out of memory — often a zombie process (below) |
| `139` | segfault | native crash (e.g. torch 2.6 inductor race — pin 2.2.0) |
| `134` | abort | assertion / native abort |

**1. Disk full (the #1 culprit).** TFDS writes an sqlite cache while reading the
dataset; if the disk is full the native layer kills the process with `exit(1)` and
**no Python traceback**. Free space and the silent deaths stop:

```bash
df -h ~                    # check free space
rm -rf ~/.odyssey/runs/*   # old run outputs
rm -rf /tmp/ovla_*         # OpenVLA temp dirs
```

**2. GPU zombies → OOM.** OpenVLA can die without calling
`destroy_process_group()`, leaking its CUDA context. A leftover `finetune.py` /
`torchrun` process keeps holding ~15–20 GB, so the *next* run OOMs. Always clear
the GPU before a run:

```bash
pkill -9 -f finetune.py
pkill -9 -f torchrun
nvidia-smi                 # confirm 0 MiB used before re-running
```

### `network gIB not found` — the GCP NCCL gotcha

The pilot trains via PyTorch DDP (NCCL backend). **GCP GPU VMs ship a custom NCCL
plugin** (`/usr/local/gib/lib64/libnccl-net.so`) that registers a virtual `gIB`
(Google InfiniBand) network for multi-node RDMA. On a **single-GPU VM** the RDMA
hardware is absent, so the plugin fails with `network gIB not found`, surfacing as
`Default process group has not been initialized` and a non-obvious `exit code 1`.

**`export NCCL_NET=Socket`** forces NCCL onto TCP sockets and fixes it (zero perf
impact on a single GPU) — the `.env` already sets this. `NCCL_IB_DISABLE=1` does
**not** help (it only disables NCCL's built-in IB transport, not the external
plugin). On AWS/Azure you likely won't hit this.

### Multi-agent-specific failures

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError` at eval about loading the planner | `ODYSSEY_SPECIALIST_PYTHON` unset or wrong | `source .env` again; confirm it points at `env_specialist/bin/python` |
| Planner OOMs / "modules dispatched on the CPU" | tried E4B, or a zombie holds VRAM | use **E2B-it** (the shipped default); `nvidia-smi` shows 0 MiB before the run |
| Planner imports fail in its venv | `env_specialist` built against the wrong stack | rebuild: `setup.sh --skip-pilot` (re-pins `constraints/specialist-known-good.txt`) |
| Planner emits NaN / garbage plans | Gemma 3 4B under int4 emits NaN logits on this stack | use **Gemma 4 E2B-it** (the default) — Gemma 3 can't run quantized here |

### Headless rendering: `Cannot initialize a EGL device display` (no NVIDIA EGL)

The eval needs an offscreen GL context. On a VM whose NVIDIA driver was installed
**compute-only** (CUDA + `nvidia-smi` work, training runs — but no OpenGL/EGL
userspace), Robosuite/MuJoCo can't create an EGL context and the eval dies with:

```text
libEGL warning: failed to open /dev/dri/renderD128: Permission denied
ImportError: Cannot initialize a EGL device display ... PLATFORM_DEVICE extension ... required for headless
```

Diagnose — there is **no NVIDIA EGL**, only Mesa (and Mesa needs `/dev/dri`, which
is `render`-group-only):

```bash
ls /usr/share/glvnd/egl_vendor.d/   # only 50_mesa.json (no 10_nvidia.json) → no NVIDIA EGL
ldconfig -p | grep -i libEGL        # only libEGL_mesa, no libEGL_nvidia
```

Fix — fall back to **OSMesa** (CPU software rendering: no EGL, no GPU, no
`/dev/dri`). Slower than GPU EGL but reliable, and fine for eval rollouts:

```bash
sudo apt-get install -y libosmesa6-dev
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa     # overrides the egl values the .env exports
```

(GPU EGL is faster but requires installing the NVIDIA OpenGL/EGL userspace that
matches the driver — fiddly; OSMesa just works.)

### Disk fills up across runs — clean `~/.odyssey/runs/`

Every `odyssey run` stages a full copy of the base model (~15 GB) into
`~/.odyssey/runs/<mission>/<task>/model` and OpenVLA saves the full **merged** 7B
checkpoint (~14 GB) — so a few failed attempts silently eat 50-100 GB and you hit
`No space left on device` at `save_pretrained` (or a silent `exit(1)`). Clean
between attempts:

```bash
du -sh ~/.odyssey/runs/* | sort -rh   # see what's accumulated
rm -rf ~/.odyssey/runs/*              # all failed/old runs
df -h /
```

### `finetune.py` dies at import (protobuf / tensorflow-metadata / wandb)

`ImportError: cannot import name 'runtime_version' from google.protobuf` (tfds) or
`cannot import name 'Imports' from wandb_telemetry_pb2` (wandb) means the env was
built **without** the pinned constraints, so pip pulled protobuf-5-era packages
that the TF 2.15 stack can't load. `setup.sh` installs with
`-c constraints/openvla-known-good.txt` to prevent this; if you hand-built the
venv, reinstall through that constraints file (it pins
`protobuf==3.20.3`, `tensorflow-metadata==1.15.0`, `wandb==0.16.6`).

> 💡 **Re-using a snapshot/custom image?** System deps don't travel in `setup.sh`.
> Re-run the §3 `apt-get` (EGL/GL libs) and the OSMesa install above on any VM
> restored from a snapshot — the driver may also be compute-only there.

### Per-session checklist

Before each run after an SSH reconnect:

- [ ] `source examples/multiagent-openvla-gemma/.env` (re-export the env vars — they're per-shell)
- [ ] `nvidia-smi` shows **0 MiB** used (no zombies)
- [ ] `df -h ~` shows free disk (clean `~/.odyssey/runs/*` and `/tmp/ovla_*`)
- [ ] `echo $ODYSSEY_SPECIALIST_PYTHON` points at the specialist venv

---

## 8. Wrap up: get your results and stop the VM

Runs land under `~/.odyssey/runs/<mission-id>/<task-id>/` on the VM (checkpoints,
and the rollout MP4s if you enabled `capture_video`). Pull what you want to keep
back to your machine with `scp`:

```bash
# from your laptop — copy a task's output dir (checkpoint + videos) locally
gcloud compute scp --recurse \
  <VM_NAME>:~/.odyssey/runs/<mission-id> ./odyssey-results --zone=<ZONE>
```

Then **stop the VM** so it stops billing the GPU/compute:

```bash
gcloud compute instances stop <VM_NAME> --zone=<ZONE>
```

> 💸 A **stopped** VM no longer bills for compute, but **still bills for its disk**
> (the 300 GB). If you're done for good, either delete the instance *and* its disk,
> or snapshot the disk first and delete it. Don't just close the SSH session: that
> leaves the VM running.

---

## Appendix: environment variable reference

| Variable | Stage | Purpose |
|---|---|---|
| `OPENVLA_REPO_PATH` | train | Path to the cloned OpenVLA repo (locates `finetune.py`) |
| `ODYSSEY_SPECIALIST_PYTHON` | eval | **Required for multi-agent** — path to `env_specialist/bin/python` (the Gemma planner) |
| `NCCL_NET=Socket` | train | **GCP:** bypass the `gIB` NCCL plugin on single-GPU VMs |
| `WANDB_MODE=disabled` | train | OpenVLA calls `wandb.init()` unconditionally |
| `HF_TOKEN` | optional | Only to avoid anonymous HF rate limits — default models are ungated |
| `MUJOCO_GL=egl` | eval | Headless GL backend for MuJoCo |
| `PYOPENGL_PLATFORM=egl` | eval | Headless PyOpenGL platform |

## Appendix: the two known-good stacks

The venv split means the two environments are pinned independently:

```text
env_pilot  (constraints/openvla-known-good.txt)   env_specialist (constraints/specialist-known-good.txt)
  Python        3.10                                 Python        3.10
  torch         2.2.0 (+cu121)                       modern transformers + torchvision
  transformers  4.40.1                               (Gemma 4 multimodal stack)
  tokenizers    0.19.1
  accelerate    0.30.1
  bitsandbytes  0.43.1
  peft          ~= 0.11
```

## See also

- [Single-agent GCP tutorial](./gcp-training-tutorial.md) — the simpler OpenVLA-only path
- [Multi-agent example README](../examples/multiagent-openvla-gemma/README.md) — architecture + gotchas in depth
- `examples/multiagent-openvla-gemma/mission.yaml` — the mission spec, heavily commented
- `examples/multiagent-openvla-gemma/setup.sh` — the two-venv setup script
