# Training OpenVLA on a GCP GPU VM — end-to-end tutorial

This is the validated, reproducible procedure for running the OpenVLA training +
evaluation pipeline (single-agent quickstart **or** the multi-agent
OpenVLA-pilot + Gemma-planner mission) on a **Google Cloud GPU VM**.

It is GCP-specific on purpose: a couple of things bite you only on GCP (the NCCL
`gIB` plugin, L4 stockouts, disk sizing) and this guide front-loads them so you
don't lose a day to them like we did. The hard-won details come from the
end-to-end validation runs in issues
[#5](https://github.com/lovellai-dev/odyssey/issues/5) and
[#22](https://github.com/lovellai-dev/odyssey/issues/22).

> **Validated on:** `g2-standard-8` · NVIDIA **L4 (24 GB)** · Ubuntu · `us-central1-a`.
> Other clouds (AWS/Azure) or GPUs may not need the NCCL workaround — see
> [the NCCL section](#5-the-gcp-critical-environment-variables).

---

## What you'll do

1. [Provision a GPU VM](#1-provision-the-vm) (with enough disk — this matters)
2. [Connect and install system deps](#2-connect--system-dependencies)
3. [Install Odyssey + OpenVLA](#3-install-odyssey--openvla)
4. [Download the Bridge V2 dataset](#4-the-dataset-bridge-v2-in-rlds-format) (RLDS, ~124 GB)
5. [Set the GCP-critical environment variables](#5-the-gcp-critical-environment-variables)
6. [Run the mission](#6-run-the-mission)
7. [Troubleshoot](#7-troubleshooting--debugging-playbook) when it silently dies
8. [Get your results and stop the VM](#8-wrap-up-get-your-results-and-stop-the-vm)

Plan for **most of a morning**: the steps themselves are quick, but the 124 GB
dataset download dominates and scales with your bandwidth (anywhere from ~30 min
to several hours).

> 💸 **Validate for free first.** A GPU VM costs real money the whole time it's
> running. Before you provision anything, confirm the mission spec and the whole
> orchestration flow on your laptop with the CPU mock — no GPU, no cost:
> ```bash
> odyssey run examples/quickstart-openvla/mission.yaml --use-mock-runner
> ```
> Only spin up the VM once that runs clean. And see [§8](#8-wrap-up-get-your-results-and-stop-the-vm)
> — **stop the VM when you're done** so it stops billing.

---

## 0. Prerequisites

- A **GCP project** with billing enabled and the [`gcloud` CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated.
- **GPU quota for L4.** New projects start with **zero** GPU quota — you must
  request an increase for `NVIDIA_L4_GPUS` (and/or `GPUS_ALL_REGIONS`) in your
  target region via *IAM & Admin → Quotas*. Approval can take minutes to a day, so
  **request it before you need it**. Hitting `Quota 'NVIDIA_L4_GPUS' exceeded`
  at VM-creation time means this step was skipped.
- A **HuggingFace account** with the gated model licenses accepted (see [§5](#huggingface-authentication-gated-models)).
- Basic familiarity with SSH and the Linux shell.

---

## 1. Provision the VM

| Setting | Value | Why |
|---|---|---|
| Machine type | `g2-standard-8` | 8 vCPU / 32 GB RAM, pairs with one L4 |
| GPU | 1 × NVIDIA **L4 (24 GB)** | Enough VRAM for OpenVLA-7B LoRA (~15 GB) + headroom |
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

> ⚠️ **L4 stockouts are frequent** in `us-central1-a`. If VM creation fails with a
> stockout, try another zone (`us-central1-b`, `us-west1-a`, …) — though those
> stock out too. **Take a [disk snapshot](https://cloud.google.com/compute/docs/disks/create-snapshots)
> before stopping or resizing a working VM**: we once lost a VM to a resize that
> left it unbootable, then couldn't get a new L4 for hours.

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

## 3. Install Odyssey + OpenVLA

OpenVLA's fine-tune runs through the **upstream OpenVLA repo**, which carries its
own dependency set. Most onboarding friction comes from version drift there, not
from Odyssey — so we pin a known-good stack.

```bash
# Odyssey
git clone https://github.com/lovellai-dev/odyssey.git ~/odyssey
cd ~/odyssey
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all,dev]"

# Upstream OpenVLA (provides draccus + the finetune.py entry point)
git clone https://github.com/openvla/openvla.git ~/openvla
pip install -e ~/openvla
```

### Known-good stack (pin these)

This is the exact combination validated end-to-end (train + eval) on the L4.
It is **not** hard-pinned in `pyproject.toml` (deferred until verified in CI), so
pin it yourself in the venv:

```bash
pip install \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  "numpy<2" \
  transformers==4.40.1 tokenizers==0.19.1 \
  accelerate==0.30.1 bitsandbytes==0.43.1 "peft~=0.11"
# Python 3.10
```

> 🔥 **Do not run on torch 2.6.** It caused a **silent `exit(1)`** during training
> (an inductor compile-worker fork-after-CUDA race). OpenVLA's pinned **`torch==2.2.0`**
> fixes it. Likewise the `transformers>=4.40` *floor* in the `[openvla]` extra can
> pull a too-new version on reinstall — pin `4.40.1` explicitly.

> 💡 **Avoid re-downloading the 7B base each run** by pointing its path env var at
> a local copy (HF id upper-cased, `/` and `-` → `_`, suffixed `_PATH`):
> ```bash
> export OPENVLA_OPENVLA_7B_PATH=/path/to/openvla-7b   # for base: openvla/openvla-7b
> ```

---

## 4. The dataset: Bridge V2 in RLDS format

OpenVLA's `finetune.py` expects datasets in **RLDS / TensorFlow-Datasets format**,
**not** LeRobot/HuggingFace format. Download Bridge V2 and rename it:

```bash
# ~124 GB — make sure the disk has room (see §1)
wget -r -nH --cut-dirs=4 --reject="index.html*" \
  https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# OpenVLA's OXE registry key for this is `bridge_orig`
mv bridge_dataset bridge_orig
```

> ⚠️ **`data_root_dir` is the *parent* of the dataset folder, not the folder
> itself.** OpenVLA resolves the data at `<data_root_dir>/<dataset_name>/<version>`
> (e.g. `/home/<user>/bridge_orig/1.0.0/`). So if the dataset lives at
> `/home/gema/bridge_orig/`, set `data_root_dir: /home/gema` — **not**
> `/home/gema/bridge_orig`. A wrong path fails right after model load + LoRA wrap,
> at dataset creation.

In your `mission.yaml`, the dataset maps through as a pass-through (Odyssey does
**not** download it):

| `mission.yaml` | becomes the flag | meaning |
|---|---|---|
| `dataset.ref: bridge_orig` | `--dataset_name bridge_orig` | the OXE **registry key** OpenVLA looks up |
| `config.data_root_dir: <path>` | `--data_root_dir <path>` | the **parent dir** of the RLDS folder |

> 💡 **Keep the shuffle buffer small.** The upstream RLDS default (256k) eats all
> 32 GB of RAM on `g2-standard-8` and freezes the VM. Set `shuffle_buffer_size: 10000`
> in the training task's `config` (the example missions already do).

---

## 5. The GCP-critical environment variables

These are per-shell — **you must re-export them after every SSH reconnect.** See
the [reusable `env.sh`](#tip-put-them-in-an-envsh) tip below.

```bash
cd ~/odyssey && source .venv/bin/activate

# --- Training (OpenVLA) ---
export OPENVLA_REPO_PATH=~/openvla
export NCCL_NET=Socket          # GCP single-GPU: bypass the gIB NCCL plugin (see below)
export WANDB_MODE=disabled      # OpenVLA calls wandb.init() unconditionally

# --- HuggingFace auth (gated models) ---
export HF_TOKEN=hf_xxxxxxxx

# --- Evaluation (Robosuite / MuJoCo, headless) ---
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### `NCCL_NET=Socket` — the GCP gotcha

OpenVLA trains via PyTorch DDP, which uses the NCCL backend. **GCP GPU VMs ship a
custom NCCL plugin** at `/usr/local/gib/lib64/libnccl-net.so` that registers a
virtual network called **`gIB`** (Google InfiniBand) for GPUDirect RDMA between
multi-node GPU clusters. On a **single-GPU VM** the RDMA hardware is absent, so
the plugin fails:

```
Error: network gIB not found
```

surfacing as `Default process group has not been initialized` and a non-obvious
`exit code 1`.

- `NCCL_IB_DISABLE=1` does **not** fix it — that only disables NCCL's *built-in*
  IB transport, not the external plugin.
- **`export NCCL_NET=Socket`** forces NCCL onto TCP sockets, bypassing the plugin.
  **Zero performance impact on a single-GPU VM.**

> On **AWS/Azure** you likely won't see the `gIB` error — if so, skip this step.

### HuggingFace authentication (gated models)

`openvla/openvla-7b` (the PILOT) is **gated** — accept its license on the model
page, then authenticate. The multi-agent planner uses `google/gemma-4-E2B-it`,
which is **ungated** (Apache-2.0), so it needs no token.

```bash
export HF_TOKEN=hf_xxxxxxxx      # most robust
# or: hf auth login
```

> ⚠️ The deprecated `huggingface-cli login` with a **named** token prints a
> non-fatal traceback at the end (`Token <name> not found ...`). The token is
> still saved and valid — but `export HF_TOKEN=...` sidesteps it entirely.

### Tip: put them in an `env.sh`

Because the vars are per-shell and easy to forget after a reconnect, drop them in
a file you `source` each session:

```bash
# ~/odyssey/env.sh  — run with:  source env.sh   (NOT ./env.sh)
source ~/odyssey/.venv/bin/activate
export OPENVLA_REPO_PATH=~/openvla
export NCCL_NET=Socket
export WANDB_MODE=disabled
export HF_TOKEN=hf_xxxxxxxx
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

> **Why `source`, not `./`:** variables `export`ed inside a script run with `./`
> live only in that script's subshell and vanish when it exits. `source env.sh`
> (or `. env.sh`) runs it in your current shell, so the vars stick.

---

## 6. Run the mission

Start with the **single-agent** quickstart — it's the spine of this tutorial:
OpenVLA LoRA fine-tune → Robosuite Lift eval, one venv, no extra moving parts.

```bash
# Sanity-check the spec first (instant, no GPU)
odyssey validate examples/quickstart-openvla/mission.yaml

# OpenVLA LoRA fine-tune → Robosuite Lift eval
odyssey run examples/quickstart-openvla/mission.yaml
```

### What success looks like

The run chains `training → evaluation` and ends with mission status **COMPLETED**.
The tail of the output looks roughly like:

```
finetune-openvla      -> training, OpenVLA LoRA finetune, checkpoint saved
eval-on-robosuite-lift -> COMPLETED, used the freshly-trained checkpoint
{"event": "mission.completed", "overall_grade": 0.0}
COMPLETED  <mission-id>
```

The eval's `checkpoint_path` referencing the model the training task just produced
is the proof that **train → eval chained correctly through the engine**.

> A low score / `grade F` on a short fine-tune is **expected** — a few hundred LoRA
> steps don't solve Lift, and there's a real domain gap (Bridge V2 real-robot data
> → Robosuite sim). The goal of this tutorial is a clean **end-to-end run**, not a
> high score. To *see* what the policy actually does, enable `capture_video: true`
> in the eval task's `config` (see the README) — it saves an MP4 per episode.

### Optional: the multi-agent mission

`examples/multiagent-openvla-gemma/mission.yaml` adds a **Gemma vision-language
planner** alongside the OpenVLA pilot. It needs **a second Python environment**
(the planner can't share OpenVLA's pinned `transformers==4.40.1`), reached via
`ODYSSEY_SPECIALIST_PYTHON`. Set that up first per the README's **"Multi-agent"**
section, then:

```bash
odyssey run examples/multiagent-openvla-gemma/mission.yaml
```

---

## 7. Troubleshooting / debugging playbook

The recurring theme on GCP: **training dies with `exit code 1` and no Python
traceback.** That's almost always the environment, not the code.

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

### Per-session checklist

Before each run after an SSH reconnect:

- [ ] `source env.sh` (re-export the env vars — they're per-shell)
- [ ] `nvidia-smi` shows **0 MiB** used (no zombies)
- [ ] `df -h ~` shows free disk (clean `~/.odyssey/runs/*` and `/tmp/ovla_*`)
- [ ] correct stack pinned (`torch==2.2.0`, `transformers==4.40.1`)

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
> or snapshot the disk first and delete it — a snapshot is far cheaper to park than
> a live 300 GB disk. Don't just close the SSH session: that leaves the VM running.

## Appendix: environment variable reference

| Variable | Stage | Purpose |
|---|---|---|
| `OPENVLA_REPO_PATH` | train | Path to the cloned OpenVLA repo (locates `finetune.py`) |
| `NCCL_NET=Socket` | train | **GCP:** bypass the `gIB` NCCL plugin on single-GPU VMs |
| `WANDB_MODE=disabled` | train | OpenVLA calls `wandb.init()` unconditionally |
| `HF_TOKEN` | train | Auth for gated models (`openvla/openvla-7b`) |
| `MUJOCO_GL=egl` | eval | Headless GL backend for MuJoCo |
| `PYOPENGL_PLATFORM=egl` | eval | Headless PyOpenGL platform |
| `ODYSSEY_SPECIALIST_PYTHON` | eval (multi-agent) | Path to the specialist venv's python (Gemma planner) |
| `OPENVLA_OPENVLA_7B_PATH` | train (optional) | Local cache of the 7B base to skip re-download |

## Appendix: known-good stack

```text
Python        3.10
torch         2.2.0 (+cu121)
torchvision   0.17.0
torchaudio    2.2.0
numpy         < 2
transformers  4.40.1
tokenizers    0.19.1
accelerate    0.30.1
bitsandbytes  0.43.1
peft          ~= 0.11
```
