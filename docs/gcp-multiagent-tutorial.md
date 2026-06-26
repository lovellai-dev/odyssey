# Multi-agent on a GCP GPU VM — OpenVLA pilot + Gemma planner (end-to-end)

This is the complete, self-contained procedure for running the **multi-agent
mission** on a **Google Cloud GPU VM**: an OpenVLA **PILOT** fine-tuned on Bridge
V2, guided at evaluation time by a vision-language **SPECIALIST** task planner
(multimodal **Gemma 4 E2B-it**) that runs **out of process**, evaluated on
Robosuite **Lift**.

It is GCP-specific on purpose: a few things bite you only on GCP (the NCCL `gIB`
plugin, L4 stockouts, disk sizing) and this guide front-loads them so you don't
lose a day to them. The hard-won details come from the end-to-end validation runs
in issues [#5](https://github.com/lovellai-dev/odyssey/issues/5) and
[#22](https://github.com/lovellai-dev/odyssey/issues/22).

> **Validated on:** `g2-standard-8` · NVIDIA **L4 (24 GB)** · Ubuntu · `us-central1-a`.
> Other clouds (AWS/Azure) or GPUs may not need the NCCL workaround — see
> [§6](#6-the-gcp-critical-environment-variables).

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

Hence **two venvs, named by role** so a plain `ls` tells them apart:

| venv | hosts | stack |
|---|---|---|
| **`env_pilot`** | OpenVLA pilot + the Robosuite eval | OpenVLA pinned (`torch 2.2.0`, `transformers 4.40.1`) |
| **`env_specialist`** | the Gemma 4 planner | modern `transformers` + `torchvision` |

Building both correctly is the **fragile part of multi-agent** — so this tutorial
leans on the **`setup.sh` script**, which builds both idempotently and pins each
to its known-good stack. It saves a lot of fiddling.

---

## What you'll do

1. [Provision a GPU VM](#1-provision-the-vm) (with enough disk — this matters)
2. [Connect and install system deps](#2-connect--system-dependencies)
3. [Build both venvs with the setup script](#3-build-both-venvs-with-the-setup-script)
4. [Download the Bridge V2 dataset](#4-the-dataset-bridge-v2-in-rlds-format) (RLDS, ~124 GB)
5. [Load the environment + smoke-test the planner](#5-load-the-environment--smoke-test-the-planner)
6. [Set the GCP-critical environment variables](#6-the-gcp-critical-environment-variables)
7. [Run the multi-agent mission](#7-run-the-multi-agent-mission)
8. [Troubleshoot](#8-troubleshooting--debugging-playbook) when it silently dies
9. [Get your results and stop the VM](#9-wrap-up-get-your-results-and-stop-the-vm)

Plan for **most of a morning**: the steps themselves are quick, but the 124 GB
dataset download dominates and scales with your bandwidth (~30 min to several hours).

> 💸 **Validate for free first.** A GPU VM costs real money the whole time it's
> running. Before you provision anything, confirm the mission spec and the whole
> orchestration flow on your laptop with the CPU mock — no GPU, no cost:
> ```bash
> odyssey run examples/multiagent-openvla-gemma/mission.yaml --use-mock-runner
> ```
> Only spin up the VM once that runs clean. And see [§9](#9-wrap-up-get-your-results-and-stop-the-vm)
> — **stop the VM when you're done** so it stops billing.

---

## 0. Prerequisites

- A **GCP project** with billing enabled and the [`gcloud` CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated.
- **GPU quota for L4.** New projects start with **zero** GPU quota — request an
  increase for `NVIDIA_L4_GPUS` (and/or `GPUS_ALL_REGIONS`) in your target region
  via *IAM & Admin → Quotas*. Approval can take minutes to a day, so **request it
  before you need it**. Hitting `Quota 'NVIDIA_L4_GPUS' exceeded` at VM-creation
  time means this step was skipped.
- *(Optional)* a **HuggingFace token** — both default models are ungated, so it's
  only needed to avoid anonymous download rate limits or to use a gated model
  (see [§6](#huggingface-access-optional)).
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
> [§8](#silent-exit1--the-decoder)). Provision **300 GB** from the start.
>
> Disk-light alternative: instead of a 300 GB disk you can **stream the dataset
> from a GCS bucket via [`gcsfuse`](https://cloud.google.com/storage/docs/gcsfuse-quickstart-mount-bucket)**
> (mount the bucket and point `data_root_dir` at the mount). Slower per-step I/O,
> but no large local disk.

> 💸 **Cost & quota.** A `g2-standard-8` + L4 runs on the order of **~$0.70–1/hour**
> (varies by region; check the [pricing page](https://cloud.google.com/compute/gpus-pricing)),
> plus a few $/month for the 300 GB disk. **You pay while the VM is running, GPU
> idle or not** — so stop it between sessions ([§9](#9-wrap-up-get-your-results-and-stop-the-vm)).
> A stopped VM still bills for its disk.

> ⚠️ **L4 stockouts are frequent** in `us-central1-a`. If VM creation fails with a
> stockout, try another zone (`us-central1-b`, `us-west1-a`, …). **Take a
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
3. **HuggingFace auth** — optional and non-blocking. Both default models are ungated,
   so no token is needed; if `HF_TOKEN` is set it logs in non-interactively.
4. **Writes `examples/multiagent-openvla-gemma/.env`** — the file you `source`
   before every run (details in [§5](#5-load-the-environment--smoke-test-the-planner)).

Useful flags:

| Flag | Effect |
|---|---|
| `--skip-pilot` | skip `env_pilot` + OpenVLA (you already built it, e.g. for single-agent) |
| `--pilot-venv PATH` | relocate the pilot venv (default `<repo>/env_pilot`) |
| `--specialist-venv PATH` | relocate the specialist venv (default `<repo>/env_specialist`) |
| `--openvla-repo PATH` | where to clone OpenVLA (default `~/openvla`) |
| `--smoke` | run the planner smoke check at the end (downloads Gemma) |
| `-h`, `--help` | show all options |

> 💡 **Why a script and not copy-paste.** The two-venv split with mutually
> incompatible `transformers` versions is exactly where manual setups break. The
> script pins both stacks from `constraints/*.txt`, so you get the validated
> combination every time and can re-run it safely after a botched attempt.

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

---

## 4. The dataset: Bridge V2 in RLDS format

The pilot's `finetune.py` expects datasets in **RLDS / TensorFlow-Datasets format**,
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
> `/home/gema/bridge_orig`. A wrong path fails right after model load + LoRA wrap.

In the mission, the dataset maps through as a pass-through (Odyssey does **not**
download it):

| `mission.yaml` | becomes the flag | meaning |
|---|---|---|
| `dataset.ref: bridge_orig` | `--dataset_name bridge_orig` | the OXE **registry key** OpenVLA looks up |
| `config.data_root_dir: <path>` | `--data_root_dir <path>` | the **parent dir** of the RLDS folder |

> 💡 **Keep the shuffle buffer small.** The upstream RLDS default (256k) eats all
> 32 GB of RAM on `g2-standard-8` and freezes the VM. The example mission already
> sets `shuffle_buffer_size: 10000` in the training task's `config`.

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
export NCCL_NET=Socket          # GCP single-GPU: bypass the gIB NCCL plugin (see §6)
export WANDB_MODE=disabled      # OpenVLA calls wandb.init() unconditionally

# --- planner / specialist (out-of-process Gemma venv) ---
export ODYSSEY_SPECIALIST_PYTHON="<repo>/env_specialist/bin/python"

# --- evaluation (Robosuite / MuJoCo, headless) ---
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# --- HuggingFace (OPTIONAL — default models are ungated) ---
# export HF_TOKEN=hf_xxxxxxxx
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

## 6. The GCP-critical environment variables

The `.env` from [§5](#5-load-the-environment--smoke-test-the-planner) already
exports these. This section explains the two that bite on GCP.

### `NCCL_NET=Socket` — the GCP gotcha

The pilot trains via PyTorch DDP, which uses the NCCL backend. **GCP GPU VMs ship a
custom NCCL plugin** at `/usr/local/gib/lib64/libnccl-net.so` that registers a
virtual network called **`gIB`** (Google InfiniBand) for GPUDirect RDMA between
multi-node GPU clusters. On a **single-GPU VM** the RDMA hardware is absent, so the
plugin fails:

```
Error: network gIB not found
```

surfacing as `Default process group has not been initialized` and a non-obvious
`exit code 1`.

- `NCCL_IB_DISABLE=1` does **not** fix it — that only disables NCCL's *built-in*
  IB transport, not the external plugin.
- **`export NCCL_NET=Socket`** forces NCCL onto TCP sockets, bypassing the plugin.
  **Zero performance impact on a single-GPU VM.**

> On **AWS/Azure** you likely won't see the `gIB` error — if so, this var is harmless.

### HuggingFace access (optional)

Both default models are **ungated** — they download without a token:

- [`openvla/openvla-7b`](https://huggingface.co/openvla/openvla-7b) — the PILOT
- [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) — the SPECIALIST (Apache-2.0)

So **no login is required** for the shipped mission. A token only raises the
anonymous download rate limit, and becomes **required** only if you swap in a gated
model — e.g. the larger Gemma 4 **E4B**, or `gemma-2` / `gemma-3`, which 401/403
until you accept Google's terms:

```bash
export HF_TOKEN=hf_xxxxxxxx      # optional
# or: hf auth login
```

---

## 7. Run the multi-agent mission

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

> **A low score / 0 lifts is expected.** The pilot is OpenVLA fine-tuned on
> `bridge_orig` (real-robot Bridge data), so there's a real sim-to-real domain gap
> into Robosuite Lift — 0 successful lifts are common even at higher step counts,
> while the Gemma planner still produces correct per-episode plans. Treat the lift
> as aspirational; the point of this tutorial is a clean end-to-end multi-agent run.
> To *see* the rollout, enable `capture_video: true` in the eval task's `config` —
> it saves an MP4 per episode.

---

## 8. Troubleshooting / debugging playbook

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

### Multi-agent-specific failures

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError` at eval about loading the planner | `ODYSSEY_SPECIALIST_PYTHON` unset or wrong | `source .env` again; confirm it points at `env_specialist/bin/python` |
| Planner OOMs / "modules dispatched on the CPU" | tried E4B, or a zombie holds VRAM | use **E2B-it** (the shipped default); `nvidia-smi` shows 0 MiB before the run |
| Planner imports fail in its venv | `env_specialist` built against the wrong stack | rebuild: `setup.sh --skip-pilot` (re-pins `constraints/specialist-known-good.txt`) |
| Planner emits NaN / garbage plans | Gemma 3 4B under int4 emits NaN logits on this stack | use **Gemma 4 E2B-it** (the default) — Gemma 3 can't run quantized here |

### Per-session checklist

Before each run after an SSH reconnect:

- [ ] `source examples/multiagent-openvla-gemma/.env` (re-export the env vars — they're per-shell)
- [ ] `nvidia-smi` shows **0 MiB** used (no zombies)
- [ ] `df -h ~` shows free disk (clean `~/.odyssey/runs/*` and `/tmp/ovla_*`)
- [ ] `echo $ODYSSEY_SPECIALIST_PYTHON` points at the specialist venv

---

## 9. Wrap up: get your results and stop the VM

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
