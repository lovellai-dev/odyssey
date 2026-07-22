# GR00T + LIBERO on a GCP GPU VM — eval-only, end-to-end tutorial

This is the validated, reproducible procedure for scoring the published **NVIDIA
GR00T-N1.7** checkpoint on the **LIBERO** simulation benchmark on a **Google Cloud
GPU VM**. It is the GR00T counterpart to the
[OpenVLA training tutorial](./gcp-training-tutorial.md): where that one fine-tunes
OpenVLA on Bridge V2 then evaluates on Robosuite, this one is **eval-only** — a
Franka Panda doing language-conditioned pick-and-place ("pick up the X …") driven
by a **chunk-emitting GR00T pilot** served out-of-process over ZMQ.

It is GCP-specific on purpose: a couple of things bite you only on GCP (L4
stockouts, disk sizing, headless GL) and this guide front-loads them. The
hard-won details come from the end-to-end validation run in issue
[#65](https://github.com/lovellai-dev/odyssey/issues/65), where the mission scored
a clean **10/10 on `libero_object`**.

> **Validated on:** `g2-standard-8` · NVIDIA **L4 (24 GB)** · Ubuntu · `us-west1-a`
> (on-demand — the most reliable zone for L4; `us-central1-a` is chronically
> stocked out). The steps are zone-independent.

---

## What's different from the OpenVLA tutorial

If you've done the [OpenVLA one](./gcp-training-tutorial.md), here's the mental diff:

| | OpenVLA tutorial | This (GR00T + LIBERO) |
|---|---|---|
| **Work done** | Train (LoRA fine-tune) → eval | **Eval only** — checkpoint pulled from HF |
| **Dataset on disk** | Bridge V2, **~124 GB** | **None** — the published checkpoint auto-downloads |
| **Pilot** | OpenVLA-7B, **in-process**, 1 action/step | GR00T-N1.7, **out-of-process** ZMQ server, **action chunks** (16 steps/call) |
| **Sim** | Robosuite (Lift) | **LIBERO** (Franka pick-and-place, MuJoCo) |
| **Environments** | 1 venv (`env_pilot`) | **2 venvs** — LIBERO client + GR00T model server (ABIs clash) |
| **GCP gotcha** | `NCCL_NET=Socket` (DDP training) | none for NCCL (no training); **headless GL** + a **gated backbone** instead |

Because there's no 124 GB download and no training, this is **much faster** than
the OpenVLA tutorial — plan for **under an hour**, most of it the one-time venv
build and model download (~13 GB: the 6.5 GB suite checkpoint + the ~6 GB Cosmos
backbone).

---

## What you'll do

1. [Provision a GPU VM](#1-provision-the-vm)
2. [Connect and install system deps](#2-connect--system-dependencies)
3. [Get access to the gated GR00T backbone](#3-huggingface-access-the-gated-backbone) (do this early — approval takes time)
4. [Build both environments with `setup.sh`](#4-build-both-environments-one-command)
5. [Wire the server interpreter + set env vars](#5-wire-the-server-interpreter--environment)
6. [Run the mission](#6-run-the-mission)
7. [Troubleshoot](#7-troubleshooting--debugging-playbook)
8. [Get your results and stop the VM](#8-wrap-up-get-your-results-and-stop-the-vm)

> 💸 **Validate for free first.** A GPU VM costs real money the whole time it's
> running. Before you provision anything, confirm the mission spec parses on your
> laptop — no GPU, no cost:
> ```bash
> odyssey validate examples/franka-libero/mission-gr00t.yaml
> ```
> Only spin up the VM once that's clean. And see
> [§8](#8-wrap-up-get-your-results-and-stop-the-vm) — **stop the VM when you're
> done** so it stops billing.

---

## 0. Prerequisites

- A **GCP project** with billing enabled and the [`gcloud` CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated.
- **GPU quota for L4.** New projects start with **zero** GPU quota — request an
  increase for `NVIDIA_L4_GPUS` (and/or `GPUS_ALL_REGIONS`) in your target region
  via *IAM & Admin → Quotas*. Approval can take minutes to a day, so **request it
  before you need it**. `Quota 'NVIDIA_L4_GPUS' exceeded` at VM-creation time means
  this step was skipped.
- **A HuggingFace account with access to `nvidia/Cosmos-Reason2-2B`** — this is the
  **gated VLM backbone every GR00T checkpoint loads**. Request access early
  ([§3](#3-huggingface-access-the-gated-backbone)); it's a manual approval and the
  server will not start without it.
- Basic familiarity with SSH and the Linux shell.

---

## 1. Provision the VM

| Setting | Value | Why |
|---|---|---|
| Machine type | `g2-standard-8` | 8 vCPU / 32 GB RAM, pairs with one L4 |
| GPU | 1 × NVIDIA **L4 (24 GB)** | GR00T-3B (~6 GB) leaves ample headroom (multi-agent peak ≈ 12–14 GB) |
| OS image | Ubuntu (Deep Learning VM image works well) | CUDA drivers preinstalled |
| **Boot disk** | **≥ 150 GB** | ~13 GB models + venvs (GR00T's torch+flash-attn stack is large) + run outputs |

> ⚠️ **You don't need 300 GB here** — there's no Bridge V2 dataset. **150 GB is
> plenty** for the two venvs, the ~13 GB of model weights, and rollout videos. (The
> OpenVLA tutorial needs 300 GB *only* because of the 124 GB dataset.)

> 💸 **Cost & quota.** A `g2-standard-8` + L4 runs on the order of **~$0.70–1/hour**
> (varies by region; check the [pricing page](https://cloud.google.com/compute/gpus-pricing)),
> plus a few $/month for the disk. **You pay while the VM is running, GPU idle or
> not** — stop it between sessions ([§8](#8-wrap-up-get-your-results-and-stop-the-vm)).

> ⚠️ **L4 stockouts are frequent** in `us-central1-a`. Prefer an **on-demand** L4
> in **`us-west1-a`** (the most reliable in our runs). If creation fails with a
> stockout, try another zone. **Snapshot a working disk before stopping/resizing** —
> we once lost a VM to a resize that left it unbootable.

---

## 2. Connect & system dependencies

```bash
gcloud compute ssh <VM_NAME> --zone=<ZONE>
nvidia-smi      # expect: NVIDIA L4, 24 GB
```

Two things need installing that Odyssey's `setup.sh --pilot gr00t` will handle for
you, but which are worth knowing:

- **`uv`** — the GR00T server venv is built with [uv](https://docs.astral.sh/uv/)
  (it fetches a managed CPython 3.12 and resolves the PyTorch cu128 index cleanly).
  Install it first, because `setup.sh` errors out early if it's missing:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
  ```
- **System build + render deps** — LIBERO's `egl_probe` (via `robomimic`) compiles
  from C source, and MuJoCo needs a headless GL backend. `setup.sh --pilot gr00t`
  installs these with `sudo` for you; listed here for reference:
  ```bash
  sudo apt-get update && sudo apt-get install -y cmake build-essential \
    python3-dev python3.10-dev \
    libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libosmesa6-dev
  ```

---

## 3. HuggingFace access: the gated backbone

**This is the single most common reason a first GR00T run fails**, so do it early.

Every GR00T-N1.7 checkpoint loads the **gated** VLM backbone
[`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B) at
serve time. If it isn't in your HF cache, the policy server never comes up.

```bash
# 1. Request access on the Hub (manual approval — do this first, it can take a while):
#    https://huggingface.co/nvidia/Cosmos-Reason2-2B   → "Agree and access repository"

# 2. Authenticate on the VM once access is granted:
huggingface-cli login          # paste a token from https://huggingface.co/settings/tokens
```

The suite checkpoint itself (`nvidia/GR00T-N1.7-LIBERO`) is **ungated** —
`setup.sh` pre-downloads it. Only the backbone needs the access request.

> `setup.sh` pre-caches the backbone for you and prints a **clear warning with the
> fix steps** if it can't (access not yet granted). If you see that warning, finish
> the access request, `huggingface-cli login`, then
> `huggingface-cli download nvidia/Cosmos-Reason2-2B` (or just re-run setup).

---

## 4. Build both environments (one command)

GR00T + LIBERO spans **two Python environments that cannot share an interpreter** —
their `torch` / CUDA / Python ABIs conflict:

| Env | Interpreter | Holds |
|---|---|---|
| **`env_pilot_libero`** | `<repo>/env_pilot_libero` (py3.10) | the LIBERO/robosuite recipe **+ the lightweight GR00T ZMQ client** |
| **GR00T server** | `~/Isaac-GR00T/.venv` (py3.12) | the GR00T **model**, `torch 2.9.0+cu128`, flash-attn |

They talk over **ZMQ**: the LIBERO recipe runs the MuJoCo rollout and drives GR00T
as a **client** of the model server. `make_gr00t_policy` is a thin ZMQ client and
never imports the GR00T model in-process — that's why the split is required.

Clone Odyssey, then let the **end-to-end** `setup.sh` build everything —
system deps, both venvs, and the model download — in one command:

```bash
git clone https://github.com/lovellai-dev/odyssey.git ~/odyssey
cd ~/odyssey

# END-TO-END for the GR00T pilot: installs system deps, builds env_pilot_libero,
# builds the GR00T server venv (~/Isaac-GR00T/.venv), and pre-downloads the suite
# checkpoint + gated backbone. Idempotent — safe to re-run.
bash examples/franka-libero/setup.sh --pilot gr00t
```

Under the hood `--pilot gr00t` (see the script header for the full rationale):

- installs the system build/render deps (`cmake`, GL/EGL/OSMesa headers, `python3-dev`);
- builds `env_pilot_libero` with the pilot stack via
  `examples/multiagent-openvla-gemma/setup.sh --pilot-venv`;
- clones **LIBERO** to `~/LIBERO` and installs its deps **minus `transformers` and
  `numpy`** (LIBERO hard-pins `transformers==4.21.1`, which would break the stack),
  registering the repo on the venv path via a `.pth` (LIBERO is a PEP 420 namespace
  package — `pip install -e .` registers nothing);
- pre-initializes `~/.libero/config.yaml` **non-interactively** (the first
  `import libero.libero` otherwise prompts on stdin and would hang `odyssey run`);
- installs the ZMQ client transport (`msgpack`, `msgpack-numpy`, `pyzmq`) into the
  pilot venv;
- clones **`NVIDIA/Isaac-GR00T`** to `~/Isaac-GR00T` and builds its server venv with
  `uv` (py3.12, `constraints/gr00t-server-known-good.txt`; flash-attn arrives as a
  prebuilt cu128/torch2.9 wheel — no separate build);
- pre-downloads the `libero_object` suite checkpoint and the gated Cosmos backbone.

> **A different suite?** Pass `--suite libero_spatial` (or `libero_goal` /
> `libero_10`) to download that checkpoint instead, and change `benchmark_name` +
> `checkpoint` in the mission to match. `libero_object` is the default and what this
> tutorial validates.

> **NOT Isaac Sim.** LIBERO is MuJoCo — there is **no Isaac Sim / Isaac Lab install
> here.** (The Isaac-GR00T repo is cloned only for its *server* code + venv.) That's
> what makes this far lighter than the Isaac-Lab GR00T path in
> [`examples/quickstart-gr00t`](../examples/quickstart-gr00t/README.md).

---

## 5. Wire the server interpreter + environment

When `setup.sh --pilot gr00t` finishes it prints the exact next steps. Two pieces:

**a) Point the mission at the GR00T server's Python.** The mission needs to know
which interpreter hosts the model server. `setup.sh` auto-wires this from
`$GR00T_VENV_PYTHON` / `$ISAAC_GR00T_DIR/.venv/bin/python`, but you can also set it
explicitly in `examples/franka-libero/mission-gr00t.yaml` under the task `config`:

```yaml
    config:
      # ...
      server_python: /home/<user>/Isaac-GR00T/.venv/bin/python
```

**b) Set the per-shell environment.** These are per-shell — **re-export them after
every SSH reconnect**:

```bash
cd ~/odyssey && source env_pilot_libero/bin/activate

# Headless render. OSMesa (CPU) always works on compute-only drivers; use egl only
# if the VM exposes NVIDIA EGL (libEGL_nvidia). OSMesa is the safe default.
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

# Run ONLINE. The server does an online metadata check for the backbone; a forced
# offline mode breaks server startup with recent transformers (OfflineModeIsEnabled
# at Qwen3VLProcessor). The WEIGHTS still come from the HF cache — this is metadata
# only, and it's why §3's pre-cache is about weights, not network access at runtime.
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0

# Where the GR00T server venv lives (auto-wires server_python).
export ISAAC_GR00T_DIR=~/Isaac-GR00T
```

> **Why `osmesa`, not `egl`?** On a compute-only NVIDIA driver (no
> `libEGL_nvidia`) the LIBERO/MuJoCo render can't make an EGL GL context and dies
> with `Cannot initialize a EGL device display`. OSMesa software rendering always
> works. Switch to `egl` only if you've confirmed NVIDIA EGL is present (it's
> faster).

> 💡 **Put these in a sourceable `env.sh`** (run with `source env.sh`, not `./`) so
> you don't forget them after a reconnect — same pattern as the
> [OpenVLA tutorial's tip](./gcp-training-tutorial.md#tip-put-them-in-an-envsh).

---

## 6. Run the mission

```bash
# Sanity-check the spec (instant, no GPU)
odyssey validate examples/franka-libero/mission-gr00t.yaml

# Score the GR00T-N1.7 checkpoint on LIBERO object (10 episodes)
odyssey run examples/franka-libero/mission-gr00t.yaml
```

### What the mission does

[`examples/franka-libero/mission-gr00t.yaml`](../examples/franka-libero/mission-gr00t.yaml)
is **eval-only** (no training task). At run time Odyssey:

1. launches the GR00T **policy server** under `server_python` and waits for it to
   come up (it loads `nvidia/GR00T-N1.7-LIBERO` → the `libero_object/` subdir + the
   Cosmos backbone);
2. runs the LIBERO `libero_object` rollout in MuJoCo, querying the server for an
   **action chunk** (`n_action_steps: 16`) and executing it open-loop — so the pilot
   is queried ~an order of magnitude fewer times per episode than a single-step
   pilot;
3. reports `success_rate` across the episodes.

Key config knobs (already set in the mission):

| Key | Value | Meaning |
|---|---|---|
| `pilot` | `gr00t` | routes to `gr00t_libero_eval.py` + the policy server |
| `checkpoint` | `nvidia/GR00T-N1.7-LIBERO` | server loads the `<suite>/` subdir |
| `serve_checkpoint` | `true` | Odyssey auto-serves the policy server |
| `embodiment_tag` | `LIBERO_PANDA` | **must match the checkpoint** — the server can't infer it |
| `sim_policy_wrapper` | `true` | serve through the sim wrapper (the flat `video.*`/`state.*` obs are built for it) |
| `n_action_steps` | `16` | action-chunk length executed open-loop |
| `capture_video` | `true` | one MP4 per episode |

### What success looks like

The run ends with mission status **COMPLETED** and a non-trivial `success_rate`.
On the validated run this was **1.000 (10/10)** on `libero_object` task 0 — GR00T
checkpoints are *trained on* LIBERO, so unlike the OpenVLA→Robosuite domain gap you
should expect **real, high** scores here.

Per-episode videos land under the run's output dir:

```bash
find ~/.odyssey/runs -path "*/videos/*.mp4" -exec ls -lh {} \;
```

### Optional: the multi-agent arms

The GR00T pilot is the one that supports **multi-agent** (OpenVLA's per-step latency
is unviable for it). Three arms live alongside the single-agent mission — each adds
a Gemma **specialist** and needs its venv loaded:

```bash
source examples/multiagent-openvla-gemma/.env   # sets ODYSSEY_SPECIALIST_PYTHON
odyssey run examples/franka-libero/mission-gr00t-multiagent-planning.yaml       # planner authors the plan up front
# odyssey run examples/franka-libero/mission-gr00t-multiagent-delegation.yaml   # fixed pick→place, specialist grounds each phase
# odyssey run examples/franka-libero/mission-gr00t-multiagent-orchestration.yaml # LLM routes the next sub-instruction
```

See [`examples/franka-libero/README.md`](../examples/franka-libero/README.md) and
[`docs/multiagent-execution-flow.md`](./multiagent-execution-flow.md) for the
per-arm flow. Set `config.trace: true` to log who acts when.

---

## 7. Troubleshooting / debugging playbook

The GR00T server runs **out of process**, so when a run stalls or dies the first
place to look is its log — **not** the `odyssey run` output.

### The server log is the source of truth

```bash
tail -n 60 /tmp/gr00t_server_5555.log     # 5555 = the mission's `port`
```

### Server won't start — the gated backbone isn't cached

```
OSError: nvidia/Cosmos-Reason2-2B is not a local folder ...   (or a 401/403)
```

The most common failure. The gated backbone isn't in your HF cache. Fix
([§3](#3-huggingface-access-the-gated-backbone)):

```bash
huggingface-cli whoami                          # confirm you're logged in
huggingface-cli download nvidia/Cosmos-Reason2-2B   # after access is granted
```

### Server startup dies with `OfflineModeIsEnabled`

```
OfflineModeIsEnabled ... at Qwen3VLProcessor / is_base_mistral
```

A forced-offline env breaks the server's online metadata check on recent
`transformers`. **Run online** — the weights still come from cache:

```bash
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
```

### `embodiment_tag` rejected by the server's `EmbodimentTag` enum

The mission serves with `LIBERO_PANDA` (what NVIDIA's own LIBERO eval uses). If the
server rejects it, the checkpoint's `experiment_cfg/conf.yaml` names its dataset
embodiment `libero_sim` — try that instead in the mission config.

### Eval can't render headless — `Cannot initialize a EGL device display`

Compute-only NVIDIA driver (no `libEGL_nvidia`). Use OSMesa software rendering:

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```

(`setup.sh` installs `libosmesa6-dev` for exactly this.)

### `ModuleNotFoundError: No module named 'libero'` (or it prompts on stdin)

Both handled by `setup.sh`, but if you built the venv by hand: LIBERO is a PEP 420
namespace package — it must be registered via a `.pth` on the venv path (not
`pip install -e .`), and `import libero.libero` prompts on first run unless
`~/.libero/config.yaml` is pre-initialized. `libero.__file__` being `None` is
**normal** for a namespace package — check `libero.__path__` instead.

### Disk fills up across runs

Each run stages models and saves videos under `~/.odyssey/runs/`. Clean between
attempts (check first with `du -sh ~/.odyssey/runs/* | sort -rh`):

```bash
rm -rf ~/.odyssey/runs/*
```

### Per-session checklist

Before each run after an SSH reconnect:

- [ ] `source env_pilot_libero/bin/activate`
- [ ] `export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`
- [ ] `export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0`
- [ ] `huggingface-cli whoami` shows you're logged in (backbone access)
- [ ] `nvidia-smi` shows the GPU free (no zombie server from a previous run — `pkill -f gr00t` if needed)

---

## 8. Wrap up: get your results and stop the VM

Runs land under `~/.odyssey/runs/<mission-id>/<task-id>/` on the VM (the
`success_rate` summary and the per-episode MP4s). Pull what you want to keep back to
your machine with `scp`:

```bash
# from your laptop — copy a task's output dir (summary + videos) locally
gcloud compute scp --recurse \
  <VM_NAME>:~/.odyssey/runs/<mission-id> ./gr00t-libero-results --zone=<ZONE>
```

Then **stop the VM** so it stops billing the GPU/compute:

```bash
gcloud compute instances stop <VM_NAME> --zone=<ZONE>
```

> 💸 A **stopped** VM no longer bills for compute, but **still bills for its disk**.
> If you're done for good, delete the instance *and* its disk, or snapshot the disk
> first (far cheaper to park than a live disk). Don't just close the SSH session —
> that leaves the VM running.

---

## Appendix: environment variable reference

| Variable | Purpose |
|---|---|
| `MUJOCO_GL=osmesa` | Headless GL backend for MuJoCo (software; use `egl` only with NVIDIA EGL) |
| `PYOPENGL_PLATFORM=osmesa` | Matching headless PyOpenGL platform |
| `HF_HUB_OFFLINE=0` / `TRANSFORMERS_OFFLINE=0` | **Run online** — server's backbone metadata check breaks under forced-offline |
| `ISAAC_GR00T_DIR` | Location of the Isaac-GR00T checkout (auto-wires `server_python`) |
| `GR00T_VENV_PYTHON` | Explicit path to the GR00T server venv's python (overrides the auto-wire) |
| `ODYSSEY_SPECIALIST_PYTHON` | (multi-agent only) path to the specialist venv's python (Gemma) |
| `LIBERO_DIR` | Override the LIBERO clone location (default `~/LIBERO`) |
| `HF_TOKEN` | Auth for the gated backbone (or use `huggingface-cli login`) |

## Appendix: the two environments

| Env | Interpreter | Python | Key pins |
|---|---|---|---|
| `env_pilot_libero` | `<repo>/env_pilot_libero` | 3.10 | robosuite 1.4.0, LIBERO deps (minus transformers/numpy), ZMQ client |
| GR00T server | `~/Isaac-GR00T/.venv` | 3.12 | torch 2.9.0+cu128, transformers 4.57.3, flash-attn (cu128 wheel) — see `constraints/gr00t-server-known-good.txt` |
</content>
</invoke>
