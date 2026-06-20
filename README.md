<h1 align="center">Odyssey</h1>

<p align="center">
  <a href="https://github.com/lovellai-dev/odyssey/actions/workflows/ci.yml"><img src="https://github.com/lovellai-dev/odyssey/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/status-pre--alpha-orange.svg" alt="Status: Pre-Alpha">
</p>

> **Status: pre-alpha (v0.0.x).** Not yet on PyPI. The first public alpha is
> targeted at `v0.1.0-alpha`. The API, CLI, schemas, and wire protocols are
> still subject to change without notice. See `docs/` for the design refs.

## Install

> [!IMPORTANT]
> **Linux only** — install build dependencies before proceeding (needed by `.[all]`):
> ```bash
> sudo apt update && sudo apt install build-essential python3-dev -y
> ```

```bash
git clone https://github.com/lovellai-dev/odyssey.git
cd odyssey
python3 -m venv .venv
source .venv/bin/activate
pip install -e .              # CLI, validate, mock runs (lightweight)
pip install -e ".[all]"       # real training + evaluation (torch, robosuite…)
pip install -e ".[all,dev]"   # + pytest, ruff, mypy
```

The base install pulls in pydantic, click, pyyaml, and aiosqlite — enough to
run `validate`, `list`, `status`, and `run --use-mock-runner` against any
mission spec without a GPU. `.[all]` adds everything needed for real training
and evaluation runs.

## Quick start (no GPU, no network)

```bash
# Validate the mission spec
$ odyssey validate examples/quickstart-openvla/mission.yaml
OK  examples/quickstart-openvla/mission.yaml
  spec version : 0.1
  mission name : openvla-bridge-lift
  robot        : embodiment=franka_panda
  tasks        : 1 training, 1 evaluation

# Run the full mission with a CPU mock (no GPU needed)
$ odyssey run examples/quickstart-openvla/mission.yaml --use-mock-runner
{"ts": "...", "event": "mission.created", ...}
{"ts": "...", "event": "mission.queued", ...}
{"ts": "...", "event": "mission.started", ...}
...
{"ts": "...", "event": "mission.completed", "overall_grade": 1.0}

COMPLETED  c1756bad855e45cc9a95b5b0566c948b
  overall_grade : 1.000

# List all missions from the local DB
$ odyssey list
c1756bad855e  COMPLETED   openvla-bridge-lift   2026-05-17T23:18:34+00:00  grade=1.000

# Show details for a specific mission (prefix match)
$ odyssey status c1756bad
COMPLETED  c1756bad855e45cc9a95b5b0566c948b
  name         : openvla-bridge-lift
  ...
  tasks:
    COMPLETED    training    finetune-openvla
    COMPLETED    evaluation  eval-on-robosuite-lift
```

`--use-mock-runner` swaps in the CPU mock for every task, so this works on a
laptop without a GPU. State is persisted to `~/.odyssey/missions.db`;
artifacts under `~/.odyssey/runs/<mission-id>/<task-id>/`.

## What it is

You train an agent by describing a mission in YAML — a robot, a model, a dataset to train on, an
evaluation benchmark to score against — and `odyssey run` walks it through the
full lifecycle: load → validate → execute training tasks → execute the
evaluation task → persist results. Local-mode by default; the hosted Lovell
services (leaderboard, learning graph, hosted runners) are optional layers
that land in later releases.

## Launching a training mission

Two training paths ship today: **GR00T** (NVIDIA Isaac GR00T, the newer path)
and **OpenVLA** (the original). Both run through `odyssey run <mission.yaml>` —
pick the quickstart that matches your model.

<details open>
<summary><b>GR00T</b> (Isaac-GR00T + Isaac Lab) — the newer path</summary>

Fine-tunes `nvidia/GR00T-N1.7-3B` on the LeRobot-format demo set that ships
inside the Isaac-GR00T repo (no separate download), evaluated in the Isaac Lab
cube-lift environment.

**Prerequisites:**

1. Install the upstream Isaac-GR00T package — it carries the training entry
   point (`gr00t.experiment.launch_finetune`) and the demo dataset. Accept
   NVIDIA's weight license:
   ```bash
   git clone https://github.com/NVIDIA/Isaac-GR00T.git /srv/Isaac-GR00T
   pip install -e /srv/Isaac-GR00T
   export ISAAC_GR00T_REPO_PATH=/srv/Isaac-GR00T   # resolves the demo dataset
   ```
2. For the Isaac Lab evaluation, install Isaac Lab and point Odyssey at its
   launcher:
   ```bash
   export ISAACLAB_PATH=/srv/IsaacLab              # provides isaaclab.sh
   ```

**Run:**

```bash
odyssey run examples/quickstart-gr00t/mission.yaml
```

The mission routes its training task to the GR00T runner with
`config: { runner: gr00t }` — OpenVLA and GR00T both serve wildcard training
tasks, so the family is selected explicitly.

</details>

<details>
<summary><b>OpenVLA</b> (Bridge V2 + Robosuite)</summary>

**Prerequisites:**

1. Install the training extras:
   ```bash
   pip install -e ".[huggingface,openvla,robosuite]"
   ```
2. Clone the upstream OpenVLA repo and install its dependencies (needed for
   `draccus` and the fine-tuning script):
   ```bash
   git clone https://github.com/openvla/openvla.git /srv/openvla
   pip install -e /srv/openvla
   export OPENVLA_REPO_PATH=/srv/openvla
   ```
3. Download the Bridge V2 dataset in RLDS format (~124 GB):
   ```bash
   wget -r -nH --cut-dirs=4 --reject="index.html*" \
     https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/
   mv bridge_dataset bridge_orig
   ```
   Set `--data_root_dir` to the parent directory containing `bridge_orig/`.

**Run:**

```bash
odyssey run examples/quickstart-openvla/mission.yaml
```

Hardware: 24 GB GPU (RTX 4090-class or better) for the OpenVLA LoRA fine-tune.

> [!NOTE]
> **GCP users:** single-GPU VMs require `export NCCL_NET=Socket` before
> running, to bypass Google's NCCL plugin. See [issue #5](https://github.com/lovellai-dev/odyssey/issues/5) for details.

> [!NOTE]
> **Evaluation:** the Robosuite runner auto-wires an OpenVLA→robosuite-action
> adapter (`make_openvla_policy` in `runners/models/openvla.py`) when no custom
> `policy_factory` is injected — it loads either a LoRA adapter or a full merged
> checkpoint, so eval works without extra glue. Full episode-completion
> validation on a real GPU is still in progress.

#### Known-good OpenVLA stack

The fine-tune runs through the **cloned OpenVLA repo**, which carries its own
dependency set — most onboarding friction comes from there, not from Odyssey.
Mixing versions surfaces as protobuf / TensorFlow / `tensorflow-metadata`
conflicts or `draccus` import errors. Known-good versions (from OpenVLA's own
requirements — treat its repo as the source of truth):

```text
Python        3.10
torch         2.2.0
torchvision   0.17.0
transformers  4.40.1
tokenizers    0.19.1
timm          0.9.10
flash-attn    2.5.5
```

To avoid re-downloading the 7B base model each run, point its path env var at a
local copy (HF id upper-cased, `/` and `-` → `_`, suffixed `_PATH`):

```bash
export OPENVLA_OPENVLA_7B_PATH=/path/to/openvla-7b   # for base: openvla/openvla-7b
```

#### Dataset: how `source: oxe` / `ref: bridge_orig` resolves

**Odyssey does not download the dataset** — `oxe` is a *pass-through*. The runner
forwards two values to OpenVLA's `finetune.py`, which loads via TFDS/RLDS:

| mission.yaml | becomes the flag | meaning |
|---|---|---|
| `dataset.ref: bridge_orig` | `--dataset_name bridge_orig` | the OXE **registry key** OpenVLA looks up |
| `config.data_root_dir: <path>` | `--data_root_dir <path>` | the **parent dir** containing the RLDS dataset folder |

⚠️ **Naming gotcha:** the registry key and the on-disk folder name can differ.
In validation, `ref: bridge_orig` resolved to data under `~/bridge_dataset/1.0.0/`,
so `data_root_dir` had to point at the **parent** of that folder — not the key
name. Check where your download actually landed and set `data_root_dir` to its
parent.

#### Weights & Biases (W&B)

OpenVLA's `finetune.py` calls `wandb.init()` unconditionally, so a run stalls or
fails if W&B isn't reachable. Control it yourself:

```bash
# Disable for local / smoke runs:
export WANDB_MODE=disabled
# Or log to your account, then pass project/entity via mission config:
#   config: { wandb_project: my-project, wandb_entity: my-entity }
```

Any `config:` key Odyssey doesn't consume is forwarded verbatim as
`--<key> <value>` to `finetune.py`.

#### What to expect during a run

Timing varies widely with hardware, disk, and network — treat these as
orientation, not promises:

1. **Base model download** — `openvla-7b` (~14 GB) on first run, unless
   `OPENVLA_OPENVLA_7B_PATH` is set.
2. **Dataset load / indexing** — Bridge V2 (~124 GB); RLDS indexing on a cold
   cache takes a while.
3. **Training startup** — model load + LoRA wrap, then steps begin.
4. **Steady state** — throughput logs as `it/s` (~1.49 it/s on an NVIDIA L4 for
   the quickstart config).

If a stage seems stuck, it's almost always a download in progress or a
dataset-path / W&B issue rather than a training bug — check those first.

</details>

## Multi-agent evaluation (PILOT + SPECIALIST)

<details>
<summary><b>Show setup &amp; how it works</b></summary>

### HuggingFace login (gated models)

The models pulled from the Hub are **gated** — you must accept each model's
license on its HuggingFace page, then authenticate on the machine before the
first run, or the download fails with `401/403`:

- [`openvla/openvla-7b`](https://huggingface.co/openvla/openvla-7b) — the PILOT
- [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) — the
  SPECIALIST in the multi-agent example (Apache-2.0, no gating)

```bash
huggingface-cli login          # paste a token from https://huggingface.co/settings/tokens
# or, non-interactive (CI / headless VM):
export HF_TOKEN=hf_xxx          # a read token on an account that accepted the licenses
```

A mission with a **SPECIALIST** agent (a task planner) in addition to the
**PILOT** runs a plan-then-execute loop during eval: the SPECIALIST decomposes
the instruction into sub-steps once per episode, and the PILOT executes each.
Only the PILOT produces actions and only the PILOT is trained — the SPECIALIST
is **inference-only** (it runs its base checkpoint to plan and has no training
task).

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

The SPECIALIST is a **vision-grounded multimodal Gemma 4** planner: it sees the
first camera frame of each episode and grounds its plan in the scene. Gemma 4
needs a modern `transformers` + `torchvision`, which conflicts with OpenVLA's
pinned `transformers==4.40.1`, so the SPECIALIST **must run out of process** in a
separate venv. The PILOT stays in the main venv; the two talk over a JSON-lines
subprocess protocol (the planner runs once per episode, off the per-step hot loop).

### Setting up the out-of-process SPECIALIST

1. Create the specialist venv (modern transformers + torchvision + Gemma deps):

   ```bash
   python -m venv ~/specialist-venv
   ~/specialist-venv/bin/pip install -e ".[specialist]" \
     -c constraints/specialist-known-good.txt
   ```

2. Point Odyssey at that venv's python. It is read per-process from the
   environment, so export it in every shell that runs a mission — or add it to
   your shell profile / VM startup script so it persists:

   ```bash
   export ODYSSEY_SPECIALIST_PYTHON=~/specialist-venv/bin/python
   ```

> **`ODYSSEY_SPECIALIST_PYTHON` is required for any mission with a SPECIALIST.**
> The planner is launched in that venv (`RemotePlanner` →
> `python -m odyssey.runners.agents.planner_server`). If it is unset, multi-agent
> eval fails fast with a clear `RuntimeError`: the multimodal Gemma 4 planner
> cannot load in the main venv, which pins `transformers==4.40.1` for OpenVLA.

Quick check without a simulator (launches the planner in the specialist venv and
prints a decomposition — no OpenVLA or simulator needed):

```bash
python tests/manual/smoke_remote_planner.py
```

> **Why Gemma 4, not Gemma 3, for multimodal.** Gemma 3 4B emits **NaN logits
> under int4 bitsandbytes** on this stack (verified across eager/sdpa attention,
> text-only and with-image), so it can't run quantized here. Gemma 4 (Apache-2.0,
> ungated) loads cleanly in int4 and grounds plans in the scene image.

> **VRAM note.** Both models still share the GPU — the venv split solves the
> *dependency* conflict, not VRAM. The SPECIALIST is pinned to **GPU 0**
> (`device_map={"": 0}`) so bitsandbytes never silently offloads layers to CPU.
> Gemma 4 **E4B-it** int4 (~9.3 GB) alongside bf16 OpenVLA (~14 GB) peaks at
> ~23 GB — tight on a 24 GB L4; drop to **E2B-it** for headroom (this is what the
> multimodal example mission uses).

> **Two known-good stacks.** The main venv pins OpenVLA's stack
> (`constraints/openvla-known-good.txt`: torch 2.2.0, transformers 4.40.1); the
> specialist venv pins a modern one with torchvision
> (`constraints/specialist-known-good.txt`). They no longer need to be mutually
> compatible.

</details>

## CLI reference

| Command | What it does |
|---|---|
| `odyssey init [DIR]` | Scaffold a new mission directory. `--template openvla\|cpu_mock`. |
| `odyssey validate <mission.yaml>` | Parse + validate a spec. Exits 0 if clean. |
| `odyssey run <mission.yaml>` | Execute end-to-end. `--use-mock-runner` for no-GPU smoke. |
| `odyssey list` | Recent missions from the local SQLite DB. `--status` to filter. |
| `odyssey status <mission_id>` | One mission's detail. Accepts an id prefix. |

All commands respect `--db` and `--working-dir` to override the
`~/.odyssey/` defaults.

## Status snapshot (v0.0.x)

| Area | Done | Deferred |
|---|---|---|
| Spec + validate | ✓ | — |
| Engine + lifecycle | ✓ | watchdog timers, materialized profiles |
| In-memory + SQLite persistence | ✓ | — |
| Provider ABCs + Local + HF | ✓ | OXE, Lovell-mode |
| CPU mock runner | ✓ | — |
| OpenVLA training runner | ✓ (validated on L4) | — |
| GR00T training runner | ✓ skeleton + tests, task-level `runner: gr00t` routing | end-to-end smoke with real Isaac-GR00T |
| Robosuite eval runner | ✓ (auto-wired OpenVLA adapter) | full GPU end-to-end validation |
| Isaac Lab eval runner | ✓ skeleton + tests, subprocess launch + `ODYSSEY_*` stdout protocol | blessed eval script (GR00T/VLA recipe), real-Isaac smoke |
| Multi-agent eval (PILOT + SPECIALIST) | ✓ (out-of-process Gemma 4 planner) | full GPU end-to-end validation |
| `odyssey init / run / list / status / validate` | ✓ | `logs`, `publish` |
| Leaderboard publish, Learning Graph, Anonymizer, Auth | — | post-v0.1.0-alpha |

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). DCO sign-off required on every
commit. Open an issue before non-trivial PRs — the API surface is moving
weekly until v0.1.0-alpha freezes.
