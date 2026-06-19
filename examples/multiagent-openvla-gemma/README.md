# Multi-agent evaluation — OpenVLA PILOT + Gemma 4 SPECIALIST

This example runs a **multi-agent mission**: an OpenVLA **PILOT** fine-tuned on
Bridge V2, guided at evaluation time by a vision-language **SPECIALIST** task
planner (multimodal Gemma 4 E2B-it) that runs **out of process**. Evaluated on
the Robosuite **Lift** benchmark.

```bash
odyssey run examples/multiagent-openvla-gemma/mission.yaml
```

A mission with a SPECIALIST agent (a task planner) in addition to the PILOT runs
a **plan-then-execute** loop during eval: the SPECIALIST decomposes the
instruction into sub-steps once per episode, and the PILOT executes each. Only
the PILOT produces actions and only the PILOT is trained — the SPECIALIST is
**inference-only** (it runs its base checkpoint to plan and has no training task).

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

## Prerequisites

> All commands are run from the **repository root**.

### 1. Main venv (PILOT)

```bash
pip install -e ".[huggingface,openvla,robosuite]"
git clone https://github.com/openvla/openvla.git /srv/openvla
export OPENVLA_REPO_PATH=/srv/openvla
pip install -e "$OPENVLA_REPO_PATH"   # pulls OpenVLA's pinned deps
```

### 2. Out-of-process SPECIALIST venv

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

### 3. HuggingFace login (gated models)

**Both** models are **gated** — you must accept each one's license on its
HuggingFace page, then authenticate on the machine before the first run, or the
download fails with `401/403`:

- [`openvla/openvla-7b`](https://huggingface.co/openvla/openvla-7b) — the PILOT
- [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) — the
  SPECIALIST (Apache-2.0 license, but gated like every Gemma release — accept
  Google's terms on the model page first)

```bash
huggingface-cli login          # paste a token from https://huggingface.co/settings/tokens
# or, non-interactive (CI / headless VM):
export HF_TOKEN=hf_xxx          # a read token on an account that accepted the licenses
```

### 4. Hardware

24 GB GPU (RTX 4090 / L4 class). The SPECIALIST **shares the GPU** with the
~14 GB bf16 PILOT, so it must fit in the remaining headroom.

## Notes / gotchas

> **Why Gemma 4, not Gemma 3, for multimodal.** Gemma 3 4B emits **NaN logits
> under int4 bitsandbytes** on this stack (verified across eager/sdpa attention,
> text-only and with-image), so it can't run quantized here. Gemma 4 (Apache-2.0,
> gated like every Gemma release) loads cleanly in int4 and grounds plans in the scene image.

> **VRAM note.** Both models share the GPU — the venv split solves the
> *dependency* conflict, not VRAM. The SPECIALIST is pinned to **GPU 0**
> (`device_map={"": 0}`) so bitsandbytes never silently offloads layers to CPU.
> E4B-it int4 (~9.3 GB) alongside bf16 OpenVLA (~14 GB) is too tight on a 24 GB
> card; this mission uses the smaller **E2B-it** (~5 GB int4) → ~19 GB peak,
> comfortable. (E4B works standalone, e.g. the smoke test, just not alongside
> the pilot.)

> **Two known-good stacks.** The main venv pins OpenVLA's stack
> (`constraints/openvla-known-good.txt`: torch 2.2.0, transformers 4.40.1); the
> specialist venv pins a modern one with torchvision
> (`constraints/specialist-known-good.txt`). They no longer need to be mutually
> compatible.
