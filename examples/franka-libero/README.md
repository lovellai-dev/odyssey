# Franka + LIBERO eval (eval-only)

Score a **published** OpenVLA-7B checkpoint on the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
simulation benchmark — a Franka Panda doing language-conditioned pick-and-place
("put the X …"). LIBERO is the standard sim benchmark open VLAs are fine-tuned for,
so this is the fastest way to see a VLA pilot actually **succeed** in sim (the
published checkpoints score ~70–90%).

**Eval-only:** there is no training task. The checkpoint is pulled straight from
HuggingFace, so you do **not** need a dataset on disk (the 10.2 GB
`openvla/modified_libero_rlds` is only for fine-tuning).

| Mission | Agents | What it adds |
|---|---|---|
| `mission.yaml` | 1 — OpenVLA **pilot** | Baseline single-agent eval (OpenVLA) on the LIBERO `object` suite. |
| `mission-gr00t.yaml` | 1 — GR00T **pilot** | Single-agent eval with the chunk-emitting GR00T-N1.7 pilot (out-of-process policy server). |
| `mission-gr00t-multiagent-planning.yaml` | 2 — GR00T pilot + Gemma **specialist** | Multi-agent, **planning** arm: the SPECIALIST authors the plan up front + completion-gates phases. |
| `mission-gr00t-multiagent-delegation.yaml` | 2 — GR00T pilot + Gemma **specialist** | Multi-agent, **delegation** arm: a fixed `pick → place` template; the SPECIALIST grounds each phase's target. |
| `mission-gr00t-multiagent-orchestration.yaml` | 2 — GR00T pilot + Gemma **orchestrator** | Multi-agent, **orchestration** arm (regime D): an LLM ORCHESTRATOR routes the next sub-instruction dynamically. |

> All are eval-only (no training task). **Multi-agent runs on the GR00T pilot** —
> OpenVLA is single-agent only (its per-step latency is unviable for MA). See
> [`docs/multiagent-execution-flow.md`](../../docs/multiagent-execution-flow.md) for
> the per-arm flow diagrams, and set `config.trace: true` to log who acts when.

> **No `task_instruction` — that's intentional.** LIBERO is a *language-conditioned*
> benchmark: every task in a suite ships its own natural-language instruction (keyed
> by `task_id`, e.g. *"pick up the alphabet soup and place it in the basket"*), which
> the runner reads automatically from the task. So the missions set **no
> `task_instruction`** — it would only act as a fallback if the task had no language.

---

## Prerequisites

- **GPU:** 24 GB (L4 / RTX 4090 class). GR00T multi-agent peak ≈ 12–14 GB (GR00T-3B + int4 Gemma).
- **System build + render deps** (needs `sudo`) — LIBERO's `egl_probe` (via `robomimic`)
  compiles from C source, and MuJoCo needs a headless GL backend:
  ```bash
  sudo apt-get update && sudo apt-get install -y cmake build-essential \
    libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libosmesa6-dev
  ```

---

## Setup (validated on a GCP L4, 2026-07-13)

LIBERO pins an **older robosuite (1.4.0)** and would downgrade `transformers` if
installed naively — so we install it into a **dedicated venv `env_pilot_libero`**,
leaving the validated `env_pilot` (OpenVLA + robosuite 1.5.2 eval) untouched. See
[Why the extra care](#why-the-extra-care-dependency-notes) below.

```bash
# 1. build the dedicated venv with the OpenVLA stack (also builds env_specialist)
examples/multiagent-openvla-gemma/setup.sh --pilot-venv "$PWD/env_pilot_libero"

# 2. install LIBERO into it (clone + safe deps + .pth + config; see setup.sh header)
source env_pilot_libero/bin/activate
examples/franka-libero/setup.sh
```

`setup.sh` is idempotent and does the fiddly bits for you:
- clones LIBERO to `$HOME/LIBERO` (override with `LIBERO_DIR`),
- installs LIBERO's deps **minus `transformers` and `numpy`** (so the OpenVLA stack
  stays at transformers 4.40.x),
- registers the LIBERO repo root on the venv path via a `.pth` (LIBERO is a PEP 420
  namespace package — `pip install -e .` does **not** work),
- pre-initializes `~/.libero/config.yaml` non-interactively (the first import
  otherwise prompts on stdin and would block `odyssey run`),
- installs `imageio[ffmpeg]` for MP4 capture, and verifies the imports.

---

## Run

```bash
# headless render (OSMesa = CPU, always works on compute-only drivers; egl if NVIDIA EGL)
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

# single-agent OpenVLA — checkpoint auto-downloads from HF on first run
odyssey validate examples/franka-libero/mission.yaml
odyssey run      examples/franka-libero/mission.yaml

# GR00T missions need the policy-server venv:
#   bash examples/franka-libero/setup.sh --pilot gr00t   (see examples/quickstart-gr00t)
# ...and access to the GATED backbone nvidia/Cosmos-Reason2-2B (every GR00T checkpoint
# loads it): request it once at https://huggingface.co/nvidia/Cosmos-Reason2-2B, then
# `huggingface-cli login`.
# Run ONLINE — the server does an online metadata check for the backbone, so a forced
# offline mode breaks server startup with recent transformers (OfflineModeIsEnabled at
# Qwen3VLProcessor/is_base_mistral). The weights still come from the HF cache.
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
odyssey run examples/franka-libero/mission-gr00t.yaml

# multi-agent (GR00T pilot + Gemma) — also load the specialist venv. Pick an arm:
source examples/multiagent-openvla-gemma/.env       # sets ODYSSEY_SPECIALIST_PYTHON
odyssey run examples/franka-libero/mission-gr00t-multiagent-planning.yaml        # planner
# odyssey run examples/franka-libero/mission-gr00t-multiagent-delegation.yaml    # delegation
# odyssey run examples/franka-libero/mission-gr00t-multiagent-orchestration.yaml # orchestration

# per-episode videos
find ~/.odyssey/runs -path "*/videos/*.mp4" -exec ls -lh {} \;
```

### Other LIBERO suites

Change `benchmark_name`, `config.checkpoint`, and `config.unnorm_key` **together**:

| `benchmark_name` | checkpoint | `unnorm_key` |
|---|---|---|
| `libero_object`  | `openvla/openvla-7b-finetuned-libero-object`  | `libero_object`  |
| `libero_spatial` | `openvla/openvla-7b-finetuned-libero-spatial` | `libero_spatial` |
| `libero_goal`    | `openvla/openvla-7b-finetuned-libero-goal`    | `libero_goal`    |
| `libero_10`      | `openvla/openvla-7b-finetuned-libero-10`      | `libero_10`      |

---

## Why the extra care (dependency notes)

The OpenVLA pilot and the LIBERO/robosuite simulator run **in the same process** during
eval, so they must co-install in one venv. LIBERO's packaging fights the OpenVLA stack
in a few ways — all handled by `setup.sh`, but worth knowing:

| Gotcha | Symptom if unhandled | Fix in `setup.sh` |
|---|---|---|
| LIBERO has **no top-level `__init__.py`** (namespace package) | `ModuleNotFoundError: No module named 'libero'` even after `pip install` | clone + `.pth` on the venv path |
| `install_requires` empty; deps in `requirements.txt` pin **`transformers==4.21.1`** | OpenVLA breaks (needs 4.40.x); scipy also downgrades | install deps **excluding transformers/numpy** |
| `robosuite` downgraded **1.5.2 → 1.4.0** | breaks the RobosuiteRunner eval in a shared env | isolate in `env_pilot_libero` (`--venv`) |
| `egl_probe` (via `robomimic`) builds from C | `RuntimeError: CMake must be installed` | apt `cmake` + `build-essential` + GL/EGL headers |
| First `import libero` prompts on stdin | `odyssey run` hangs waiting for input | pre-init `~/.libero/config.yaml` with `echo n` |

> `libero.__file__` is `None` — that's normal for a namespace package. Check
> `libero.__path__` and `libero.libero.benchmark.__file__` instead.
