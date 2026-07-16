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
| `mission.yaml` | 1 — OpenVLA **pilot** | Baseline single-agent eval on the LIBERO `object` suite. |
| `mission-multiagent.yaml` | 2 — pilot + Gemma **specialist** | **Planner-driven.** An out-of-process multimodal Gemma 4 planner decomposes the instruction into phases the pilot executes (`PlannedEvalRuntime`, `coordination: planning`). |
| `mission-multiagent-delegation.yaml` | 2 — pilot + Gemma **specialist** | **Delegation-driven.** A deterministic orchestrator delegates per-phase target *grounding* to the specialist and hands control back on completion (`DelegatedEvalRuntime`, `coordination: delegation`). |

> All three are eval-only — the multi-agent ones are **not** training missions,
> they just add a specialist. See the top of each YAML for the config knobs, and
> [Planner vs delegation](#planner-vs-delegation-two-multi-agent-arms) below for
> what actually differs between the two multi-agent arms.

> **No `task_instruction` — that's intentional.** LIBERO is a *language-conditioned*
> benchmark: every task in a suite ships its own natural-language instruction (keyed
> by `task_id`, e.g. *"pick up the alphabet soup and place it in the basket"*), which
> the runner reads automatically from the task. So the missions set **no
> `task_instruction`** — it would only act as a fallback if the task had no language.
> (Contrast with the robosuite [`franka-pickplace`](../franka-pickplace/) example,
> where the sim has no language annotation and you must supply the instruction.)

---

## Prerequisites

- **GPU:** 24 GB (L4 / RTX 4090 class). Multi-agent peak ≈ 19 GB (int4 planner + bf16 pilot).
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

# single-agent — checkpoint auto-downloads from HF on first run
odyssey validate examples/franka-libero/mission.yaml
odyssey run      examples/franka-libero/mission.yaml

# multi-agent — load the specialist venv first (sets ODYSSEY_SPECIALIST_PYTHON)
source examples/multiagent-openvla-gemma/.env

# planner-driven arm: specialist authors the plan, pilot executes it
odyssey run examples/franka-libero/mission-multiagent.yaml

# delegation-driven arm: orchestrator delegates grounding to the specialist
odyssey run examples/franka-libero/mission-multiagent-delegation.yaml

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

## Planner vs delegation (two multi-agent arms)

Both multi-agent missions pair the same OpenVLA **pilot** with the same Gemma 4
**specialist** and share the same completion check — the *only* difference is the
specialist's **role**, selected by the `coordination` key in the task `config:`.
This is the "planner vs delegation" comparison: run both on identical suites and
compare `success_rate`, wasted steps, and time-to-completion.

| | Planner-driven (`coordination: planning`) | Delegation-driven (`coordination: delegation`) |
|---|---|---|
| Who owns the sequence | The **specialist** — it authors the full plan once per episode | The **orchestrator** — a generic `pick → place` skill template |
| Specialist's job | `plan()`: decompose the task into ordered sub-instructions | `ground()`: locate the current phase's target in the scene, on demand |
| Phase transition | Pilot marches the fixed plan; advance on completion check / step cap | Orchestrator hands control back when the specialist confirms the sub-task is done |
| Runtime | `PlannedEvalRuntime` | `DelegatedEvalRuntime` |
| Mission | `mission-multiagent.yaml` | `mission-multiagent-delegation.yaml` |

**How delegation works per phase:** the orchestrator asks the specialist *"where
is the object to pick up in this scene?"* (`ground`), splices the returned phrase
into the pilot's instruction (e.g. `pick up the red mug on the left`), lets the
pilot act, and every `phase_check_every` steps asks *"is this sub-task done?"*
(`check_done`) — advancing to `place` only on a confirmed yes, with
`phase_max_steps` as a safety cap. The specialist **authors no plan**; it is an
on-demand perception tool.

**Zero extra VRAM either way:** the one loaded Gemma answers `plan`, `check_done`
*and* `ground` from the same out-of-process server. Grounding/check calls add
~1–2 s each, so keep `phase_check_every >= 10`.

**Reading the telemetry:** both arms emit `phase_advance` progress events. In the
delegation arm each event carries a `capability` (`grounding` when a new target
is grounded, `handback` when a phase completes) and a `reason` (`grounding` /
`completion` / `cap`). A healthy delegation run shows `completion` hand-backs, not
just `cap` ones — `cap`-only means the completion check never fired (check the
specialist server logs).

```bash
# same suite, both arms — compare the reported success_rate
odyssey run examples/franka-libero/mission-multiagent.yaml             # planning
odyssey run examples/franka-libero/mission-multiagent-delegation.yaml  # delegation
```

> Scope: the delegation orchestrator's routing is a fixed `pick → place` template
> and completion-gated hand-back — a deliberate skeleton. Capability advertising /
> an LLM router that *chooses* which agent to invoke (full delegation-driven
> regime) is not yet wired.

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
