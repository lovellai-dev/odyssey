# LIBERO evaluation integration — status & handoff

**Status:** integration code complete and green (ruff + mypy + existing tests), but
**blocked on one spec-policy decision** before the eval-only missions can `validate`.
This doc captures the work, the blocker, and how to resume.

## Why LIBERO

To get a VLA pilot that **actually succeeds in simulation** (not just exercises the
pipeline), the fastest path is **LIBERO** — a robosuite-based sim benchmark with a
**Franka Panda** and language-conditioned "put the X …" tasks, which open VLAs are
already fine-tuned for:

- **Dataset already in RLDS** (no conversion): [`openvla/modified_libero_rlds`](https://huggingface.co/datasets/openvla/modified_libero_rlds) — **10.2 GB** total (only needed for *fine-tuning*, not for eval).
- **Checkpoints that already work** (~70–90%): [`openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}`](https://huggingface.co/openvla/openvla-7b-finetuned-libero-10).

So the plan is **eval-only first**: score the published checkpoint to see the Franka
work, then optionally fine-tune later. (π0/π0.5 via `openpi` also ship LIBERO
checkpoints — a future "support another VLA" runner, bigger lift.)

## What's implemented (this branch)

| Piece | File | Notes |
|---|---|---|
| `EvaluationType.LIBERO` | `src/odyssey/spec/tasks.py` | new enum value `"libero"` |
| Eval-only checkpoint wiring | `src/odyssey/runners/evals/_common.py` | `resolve_eval_checkpoint` now honors `config.checkpoint` (local path **or** HF repo id) before falling back to a training checkpoint |
| `LiberoRunner` | `src/odyssey/runners/evals/libero.py` | mirrors `RobosuiteRunner`; LIBERO env + obs/action handling; single-agent (OpenVLA policy) **and** multi-agent (`PlannedEvalRuntime` + Gemma) |
| Registration | `src/odyssey/cli/commands/run.py` | `registry.register(LiberoRunner())` |
| mypy ignore | `pyproject.toml` | `libero.*` added to missing-imports list |
| Example missions | `examples/franka-libero/mission.yaml`, `mission-multiagent.yaml` | **eval-only** (no training task) |
| Setup script | `examples/franka-libero/setup.sh` | installs `libero` + `imageio[ffmpeg]`, sets render env, prints next steps |

Verified: `ruff check src/` clean · `mypy` clean (69 files) · `test_robosuite_runner` +
`test_spec` pass (35). `LiberoRunner` imports with `libero`/`openvla` absent (lazy imports).

## ⛔ Blocker: the spec forbids eval-only missions (by design)

`Mission` (`src/odyssey/spec/mission.py`) enforces a documented cardinality invariant:

- `tasks` length **≥ 2**,
- **≥ 1 training task**,
- **exactly 1 evaluation task** (and it must be last).

So a **single eval task with no training** does **not** validate — which is exactly
what an eval-only LIBERO mission is. The spec docstring notes these invariants
"match the CC missions-table NOT NULL columns", i.e. they may encode a **contract
with Command Center / lai-trainer**, not just a local rule. **Not relaxed unilaterally.**

## Decision needed (then resume)

**Option A — relax the spec to allow eval-only missions.**
- Change: `tasks` `min_length 2 → 1`; allow `training == 0` (keep "exactly 1 eval");
  update the docstring invariant and `tests/unit/test_spec.py::test_zero_training_tasks_rejected`.
- Pro: scores the **already-working** published checkpoint immediately, no training.
- Con/risk: changes what a "Mission" means; **confirm CC/lai-trainer tolerate a
  training-less mission** (the NOT NULL contract) before doing this.

**Option B — keep the spec: a train→eval LIBERO mission.**
- "training" = a short fine-tune on `modified_libero_rlds` (10.2 GB) → eval.
- Pro: no spec change; fits the current model.
- Con: needs the dataset + training time, and re-trains when a working checkpoint
  already exists.

**Recommendation:** Option A if CC tolerates training-less missions in v0.1.0-alpha;
otherwise B. The example missions in this branch are written for **A (eval-only)**.

## How to test on the VM (once unblocked)

```bash
# 1. install LIBERO + encoder into the OpenVLA env (DEPENDENCY SPIKE — see risks)
examples/franka-libero/setup.sh

# 2. headless render (compute-only driver → OSMesa)
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

# 3. single-agent (checkpoint auto-downloads from HF; no dataset needed)
odyssey validate examples/franka-libero/mission.yaml
odyssey run      examples/franka-libero/mission.yaml

# 4. multi-agent (Gemma planner) — load the specialist venv first
source examples/multiagent-openvla-gemma/.env
odyssey run examples/franka-libero/mission-multiagent.yaml

# 5. videos
find ~/.odyssey/runs -path "*/videos/*.mp4" -exec ls -lh {} \;
```

## Risks / to verify on first run

1. **Dependency spike (biggest):** LIBERO pins its own robosuite/robomimic versions —
   confirm it co-installs with the OpenVLA stack in `env_pilot`. If pip conflicts,
   pin (a `constraints/libero-known-good.txt`) or use a dedicated venv.
2. **obs orientation** (`_libero_image`, 180° flip) and **gripper action**
   (`_libero_action`, binarize + invert) mirror OpenVLA's `run_libero_eval.py`. If
   the arm behaves inverted, check these two against the reference script.
3. **`unnorm_key` must match the suite/checkpoint** (e.g. `libero_object`).
4. Headless EGL/OSMesa (same as the rollout-video work).

## Follow-ups (tracked separately)

- Consolidate the duplicated multi-agent helpers (`_has_specialist`,
  `_find_specialist_model`, `_build_planned_runtime`) from `robosuite.py` + `libero.py`
  into `_common.py` (avoid the cross-runner private-import coupling PR #41 removed).
- Multi-task sweep within a suite (today: one `task_id`).
- Optional: fine-tune on `modified_libero_rlds`; a π0/π0.5 (`openpi`) runner.
