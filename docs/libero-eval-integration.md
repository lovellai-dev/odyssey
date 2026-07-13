# LIBERO evaluation integration — status & handoff

**Status:** integration code complete and green (ruff + mypy + tests). The
spec-policy decision below is **resolved — Option A**: the Mission spec now allows
**eval-only** missions (zero training tasks), so the LIBERO example missions
`validate` and run. This doc captures the work and the rationale.

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

## ✅ Resolved: the spec now allows eval-only missions (Option A)

`Mission` (`src/odyssey/spec/mission.py`) previously enforced `tasks` length ≥ 2
**and** ≥ 1 training task, so a single eval task with no training did not validate.

**We relaxed it (Option A):**

- `tasks` `min_length 2 → 1`.
- `_task_cardinality` no longer requires a training task — **zero or more** training
  tasks are allowed. **Exactly one evaluation task** (and eval-is-last) is unchanged.
- Docstring invariant updated; `test_zero_training_tasks_rejected` replaced by
  `test_eval_only_mission_accepted` + `test_empty_tasks_rejected`.

**Why:** an eval-only mission is the flexibility we want — score an
already-fine-tuned checkpoint (via `config.checkpoint`) with no training, and hold
the pilot fixed while **swapping the SPECIALIST to re-run the same eval** (e.g.
comparing planners once the pilot is trained). A single eval with `config.checkpoint`
pointing at a published HF repo is the fastest route to a Franka that actually
succeeds in sim.

**Contract note (to verify downstream):** the spec docstring noted these invariants
"match the CC missions-table NOT NULL columns" — a possible Command Center /
lai-trainer contract. Eval-only missions must be accepted there too; confirm CC
tolerates a training-less mission before relying on this end-to-end.

**Option B (rejected)** — keep the spec and make LIBERO a train→eval mission (short
fine-tune on `modified_libero_rlds`, 10.2 GB). No spec change, but re-trains when a
working checkpoint already exists; slower and less flexible than A.

## How to test on the VM (once unblocked)

```bash
# 1. install LIBERO + encoder (DEPENDENCY SPIKE — see risks). Isolate into a
#    dedicated venv so LIBERO's robosuite/robomimic pins can't disturb the
#    validated env_pilot (OpenVLA + the robosuite eval). Build it once, then
#    install LIBERO into it:
examples/multiagent-openvla-gemma/setup.sh --pilot-venv "$PWD/env_pilot_libero"
source env_pilot_libero/bin/activate
examples/franka-libero/setup.sh                 # installs into the active venv
# (or plain `examples/franka-libero/setup.sh` to use the default env_pilot)

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
   confirm it co-installs with the OpenVLA stack. Note OpenVLA and LIBERO run in the
   **same process** during eval (policy + sim), so they must co-install in one venv —
   a separate venv isolates but does not resolve an inherent conflict. We install into
   a dedicated **`env_pilot_libero`** (via `franka-libero/setup.sh --venv`) to keep the
   validated `env_pilot` untouched. If pip still conflicts, pin
   (a `constraints/libero-known-good.txt`).
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
