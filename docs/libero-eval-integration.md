# LIBERO evaluation integration — status & handoff

**Status:** integration code complete and green (ruff + mypy + tests), and the eval
pipeline is **validated end-to-end on a GCP L4** (2026-07-13): deps install, model
loads, the LIBERO env builds, episodes run and are scored, videos are captured. The
spec-policy decision below is **resolved — Option A**: the Mission spec now allows
**eval-only** missions (zero training tasks), so the LIBERO example missions
`validate` and run. One open item — the published checkpoint's `success_rate` came
out 0/2 on a first tiny run (arm approaches but misses the grasp); tracked in
**issue #61** (variance vs a center-crop preprocessing gap). This doc captures the
work and the rationale; `examples/franka-libero/README.md` is the user-facing guide.

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
| `LiberoRunner` | `src/odyssey/runners/evals/libero.py` | mirrors `RobosuiteRunner`; LIBERO env + obs/action handling; OpenVLA single-agent, plus the GR00T pilot (single- and multi-agent via the out-of-process recipe) |
| Registration | `src/odyssey/cli/commands/run.py` | `registry.register(LiberoRunner())` |
| mypy ignore | `pyproject.toml` | `libero.*` added to missing-imports list |
| Example missions | `examples/franka-libero/mission.yaml`, `mission-gr00t.yaml`, `mission-gr00t-multiagent-planning.yaml` | **eval-only** (no training task) |
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

## How to test on the VM (validated procedure)

Full user-facing steps + a dependency-gotcha table are in
`examples/franka-libero/README.md`; `examples/franka-libero/setup.sh` automates the
install. In short:

```bash
# 0. system build + render deps (needs sudo). LIBERO's egl_probe (via robomimic)
#    builds from C source (needs cmake + a compiler); MuJoCo needs a headless GL lib.
sudo apt-get update && sudo apt-get install -y cmake build-essential \
  libegl1-mesa-dev libgl1-mesa-dev libgles2-mesa-dev libosmesa6-dev

# 1. dedicated venv with the OpenVLA stack (isolates the robosuite 1.4.0 downgrade),
#    then install LIBERO into it (clone + safe deps + .pth; see setup.sh header):
examples/multiagent-openvla-gemma/setup.sh --pilot-venv "$PWD/env_pilot_libero"
source env_pilot_libero/bin/activate
examples/franka-libero/setup.sh                 # installs into the active venv

# 2. headless render (compute-only driver → OSMesa; EGL failed headless on the L4)
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

# 3. single-agent (checkpoint auto-downloads from HF; no dataset needed)
odyssey validate examples/franka-libero/mission.yaml
odyssey run      examples/franka-libero/mission.yaml

# 4. multi-agent (GR00T pilot + Gemma specialist) — load the specialist venv first
source examples/multiagent-openvla-gemma/.env
odyssey run examples/franka-libero/mission-gr00t-multiagent-planning.yaml

# 5. videos
find ~/.odyssey/runs -path "*/videos/*.mp4" -exec ls -lh {} \;
```

**Install gotchas found during validation (all handled by `setup.sh`):** LIBERO ships
no top-level `__init__.py` (PEP 420 namespace package → registered via a `.pth`, not
`pip install -e .`); its `requirements.txt` hard-pins `transformers==4.21.1` +
`numpy==1.22.4` which would break the OpenVLA stack, so we install its deps **minus
those two**; `robosuite` is downgraded 1.5.2 → 1.4.0 (fine in the dedicated venv);
and the first `import libero` prompts on stdin, so the config is pre-initialized.

## Risks / to verify on first run

1. **Dependency spike (biggest):** LIBERO pins its own robosuite/robomimic versions —
   confirm it co-installs with the OpenVLA stack. Note OpenVLA and LIBERO run in the
   **same process** during eval (policy + sim), so they must co-install in one venv —
   a separate venv isolates but does not resolve an inherent conflict. We install into
   a dedicated **`env_pilot_libero`** (via `franka-libero/setup.sh --venv`) to keep the
   validated `env_pilot` untouched. If pip still conflicts, pin
   (a `constraints/libero-known-good.txt`).
2. **obs orientation** (`_libero_image`, 180° flip) and **gripper action**
   (`_libero_action`, normalize + binarize + invert) — verified during validation to
   **match** OpenVLA's `run_libero_eval.py` (incl. camera 256², 10 settle no-ops, and
   raw `predict_action` gripper `[0,1]`). Not the cause of the observed miss.
3. **`unnorm_key` must match the suite/checkpoint** (e.g. `libero_object`). Verified.
4. **`success_rate` (open, issue #61):** the published `libero-object` checkpoint
   scored 0/2 on a first run — arm approaches but misses the grasp. Leading suspect if
   systematic (rather than small-sample variance): **center-crop** — the
   `finetuned-libero-*` checkpoints trained with `center_crop=True`; audit that
   `make_openvla_policy` replicates the crop. Re-run with `num_episodes: 10` first.
5. Headless EGL/OSMesa: EGL failed headless on the L4 → OSMesa (CPU render, slower).

## Follow-ups (tracked separately)

- **`success_rate` investigation — issue #61** (variance vs center-crop). Re-run
  `num_episodes: 10`; if ~0, audit `center_crop` in `make_openvla_policy`.
- Consolidate the duplicated multi-agent helpers (`_has_specialist`,
  `_find_specialist_model`, `_build_planned_runtime`) from `robosuite.py` + `libero.py`
  into `_common.py` (avoid the cross-runner private-import coupling PR #41 removed).
- Multi-task sweep within a suite (today: one `task_id`).
- Optional: fine-tune on `modified_libero_rlds`; a π0/π0.5 (`openpi`) runner.
