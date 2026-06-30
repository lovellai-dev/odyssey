# Franka pick-and-place (multi-agent) — put the can in the bin

A **multi-agent** OpenVLA mission: an OpenVLA 7B **pilot** (the only trained agent)
guided by an out-of-process multimodal **Gemma 4 task planner** (the specialist).
Fine-tune the pilot on **Bridge V2**, then evaluate on the Robosuite **`PickPlaceCan`**
benchmark, where a **Franka Panda** picks up a can and places it in the bin (a "put an
object in a box" task).

The planner sees the first frame of each episode and decomposes the goal into phases
(reach → grasp → transport → place) that it feeds to the pilot. PickPlaceCan's
long-horizon, multi-stage nature is exactly where that decomposition is meant to help.
It's the single-object variant of `PickPlace` (which juggles four objects/bins).

## ⚠️ Read first: the domain gap

| | Training | Evaluation |
|---|---|---|
| Source | Bridge V2 (`bridge_orig`) | Robosuite (simulation) |
| Robot | WidowX (real) | Franka Panda (sim) |

A freshly fine-tuned LoRA will rarely succeed zero-/few-shot — the pilot is tested on
a different robot and domain than it learned on, and **the planner sequences
sub-instructions but does not close that gap**. Treat this as a coherent, visually
attractive *pipeline* demo; real successes need serious training. Bridge does contain
many "put X in a container" demos, so this task is *more* aligned with the training
data than e.g. `Lift`.

## Prerequisites

This mission needs **two Python environments** (the pilot's OpenVLA stack and the
specialist's modern Gemma stack are mutually incompatible), so the specialist runs
out-of-process. Build both with the multi-agent setup script and load its env file:

```bash
examples/multiagent-openvla-gemma/setup.sh        # builds env_pilot + env_specialist
source examples/multiagent-openvla-gemma/.env      # activates env_pilot, exports ODYSSEY_SPECIALIST_PYTHON
```

Without `ODYSSEY_SPECIALIST_PYTHON` the eval fails fast with a clear error.

## Time & hardware

- **GPU:** 24 GB (L4) — int4 E2B-it planner (~5 GB) + ~14 GB bf16 pilot ≈ 19 GB peak.
- **Training:** `max_steps: 5000` (~1 h on an L4) for a real attempt; drop to
  `max_steps: 10` for a quick **pipeline smoke** (no real learning).
- **Dataset:** the Bridge V2 RLDS dataset on disk (~124 GB). See
  `docs/gcp-training-tutorial.md` for downloading it.

## Run it

```bash
# headless render — EGL on a GPU with NVIDIA EGL, else OSMesa (CPU) on compute-only drivers
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl     # or =osmesa

# point the mission at your dataset
#   edit examples/franka-pickplace/mission.yaml -> config.data_root_dir

odyssey validate examples/franka-pickplace/mission.yaml
odyssey run      examples/franka-pickplace/mission.yaml
```

## Output

- Checkpoint + per-task artifacts under `~/.odyssey/runs/<mission_id>/<task_id>/`.
- One MP4 per episode in the eval task's `videos/` dir (`capture_video: true`), also
  surfaced in `result_summary.artifacts.videos`. The planner's per-episode phase plan
  shows up in the `episode_plan` progress events.

## Want a *custom* box (not Robosuite's bins)?

`PickPlaceCan` uses Robosuite's built-in bins. A bespoke "box" object with its own
success condition would need a custom Robosuite environment (or a custom `env_factory`
injected into `RobosuiteRunner`) — a separate, larger piece of work.
