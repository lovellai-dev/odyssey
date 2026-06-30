# Franka pick-and-place — put the can in the bin

A single-agent OpenVLA mission: fine-tune the pilot on **Bridge V2**, then evaluate
on the Robosuite **`PickPlaceCan`** benchmark, where a **Franka Panda** picks up a
can and places it in the bin (a "put an object in a box" task).

`PickPlaceCan` is the single-object variant of `PickPlace` (which juggles four
objects/bins), so it's the most attainable place-in-bin task to start from.

## ⚠️ Read first: the domain gap

This mission **trains and evaluates in different settings**:

| | Training | Evaluation |
|---|---|---|
| Source | Bridge V2 (`bridge_orig`) | Robosuite (simulation) |
| Robot | WidowX (real) | Franka Panda (sim) |

A freshly fine-tuned LoRA will rarely succeed zero-/few-shot — the pilot is tested
on a different robot and domain than it learned on. Treat this as a **coherent,
visually attractive pipeline demo**, not a guaranteed-success benchmark. Bridge does
contain many "put X in a container" demonstrations, so this task is *more* aligned
with the training data than e.g. `Lift`.

## Time & hardware

- **GPU:** 24 GB (L4 / RTX 4090 class).
- **Training:** the `mission.yaml` ships `max_steps: 5000` (~1 h on an L4) for a real
  attempt. For a quick **pipeline smoke** (no real learning), drop it to `max_steps: 10`.
- **Dataset:** the Bridge V2 RLDS dataset on disk (~124 GB). See the GCP tutorial
  (`docs/gcp-training-tutorial.md`) for downloading it.

## Run it

```bash
# 1. Environment (reuse the OpenVLA env; e.g. the one built by the multi-agent setup.sh)
#    + the GCP-critical vars. On a compute-only driver use OSMesa for headless render.
export NCCL_NET=Socket
export WANDB_MODE=disabled
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl     # or =osmesa if the VM has no NVIDIA EGL

# 2. Point the mission at your dataset
#    edit examples/franka-pickplace/mission.yaml -> config.data_root_dir

# 3. Validate, then run
odyssey validate examples/franka-pickplace/mission.yaml
odyssey run      examples/franka-pickplace/mission.yaml
```

## Output

- Checkpoint + per-task artifacts under `~/.odyssey/runs/<mission_id>/<task_id>/`.
- One MP4 per episode in the eval task's `videos/` dir (`capture_video: true`),
  also surfaced in `result_summary.artifacts.videos`.

## Want a *custom* box (not Robosuite's bins)?

`PickPlaceCan` uses Robosuite's built-in bins. A bespoke "box" object with its own
success condition would need a custom Robosuite environment (or a custom
`env_factory` injected into `RobosuiteRunner`) — a separate, larger piece of work.
