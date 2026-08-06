# π0.5 fine-tune quickstart (training)

Fine-tune a Physical Intelligence **π0.5** (openpi) checkpoint on a LeRobot-format
demonstration set through the odyssey `pi05` training runner. The runner shells
out to openpi's own entry points — `scripts/compute_norm_stats.py` then
`scripts/train.py` — so the actual training happens in your openpi checkout.

This is the **training** counterpart to `examples/quickstart-pi05/` (eval-only).

> **Status:** the odyssey-side wiring (argv build, norm-stats + train subprocess
> orchestration, checkpoint capture, stdout→progress parsing) is complete and
> unit-tested on CPU. It has **not** been run end-to-end on a GPU box yet —
> treat the openpi flag names below as the contract to confirm on the first smoke.

## What you need

- A GPU box (π0.5 fine-tune is memory-hungry — an 80 GB A100/H100 class card).
- **openpi** installed and checked out:
  ```bash
  git clone https://github.com/Physical-Intelligence/openpi
  cd openpi && pip install -e .
  export OPENPI_REPO_PATH=$(pwd)      # runner defaults to /srv/openpi otherwise
  ```
- The LeRobot dataset unpacked on the box, e.g.
  `/data/ur10e_drugsort_v0/ur10e_partial_cond_aug/`.

## Why openpi is config-name-driven (and GR00T isn't)

GR00T / OpenVLA take a flat bag of `--flag value` overrides. openpi selects a
**registered `TrainConfig`** by name — it bundles the data transforms, the base
weight loader and the optimizer. So fine-tuning a *new* dataset/embodiment means
adding one config entry (the π0.5 analogue of GR00T's `modality_config_path`).

### 1. Register a `TrainConfig` in openpi

Add an entry to `src/openpi/training/config.py` (name it `pi05_ur10e_drugsort`, to
match `config_name` in the mission). Base it on the shipped `pi05_libero` config,
then point its data config at your LeRobot dataset and describe the state/action
mapping:

- **state/action**: 7-DoF (6 joints + gripper) — π0.5 pads to the model's action
  dim; no observer-conditioned `grasp_target` channel (that's GR00T-specific —
  drop it here).
- **images**: map `observation.images.exterior` → `base_0_rgb` and
  `observation.images.wrist` → `left_wrist_0_rgb` in the repack transform.
- **repo_id**: the dataset folder name, `ur10e_partial_cond_aug`.

### 2. Dataset resolution

For a **local absolute** dataset ref, the runner sets `HF_LEROBOT_HOME` to the
dataset's parent dir, so openpi's LeRobot loader resolves `data.repo_id` == the
folder name. Keep the two in sync:

```
ref:            /data/ur10e_drugsort_v0/ur10e_partial_cond_aug
data.repo_id:   ur10e_partial_cond_aug          # HF_LEROBOT_HOME=/data/ur10e_drugsort_v0
```

## Run

```bash
# CPU validation (no training):
odyssey validate mission.yaml
odyssey run mission.yaml --use-mock-runner

# Real fine-tune on the GPU box:
export OPENPI_REPO_PATH=/path/to/openpi
odyssey run mission.yaml
```

The runner executes, in order:

1. `python scripts/compute_norm_stats.py --config-name pi05_ur10e_drugsort`
   (skip with `config: {compute_norm_stats: false}` if `./assets` already holds them)
2. `python scripts/train.py pi05_ur10e_drugsort --exp-name finetune-pi05-ur10e --overwrite \
      --data.repo-id ur10e_partial_cond_aug --num-train-steps 30000 --batch-size 32`

> The norm-stats step takes no `data.*` overrides — it reads the dataset fixed by
> the config's `data` factory. So the mission's `data.repo_id` **must equal** the
> registered config's default `repo_id`, or step 1 writes norm stats under one
> `repo_id` and step 2 reads another and fails (loudly, after the full dataset scan).

> Norm stats are cached across runs at `~/.odyssey/pi05_assets/<config_name>/<config_name>/<repo_id>/`
> and step 1 is skipped on a hit. The hit is keyed on `config_name` + `repo_id`
> only — it has **no content fingerprint**, so a dataset recaptured under an
> unchanged `repo_id` would silently reuse stale statistics. When the dataset
> content changed, set `config: {norm_stats_cache: false}` to force a fresh
> recompute into the per-run dir.

Both run with `cwd` = the task's `output_dir`, so openpi's cwd-relative `./assets`
and `./checkpoints` land under the odyssey run dir. The trained checkpoint is
captured from `checkpoints/<config_name>/<exp_name>/<step>/` (highest step).

## Config keys the runner interprets

| key                  | meaning                                                            |
|----------------------|-------------------------------------------------------------------|
| `runner: pi05`       | routes the wildcard training task to `Pi05Runner`                 |
| `config_name`        | **required** — the registered openpi `TrainConfig` (positional to `train.py`, `--config-name` flag to `compute_norm_stats.py`) |
| `exp_name`           | openpi `--exp-name`; defaults to the task name                     |
| `overwrite`          | emit `--overwrite` (default `true`); clobbers the prior exp dir    |
| `resume`             | emit `--resume` instead of `--overwrite`; continue latest ckpt    |
| `compute_norm_stats` | run the norm-stats pre-step (default `true`)                       |
| `norm_stats_cache`   | reuse cached norm stats across runs, keyed by `config_name`+`repo_id` (default `true`); set `false` to force a recompute when the dataset content changed under the same `repo_id` |
| *anything else*      | forwarded as a tyro override (`a_b` → `--a-b`; nested `x: {y: 1}` → `--x.y 1`; bools → `--flag` / `--no-flag`) |

## Can I then evaluate on LIBERO?

Not honestly with *this* checkpoint. LIBERO is a **Franka Panda** sim benchmark;
a π0.5 fine-tuned on **UR10e** demos is a different embodiment, action space and
task, so a LIBERO score would measure a sim-to-sim gap, not this cell. The
`examples/quickstart-pi05/` LIBERO eval is for a **LIBERO-compatible** π0.5
checkpoint (e.g. the shipped `pi05_libero`). For the drug-sort model, the honest
metric is the standalone closed-loop grasp+place eval against physics ground
truth (`meta/gt`).
