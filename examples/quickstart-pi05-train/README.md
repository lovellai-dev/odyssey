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

1. `python <odyssey>/runners/models/openpi_bootstrap.py scripts/compute_norm_stats.py \
      pi05_ur10e_drugsort -- --data.repo-id ur10e_partial_cond_aug --num-train-steps 30000 \
      --batch-size 32 --exp-name finetune-pi05-ur10e`
   (this mission sets overrides, so norm stats see the same config as step 2; skip the
   step with `config: {compute_norm_stats: false}` if `./assets` already holds them)
2. `python scripts/train.py pi05_ur10e_drugsort --exp-name finetune-pi05-ur10e --overwrite \
      --data.repo-id ur10e_partial_cond_aug --num-train-steps 30000 --batch-size 32`

> openpi's `compute_norm_stats.py` only accepts `--config-name`; it can't take the
> mission's overrides the way `train.py` does. So when the mission sets any override
> (e.g. `data.repo_id`), step 1 runs through Odyssey's `openpi_bootstrap.py`. It
> builds the config with the same parser and overrides as `train.py`, and runs the
> script against it. The norm stats are then computed for the dataset that training
> reads. Without overrides, step 1 calls the script directly, as before.

> Norm stats are cached across runs at `~/.odyssey/pi05_assets/<config_name>/<config_name>/<repo_id>/`
> and step 1 is skipped on a hit. The hit is keyed on `config_name` + `repo_id`
> and, when the task's `dataset` pins a commit-sha `revision` that was verified on
> disk (below), on that revision too (the cache root becomes
> `<config_name>@<revision>`). A new version of a dataset under the same `repo_id`
> then misses the cache and its statistics are recomputed. A branch or tag
> (`revision: main`) moves, so it is recorded but never used as a cache key.
>
> **Without a verified revision there is no content fingerprint**, so a dataset
> recaptured under an unchanged `repo_id` silently reuses stale statistics. Pin the
> revision, or set `config: {norm_stats_cache: false}` to force a fresh recompute
> into the per-run dir.

> **Pinning the dataset revision.** `dataset: {source: local, ref: /data/my_dataset,
> revision: <commit sha>}` records the exact version trained on. A copy made with
> `hf download <repo> --repo-type dataset --revision <sha> --local-dir /data/my_dataset`
> carries Hub download metadata for **every file**, and the runner checks all of
> them in the directory training actually loads (the pre-flight resolves
> `HF_LEROBOT_HOME / data.repo_id` with openpi and refuses a `dataset.ref` that
> isn't it). Each file must come from the pinned commit and be unmodified since
> download, the rule `huggingface_hub` itself uses. A file from another revision,
> edited locally or added by hand, or a copy with no download metadata, fails the
> task before any GPU work. The copy must also be complete: every file the LeRobot
> manifest (`meta/info.json`, `meta/episodes.jsonl`) says training reads must be
> there, so a partial `--include` download can't pass; move copies with `cp -p` / `rsync -a` so mtimes
> survive. The result records `dataset_revision_verified`. openpi loads a
> `source: huggingface` dataset by `repo_id` and cannot be pinned, so a pinned hub
> dataset is refused; download the revision and use `source: local`.

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
| `norm_stats_cache`   | reuse cached norm stats across runs, keyed by `config_name`+`repo_id` (+ the dataset `revision` when pinned to a verified commit sha) (default `true`); set `false` to force a recompute when the dataset content changed under the same unpinned `repo_id` |
| *anything else*      | forwarded as a tyro override (`a_b` → `--a-b`; nested `x: {y: 1}` → `--x.y 1`; bools → `--flag` / `--no-flag`) |

## Can I then evaluate on LIBERO?

Not honestly with *this* checkpoint. LIBERO is a **Franka Panda** sim benchmark;
a π0.5 fine-tuned on **UR10e** demos is a different embodiment, action space and
task, so a LIBERO score would measure a sim-to-sim gap, not this cell. The
`examples/quickstart-pi05/` LIBERO eval is for a **LIBERO-compatible** π0.5
checkpoint (e.g. the shipped `pi05_libero`). For the drug-sort model, the honest
metric is the standalone closed-loop grasp+place eval against physics ground
truth (`meta/gt`).
