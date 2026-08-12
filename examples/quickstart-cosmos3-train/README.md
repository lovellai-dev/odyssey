# Cosmos 3 (WAM) action-policy fine-tune quickstart (training)

Fine-tune an NVIDIA **Cosmos 3** action policy on a LeRobot-v3 demonstration set
through the odyssey `cosmos3` training runner. The runner shells out to
**cosmos-framework**'s own entry points — `convert_model_to_dcp` → `train`
(under `torchrun`) → `export_model` — so the actual training happens in your
cosmos-framework checkout.

This is the **training** counterpart to the eval-only `pilot: cosmos3` path
(LIBERO / RoboLab). Once this produces an export, `pilot: cosmos3` scores it for
real instead of running out-of-distribution against a DROID-only checkpoint.

> **Status:** the odyssey-side wiring (three-step subprocess orchestration, argv
> build, env bridging, run-dir + checkpoint capture, stdout→progress parsing) is
> complete and unit-tested on CPU (`tests/unit/test_cosmos3_train.py`). It has
> **not** been run end-to-end on a GPU box yet — treat the cosmos-framework flag
> names and env vars below as the contract to confirm on the first smoke.

## What you need

- A multi-GPU box. The reference LIBERO-Nano recipe is 8 GPUs/node (the paper
  used far more); a single-node **smoke** with `trainer.max_iter=10` fits fewer.
- **cosmos-framework** installed and checked out:
  ```bash
  git clone https://github.com/NVIDIA/cosmos-framework
  cd cosmos-framework && pip install -e .
  export COSMOS_FRAMEWORK_REPO_PATH=$(pwd)   # runner defaults to /srv/cosmos-framework
  ```
- The **LeRobot v3.0** dataset unpacked on the box (e.g.
  `nvidia/LIBERO_LeRobot_v3` / `nvidia/Cosmos3-DROID`). FOOTGUN: `DATASET_PATH`
  must point at the **PARENT of the `success/` folder** — i.e. the named dataset
  dir itself — and the loader infers its schema from that **dir name**, so keep
  the vendor naming (e.g. `droid_plus_lerobot_640x360_20260412`).
- The Wan2.2 VAE weights (`WAN_VAE_PATH`) and, for DROID recipes, the DROID
  filters dir (`FILTER_DIR`).

## Option A: run `odyssey` from inside the cosmos-framework venv

cosmos-framework lives in its own heavy venv. `run_training_subprocess` launches
every child (torchrun workers included) via `sys.executable`, so the simplest,
zero-core-change setup is to **install odyssey into that venv and run it from
there** (the UR10e remote-venv pattern):

```bash
source /path/to/cosmos-framework/.venv/bin/activate
pip install -e /path/to/odyssey
odyssey run examples/quickstart-cosmos3-train/mission.yaml
```

## Why cosmos-framework is config-name-driven (and GR00T isn't)

GR00T / OpenVLA take a flat bag of `--flag value` overrides. cosmos-framework —
like π0.5 — selects a **registered SFT recipe TOML** by name; it bundles the data
pipeline, weight loader and optimizer. So fine-tuning a *new* dataset/embodiment
means adding a recipe under `examples/toml/sft_config/` whose interpolated env
paths point at your data/checkpoints.

`config.config_name` in the mission names that recipe (bare, no `.toml`):
`action_policy_libero_nano`, `action_policy_droid_nano`, … Everything else under
`config:` (minus the control keys below) is forwarded as a **dotted CLI override**
(`trainer.max_iter` → `--trainer.max-iter`; booleans become `--flag`/`--no-flag`).

## The three steps the runner runs

| # | cosmos-framework entry | torchrun | Skipped when |
|---|------------------------|----------|--------------|
| 1 | `convert_model_to_dcp` | no  | `$BASE_CHECKPOINT_PATH` exists, or `config: {convert_dcp: false}` |
| 2 | `train --sft-toml=<recipe>` | **yes** (`nproc_per_node`) | never |
| 3 | `export_model` | no  | `config: {export: false}` |

### Env vars the recipe TOML interpolates (`${oc.env:...}`)

The runner sets these from the resolved mission refs; anything omitted falls
through from your shell:

| Var | Set by | Meaning |
|-----|--------|---------|
| `DATASET_PATH` | `dataset.ref` (or `config.dataset_path`) | LeRobot-v3 dataset root (parent of `success/`) |
| `BASE_CHECKPOINT_PATH` | `config.base_checkpoint_path`, else `<output>/base_checkpoint_dcp` | DCP dir (step 1 output) |
| `IMAGINAIRE_OUTPUT_ROOT` | task `output_dir` | where train writes the run dir |
| `NPROC_PER_NODE` | `config.nproc_per_node` (default 8) | torchrun workers |
| `WAN_VAE_PATH` | `config.wan_vae_path` (or shell) | Wan2.2 VAE weights |
| `FILTER_DIR` | `config.filter_dir` (or shell) | DROID filters (DROID recipes) |
| `HF_TOKEN` | shell | HF auth (gated weights) |

### Control keys (consumed by the runner, never forwarded as overrides)

`config_name`, `runner`, `base_model`, `convert_dcp`, `export`, `nproc_per_node`,
`base_checkpoint_path`, `wan_vae_path`, `filter_dir`, `dataset_path`.

## Validate without a GPU

```bash
odyssey validate examples/quickstart-cosmos3-train/mission.yaml
odyssey run examples/quickstart-cosmos3-train/mission.yaml --use-mock-runner
```

## train → serve → eval hand-off

The mission's second task scores the export on LIBERO with `pilot: cosmos3`.
Cosmos 3 does **not** auto-serve in-process (contrast GR00T): you start the
cosmos-framework HTTP server on the export, then run the eval against it.

```bash
# 1. Train (produces <task output_dir>/.../model — the exported HF safetensors).
odyssey run examples/quickstart-cosmos3-train/mission.yaml

# 2. Serve the export (in its own shell; note the printed export path).
python -m cosmos_framework.scripts.action_policy_server_libero \
    --checkpoint-path <run>/model --port 8000

# 3. Eval — the LIBERO task connects to 127.0.0.1:8000 (host/port in the mission).
#    If you ran step 1 as a full mission, the eval task will have already tried to
#    connect; run just the eval once the server is up, or split into two missions.
```

The `pilot: cosmos3` eval knobs (`host`, `port`, `n_action_steps`,
`domain_name`, `task_id`, …) pass through verbatim to `cosmos3_libero_eval.py`;
`n_action_steps: 0` adopts the server's own `action_chunk_size` from `GET /info`.
