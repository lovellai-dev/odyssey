# Cosmos 3 × RoboLab (eval-only, first-class `evaluation_type: robolab`)

Scores a published Cosmos 3 DROID world-action policy on
[NVlabs/RoboLab](https://github.com/NVlabs/RoboLab) — the **Isaac Lab-based
simulation twin of the DROID rig** (the real-robot cell in NVIDIA's Cosmos
videos: Franka + two mounted cameras + tabletop tasks). The `RobolabRunner`
launches RoboLab's `policies/cosmos3/run.py` headless and scores its
`episode_results.jsonl` into the standard eval summary (per-task rates +
failure reasons in `metrics`).

## Prerequisites (two external processes)

1. **Policy server** (cosmos-framework env, GPU box — Nano 16B needs ≥32 GB
   VRAM in bf16; Edge 4B fits smaller GPUs):

   ```bash
   # --checkpoint-path takes a LOCAL dir — stage it first (setup.sh does this):
   hf download nvidia/Cosmos3-Nano-Policy-DROID --local-dir ~/checkpoints/Cosmos3-Nano-Policy-DROID
   python -m cosmos_framework.scripts.action_policy_server_robolab \
       --checkpoint-path ~/checkpoints/Cosmos3-Nano-Policy-DROID --port 8000
   # Edge variant: same, plus --format-prompt-as-json True
   ```

   Guardrails are gated (`nvidia/Cosmos-Guardrail1`) — see setup.sh for the
   HF-access / disable options.

2. **RoboLab checkout with its Isaac Sim env** (client side; the docker
   images work):

   ```bash
   git clone https://github.com/NVlabs/RoboLab.git
   cd RoboLab && ./docker/build_docker.sh latest
   ```

## Run

Edit `mission.yaml` (`robolab_root`, `eval_python` for the Isaac interpreter,
`remote_host`/`remote_port` for the server), then:

```bash
odyssey run examples/cosmos3-robolab/mission.yaml
```

Contract details (pinned from RoboLab source): `benchmark_name` → `--task`;
`num_episodes` → `--num-runs ceil(episodes / num_envs)`; results are read from
`<robolab_root>/output/odyssey_<task-id>/episode_results.jsonl` and the raw
jsonl is copied into the task's output dir as an artifact. Passthrough config
keys reach `run.py` with underscores dashed (`remote_port` → `--remote-port`).

## Family notes

Any Cosmos 3 policy member works — the server loads the weights; the mission
only records the id. `config.entry_script` selects the RoboLab policy backend
(default `policies/cosmos3/run.py`) — a future backend is a mission edit, not
a new runner. For LIBERO evals of the family use `examples/quickstart-cosmos3/`
(`pilot: cosmos3`); for the no-sim decode check use
`examples/cosmos3-droid-openloop/`.
