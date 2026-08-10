# Cosmos 3 × RoboLab (eval-only, custom runner)

Scores a published Cosmos 3 DROID world-action policy on
[NVlabs/RoboLab](https://github.com/NVlabs/RoboLab) via Odyssey's
`evaluation_type: custom` contract — no first-class RoboLab runner (yet).

## Prerequisites (two external processes)

1. **Policy server** (cosmos-framework env, GPU box — Nano 16B needs ≥32 GB
   VRAM in bf16; Edge 4B fits smaller GPUs):

   ```bash
   python -m cosmos_framework.scripts.action_policy_server_robolab \
       --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8000
   # Edge variant: add --checkpoint-path nvidia/Cosmos3-Edge-Policy-DROID \
   #               --format-prompt-as-json True
   ```

2. **RoboLab checkout** (client side; NVIDIA's docker images work):

   ```bash
   git clone https://github.com/NVlabs/RoboLab.git
   cd RoboLab && ./docker/build_docker.sh latest
   ```

## Run

Edit `mission.yaml` (`robolab_root`, optionally `robolab_python` /
`extra_args` for the server address), then:

```bash
odyssey run examples/cosmos3-robolab/mission.yaml
```

`robolab_eval.py` launches `policies/cosmos3/run.py` per task (headless,
`--num-envs`), parses the results, and writes the `--out-json` metrics the
`CustomEvalRunner` folds into the mission summary.

⚠ First GPU smoke: pin RoboLab's actual result output (JSON files vs stdout)
and tighten the bridge's tolerant parser (`results_glob` preferred).

## Family notes

Any Cosmos 3 policy member works — the server loads the weights; the mission
only records the id. For LIBERO evals of the family use
`examples/quickstart-cosmos3/` (`pilot: cosmos3`) instead.
