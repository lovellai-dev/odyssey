# Examples

| Directory | What it does | Hardware |
|---|---|---|
| `quickstart-openvla/` | OpenVLA 7B LoRA fine-tune on Bridge V2 + Robosuite Lift eval. | 24 GB GPU |
| `quickstart-gr00t/` | GR00T N1.7 3B fine-tune on the Isaac-GR00T demo set + Isaac Lab eval. | 24 GB+ GPU for training; eval mock-only until the Isaac Lab runner lands |

More quickstarts (Octo, Pi0.5) arrive in later releases. See the
publication plan for the cadence.

## Validating and mock-running an example

Once `lovell-odyssey` is installed:

```bash
odyssey validate examples/quickstart-openvla/mission.yaml
odyssey validate examples/quickstart-gr00t/mission.yaml
```

Every example also runs end-to-end with the CPU mock (no GPU, no
network):

```bash
odyssey run examples/quickstart-gr00t/mission.yaml --use-mock-runner
```

For a real GR00T fine-tuning run, see the prerequisites in the top-level
README (Isaac-GR00T repo clone, a LeRobot-format dataset, and a CUDA GPU).

## GR00T quickstart status

`quickstart-gr00t/` ships with real runner skeletons on both sides:
`validate` and `--use-mock-runner` work everywhere; real training runs
work once the [Isaac-GR00T repo](https://github.com/NVIDIA/Isaac-GR00T)
is installed (`pip install -e` of the checkout, plus
`$ISAAC_GR00T_REPO_PATH` pointing at it for the demo data); and real
evaluation runs work once you have Isaac Lab installed
(`$ISAACLAB_PATH`) and supply an eval script speaking the `ODYSSEY_*`
stdout protocol via `config: {eval_script: ...}` — see
`src/odyssey/runners/evals/isaac_lab.py` for the contract. The blessed
GR00T-policy eval recipe now ships at
`src/odyssey/runners/evals/gr00t_isaac_eval.py` — see **Closed-loop GR00T
eval** below.

The moving parts:

* **Model** — `nvidia/GR00T-N1.7-3B` from HuggingFace (the base
  checkpoint; zero-shot on pretrain embodiments, fine-tunable for new
  tasks). NVIDIA's license applies to the weights.
* **Data** — the LeRobot-v2-flavor demo set shipped inside Isaac-GR00T
  (`demo_data/cube_to_bowl_5`), resolved against
  `$ISAAC_GR00T_REPO_PATH`. GR00T's format adds a `meta/modality.json`
  on top of LeRobot v2.
* **Runner routing** — OpenVLA and GR00T are both wildcard training
  runners, so the mission selects GR00T explicitly with
  `config: {runner: gr00t}` on the training task.
* **Training config** — remaining keys mirror the
  `gr00t/experiment/launch_finetune.py` flags (`embodiment_tag`,
  `global_batch_size`, `max_steps`); the runner maps them 1:1 onto the
  upstream CLI with underscores translated to dashes.
* **Eval** — Isaac Lab task `Isaac-Lift-Cube-Franka-v0`,
  `evaluation_type: isaac_lab`. The runner launches your eval script
  under `isaaclab.sh -p`, passes
  `--task/--num_episodes/--checkpoint/--headless`, and scores the
  `ODYSSEY_EPISODE` / `ODYSSEY_RESULT` lines the script prints.

## Closed-loop GR00T eval (train → deploy → grade in one run)

`quickstart-gr00t/closed_loop_mission.yaml` runs the whole loop in a single
`odyssey run`:

1. **Train** — a `runner: gr00t` training task fine-tunes GR00T N1.7 on the
   demo set, producing a checkpoint.
2. **Deploy + grade** — the `isaac_lab` eval task uses the blessed GR00T eval
   recipe at **`src/odyssey/runners/evals/gr00t_isaac_eval.py`** (relocated from
   the top-level `scripts/` folder, so it lives beside the `isaac_lab` runner it
   serves; `gr00t_transforms.py` is its sibling). With `serve_checkpoint: true`
   the recipe **auto-serves the just-trained checkpoint** as a GR00T policy
   server, waits for it to be ready, drives the Franka visuomotor env in Isaac
   Lab, and reports per-episode `ODYSSEY_*`. The runner auto-passes the training
   task's checkpoint, so the deployed policy is exactly what was just trained —
   no manually-started external server.

The recipe keeps its heavy deps (isaaclab / gr00t / torch / numpy) lazy in the
run path, so its argv + `ODYSSEY_*` surface stay unit-testable on a CPU box.
Key eval-task `config:` keys:

| key | meaning |
|---|---|
| `eval_script` | the relocated recipe path (`src/odyssey/runners/evals/gr00t_isaac_eval.py`) |
| `serve_checkpoint: true` | boot a GR00T server on the trained checkpoint here, vs. connecting to an external one |
| `embodiment_tag` | **must match** the training `embodiment_tag` — the server can't infer it |
| `server_python` | interpreter that has `gr00t` installed (the GR00T venv) |
| `served_model_path` | optional explicit checkpoint to serve (e.g. a Command-Center-delegated eval whose runner `--checkpoint` is a stub) |
| `pos_scale` | scales the raw GR00T eef deltas into the IK-rel action |

Plumbing-only (no Isaac / GR00T):

```bash
odyssey run examples/quickstart-gr00t/closed_loop_mission.yaml --use-mock-runner
```

A real run needs the GR00T venv plus an Isaac Lab install with the Cosmos
visuomotor env.
