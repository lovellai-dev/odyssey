# Examples

| Directory | What it does | Hardware |
|---|---|---|
| `quickstart-openvla/` | OpenVLA 7B LoRA fine-tune on Bridge V2 + Robosuite Lift eval. | 24 GB GPU |
| `quickstart-gr00t/` | GR00T N1.7 3B fine-tune on the Isaac-GR00T demo set + Isaac Lab eval. | 24 GB+ GPU for training; eval mock-only until the Isaac Lab runner lands |
| `multiagent-openvla-gemma/` | Multi-agent eval: OpenVLA **PILOT** + an out-of-process multimodal Gemma 4 **SPECIALIST** planner, on Robosuite Lift. Needs extra setup — [see its README →](multiagent-openvla-gemma/README.md). | 24 GB GPU (PILOT + SPECIALIST share it) |

More quickstarts (Octo, Pi0.5) arrive in later releases. See the
publication plan for the cadence.

## Validating and mock-running an example

Once `lovell-odyssey` is installed:

```bash
odyssey validate examples/quickstart-openvla/mission.yaml
odyssey validate examples/quickstart-gr00t/mission.yaml
odyssey validate examples/multiagent-openvla-gemma/mission.yaml
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
`src/odyssey/runners/isaac_lab.py` for the contract. The built-in
GR00T-policy eval script is the remaining v0.2.x piece.

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
