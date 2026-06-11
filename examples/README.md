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

For a real OpenVLA training run, see the prerequisites in the top-level
README (OpenVLA repo clone, Bridge V2 download, 24 GB GPU).

## GR00T quickstart status

`quickstart-gr00t/` ships with a real training runner skeleton
(`src/odyssey/runners/gr00t.py`): `validate` and `--use-mock-runner`
work everywhere, and real training runs work once the
[Isaac-GR00T repo](https://github.com/NVIDIA/Isaac-GR00T) is installed
(`pip install -e` of the checkout, plus `$ISAAC_GR00T_REPO_PATH`
pointing at it for the demo data). The Isaac Lab *evaluation* runner is
still a v0.2.x line item — the eval task is spec-pinned but mock-only.

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
  `evaluation_type: isaac_lab` (already a first-class enum in the
  mission spec).
