# UR5e drug-sorting — GR00T fine-tune for the Lovell AI Robot Playground

Fine-tune NVIDIA Isaac **GR00T N1.7 (3B)** on the **AseptiPack** fill-finish cell
(UR5e + Robotiq 2F-85) so the resulting policy drops straight into the browser
**Lovell AI Robot Playground** Pilot slot and drives a real pick-and-place of a
vial into the rack.

The whole pipeline is:

```
scripts/gen_ur5e_drugsort_demos.py   →  GR00T LeRobot v2.1 dataset  (CPU, no GPU)
        │  (headless MuJoCo replay of the scripted pick-place policy + DR)
        ▼
examples/ur5e-drugsort/ur5e_config.py  +  mission.yaml
        │  (GR00T new_embodiment modality config + odyssey `runner: gr00t` task)
        ▼
gr00t.experiment.launch_finetune       →  fine-tuned checkpoint   (GPU / H100)
        ▼
GR00T policy server  →  agent-service inference client  →  Playground setTargets bridge
```

See **[RUNBOOK.md](RUNBOOK.md)** for the exact end-to-end commands.

## The obs/action contract (why it transfers)

The dataset is recorded in **exactly** the space the Playground consumes at
inference, so the checkpoint is drop-in:

| | Playground (browser) | Recorded demo | GR00T modality |
|---|---|---|---|
| **action** | `setTargets(q, grip)` — 6 UR5e joint targets (rad) + `grip` 0..1 (→ `gr_fingers_actuator` 0..255) | `action` = 6 joint targets (rad) + grip 0..1 | `action.single_arm` (6) + `action.gripper` (1) |
| **proprio** | 6 arm joint qpos + gripper closure | `observation.state` = 6 qpos + grip 0..1 | `state.single_arm` (6) + `state.gripper` (1) |
| **camera** | `TrueSensorRenderer.capturePrimary()` → `exterior` mount, 256×256 | `observation.images.exterior` (MJCF `room` cam), 256×256 | `video.exterior` |
| **language** | task prompt | `annotation.human.task_description` | `annotation.human.task_description` |

The action space mirrors the Phase-1 pilot seam in
`lai-agent .../aseptipack-pickplace.js` (`setTargets(q, grip)`), and the camera
is the single **primary** view the inference loop publishes — we deliberately do
**not** record a wrist view the browser bridge doesn't emit, which would break
transfer.

## Files

| File | What |
|---|---|
| `ur5e_config.py` | GR00T `new_embodiment` modality config (registered under `EmbodimentTag.NEW_EMBODIMENT`). Passed via `--modality-config-path`. |
| `mission.yaml` | Odyssey mission: a `runner: gr00t` demonstration training task + a (secondary, optional) closed-loop Isaac Lab eval. |
| `RUNBOOK.md` | Exact commands: generate data → fine-tune on GPU → deploy to the Playground. |

The generator, the LeRobot writer, and the embodiment spec live in the library:

- `scripts/gen_ur5e_drugsort_demos.py` — the CLI data generator.
- `src/odyssey/embodiments/ur5e_drugsort/` — `embodiment.py` (modality/info spec)
  + `policy.py` (the mujoco-free scripted pick-place FSM, ported from the browser).
- `src/odyssey/datasets/lerobot_writer.py` — the GR00T LeRobot v2.1 writer.

## Validate without a GPU

```bash
odyssey validate examples/ur5e-drugsort/mission.yaml
odyssey run      examples/ur5e-drugsort/mission.yaml --use-mock-runner
```
