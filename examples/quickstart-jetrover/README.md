# JetRover quickstart — GR00T on the Hiwonder 6DoF arm (real hardware)

Fine-tune GR00T N1.7 as a `NEW_EMBODIMENT` on teleop demos captured on the
[Hiwonder JetRover](https://www.hiwonder.com/products/jetrover)'s 6DoF arm, then
evaluate closed-loop **on the physical arm**. The built-in eval runners
(Robosuite / LIBERO / Isaac Lab) are sim-only and ship no JetRover model, so the
eval task uses `evaluation_type: custom` with the `eval_jetrover.py` script in
this directory.

Out of scope here: dataset-capture tooling (record with LeRobot or your own
teleop stack) and a sim eval (would need Hiwonder's URDF ported into a sim —
possible later phase).

## What's in this directory

| File | Purpose |
| --- | --- |
| `mission.yaml` | Training (GR00T NEW_EMBODIMENT, LeRobot dataset) + real-arm custom eval |
| `jetrover_modality_config.py` | GR00T modality config (6-D arm + 1-D gripper, front camera) — copy into your Isaac-GR00T checkout |
| `eval_jetrover.py` | The custom eval script: GR00T policy server client + arm backends (`mock`/`ros2`/`hiwonder`) + operator scoring |

Validate everything without a GPU or the robot:

```bash
odyssey validate examples/quickstart-jetrover/mission.yaml
odyssey run examples/quickstart-jetrover/mission.yaml --use-mock-runner
```

## 1. Capture the dataset (~50 episodes, LeRobot layout)

GR00T's own SO-101 tutorial uses ~50 episodes as the few-shot floor; plan for at
least that. Record teleoperated demonstrations of ONE task (e.g. "pick up the
object") into a LeRobot v2 dataset with, per frame:

| Key | Shape | Meaning |
| --- | --- | --- |
| `state.single_arm` | (6,) | arm joint positions, rad |
| `state.gripper` | (1,) | gripper position |
| `action.single_arm` | (6,) | target joint positions, rad |
| `action.gripper` | (1,) | target gripper position |
| `video.front` | HxWx3 | the depth cam's RGB stream |
| `annotation.human.task_description` | str | the language instruction |

GR00T additionally needs a `meta/modality.json` describing these keys (see the
`demo_data/` sets inside the Isaac-GR00T repo for the exact shape). Copy the
finished dataset to the GPU box, e.g. `/data/jetrover_pick_v0/jetrover_pick`.

## 2. Drop in the modality config

```bash
mkdir -p $ISAAC_GR00T_REPO_PATH/examples/JETROVER
cp jetrover_modality_config.py $ISAAC_GR00T_REPO_PATH/examples/JETROVER/jetrover_config.py
```

Upstream transform/import names drift between GR00T releases — diff against
`examples/SO100/so100_config.py` in *your* checkout and align if needed.

## 3. Fine-tune (GPU box)

```bash
export ISAAC_GR00T_REPO_PATH=~/Isaac-GR00T
odyssey run examples/quickstart-jetrover/mission.yaml
```

The training task maps 1:1 onto `gr00t/experiment/launch_finetune.py` flags
(`embodiment_tag: new_embodiment`, `modality_config_path`, batch/steps knobs in
`mission.yaml`). The checkpoint lands under the task's output dir.

## 4. Serve the checkpoint (GPU box)

GR00T inference does not fit on the Jetson — serve remotely and let the eval
client connect:

```bash
python -m gr00t.eval.run_gr00t_server \
  --model_path <task-output-dir-checkpoint> \
  --embodiment_tag new_embodiment \
  --port 5555
```

## 5. Evaluate on the real arm

Set `policy_host` in `mission.yaml` to the GPU box and run the eval task (on
whichever machine reaches both the arm and the server — typically the Jetson
with `arm_backend: ros2`). Sanity-check the wire first without moving anything:

```bash
python eval_jetrover.py --checkpoint <ckpt> --out-json /tmp/m.json \
  --arm_backend mock --policy_backend zmq --policy_host <gpu-box> --num_episodes 1 --scorer none
```

Then the real thing via `odyssey run` (or the script directly with
`--arm_backend ros2`). After each episode the operator answers
`Episode N success? [y/n]`; `--auto_timeout <sec>` scores unanswered prompts as
failures so unattended runs terminate honestly. `--scorer none` skips scoring
and reports metrics only.

Backends:

- `ros2` — publishes `trajectory_msgs/JointTrajectory` to the JetRover arm
  controller and reads `/joint_states` + the depth cam RGB topic. Topic names at
  the top of `Ros2Arm` match Hiwonder's shipped packages; verify with
  `ros2 topic list` and adjust for your image.
- `hiwonder` — direct bus-servo control via the on-board
  `ros_robot_controller_sdk` (state-only: no camera frames, the policy sees
  zeros — prefer `ros2` for visuomotor policies).
- `mock` — deterministic, dependency-free; used by CI and for wire checks.

## Safety on the real arm

- Clear the workspace and keep the e-stop (or the power switch) at hand.
- `--max_joint_delta` (default 0.15 rad/step) rate-limits every commanded step;
  `--control_hz` (default 5) bounds the step rate. Start conservative.
- First run: `--num_episodes 1 --max_steps 50` and watch the arm.
