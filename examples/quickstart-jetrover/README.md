# JetRover quickstart — GR00T on the Hiwonder arm (real hardware)

Fine-tune GR00T N1.7 as a `NEW_EMBODIMENT` on teleop demos captured on the
[Hiwonder JetRover](https://www.hiwonder.com/products/jetrover)'s arm, then
evaluate closed-loop **on the physical arm**.

> **DoF note:** Hiwonder markets the arm as "6DOF" *counting the gripper*. Per
> the [vendor docs](https://docs.hiwonder.com/projects/JetRover/en/jetson-orin-nano/docs/7.Robot_Arm_Control_Course.html),
> servo IDs 1–5 position the arm (pan-tilt base + 3 body joints + wrist) and
> ID 10 is the gripper — a **5-joint kinematic chain + 1 gripper**, i.e. 6-dim
> actions. Everything in this example uses 5 + 1; if your unit differs, adjust
> the modality config, the dataset, and the eval script's `--arm_dof` together.

The built-in eval runners
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
| `jetrover_modality_config.py` | GR00T modality config (5-D arm + 1-D gripper, front camera) — copy into your Isaac-GR00T checkout |
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
| `state.single_arm` | (5,) | arm joint positions, rad (servos 1–5) |
| `state.gripper` | (1,) | gripper position, rad (servo 10) |
| `action.single_arm` | (5,) | target joint positions, rad |
| `action.gripper` | (1,) | target gripper position, rad |
| `video.front` | HxWx3 | the depth cam's RGB stream |
| `annotation.human.task_description` | str | the language instruction |

Record joint values in the same rad convention the stock controller publishes on
`/controller_manager/joint_states` — that is what the eval feeds the policy.

GR00T additionally needs a `meta/modality.json` describing these keys — the
upstream [`examples/SO100/modality.json`](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/examples/SO100/modality.json)
is the same 5 + 1 layout (drop its `wrist` camera). Copy the finished dataset to
the GPU box, e.g. `/data/jetrover_pick_v0/jetrover_pick`.

## 2. Drop in the modality config

```bash
mkdir -p $ISAAC_GR00T_REPO_PATH/examples/JETROVER
cp jetrover_modality_config.py $ISAAC_GR00T_REPO_PATH/examples/JETROVER/jetrover_config.py
```

It targets the GR00T **N1.7** API (`gr00t.data.types.ModalityConfig` +
`register_modality_config(..., NEW_EMBODIMENT)`), mirroring upstream
`examples/SO100/so100_config.py`. `launch_finetune.py` imports the file for that
registration side effect. The arm is learned as RELATIVE actions and the
gripper as ABSOLUTE (as in SO100); the server converts back to absolute
targets, so the eval client is unaffected.

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
cd $ISAAC_GR00T_REPO_PATH
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path <task-output-dir-checkpoint> \
  --embodiment-tag NEW_EMBODIMENT \
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
failures so unattended runs terminate honestly. All prompts share one stdin
reader, so an answer typed after a prompt expired is discarded rather than
counted for the next episode. `--scorer none` skips scoring and reports metrics
only.

Backends:

- `ros2` — talks to the stock JetRover `controller_manager` (Hiwonder's
  `servo_controller` package): publishes `servo_controller_msgs/ServosPosition`
  (`position_unit: rad`, servo IDs 1–5 + 10) on `/servo_controller` and reads
  `/controller_manager/joint_states` + the depth cam RGB topic. The controller
  applies each joint's calibration and limits. Topic names at the top of
  `Ros2Arm`; verify with `ros2 topic list` if your image namespaces them.
- `hiwonder` — direct bus-servo control via the on-board
  `ros_robot_controller_sdk` (state-only: no camera frames, the policy sees
  zeros — prefer `ros2` for visuomotor policies). Don't run it while the
  `controller_manager` owns the bus.
- `mock` — deterministic, dependency-free; used by CI and for wire checks.

## Safety on the real arm

- Clear the workspace and keep the e-stop (or the power switch) at hand.
- `--max_joint_delta` (default 0.15 rad/step, max 0.5) rate-limits every
  policy step; `--control_hz` (default 5) bounds the step rate and is also the
  duration each move is given. Start conservative. Only the joint *rate* is
  bounded here: gripper commands pass through (open/close must stay crisp), and
  absolute joint ranges are enforced by the vendor controller's per-joint
  limits (`ros2`) or the servos' ±120° pulse range (`hiwonder`). Non-finite
  policy actions abort the run.
- **Episode reset never jumps.** `--reset_mode bounded` (default) walks the arm
  from its *measured* pose to the servo centres at `--home_max_joint_delta`
  (default 0.05 rad/step, max 0.2, every axis incl. the gripper) and aborts if
  it isn't within `--home_tolerance` after `--home_max_steps` — e.g. a stalled
  servo or an obstruction. `--reset_mode operator` sends no reset motion: the
  operator places the arm and types `ready` (a bare Enter or a stray `y` is not
  accepted); the measured start pose is logged.
- If an episode aborts (homing failure, bad action), the metrics for the
  episodes already completed are still written, with `metrics.aborted` set,
  and the script exits non-zero.
- First run: `--num_episodes 1 --max_steps 50` and watch the arm.
