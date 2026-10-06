# GR00T N1.7 NEW_EMBODIMENT modality config for the Hiwonder JetRover arm
# (marketed "6DOF": 5 positional joints + gripper).
#
# HOW TO USE: copy this file into your Isaac-GR00T checkout as
#   <Isaac-GR00T>/examples/JETROVER/jetrover_config.py
# and reference it from the mission as
#   modality_config_path: examples/JETROVER/jetrover_config.py
# (relative paths resolve against $ISAAC_GR00T_REPO_PATH — see
# src/odyssey/runners/models/gr00t_train.py).
#
# N1.7's launch_finetune.py *imports* this file for its side effect: the
# register_modality_config() call at the bottom is what makes NEW_EMBODIMENT
# resolvable to these keys. A module that only defines a config object is never
# picked up. Modeled 1:1 on upstream examples/SO100/so100_config.py at
# NVIDIA/Isaac-GR00T@51d4c89 (the single-arm + gripper NEW_EMBODIMENT example).
# Serving needs no copy of it: like upstream's SO100 recipe, run_gr00t_server
# takes only --model-path and --embodiment-tag NEW_EMBODIMENT.
#
# Keys are the meta/modality.json group names (no "state." / "video." prefix)
# and must match both the dataset and the eval client's wire keys
# (eval_jetrover.py: VIDEO_KEY="front", state "single_arm" + "gripper"):
#   state  : single_arm (5) arm joint positions, rad   gripper (1)
#   action : single_arm (5) target joint positions, rad gripper (1)
#   video  : front — the depth cam's RGB stream
#   language: annotation.human.task_description
#
# Action representation follows SO100: the arm is learned RELATIVE to the
# current state (N1.7's launcher sets use_relative_action=True) and the gripper
# ABSOLUTE. The policy server decodes relative chunks back to absolute targets
# with the observed state, so the eval client always receives absolute joint
# positions — which is what its rate-limit clamp assumes.
#
# DoF note: Hiwonder markets the JetRover arm as "6DOF" *counting the gripper* —
# the vendor docs enumerate servo IDs 1-5 for the arm (pan-tilt base + 3 body
# joints + wrist) and ID 10 for the gripper. Hence 5-D single_arm. If your unit
# differs, adjust the dims consistently in the dataset's modality.json and via
# the eval script's --arm_dof flag.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

jetrover_config = {
    # Video: current frame only
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["front"],
    ),
    # State: current proprioceptive reading
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    # Action: 16-step chunk (the GR00T default); one ActionConfig per key
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(jetrover_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
