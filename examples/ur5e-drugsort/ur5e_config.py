# GR00T modality configuration for the UR5e / Robotiq 2F-85 drug-sorting cell.
#
# Passed to `gr00t.experiment.launch_finetune` via `--modality-config-path`
# (see examples/ur5e-drugsort/mission.yaml and RUNBOOK.md). It registers the
# obs/action modality layout for the custom NEW_EMBODIMENT slot; the per-
# dimension sizes come from the dataset's `meta/modality.json`
# (single_arm=6, gripper=1 -> 7-D state & action), so this file only names the
# modality KEYS and declares each action's representation.
#
# Layout mirrors the AseptiPack cell recorded by
# `scripts/gen_ur5e_drugsort_demos.py` and the Lovell AI Robot Playground deploy
# contract (setTargets(q, grip) + TrueSensorRenderer primary view):
#   * state  : single_arm (6 UR5e joint positions) + gripper (closure in [0,1])
#   * action : single_arm (6 joint-position targets) + gripper (closure in [0,1])
#   * video  : exterior (the Playground primary camera-mount view, 256x256)
#   * language: annotation.human.task_description
#
# This is the odyssey source-of-truth copy. `launch_finetune` importlib-loads
# the file by path, so point `--modality-config-path` straight at it (absolute
# path), or copy it into the Isaac-GR00T checkout as
# `examples/UR5e_DrugSort/ur5e_config.py` and use a repo-relative path (the
# odyssey GR00T runner resolves relative paths against $ISAAC_GR00T_REPO_PATH).

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

ur5e_drugsort_config = {
    # Video: current frame only; keys must match "video" entries in meta/modality.json.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["exterior"],  # single Playground primary camera-mount view
    ),
    # State: current proprioceptive reading; keys must match "state" entries.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "single_arm",  # 6 UR5e joint positions (rad)
            "gripper",     # normalised Robotiq closure in [0, 1]
        ],
    ),
    # Action: 16-step prediction horizon; one ActionConfig per modality key.
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),  # predict 16 future steps
        modality_keys=[
            "single_arm",
            "gripper",
        ],
        action_configs=[
            # single_arm: ABSOLUTE = predict the absolute joint-position target
            # directly (NOT a delta from the current state). The earlier RELATIVE
            # rep integrated its deltas on top of whatever pose the arm was in, so
            # in closed loop a small error compounded — the base joint drifted
            # monotonically past its training range to ~2π (the "shoulder_pan
            # runaway" that failed the first eval 0/20). Predicting absolute
            # targets lets the policy pull the arm back onto the manifold from an
            # off-distribution pose instead of amplifying the drift. The recorded
            # action IS already the absolute joint target (rad), so this needs no
            # dataset change — only how the fine-tune interprets it. The served
            # policy still returns absolute targets, so eval + the Playground
            # setTargets bridge apply them unchanged.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,  # joint-space, not end-effector
                format=ActionFormat.DEFAULT,
            ),
            # gripper: ABSOLUTE = target closure (binary open/close is cleaner absolute).
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction from the annotation field in the dataset.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

# Always register custom embodiments under NEW_EMBODIMENT.
register_modality_config(ur5e_drugsort_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
