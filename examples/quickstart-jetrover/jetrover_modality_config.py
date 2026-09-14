# GR00T NEW_EMBODIMENT modality config for the Hiwonder JetRover arm
# (marketed "6DOF": 5 positional joints + gripper).
#
# HOW TO USE: copy this file into your Isaac-GR00T checkout as
#   <Isaac-GR00T>/examples/JETROVER/jetrover_config.py
# and reference it from the mission as
#   modality_config_path: examples/JETROVER/jetrover_config.py
# (relative paths resolve against $ISAAC_GR00T_REPO_PATH — see
# src/odyssey/runners/models/gr00t_train.py).
#
# It is modeled 1:1 on the upstream examples/SO100/so100_config.py (the other
# single-arm + gripper NEW_EMBODIMENT example). Upstream transform/import names
# drift between GR00T releases — before training, diff this file against the
# so100 config in YOUR checkout and align the imports/class shape if they differ.
#
# The matching LeRobot dataset must carry, per frame (see this example's README):
#   state.single_arm  (5,)  arm joint positions, rad
#   state.gripper     (1,)  gripper position
#   action.single_arm (5,)  target joint positions, rad
#   action.gripper    (1,)  target gripper position
#   video.front       front RGB camera (the depth cam's RGB stream)
#   annotation.human.task_description
# declared in the dataset's meta/modality.json.
#
# DoF note: Hiwonder markets the JetRover arm as "6DOF" *counting the gripper* —
# the vendor docs enumerate servo IDs 1-5 for the arm (pan-tilt base + 3 body
# joints + wrist) and ID 10 for the gripper. Hence 5-D single_arm here. If your
# unit differs, adjust the dims consistently here, in the dataset, and via the
# eval script's --arm_dof flag.

from typing import ClassVar

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import VideoColorJitter, VideoCrop, VideoResize, VideoToTensor
from gr00t.experiment.data.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform


class JetroverDataConfig(BaseDataConfig):
    """JetRover arm (5 joints) + gripper, front camera — GR00T NEW_EMBODIMENT."""

    video_keys: ClassVar[list[str]] = ["video.front"]
    state_keys: ClassVar[list[str]] = ["state.single_arm", "state.gripper"]
    action_keys: ClassVar[list[str]] = ["action.single_arm", "action.gripper"]
    language_keys: ClassVar[list[str]] = ["annotation.human.task_description"]

    observation_indices: ClassVar[list[int]] = [0]
    # 16-step action chunks, the GR00T default
    action_indices: ClassVar[list[int]] = list(range(16))

    def modality_config(self) -> dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self) -> ComposedModalityTransform:
        transforms = [
            # video
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            # state: normalize to the dataset's min/max stats
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            ),
            # concat + model-side packing
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


DATA_CONFIG = JetroverDataConfig()
