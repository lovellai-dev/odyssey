"""OpenVLA inference policy and runtime.

Inference policy
----------------
``make_openvla_policy()`` loads a LoRA-finetuned checkpoint via
HuggingFace transformers + peft and returns a ``Policy`` callable that
maps robosuite observation dicts to 7-DoF actions using the model's
built-in ``predict_action()`` method.

All heavy inference imports (transformers, peft, torch, PIL) are deferred
so the module can be imported in environments without GPU dependencies.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Default natural-language instructions per Robosuite benchmark.
_DEFAULT_INSTRUCTIONS: dict[str, str] = {
    "Lift": "pick up the red cube",
    "Stack": "stack the red cube on top of the green cube",
    "NutAssembly": "pick up the nut and place it on the peg",
    "NutAssemblySquare": "pick up the square nut and place it on the square peg",
    "NutAssemblyRound": "pick up the round nut and place it on the round peg",
    "PickPlace": "pick up the object and place it in the bin",
    "Door": "open the door",
    "Wipe": "wipe the table",
    "ToolHang": "hang the tool on the rack",
    "TwoArmLift": "lift the pot together",
}


def _resolve_base_model(checkpoint_path: Path) -> str:
    """Read ``adapter_config.json`` to find the base model name."""
    config_file = checkpoint_path / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in {checkpoint_path}. "
            "Expected a peft LoRA checkpoint directory."
        )
    with open(config_file) as f:
        config = json.load(f)
    base = config.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            f"adapter_config.json in {checkpoint_path} is missing "
            "'base_model_name_or_path' key."
        )
    return str(base)


def _find_image_key(obs: dict[str, Any], preferred: str) -> str:
    """Find a camera image key in the observation dict."""
    if preferred in obs:
        return preferred
    for key in obs:
        if key.endswith("_image"):
            return key
    raise KeyError(
        f"No image key found in observation dict. "
        f"Looked for {preferred!r} and any key ending with '_image'. "
        f"Available keys: {sorted(obs.keys())}"
    )


def _is_lora_checkpoint(checkpoint_path: Path) -> bool:
    """Check whether the checkpoint is a LoRA adapter or a full model."""
    return (checkpoint_path / "adapter_config.json").is_file()


def _center_crop_image(image: Any, crop_scale: float = 0.9) -> Any:
    """Center-crop to ``crop_scale`` of the frame, then resize back to the original size.

    OpenVLA checkpoints fine-tuned with image augmentation (``image_aug=True`` — e.g. the
    published ``openvla-7b-finetuned-libero-*``) were trained on a RandomResizedCrop, so
    inference must apply the matching deterministic center crop (mirrors ``crop_and_resize``
    in OpenVLA's ``run_libero_eval.py``). Skipping it widens the input FOV vs. training and
    degrades spatial precision — the arm approaches the target but misses the grasp.
    """
    import math

    from PIL import Image

    w, h = image.size
    side = math.sqrt(crop_scale)
    cw, ch = round(w * side), round(h * side)
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = image.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.Resampling.BILINEAR)


def make_openvla_policy(
    checkpoint_path: Path,
    *,
    config: dict[str, Any] | None = None,
    benchmark_name: str = "Lift",
) -> Any:
    """Build an OpenVLA inference policy from a checkpoint.

    Supports both LoRA adapter checkpoints (with ``adapter_config.json``)
    and full merged model checkpoints (with ``config.json`` and safetensors).

    Returns a callable ``policy(obs_dict) -> action_array`` suitable for
    use as a ``Policy`` in ``RobosuiteRunner``.
    """
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError as e:
        raise NotImplementedError(
            "OpenVLA inference policy requires the 'openvla' extra. "
            "Install with: pip install 'lovell-odyssey[openvla]'"
        ) from e

    cfg = config or {}
    unnorm_key = cfg.get("unnorm_key", "bridge_orig")
    task_instruction = cfg.get("task_instruction") or _DEFAULT_INSTRUCTIONS.get(
        benchmark_name, "complete the task"
    )
    image_key = cfg.get("image_key", "agentview_image")
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    # OpenVLA checkpoints trained with image_aug expect a matching center crop at eval
    # (e.g. the finetuned-libero-* checkpoints). Default on; disable via config.
    center_crop = bool(cfg.get("center_crop", True))
    crop_scale = float(cfg.get("crop_scale", 0.9))

    checkpoint_path = Path(checkpoint_path)
    is_lora = _is_lora_checkpoint(checkpoint_path)

    if is_lora:
        from peft import PeftModel

        base_model_name = _resolve_base_model(checkpoint_path)
        logger.info("Loading OpenVLA base model: %s", base_model_name)
        processor = AutoProcessor.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info("Applying LoRA adapter from: %s", checkpoint_path)
        model = PeftModel.from_pretrained(model, str(checkpoint_path))

        if hasattr(model, "merge_and_unload"):
            merged = model.merge_and_unload()
            if hasattr(merged, "predict_action"):
                model = merged
                logger.info("LoRA merged and unloaded for faster inference")
            else:
                logger.info(
                    "Skipping merge_and_unload — predict_action not on merged model"
                )
    else:
        logger.info("Loading full merged model from: %s", checkpoint_path)
        processor = AutoProcessor.from_pretrained(
            str(checkpoint_path), trust_remote_code=True
        )
        model = AutoModelForVision2Seq.from_pretrained(
            str(checkpoint_path),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    model = model.to(device)

    logger.info(
        "OpenVLA policy ready — instruction=%r, unnorm_key=%r, image_key=%r",
        task_instruction,
        unnorm_key,
        image_key,
    )

    def policy(obs: dict[str, Any]) -> Any:
        import numpy as np

        key = _find_image_key(obs, image_key)
        img_array = obs[key]

        if not isinstance(img_array, Image.Image):
            img_array = Image.fromarray(img_array.astype("uint8"), "RGB")

        if center_crop:
            img_array = _center_crop_image(img_array, crop_scale)

        inputs = processor(task_instruction, img_array).to(device, dtype=torch.bfloat16)
        action = model.predict_action(
            inputs["input_ids"],
            unnorm_key=unnorm_key,
            do_sample=False,
        )
        return np.array(action, dtype=np.float64)

    return policy


class VLARuntime:
    """OpenVLA pilot runtime with per-call instruction.

    Unlike ``make_openvla_policy()`` which bakes ``task_instruction`` at
    creation time, ``VLARuntime.act()`` accepts the instruction on every
    call. This lets ``PlannedEvalRuntime`` feed different sub-instructions
    per phase without reloading the model.

    Satisfies ``PilotRuntime`` protocol.

    Parameters
    ----------
    checkpoint_path:
        Path to a LoRA adapter dir or full merged model dir.
    unnorm_key:
        Unnormalization key passed to ``predict_action``.
    device:
        Torch device string. Defaults to CUDA if available.
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        *,
        unnorm_key: str = "bridge_orig",
        device: str | None = None,
        center_crop: bool = True,
        crop_scale: float = 0.9,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as e:
            raise NotImplementedError(
                "VLARuntime requires the 'openvla' extra. "
                "Install with: pip install 'lovell-odyssey[openvla]'"
            ) from e

        self._checkpoint_path = Path(checkpoint_path)
        self._unnorm_key = unnorm_key
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._center_crop = center_crop
        self._crop_scale = crop_scale

        is_lora = _is_lora_checkpoint(self._checkpoint_path)

        if is_lora:
            from peft import PeftModel

            base_name = _resolve_base_model(self._checkpoint_path)
            logger.info("VLARuntime: loading base model %s", base_name)
            self._processor = AutoProcessor.from_pretrained(
                base_name, trust_remote_code=True
            )
            model = AutoModelForVision2Seq.from_pretrained(
                base_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            logger.info("VLARuntime: applying LoRA adapter from %s", self._checkpoint_path)
            model = PeftModel.from_pretrained(model, str(self._checkpoint_path))
            if hasattr(model, "merge_and_unload"):
                merged = model.merge_and_unload()
                if hasattr(merged, "predict_action"):
                    model = merged
                    logger.info("LoRA merged for faster inference")
        else:
            logger.info("VLARuntime: loading merged model from %s", self._checkpoint_path)
            self._processor = AutoProcessor.from_pretrained(
                str(self._checkpoint_path), trust_remote_code=True
            )
            model = AutoModelForVision2Seq.from_pretrained(
                str(self._checkpoint_path),
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

        self._model = model.to(self._device)
        logger.info("VLARuntime ready on %s", self._device)

    def act(
        self,
        image: Any,
        instruction: str,
    ) -> Any:
        """Produce a 7-DoF action from image + instruction."""
        import numpy as np
        import torch
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8), "RGB")

        if self._center_crop:
            image = _center_crop_image(image, self._crop_scale)

        inputs = self._processor(instruction, image).to(
            self._device, dtype=torch.bfloat16
        )
        action = self._model.predict_action(
            inputs["input_ids"],
            unnorm_key=self._unnorm_key,
            do_sample=False,
        )
        return np.array(action, dtype=np.float64)
