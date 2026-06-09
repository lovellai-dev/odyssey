"""OpenVLA inference policy for robosuite evaluation.

Loads a LoRA-finetuned OpenVLA checkpoint via HuggingFace transformers + peft,
and returns a ``Policy`` callable that maps robosuite observation dicts to
7-DoF actions using the model's built-in ``predict_action()`` method.

All heavy imports (transformers, peft, torch, PIL) are deferred so the module
can be imported in test environments without GPU dependencies.
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
    # Fall back to any key ending with "_image"
    for key in obs:
        if key.endswith("_image"):
            return key
    raise KeyError(
        f"No image key found in observation dict. "
        f"Looked for {preferred!r} and any key ending with '_image'. "
        f"Available keys: {sorted(obs.keys())}"
    )


def make_openvla_policy(
    checkpoint_path: Path,
    *,
    config: dict[str, Any] | None = None,
    benchmark_name: str = "Lift",
) -> Any:
    """Build an OpenVLA inference policy from a LoRA checkpoint.

    Returns a callable ``policy(obs_dict) -> action_array`` suitable for
    use as a ``Policy`` in ``RobosuiteRunner``.
    """
    try:
        import torch
        from peft import PeftModel
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore[attr-defined]
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

    checkpoint_path = Path(checkpoint_path)
    base_model_name = _resolve_base_model(checkpoint_path)

    logger.info("Loading OpenVLA base model: %s", base_model_name)
    processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
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

    # merge_and_unload() for faster inference — but only if the merged model
    # retains the custom predict_action method.
    if hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
        if hasattr(merged, "predict_action"):
            model = merged
            logger.info("LoRA merged and unloaded for faster inference")
        else:
            logger.info(
                "Skipping merge_and_unload — predict_action not on merged model"
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

        # Convert numpy HWC uint8 → PIL Image
        if not isinstance(img_array, Image.Image):
            img_array = Image.fromarray(img_array.astype("uint8"), "RGB")

        action = model.predict_action(
            processor,
            img_array,
            task_instruction,
            unnorm_key=unnorm_key,
            do_sample=False,
        )
        return np.array(action, dtype=np.float64)

    return policy
