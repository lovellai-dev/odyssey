"""VLARuntime — wraps OpenVLA for per-call instruction inference.

Unlike ``make_openvla_policy()`` which bakes ``task_instruction`` at
creation time (the policy closure captures it), ``VLARuntime.act()``
accepts the instruction on every call. This lets the planned-eval
runtime feed different sub-instructions per phase without reloading
the model.

All heavy imports (torch, transformers, peft, PIL) are deferred so
the module can be imported without GPU dependencies.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _is_lora_checkpoint(checkpoint_path: Path) -> bool:
    return (checkpoint_path / "adapter_config.json").is_file()


def _resolve_base_model(checkpoint_path: Path) -> str:
    config_file = checkpoint_path / "adapter_config.json"
    with open(config_file) as f:
        config = json.load(f)
    base = config.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            f"adapter_config.json in {checkpoint_path} is missing "
            "'base_model_name_or_path'."
        )
    return str(base)


class VLARuntime:
    """OpenVLA pilot runtime with per-call instruction.

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
    ) -> NDArray[np.floating[Any]]:
        """Produce a 7-DoF action from image + instruction."""
        from PIL import Image

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8), "RGB")

        action = self._model.predict_action(
            image,
            instruction,
            processor=self._processor,
            unnorm_key=self._unnorm_key,
            do_sample=False,
        )
        return np.array(action, dtype=np.float64)
