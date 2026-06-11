"""LLMPlanner — Gemma 4B int4 task decomposition.

Loads a quantized language model (e.g. ``google/gemma-3-4b-it`` at
int4) and decomposes a high-level task instruction into ordered
sub-instructions the pilot executes sequentially.

Satisfies ``PlannerRuntime`` protocol.

VRAM footprint: ~2.5 GB for Gemma 4B int4 via bitsandbytes,
leaving ~20 GB for the VLA on a 24 GB card.

All heavy imports (torch, transformers, bitsandbytes) are deferred.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a robot task planner. Given a high-level task instruction, "
    "decompose it into a numbered list of simple, sequential sub-instructions "
    "that a robot arm can execute one at a time. Each sub-instruction should "
    "describe a single atomic motion or action. Output ONLY the numbered list, "
    "nothing else."
)

_NUMBERED_LINE = re.compile(r"^\s*\d+[\.\)]\s*(.+)$")


def _parse_plan(text: str) -> list[str]:
    """Extract numbered sub-instructions from LLM output."""
    lines = []
    for line in text.strip().splitlines():
        m = _NUMBERED_LINE.match(line)
        if m:
            lines.append(m.group(1).strip())
    return lines


class LLMPlanner:
    """Task planner using a quantized language model.

    Parameters
    ----------
    model_name:
        HuggingFace model ID. Default: ``google/gemma-3-4b-it``.
    quantization:
        Quantization config string. ``"int4"`` uses bitsandbytes
        4-bit. ``None`` loads in bfloat16 (needs more VRAM).
    device:
        Torch device. Defaults to CUDA if available.
    max_new_tokens:
        Max tokens for the plan generation.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        *,
        quantization: str | None = "int4",
        device: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise NotImplementedError(
                "LLMPlanner requires transformers + torch. "
                "Install with: pip install transformers torch"
            ) from e

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._max_new_tokens = max_new_tokens

        quant_config: Any = None
        if quantization == "int4":
            try:
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                logger.warning(
                    "bitsandbytes not available, loading %s without quantization",
                    model_name,
                )

        logger.info(
            "LLMPlanner: loading %s (quantization=%s)", model_name, quantization
        )

        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        )
        if quant_config is None:
            self._model = self._model.to(self._device)

        logger.info("LLMPlanner ready (%s)", model_name)

    def plan(self, task_instruction: str) -> list[str]:
        """Decompose a task instruction into sub-steps."""
        import torch

        messages = [
            {"role": "user", "content": f"{_SYSTEM_PROMPT}\n\nTask: {task_instruction}"},
        ]

        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        logger.debug("LLMPlanner raw output:\n%s", text)

        steps = _parse_plan(text)
        if not steps:
            logger.warning(
                "LLMPlanner produced no parseable steps for %r, "
                "falling back to single-step",
                task_instruction,
            )
            return [task_instruction]
        return steps
