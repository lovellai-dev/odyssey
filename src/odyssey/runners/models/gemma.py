"""Gemma text generation model loader.

Loads a quantized causal LM (e.g. ``google/gemma-2b-it`` at int4)
and exposes a simple ``generate(messages) -> str`` interface. The
planning logic lives in ``runners/agents/planner.py`` — this module
only handles model loading and inference.

Model choice is constrained by OpenVLA: the PILOT (OpenVLA) pins
``transformers==4.40.1``, and the SPECIALIST runs in the same process,
so only **Gemma 1** (``gemma-2b-it`` / ``gemma-7b-it``, supported since
transformers 4.38) is compatible. Gemma 2 needs 4.42+ and Gemma 3
needs 4.49+, both of which break OpenVLA.

VRAM footprint: ~1.5 GB for Gemma 2B int4 via bitsandbytes,
leaving ~22 GB for the VLA on a 24 GB card.

All heavy imports (torch, transformers, bitsandbytes) are deferred.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GemmaTextGenerator:
    """Loads a causal LM and generates text from chat messages.

    Satisfies ``TextGenerator`` protocol.

    Parameters
    ----------
    model_name:
        HuggingFace model ID. Default: ``google/gemma-2b-it`` (Gemma 1,
        the only family compatible with OpenVLA's transformers 4.40.1 pin).
    quantization:
        Quantization config string. ``"int4"`` uses bitsandbytes
        4-bit. ``None`` loads in bfloat16 (needs more VRAM).
    device:
        Torch device. Defaults to CUDA if available.
    max_new_tokens:
        Max tokens for generation.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2b-it",
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
                "GemmaTextGenerator requires transformers + torch. "
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
            "GemmaTextGenerator: loading %s (quantization=%s)",
            model_name,
            quantization,
        )

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if quant_config is not None:
            # bitsandbytes 4-bit: let the quantization config own dtype/placement.
            # Passing torch_dtype / low_cpu_mem_usage alongside device_map makes
            # transformers' dispatch_model call model.to(), which bnb forbids
            # ("`.to` is not supported for 4-bit models").
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["low_cpu_mem_usage"] = True

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, **load_kwargs
        )
        if quant_config is None:
            self._model = self._model.to(self._device)

        logger.info("GemmaTextGenerator ready (%s)", model_name)

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate text from a list of chat messages.

        Parameters
        ----------
        messages:
            Chat messages in the format ``[{"role": "user", "content": "..."}]``.

        Returns
        -------
        Generated text (decoded, special tokens stripped).
        """
        import torch

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
        return str(self._tokenizer.decode(generated, skip_special_tokens=True))
