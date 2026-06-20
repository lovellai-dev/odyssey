"""Gemma 4 multimodal (vision-language) text generation loader.

Loads a multimodal Gemma 4 checkpoint (``google/gemma-4-E2B-it`` /
``E4B-it``) and exposes ``generate(messages, image=None) -> str``. Gemma 4
vision-language models load through ``AutoModelForMultimodalLM`` +
``AutoProcessor`` (rather than the text-only ``AutoModelForCausalLM`` +
``AutoTokenizer``) and consume chat messages whose ``content`` is a list of
typed blocks (image / text). This is the only SPECIALIST planner loader.

We use Gemma 4 (Apache-2.0, no gated license) rather than Gemma 3: Gemma 3
4B produces NaN logits under int4 bitsandbytes on this stack (verified for
both eager and sdpa attention, text-only and with-image), so it cannot run
quantized here. Gemma 4 loads cleanly in int4 and grounds plans in the scene
image.

Only viable **out of process** (in the specialist venv): Gemma 4 needs a
modern ``transformers`` (plus ``torchvision`` for its image processor),
incompatible with OpenVLA's pinned ``transformers==4.40.1`` in the main
venv. The ``planner_server`` launches this in that separate venv.

VRAM footprint: ~9.3 GB for Gemma 4 E4B-it int4 (bf16 compute) via
bitsandbytes — alongside the ~14 GB bf16 OpenVLA pilot that is ~23 GB peak
on a 24 GB card. Tight but fits; drop to E2B-it for more headroom.

All heavy imports (torch, transformers, bitsandbytes) are deferred.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _multimodal_model_class() -> Any:
    """Resolve Gemma 4's multimodal auto-class (name has drifted across versions)."""
    import transformers

    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise NotImplementedError(
        "No multimodal auto-class (AutoModelForMultimodalLM / "
        "AutoModelForImageTextToText) in this transformers — it is too old for "
        "Gemma 4. Upgrade in the specialist venv: pip install -U transformers"
    )


def _to_pil(image: Any) -> Any:
    """Normalize a PIL Image or HWC uint8 ndarray to a PIL Image."""
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    # numpy HWC uint8 (the robosuite frame format)
    return Image.fromarray(image).convert("RGB")


class GemmaVLMGenerator:
    """Loads a multimodal Gemma 4 model and generates text from chat messages.

    Satisfies the ``TextGenerator`` protocol, plus accepts an optional
    ``image`` on ``generate`` so the planner can ground its plan in the
    scene. With ``image=None`` it degrades to text-only generation.

    Parameters
    ----------
    model_name:
        HuggingFace model ID. Default ``google/gemma-4-E4B-it`` — the
        multimodal E4B variant (Apache-2.0), the sweet spot for a 24 GB
        card shared with the OpenVLA pilot. Use ``google/gemma-4-E2B-it``
        for more VRAM headroom.
    quantization:
        ``"int4"`` uses bitsandbytes 4-bit (nf4). ``None`` loads in
        bfloat16 (needs more VRAM).
    device:
        Torch device. Defaults to CUDA if available. Ignored for 4-bit
        (bitsandbytes owns placement via ``device_map="auto"``).
    max_new_tokens:
        Max tokens for generation.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-4-E4B-it",
        *,
        quantization: str | None = "int4",
        device: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as e:
            raise NotImplementedError(
                "GemmaVLMGenerator requires a modern transformers + torch + "
                "torchvision. Install the specialist extra: pip install '.[specialist]'"
            ) from e

        model_cls = _multimodal_model_class()

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._max_new_tokens = max_new_tokens

        quant_config: Any = None
        if quantization == "int4":
            try:
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except ImportError:
                logger.warning(
                    "bitsandbytes not available, loading %s without quantization",
                    model_name,
                )

        logger.info(
            "GemmaVLMGenerator: loading %s (quantization=%s)",
            model_name,
            quantization,
        )

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            logger.info(
                "GemmaVLMGenerator: GPU free %.2f GB / total %.2f GB before load",
                free / 1e9,
                total / 1e9,
            )

        load_kwargs: dict[str, Any] = {}
        if quant_config is not None:
            # bitsandbytes 4-bit: let the quantization config own dtype/placement.
            # Passing torch_dtype alongside device_map makes transformers call
            # model.to(), which bnb forbids for 4-bit.
            load_kwargs["quantization_config"] = quant_config
            # Force the whole model onto GPU 0 instead of device_map="auto".
            # When the PILOT already occupies most of the shared GPU, "auto"
            # conservatively reserves a buffer, decides the planner doesn't fit,
            # and offloads some modules to CPU — which bitsandbytes rejects for
            # 4-bit ("Some modules are dispatched on the CPU or the disk").
            # Pinning to {"": 0} uses all remaining VRAM (it OOMs cleanly if the
            # model genuinely doesn't fit, instead of the confusing offload error).
            load_kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else "auto"
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["low_cpu_mem_usage"] = True

        # padding_side="left" is required for correct generation with the
        # processor (decoder-only models pad on the left).
        self._processor = AutoProcessor.from_pretrained(
            model_name, padding_side="left"
        )
        self._model = model_cls.from_pretrained(model_name, **load_kwargs)
        if quant_config is None:
            self._model = self._model.to(self._device)
        self._model.eval()

        logger.info("GemmaVLMGenerator ready (%s)", model_name)

    def generate(
        self,
        messages: list[dict[str, Any]],
        image: Any | None = None,
    ) -> str:
        """Generate text from chat messages, optionally grounded in an image.

        Parameters
        ----------
        messages:
            Chat messages. ``content`` may be a plain string (text-only) or
            a list of typed blocks (``{"type": "text", ...}`` /
            ``{"type": "image", ...}``). Plain-string contents are wrapped
            into a single text block.
        image:
            Optional PIL Image or HWC uint8 ndarray. When provided, it is
            inserted as an image block on the last user message — the
            processor adds the ``<start_of_image>`` token automatically.

        Returns
        -------
        Generated text (decoded, prompt stripped, special tokens removed).
        """
        import torch

        chat = self._build_chat(messages, image)

        inputs = self._processor.apply_chat_template(
            chat,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        generated = outputs[0][input_len:]
        return str(self._processor.decode(generated, skip_special_tokens=True))

    def _build_chat(
        self,
        messages: list[dict[str, Any]],
        image: Any | None,
    ) -> list[dict[str, Any]]:
        """Normalize messages to block-content form and attach the image.

        Text-only ``content`` strings are wrapped into ``[{"type": "text"}]``.
        The image (if any) is prepended to the last user message's blocks.
        """
        pil = _to_pil(image) if image is not None else None
        chat: list[dict[str, Any]] = []
        last_user_idx = -1
        for msg in messages:
            content = msg["content"]
            blocks = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else list(content)
            )
            chat.append({"role": msg["role"], "content": blocks})
            if msg["role"] == "user":
                last_user_idx = len(chat) - 1

        if pil is not None and last_user_idx >= 0:
            chat[last_user_idx]["content"].insert(0, {"type": "image", "image": pil})
        return chat
