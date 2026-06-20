"""Diagnostic: verify Gemma 4 E4B-it yields valid (non-NaN) multimodal logits.

We are pivoting the SPECIALIST from Gemma 3 4B (gated, and NaN under int4
bitsandbytes on this stack) to Gemma 4 E4B-it (Apache-2, newer arch). Before
touching the production generator, confirm Gemma 4 loads and produces finite
logits + real text here.

Gemma 4 loads via ``AutoModelForMultimodalLM`` + ``AutoProcessor`` and uses the
same ``apply_chat_template`` block-content message format as Gemma 3, so the
production change is just the model class.

Modes (argv[1]):
    int4-bf16   — bitsandbytes 4-bit, compute bfloat16 (target for 24 GB)
    int4-fp16   — bitsandbytes 4-bit, compute float16
    bf16        — no quantization (correctness reference)

REQUIRES the specialist venv with a Gemma-4-capable transformers:
    ~/specialist-venv/bin/pip install -U transformers accelerate
Then:
    ~/specialist-venv/bin/python tests/manual/debug_multimodal_raw.py int4-bf16
    ~/specialist-venv/bin/python tests/manual/debug_multimodal_raw.py bf16
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

_MODEL = "google/gemma-4-E4B-it"
_TASK_TEXT = (
    "You are a robot task planner. Given the scene image and a task, decompose "
    "it into a numbered list of simple sequential sub-instructions (1., 2., "
    "3. ...). Output ONLY the numbered list.\n\nTask: pick up the red cube"
)


def _demo_image() -> object:
    import numpy as np
    from PIL import Image

    arr = np.full((256, 256, 3), 128, dtype=np.uint8)
    arr[150:210, 90:150] = (200, 30, 30)
    return Image.fromarray(arr)


def _model_class():
    """Gemma 4's multimodal auto-class, with a fallback for naming drift."""
    import transformers

    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, name, None)
        if cls is not None:
            print(f"using model class: {name}")
            return cls
    raise SystemExit(
        "Neither AutoModelForMultimodalLM nor AutoModelForImageTextToText found "
        "— transformers is too old for Gemma 4. Run: pip install -U transformers"
    )


def _load(mode: str):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    model_cls = _model_class()
    kwargs: dict = {"device_map": "auto"}
    if mode == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif mode == "int4-bf16":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif mode == "int4-fp16":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        raise SystemExit(f"unknown mode {mode!r}")

    print(f"\n=== Loading {_MODEL} (mode={mode}) ===", flush=True)
    processor = AutoProcessor.from_pretrained(_MODEL)
    model = model_cls.from_pretrained(_MODEL, **kwargs)
    model.eval()
    return processor, model


def _probe(processor, model, with_image: bool, image: object) -> None:
    import torch

    label = "WITH image" if with_image else "TEXT-ONLY"
    blocks: list[dict] = []
    if with_image:
        blocks.append({"type": "image", "image": image})
    blocks.append({"type": "text", "text": _TASK_TEXT})
    messages = [{"role": "user", "content": blocks}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        logits = model(**inputs).logits
    print(f"\n===== {label} =====")
    print(f"input_len={input_len}  logits NaN? {bool(torch.isnan(logits).any())}  "
          f"inf? {bool(torch.isinf(logits).any())}")

    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=96, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
    new_ids = out[0][input_len:]
    print("greedy decode:", repr(processor.decode(new_ids, skip_special_tokens=True)))


def main() -> None:
    import torch

    mode = sys.argv[1] if len(sys.argv) > 1 else "int4-bf16"
    processor, model = _load(mode)

    if torch.cuda.is_available():
        print(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    image = _demo_image()
    _probe(processor, model, with_image=False, image=image)
    _probe(processor, model, with_image=True, image=image)


if __name__ == "__main__":
    main()
