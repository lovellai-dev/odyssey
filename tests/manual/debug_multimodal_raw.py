"""Diagnostic: pinpoint the Gemma 3 int4 NaN-logits failure.

The model loads but the forward pass produces NaN/inf logits — greedy then
picks token 0 (<pad>) every step (empty output) and sampling crashes with
"probability tensor contains inf/nan". The Gemma family is known to NaN under
the default `sdpa` attention because of its attention soft-capping/scaling;
the documented fix is `attn_implementation="eager"`.

This script loads the model with a chosen attention impl and, for both the
with-image and text-only paths, runs ONE forward pass and reports whether the
logits contain NaN/inf (the definitive check), then does a short greedy decode.
No sampling — a device-side assert would poison the CUDA context.

REQUIRES the specialist venv. Run:
    ~/specialist-venv/bin/python tests/manual/debug_multimodal_raw.py          # eager (default)
    ~/specialist-venv/bin/python tests/manual/debug_multimodal_raw.py sdpa     # compare

Footprint: ~5 GB VRAM (Gemma 3 4B int4).
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

_MODEL = "google/gemma-3-4b-it"
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


def _messages(with_image: bool, image: object) -> list[dict]:
    blocks: list[dict] = []
    if with_image:
        blocks.append({"type": "image", "image": image})
    blocks.append({"type": "text", "text": _TASK_TEXT})
    return [{"role": "user", "content": blocks}]


def _probe(processor, model, with_image: bool, image: object) -> None:
    import torch

    label = "WITH image" if with_image else "TEXT-ONLY"
    msgs = _messages(with_image, image)
    inputs = processor.apply_chat_template(
        msgs,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    # One forward pass — inspect the raw logits for NaN/inf.
    with torch.no_grad():
        logits = model(**inputs).logits
    last = logits[0, -1]
    print(f"\n===== {label} =====")
    print(f"input_len={inputs['input_ids'].shape[-1]}  logits.shape={tuple(logits.shape)}")
    print(f"logits NaN? {bool(torch.isnan(logits).any())}   inf? {bool(torch.isinf(logits).any())}")
    print(f"last-token logits: min={last.min().item():.3f} max={last.max().item():.3f} "
          f"argmax={int(last.argmax())}")

    # Short greedy decode (safe even if NaN — just yields pads).
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=64, do_sample=False,
            temperature=None, top_p=None, top_k=None,
        )
    new_ids = out[0][inputs["input_ids"].shape[-1]:]
    print("greedy decode:", repr(processor.decode(new_ids, skip_special_tokens=True))[:300])


def main() -> None:
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

    attn = sys.argv[1] if len(sys.argv) > 1 else "eager"

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"\n=== Loading {_MODEL} (int4, attn_implementation={attn!r}) ===", flush=True)
    processor = AutoProcessor.from_pretrained(_MODEL, padding_side="left")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        _MODEL,
        quantization_config=quant,
        device_map="auto",
        attn_implementation=attn,
    )
    model.eval()

    image = _demo_image()
    _probe(processor, model, with_image=False, image=image)
    _probe(processor, model, with_image=True, image=image)


if __name__ == "__main__":
    main()
