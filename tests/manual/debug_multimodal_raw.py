"""Diagnostic: low-level Gemma 3 multimodal generation (token-level).

The smoke produced an EMPTY string — the model emits EOS immediately and
``skip_special_tokens`` strips it to "". This script goes under
``GemmaVLMGenerator`` and drives ``Gemma3ForConditionalGeneration`` +
``AutoProcessor`` directly so we can see, for several prompt/generation
variants:

  * input token count, total output count, and how many NEW tokens were
    generated (this is the key number — if it's ~0, the model stops at once);
  * the new tokens decoded WITH and WITHOUT special tokens (to tell an
    immediate-EOS from a real-but-stripped answer);
  * a greedy vs sampled comparison (to spot greedy degeneracy under int4).

REQUIRES the specialist venv (modern transformers >=4.50):
    ~/specialist-venv/bin/python tests/manual/debug_multimodal_raw.py

Footprint: ~5 GB VRAM (Gemma 3 4B int4 incl. SigLIP vision tower).
"""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

_MODEL = "google/gemma-3-4b-it"
_INSTRUCTION = "pick up the red cube"
_TASK_TEXT = (
    "You are a robot task planner. Given the scene image and a task, decompose "
    "it into a numbered list of simple sequential sub-instructions (1., 2., "
    f"3. ...). Output ONLY the numbered list.\n\nTask: {_INSTRUCTION}"
)

# Variant A: single user turn (what LLMPlanner + GemmaVLMGenerator build today).
_MESSAGES_SINGLE_USER = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _TASK_TEXT}]},
]

# Variant B: official-example shape — system turn split from the user turn.
_MESSAGES_SYS_SPLIT = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": _TASK_TEXT}]},
]


def _demo_image() -> object:
    import numpy as np
    from PIL import Image

    arr = np.full((256, 256, 3), 128, dtype=np.uint8)
    arr[150:210, 90:150] = (200, 30, 30)  # a red "cube"
    return Image.fromarray(arr)


def _run(processor, model, messages, image, label, *, do_sample: bool) -> None:
    import torch

    # Attach the PIL image to the image block (apply_chat_template wires the
    # <start_of_image> placeholder; passing the image via the content block).
    msgs = [dict(m) for m in messages]
    for m in msgs:
        if m["role"] == "user":
            m["content"] = [
                ({"type": "image", "image": image} if b.get("type") == "image" else b)
                for b in m["content"]
            ]

    inputs = processor.apply_chat_template(
        msgs,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs: dict = {"max_new_tokens": 128}
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.95, top_k=64)
    else:
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    total_len = out.shape[-1]
    new_ids = out[0][input_len:]
    n_new = new_ids.shape[-1]

    print(f"\n===== {label} | do_sample={do_sample} =====")
    print(f"input_len={input_len}  total_len={total_len}  n_new={n_new}")
    print(f"new token ids (first 20): {new_ids[:20].tolist()}")
    print("decode(skip_special_tokens=True) :", repr(processor.decode(new_ids, skip_special_tokens=True)))
    print("decode(skip_special_tokens=False):", repr(processor.decode(new_ids, skip_special_tokens=False)))


def main() -> None:
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print(f"\n=== Loading {_MODEL} (int4) ===", flush=True)
    processor = AutoProcessor.from_pretrained(_MODEL, padding_side="left")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        _MODEL, quantization_config=quant, device_map="auto"
    )
    model.eval()

    # Useful context for diagnosing immediate-EOS.
    print("eos_token_id:", processor.tokenizer.eos_token_id)
    print("bos_token_id:", processor.tokenizer.bos_token_id)
    print("model.generation_config.eos_token_id:", model.generation_config.eos_token_id)

    image = _demo_image()
    _run(processor, model, _MESSAGES_SINGLE_USER, image, "single-user (current)", do_sample=False)
    _run(processor, model, _MESSAGES_SINGLE_USER, image, "single-user (current)", do_sample=True)
    _run(processor, model, _MESSAGES_SYS_SPLIT, image, "system-split (official)", do_sample=False)


if __name__ == "__main__":
    main()
