#!/usr/bin/env python3
"""Cosmos-Reason intent-trace sidecar for GR00T evaluations.

GR00T-N1.7 runs on the **Cosmos-Reason2-2B** backbone — a generative Qwen3-VL.
This sidecar queries that same backbone *as a VLM* with the current scene frame +
the task instruction, to emit a short natural-language **intent trace** alongside
the action policy: the "it's reasoning *and* acting" surface for demos and Episode
Review.

Honest framing: the **action policy** (``run_gr00t_server``) is what drives the
robot. This trace is a *parallel intent narration* from the same backbone family —
not a literal, action-by-action rationale of the executed motions. Query it **once
per episode** (cheap; ~1-2 s) rather than per control step.

Lives in ``odyssey.runners.evals`` beside the GR00T eval recipe and is **launched /
imported by path** (the package ``__init__`` pulls heavy sim deps), so the prompt +
``ODYSSEY_REASONING`` protocol surface stay unit-testable on a CPU box. Heavy deps
(torch / transformers / PIL / numpy) are imported lazily in the load/generate path.

CLI (standalone, for demo narration over a captured frame)::

    python cosmos_reason.py --image frame.png \
        --instruction "stack the red cube on the green cube"
    # -> prints the trace + an ODYSSEY_REASONING line
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any

log = logging.getLogger("cosmos_reason")

# The GR00T-N1.7 backbone, queried as a generative VLM for the intent trace.
DEFAULT_REASON_MODEL = "nvidia/Cosmos-Reason2-2B"
_REASONING_PREFIX = "ODYSSEY_REASONING "


# ---------------------------------------------------------------------------
# Prompt + protocol surface (stdlib only -> unit-testable anywhere)
# ---------------------------------------------------------------------------

def build_reasoning_prompt(instruction: str) -> str:
    """The VLM user text — asks for a short, scene-grounded intent trace."""
    return (
        f'You are a robot arm executing the task: "{instruction}". '
        "Looking at the current scene, state your plan in 3 short steps "
        "(what to locate, what to grasp, where to place it). Be concise."
    )


def clean_reasoning(text: str, *, max_chars: int = 400) -> str:
    """Collapse whitespace + truncate the raw generation into a one-line trace."""
    return " ".join((text or "").split()).strip()[:max_chars]


def _to_pil(frame: Any) -> Any:
    """Normalise a frame (PIL image or HxWx3 array) to a contiguous RGB PIL image.

    Two real-frame hazards this guards against (both caught by a live LIBERO
    rollout): a numpy array also has ``.size`` so it can't be distinguished from a
    PIL image by attribute, and sim agentviews are often flipped (``[::-1, ::-1]``)
    into a negative-stride view that transformers' fast image processor rejects —
    ``np.ascontiguousarray`` makes a positive-stride copy.
    """
    import numpy as np
    from PIL import Image

    if isinstance(frame, Image.Image):
        return frame
    return Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8))


def reasoning_line(*, episode: int, instruction: str, reasoning: str) -> str:
    """One ``ODYSSEY_REASONING`` protocol line — mirrors the recipe's
    ``episode_line`` / ``result_line`` so a runner/collector can pick it up and
    surface it next to the episode grade in Episode Review."""
    return _REASONING_PREFIX + json.dumps(
        {
            "episode": int(episode),
            "instruction": str(instruction),
            "reasoning": str(reasoning),
        }
    )


# ---------------------------------------------------------------------------
# The sidecar (heavy model deferred to load()/reason())
# ---------------------------------------------------------------------------

class ReasoningSidecar:
    """Loads Cosmos-Reason2-2B once and returns an intent trace per call.

    Lazy: the model loads on first ``reason()`` (or via ``load()``), so importing
    this module + the helpers above needs no GPU / torch / transformers.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_REASON_MODEL,
        device: str = "cuda",
        max_new_tokens: int = 70,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._proc: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import os

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # Gated Cosmos-Reason2 backbone -> load from the local cache offline.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        log.info("cosmos-reason: loading %s on %s", self.model_id, self.device)
        self._proc = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        # Load then ``.to(device)`` rather than ``device_map=`` — the latter requires
        # `accelerate`, which isn't in every eval env (the LIBERO/Isaac sim venvs lack
        # it). A live LIBERO rollout caught device_map raising -> guarded -> silent
        # empty trace; .to() keeps the sidecar portable to any torch+transformers env.
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device).eval()

    def reason(self, frame: Any, instruction: str) -> str:
        """Return a short intent trace for ``instruction`` grounded in ``frame``
        (an HxWx3 uint8 numpy array or a ``PIL.Image``).

        Never raises — returns ``""`` on any failure. The trace is a
        nice-to-have narration; it must never break or fail an eval.
        """
        try:
            import torch

            self.load()
            img = _to_pil(frame)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": build_reasoning_prompt(instruction)},
                    ],
                }
            ]
            text = self._proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._proc(text=[text], images=[img], return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
            gen = self._proc.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]
            return clean_reasoning(gen)
        except Exception:
            log.warning("cosmos-reason: generation failed; skipping trace", exc_info=True)
            return ""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Cosmos-Reason intent-trace sidecar.")
    ap.add_argument("--model-path", default=DEFAULT_REASON_MODEL)
    ap.add_argument("--image", required=True, help="Path to a scene frame (PNG/JPG).")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--episode", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=70)
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    from PIL import Image

    sidecar = ReasoningSidecar(
        model_id=args.model_path, device=args.device, max_new_tokens=args.max_new_tokens
    )
    trace = sidecar.reason(Image.open(args.image).convert("RGB"), args.instruction)
    print("REASONING:", trace, flush=True)
    print(
        reasoning_line(episode=args.episode, instruction=args.instruction, reasoning=trace),
        flush=True,
    )


if __name__ == "__main__":
    main()
