"""SpecialistTurn — one advisory-producing cognitive turn.

Wraps *any* ``TextGenerator`` (in-process ``GemmaVLMGenerator`` or the
out-of-process ``RemoteGenerator`` — both satisfy the same
``runtime.py:TextGenerator`` protocol) and turns ``(instruction, image)`` into
a single grounded **advisory** string.

The advisory prompt lives HERE, in the brain — never in the model host. The
host just runs the model on whatever messages it is handed; deciding what to
ask (a grounded advisory, not a numbered plan) is a control-plane concern.
"""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Advisory system prompt. Reuses the vision-grounding guardrails of the old
# planner (``agents/planner.py:_SYSTEM_PROMPT_VISION``) but asks for ONE grounded
# advisory the Pilot can act on — not a numbered decomposition.
_ADVISORY_SYSTEM = (
    "You are a robot task SPECIALIST. You advise the pilot; you do not act. "
    "Given the task instruction, produce a single short advisory (1-3 sentences) "
    "that helps the pilot execute the task. Be concrete and actionable. Output "
    "ONLY the advisory text, no preamble, no numbered list."
)

_ADVISORY_SYSTEM_VISION = (
    "You are a robot task SPECIALIST with vision. You advise the pilot; you do "
    "not act. You are given the current scene image and the task instruction. "
    "Ground your advisory STRICTLY in what you actually see — name concrete "
    "objects, colors, and spatial relationships relevant to the task. Do NOT "
    "invent objects or details that are not visible. Produce a single short "
    "advisory (1-3 sentences) that helps the pilot execute the task. Output ONLY "
    "the advisory text, no preamble, no numbered list."
)

_WS = re.compile(r"\s+")


@runtime_checkable
class TextGenerator(Protocol):
    """The model-layer port a specialist runs on (mirrors ``runtime.py``)."""

    def generate(self, messages: list[dict[str, Any]]) -> str: ...


class SpecialistTurn:
    """An advisory-only specialist over a ``TextGenerator``.

    Parameters
    ----------
    name:
        Human-readable specialist name (used for advisory attribution + dispatch).
    generator:
        Any ``TextGenerator`` (in-process or out-of-process). A multimodal one
        (``generate`` accepts an ``image`` kwarg) is fed the scene image.
    is_vision:
        Whether this specialist can consume the camera image (multimodal model).
    keywords:
        Optional extra routing vocabulary, merged with tokens from ``name``.
    """

    def __init__(
        self,
        name: str,
        generator: TextGenerator,
        *,
        is_vision: bool,
        keywords: set[str] | None = None,
    ) -> None:
        self._name = name
        self._generator = generator
        self._is_vision = is_vision
        self._accepts_image = "image" in inspect.signature(generator.generate).parameters
        toks = {t for t in re.split(r"[^a-z0-9]+", name.lower()) if len(t) >= 3}
        self._vocab = toks | (keywords or set())

    @property
    def name(self) -> str:
        return self._name

    @property
    def vocab(self) -> set[str]:
        return self._vocab

    @property
    def is_vision(self) -> bool:
        return self._is_vision

    def run(self, instruction: str, image: Any | None = None) -> str:
        """Produce one advisory string for the task. Empty string on failure.

        The image is forwarded only when this specialist is vision-capable AND
        its generator accepts an ``image`` argument.
        """
        use_vision = image is not None and self._is_vision and self._accepts_image
        system = _ADVISORY_SYSTEM_VISION if use_vision else _ADVISORY_SYSTEM
        messages = [
            {"role": "user", "content": f"{system}\n\nTask: {instruction}"},
        ]
        try:
            if use_vision:
                text = self._generator.generate(messages, image=image)  # type: ignore[call-arg]
            else:
                text = self._generator.generate(messages)
        except Exception as e:
            logger.warning("SpecialistTurn %r failed (%s) — empty advisory", self._name, e)
            return ""
        return _WS.sub(" ", text).strip()

    def close(self) -> None:
        """Release the generator if it owns out-of-process resources. Safe to
        call multiple times; no-op for in-process generators."""
        closer = getattr(self._generator, "close", None)
        if callable(closer):
            closer()
