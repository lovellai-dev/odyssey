"""LLMPlanner — task decomposition logic.

Takes any ``TextGenerator`` (the model-layer interface defined in
``runtime.py``) and uses it to decompose a high-level task instruction
into ordered sub-instructions. The model loading lives in
``runners/models/`` (e.g. ``GemmaVLMGenerator``) — this module only
handles the planning prompt and output parsing.

Satisfies ``PlannerRuntime`` protocol.
"""

from __future__ import annotations

import inspect
import logging
import re
from typing import Any

from odyssey.runners.agents.runtime import TextGenerator

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a robot task planner. Given a high-level task instruction, "
    "decompose it into a numbered list of simple, sequential sub-instructions "
    "that a robot arm can execute one at a time. Each sub-instruction should "
    "describe a single atomic motion or action. Output ONLY the numbered list, "
    "nothing else."
)

_SYSTEM_PROMPT_VISION = (
    "You are a robot task planner. You are given the current scene image and a "
    "high-level task instruction. Using what you can see in the image, decompose "
    "the task into a numbered list of ALL the simple, sequential sub-instructions "
    "a robot arm must execute to complete it, from start to finish "
    "(1., 2., 3., ...). Each line is one atomic motion that refers to the objects "
    "visible in the scene (e.g. locate, move to, align, grasp, lift, place). "
    "Give every step needed — do not stop after the first. Output ONLY the "
    "numbered list, nothing else."
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
    """Task planner that decomposes instructions into sub-steps.

    Satisfies ``PlannerRuntime`` protocol.

    Parameters
    ----------
    generator:
        Any ``TextGenerator`` implementation (e.g. ``GemmaVLMGenerator``).
        The planner doesn't care which model is behind it.
    """

    def __init__(self, generator: TextGenerator) -> None:
        self._generator = generator
        # A multimodal generator (e.g. GemmaVLMGenerator) accepts an ``image``
        # argument on ``generate``; a text-only one does not. Detect once so
        # ``plan`` forwards the scene image only when it's supported.
        self._accepts_image = "image" in inspect.signature(generator.generate).parameters

    def plan(self, task_instruction: str, image: Any | None = None) -> list[str]:
        """Decompose a task instruction into sub-steps.

        When ``image`` is given and the underlying generator is multimodal,
        the plan is grounded in the scene; otherwise the image is ignored.
        """
        use_vision = image is not None and self._accepts_image
        system_prompt = _SYSTEM_PROMPT_VISION if use_vision else _SYSTEM_PROMPT
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\nTask: {task_instruction}"},
        ]

        if use_vision:
            text = self._generator.generate(messages, image=image)  # type: ignore[call-arg]
        else:
            text = self._generator.generate(messages)
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
