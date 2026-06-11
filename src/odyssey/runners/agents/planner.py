"""LLMPlanner — task decomposition logic.

Takes any ``TextGenerator`` (the model-layer interface defined in
``runtime.py``) and uses it to decompose a high-level task instruction
into ordered sub-instructions. The model loading lives in
``runners/models/`` (e.g. ``GemmaTextGenerator``) — this module only
handles the planning prompt and output parsing.

Satisfies ``PlannerRuntime`` protocol.
"""

from __future__ import annotations

import logging
import re

from odyssey.runners.agents.runtime import TextGenerator

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
    """Task planner that decomposes instructions into sub-steps.

    Satisfies ``PlannerRuntime`` protocol.

    Parameters
    ----------
    generator:
        Any ``TextGenerator`` implementation (e.g. ``GemmaTextGenerator``).
        The planner doesn't care which model is behind it.
    """

    def __init__(self, generator: TextGenerator) -> None:
        self._generator = generator

    def plan(self, task_instruction: str) -> list[str]:
        """Decompose a task instruction into sub-steps."""
        messages = [
            {"role": "user", "content": f"{_SYSTEM_PROMPT}\n\nTask: {task_instruction}"},
        ]

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
