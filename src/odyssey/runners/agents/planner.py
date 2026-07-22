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

from odyssey.runners.agents.runtime import RouteDecision, TextGenerator

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

# Criteria appended to the completion question (kept separate for reuse/testing).
_COMPLETION_PROMPT = (
    "A 'pick up'/'grasp'/'lift' counts as completed once the gripper is clearly "
    "holding the object OR the object is off the table; a 'place'/'put'/'drop' counts "
    "as completed once the object is resting at the target location."
)

_GROUNDING_PROMPT = (
    "You are a robot perception module. You are given the current scene image "
    "and a description of a target the robot must act on. Look at the image and "
    "identify the single concrete object or location that best matches the "
    "description. Answer with a SHORT noun phrase naming it and, if helpful, its "
    "distinguishing visual feature or position (e.g. 'the red mug on the left', "
    "'the top drawer', 'the black bowl'). Output ONLY that phrase — no full "
    "sentence, no explanation, no quotes."
)

_ROUTE_PROMPT = (
    "You are a robot task orchestrator. You are given the current scene image, "
    "the overall task, and the sub-instructions already completed. Decide the "
    "SINGLE next sub-instruction the robot arm should execute now to make "
    "progress, grounded in what you can see. If the overall task is already fully "
    "accomplished in the image, answer with EXACTLY the word DONE. Otherwise "
    "answer with ONLY that one sub-instruction — a single atomic action naming "
    "the visible objects (e.g. 'pick up the red mug', 'place it in the basket') — "
    "no numbering, no explanation, no quotes."
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


def _parse_yes_no(text: str) -> bool:
    """True only when the output's first word is an explicit yes.

    Takes the first alphabetic run (ignoring surrounding markdown/punctuation
    like ``**YES**``). Conservative by design: anything ambiguous (``no``,
    empty, ``maybe``, a sentence not starting with yes) → False, so a phase
    never advances on an unclear answer (the runtime's step cap still
    guarantees progress).
    """
    m = re.search(r"[a-zA-Z]+", text or "")
    return m is not None and m.group(0).lower() in ("yes", "y")


def _parse_grounding(text: str) -> str:
    """Extract a single grounded phrase from the VLM output.

    Takes the first non-empty line and strips surrounding markdown/quotes/
    trailing punctuation, keeping a compact noun phrase. Returns "" when the
    output has nothing usable, so ``ground`` can fall back to the raw query.
    """
    for line in (text or "").strip().splitlines():
        # Drop a leading list marker ("- ", "1. ") the model may add.
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        cleaned = cleaned.strip("\"'`*.").strip()
        if cleaned:
            return cleaned
    return ""


def _parse_route(text: str) -> tuple[str, bool]:
    """Parse a routing reply into ``(subtask, done)``.

    Reuses ``_parse_grounding`` to pull a clean first phrase. ``DONE`` (case-
    insensitive, alone) → ``("", True)``; an empty/unusable reply → ``("", False)``
    so ``route`` fails safe to the task (never a spurious done).
    """
    phrase = _parse_grounding(text)
    if not phrase:
        return "", False
    if phrase.strip().upper() == "DONE":
        return "", True
    return phrase, False


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
        self._last_check_raw: str = ""  # raw text of the last check_done (diagnostics)
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

    def check_done(self, instruction: str, image: Any) -> bool:
        """Return True if ``instruction`` looks completed in ``image``.

        Satisfies ``CompletionDetector``. Requires a multimodal generator and a
        frame — without either, the check is impossible, so it returns False
        (never a false positive; the runtime's step cap still advances phases).
        """
        if not self._accepts_image or image is None:
            logger.debug(
                "check_done unavailable (accepts_image=%s, image=%s) — returning False",
                self._accepts_image,
                image is not None,
            )
            return False
        messages = [
            {
                "role": "user",
                "content": (
                    f'Look at this camera frame of a robot arm on a tabletop. Has the '
                    f'robot completed this action: "{instruction}"? {_COMPLETION_PROMPT} '
                    f"Answer with ONE word: YES or NO."
                ),
            },
        ]
        text = self._generator.generate(messages, image=image)  # type: ignore[call-arg]
        logger.debug("check_done(%r) raw output: %r", instruction, text)
        self._last_check_raw = text
        return _parse_yes_no(text)

    def ground(self, target_query: str, image: Any) -> str:
        """Return a scene-grounded phrase for ``target_query`` in ``image``.

        Satisfies ``GroundingProvider``. Requires a multimodal generator and a
        frame; without either, grounding is impossible, so it returns
        ``target_query`` unchanged (fail-safe — the pilot still gets an
        actionable instruction, it just isn't scene-specialised).
        """
        if not self._accepts_image or image is None:
            logger.debug(
                "ground unavailable (accepts_image=%s, image=%s) — echoing query",
                self._accepts_image,
                image is not None,
            )
            return target_query
        messages = [
            {"role": "user", "content": f"{_GROUNDING_PROMPT}\n\nTarget: {target_query}"},
        ]
        text = self._generator.generate(messages, image=image)  # type: ignore[call-arg]
        logger.debug("ground(%r) raw output: %r", target_query, text)
        return _parse_grounding(text) or target_query

    def route(
        self, task_instruction: str, image: Any, history: list[str]
    ) -> RouteDecision:
        """Decide the next sub-instruction (or done) for ``task_instruction``.

        Satisfies ``OrchestratorRuntime``. Requires a multimodal generator and a
        frame; without either, routing is impossible, so it fails safe to the raw
        task (``done=False``) — the pilot still gets an actionable instruction and
        the episode never terminates on a missing route.
        """
        if not self._accepts_image or image is None:
            logger.debug(
                "route unavailable (accepts_image=%s, image=%s) — using task",
                self._accepts_image,
                image is not None,
            )
            return RouteDecision(subtask=task_instruction, done=False)
        done_note = "\n\nAlready completed: " + "; ".join(history) if history else ""
        messages = [
            {
                "role": "user",
                "content": f"{_ROUTE_PROMPT}\n\nTask: {task_instruction}{done_note}",
            },
        ]
        text = self._generator.generate(messages, image=image)  # type: ignore[call-arg]
        logger.debug("route(%r) raw output: %r", task_instruction, text)
        subtask, done = _parse_route(text)
        if done:
            return RouteDecision(subtask="", done=True)
        # Empty/unusable parse -> fail safe to the task so the pilot keeps acting.
        return RouteDecision(subtask=subtask or task_instruction, done=False)
