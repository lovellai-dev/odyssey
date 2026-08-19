"""DeterministicBrain — the orchestrator: dispatch -> compose -> converge.

Replaces the plan-then-execute ``PlannedEvalRuntime``. Instead of decomposing
the task into a script the pilot replays, it engages specialists by meaning
(deterministic dispatch, no LLM), composes their advisories into the pilot's
instruction, and runs a single convergent pilot inference per step over that
composed instruction. No agent framework — it calls only the ``PilotRuntime``
and ``TextGenerator`` ports.

It presents the exact surface the eval loop already drives
(``begin_episode`` / ``get_action`` / ``close``), so it is a drop-in for
``PlannedEvalRuntime`` behind the ``brain: deterministic`` flag.

v1 cadence is ``"episode"``: dispatch + compose run ONCE at episode start and
the composed instruction is reused for every step. ``cadence`` is the extension
point for a future per-tick recompose (e.g. gated on a completion detector).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from odyssey.runners.agents.brain.context import Advisory, BrainContext
from odyssey.runners.agents.brain.dispatch import select_specialists
from odyssey.runners.agents.brain.safety import SafetyBoundary
from odyssey.runners.agents.brain.specialist import SpecialistTurn
from odyssey.runners.agents.runtime import PilotRuntime

logger = logging.getLogger(__name__)


class DeterministicBrain:
    """Dispatch -> compose -> converge orchestrator (drop-in for the eval loop).

    Parameters
    ----------
    pilot:
        The convergent action agent (a ``PilotRuntime`` such as ``VLARuntime``).
    specialists:
        Advisory-only cognitive turns. Engaged per-request by deterministic
        dispatch; their outputs compose into the pilot's instruction.
    identity:
        Optional robot identity prepended to the composed instruction.
    safety:
        Optional monitor-only ``SafetyBoundary`` the action traverses.
    cadence:
        ``"episode"`` (v1): compose once per episode. Reserved for future
        higher-cadence strategies.
    """

    def __init__(
        self,
        pilot: PilotRuntime,
        specialists: list[SpecialistTurn],
        *,
        identity: str | None = None,
        safety: SafetyBoundary | None = None,
        cadence: str = "episode",
    ) -> None:
        self._pilot = pilot
        self._specialists = specialists
        self._identity = identity
        self._safety = safety
        if cadence != "episode":
            logger.warning(
                "DeterministicBrain: cadence %r not implemented; using 'episode'", cadence
            )
        self._cadence = "episode"
        self._instruction = ""
        self._composed = ""

    @property
    def current_instruction(self) -> str:
        return self._composed

    def begin_episode(
        self, task_instruction: str, image: Any | None = None
    ) -> list[str]:
        """Dispatch + compose once for the episode; cache the composed instruction.

        Returns a single-element "plan" (the task) so the eval loop's plan
        telemetry stays meaningful — this brain converges on one instruction
        rather than a multi-phase script.
        """
        self._instruction = task_instruction
        has_image = image is not None
        decisions = select_specialists(task_instruction, self._specialists, has_image)
        engaged = [d for d in decisions if d.engaged]
        logger.info(
            "brain dispatch: engaging %d/%d specialist(s)",
            len(engaged),
            len(self._specialists),
        )
        for d in decisions:
            logger.info(
                "brain   %s: %s (%s)",
                d.specialist.name,
                "ENGAGE" if d.engaged else "skip",
                d.reason,
            )

        advisories: list[Advisory] = []
        for d in engaged:
            spec = d.specialist
            assert isinstance(spec, SpecialistTurn)
            text = spec.run(task_instruction, image if d.is_vision else None)
            if text:
                advisories.append(Advisory(source=spec.name, kind="specialist", text=text))
                logger.info("brain advisory [%s]: %s", spec.name, text[:160])

        ctx = BrainContext(
            instruction=task_instruction,
            identity=self._identity,
            advisories=advisories,
        )
        self._composed = ctx.render_instruction()
        logger.info(
            "brain composed instruction for the pilot (%d advisory block(s))",
            len(advisories),
        )
        return [task_instruction]

    def get_action(self, image: Any) -> NDArray[np.floating[Any]]:
        """Converge: one pilot inference over the composed instruction, then safety."""
        instruction = self._composed or self._instruction
        action: NDArray[np.floating[Any]] = self._pilot.act(image, instruction)
        if self._safety is not None:
            action = self._safety.check(action)
        return action

    def close(self) -> None:
        """Tear down specialists (out-of-process subprocesses). Idempotent."""
        for spec in self._specialists:
            try:
                spec.close()
            except Exception as e:
                logger.warning("brain: specialist %r close failed: %s", spec.name, e)
