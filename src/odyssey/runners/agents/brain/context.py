"""BrainContext — the compose seam.

Framework-agnostic (no ADK, no torch at import) types that carry a single
Pilot turn's input: the request, the robot identity, and the advisories
composed from specialists. ``render_instruction`` is the COMPOSE step (of
dispatch -> compose -> converge) expressed as plain string composition the
model host cannot silently reshape.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Advisory:
    """One specialist's structured output block, composed into the Pilot turn.

    ``kind`` distinguishes the source class (e.g. ``"specialist"``); ``source``
    is the human-readable origin (the specialist's name), surfaced in the
    rendered instruction and in logs/telemetry.
    """

    source: str
    kind: str
    text: str


@dataclass
class BrainContext:
    """Input to one convergent Pilot turn.

    Assembled by the orchestrator from the request (``instruction``), the
    robot ``identity``, and ``advisories`` gathered from engaged specialists.
    """

    instruction: str
    identity: str | None = None
    advisories: list[Advisory] = field(default_factory=list)

    def render_instruction(self) -> str:
        """Compose identity + advisory blocks + the request into one string.

        This is the compose step (dispatch -> **compose** -> converge) as plain
        string concatenation: the Pilot sees every advisory before it acts.
        When there is nothing to compose, the bare instruction is returned so a
        no-specialist turn is byte-identical to a single-agent one.
        """
        sections: list[str] = []
        if self.identity:
            sections.append(f"[ROBOT IDENTITY]\n{self.identity}")
        for adv in self.advisories:
            if adv.text:
                sections.append(f"[ADVISORY — {adv.source}]\n{adv.text}")
        if not sections:
            return self.instruction
        sections.append(f"[TASK]\n{self.instruction}")
        return "\n\n".join(sections)
