"""GraphSpec — optional Learning Graph contribution settings.

Scope B accepts the schema but does not actually contribute records yet —
the graph backend is post-alpha. `contribute: true` is silently ignored
until the graph SDK lands.
"""

from __future__ import annotations

from odyssey.spec._base import SpecModel


class GraphSpec(SpecModel):
    contribute: bool = True
    contribute_instruction_prefix: bool = False
    notes: str | None = None
