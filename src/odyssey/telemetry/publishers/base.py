"""EventPublisher ABC.

Three implementations are planned per design §3.4: stdout (this file's
sibling), file (buffered to disk for replay), and remote (Pub/Sub /
Platform API). v0.1.0-alpha ships stdout only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EventPublisher(ABC):
    @abstractmethod
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None: ...
