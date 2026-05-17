"""Stdout event publisher — one JSON line per event."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, TextIO

from odyssey.telemetry.publishers.base import EventPublisher


class StdoutEventPublisher(EventPublisher):
    """JSON-per-line writer.

    Defaults to ``sys.stdout`` but accepts any text stream so tests can
    capture output without monkey-patching.
    """

    def __init__(self, stream: TextIO | None = None):
        self._stream = stream if stream is not None else sys.stdout

    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            **payload,
        }
        self._stream.write(json.dumps(record, default=str) + "\n")
        self._stream.flush()
