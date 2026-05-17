"""LeaderboardSpec — optional block on a Mission.

Honored only by `odyssey publish`. Scope B (v0.1.0-alpha) ships the schema
but not the publish path; setting `publish: true` here is accepted but a
no-op at runtime until the leaderboard backend exists.
"""

from __future__ import annotations

from pydantic import BaseModel


class LeaderboardSpec(BaseModel):
    publish: bool = False
    endpoint: str = "https://odyssey.lovell.ai"
    category: str | None = None
    team: str | None = None
    api_key_env: str = "ODYSSEY_API_KEY"
