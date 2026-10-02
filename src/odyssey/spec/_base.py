"""Shared base for every mission-spec model.

``extra="forbid"`` makes an unknown key a validation error instead of a
silently dropped field. A misspelled or misplaced key (``dataset.revison``,
``control_hz`` under the wrong block) would otherwise vanish while the
mission runs as if it had never been written. ``load_mission`` turns the
error into a ``LoadError`` and ``odyssey run`` exits before any task starts.

Free-form maps stay free-form: ``config: dict[str, Any]`` fields are not
models, so runner-specific keys are unaffected.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
