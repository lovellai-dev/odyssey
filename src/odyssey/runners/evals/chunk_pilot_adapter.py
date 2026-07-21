"""ChunkPilotAdapter — make a chunk-emitting VLA server look like a ``PilotRuntime``.

Chunk-emitting pilots (GR00T, π0.5) query the model ONCE and get back an action
**chunk** of ``n_action_steps`` actions executed open-loop — unlike a single-step
pilot (OpenVLA) that is queried every step. The multi-agent runtimes
(``PlannedEvalRuntime`` / ``DelegatedEvalRuntime``) drive a ``PilotRuntime`` with
``act(image, instruction)`` once per env step, so this adapter buffers a chunk and
drains it one action per ``act()``.

**Chunk-aware gating (the key behaviour).** The adapter re-queries the server when
the buffered chunk is empty, exhausted, OR **the instruction changed** since the
chunk was generated. That last condition is what makes phase transitions correct:
when a runtime advances a phase it simply feeds a new sub-instruction on the next
``act()``; the adapter detects the change, discards the now-stale chunk (generated
for the OLD sub-instruction) and re-queries with the new one. The runtime stays
completely chunk-oblivious.

**Pilot-agnostic by construction.** All model-specific knowledge lives in two
injected callables — ``obs_builder(observation, instruction) -> server_obs`` and
``action_mapper(chunk, k) -> action`` — so GR00T and π0.5 differ only in which
transforms they pass, never in this buffering logic.

**Observation threading.** ``PilotRuntime.act`` only carries the RGB image, but a
chunk pilot needs the full observation (proprioception, wrist frame, …). The
driver (the eval recipe, which owns the env) pushes the full observation via
``set_obs(observation)`` each step before calling ``get_action``; the adapter's
``obs_builder`` reads it. This keeps the runtimes and the ``PilotRuntime`` protocol
env-/pilot-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class ChunkPilotAdapter:
    """Adapt a chunk-emitting policy client to the single-step ``PilotRuntime`` API.

    Parameters
    ----------
    client:
        A connected policy client exposing ``get_action(server_obs) -> chunk`` (or
        ``(chunk, info)``). For GR00T this is the ZMQ ``PolicyClient`` from
        ``_gr00t_server.connect_policy_client``.
    obs_builder:
        ``(observation, instruction) -> server_obs``. Turns the raw env observation
        (stashed via :meth:`set_obs`) plus the current sub-instruction into the
        server's expected observation. Model/env-specific (the pilot-agnostic seam).
    action_mapper:
        ``(chunk, k) -> action``. Extracts step ``k`` of a chunk as the env's action
        vector. Model-specific (the other half of the seam).
    n_action_steps:
        Chunk length — how many actions each server query yields before a re-query.
    """

    def __init__(
        self,
        client: Any,
        *,
        obs_builder: Callable[[Any, str], Any],
        action_mapper: Callable[[Any, int], Any],
        n_action_steps: int,
    ) -> None:
        self._client = client
        self._obs_builder = obs_builder
        self._action_mapper = action_mapper
        self._n = int(n_action_steps)
        # Latest full observation, pushed by the driver via set_obs().
        self._obs: Any = None
        # Buffered chunk state.
        self._chunk: Any = None
        self._cursor = 0
        self._chunk_instruction: str | None = None

    def set_obs(self, observation: Any) -> None:
        """Push the current full env observation (called each step by the driver)."""
        self._obs = observation

    def reset(self) -> None:
        """Flush the buffer so the next :meth:`act` re-queries (call per episode)."""
        self._chunk = None
        self._cursor = 0
        self._chunk_instruction = None

    def act(self, image: Any, instruction: str) -> Any:
        """Return one action, re-querying the server only when needed.

        ``image`` is accepted for the ``PilotRuntime`` contract but unused — the
        adapter builds the server observation from the full obs pushed via
        :meth:`set_obs`. Re-queries when the chunk is empty/exhausted or the
        instruction changed (the chunk-aware phase gating).
        """
        if (
            self._chunk is None
            or self._cursor >= self._n
            or instruction != self._chunk_instruction
        ):
            server_obs = self._obs_builder(self._obs, instruction)
            result = self._client.get_action(server_obs)
            self._chunk = result[0] if isinstance(result, tuple) else result
            self._cursor = 0
            self._chunk_instruction = instruction
        action = self._action_mapper(self._chunk, self._cursor)
        self._cursor += 1
        return action
