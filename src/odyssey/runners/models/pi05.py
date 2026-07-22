"""π0.5 (openpi) pilot — chunk-emitting VLA behind the shared chunk adapter.

π0.5 runs, like GR00T, as an **out-of-process policy server** (openpi's
``WebsocketPolicyServer``); this process holds only the lightweight websocket
client + the LIBERO env. A query returns an action **chunk** (``action_horizon``
future steps, ``chunk_size`` default 50 / 10 for ``pi05_libero``), so the pilot
reuses the pilot-agnostic ``ChunkPilotAdapter`` for buffering + replay + the
flush-on-instruction-change gating — no bespoke replay loop, no duplicated
abstraction (see ``runners/agents/chunk_pilot.py``).

The wire glue is π0.5-specific but thin: ``build_pi05_libero_obs`` (openpi
``observation/*`` keys) and ``pi05_action_to_libero`` (slice to 7-D, no gripper
fix-up), both reusing the Franka Panda kinematics from ``gr00t_transforms``.

Heavy deps (``openpi_client`` for the websocket policy, numpy for the transforms)
are imported lazily inside the run path, so this module imports under the bare
stdlib and its factory stays unit-testable on a CPU box.
"""

from __future__ import annotations

from typing import Any


def _make_pi05_client(*, host: str, port: int, api_key: str | None = None) -> Any:
    """Return a connected openpi ``WebsocketClientPolicy`` (light deps).

    Deferred import: ``openpi_client`` ships with the π0.5 serving stack, which
    lives in a separate env (openpi/JAX). Raises ``NotImplementedError`` with an
    actionable message when it is not importable, matching ``VLARuntime``'s
    contract for a missing optional backend.
    """
    try:
        from openpi_client import websocket_client_policy as _wcp
    except ImportError as e:  # pragma: no cover - exercised via make_pi05_pilot test
        raise NotImplementedError(
            "The π0.5 pilot requires the openpi client ('openpi_client'). "
            "Install openpi and serve a π0.5 checkpoint "
            "(e.g. scripts/serve_policy.py), then point the mission at host:port."
        ) from e
    kwargs: dict[str, Any] = {"host": host, "port": int(port)}
    if api_key:
        kwargs["api_key"] = api_key
    return _wcp.WebsocketClientPolicy(**kwargs)


def make_pi05_pilot(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    n_action_steps: int = 10,
    api_key: str | None = None,
    client: Any = None,
    translation_only: bool = False,
) -> Any:
    """Build a chunk-aware π0.5 ``PilotRuntime`` over a running openpi server.

    Wires the π0.5 wire transforms into the shared ``ChunkPilotAdapter``:

      * ``predict_chunk`` = ``client.infer`` (openpi returns ``{"actions": ...}``);
      * ``observation_builder`` = ``build_pi05_libero_obs`` (from ``**raw_obs``);
      * ``action_decoder`` = ``pi05_action_to_libero`` (slice 7-D, no gripper fix-up).

    ``n_action_steps`` is how many steps of each chunk are replayed before the
    next query — match it to the checkpoint's ``action_horizon`` (10 for
    ``pi05_libero``, 50 default). ``client`` may be injected (tests / a
    pre-connected client); otherwise one is created against ``host:port``.
    """
    from odyssey.runners.agents.chunk_pilot import ChunkPilotAdapter
    from odyssey.runners.evals.pi05_transforms import (
        build_pi05_libero_obs,
        pi05_action_to_libero,
    )

    policy = client if client is not None else _make_pi05_client(
        host=host, port=port, api_key=api_key,
    )

    def observation_builder(raw_obs: Any, instruction: str) -> Any:
        # raw_obs is the kwargs dict already shaped for build_pi05_libero_obs
        # (image/wrist_image/eef_pos/eef_quat_xyzw/gripper_qpos), assembled by
        # the eval recipe from the LIBERO env observation.
        return build_pi05_libero_obs(instruction=instruction, **raw_obs)

    def action_decoder(chunk: Any, k: int) -> Any:
        return pi05_action_to_libero(chunk, k, translation_only=translation_only)

    return ChunkPilotAdapter(
        predict_chunk=policy.infer,
        action_decoder=action_decoder,
        observation_builder=observation_builder,
        n_action_steps=n_action_steps,
    )
