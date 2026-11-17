"""MolmoAct2 (Ai2) pilot — chunk-emitting VLA behind the shared adapter.

MolmoAct2 (Molmo2-ER backbone + flow-matching action expert, Apache-2.0,
https://github.com/allenai/molmoact2) runs as an **out-of-process policy
server** — a FastAPI ``/act`` server speaking Ai2's json_numpy protocol — the
same serving posture as π0.5 / Cosmos 3 / WALL-OSS: the user pre-starts it in a
GPU env (torch + transformers, ``trust_remote_code``) ::

    uv run python examples/quickstart-molmoact2/host_server_libero.py \
        --repo-id allenai/MolmoAct2-LIBERO --port 8000

and this process holds only a stdlib HTTP client and the sim. One ``POST /act``
returns a whole action **chunk** (10 steps for MolmoAct2-LIBERO), replayed
through the pilot-agnostic ``ChunkPilotAdapter``
(``runners/agents/chunk_pilot.py``).

The client is ``urllib`` + the json_numpy codec in ``molmoact2_transforms`` — no
new dependency — and is injectable for tests, so this module imports and
unit-tests on a CPU box.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_SERVE_HINT = (
    "Start it first in a MolmoAct2 GPU env, e.g. uv run python "
    "examples/quickstart-molmoact2/host_server_libero.py "
    "--repo-id allenai/MolmoAct2-LIBERO --port <port>."
)


class MolmoAct2HttpClient:
    """Minimal stdlib client for the MolmoAct2 FastAPI ``/act`` server.

    ``info`` reads ``GET /act`` (health + ``action_horizon``); ``infer`` posts
    one json_numpy request and returns the decoded response. The first call can
    be slow (model warm-up / CUDA-graph capture), hence the generous timeout.
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 8000,
                 timeout_seconds: float = 600.0) -> None:
        self._url = f"http://{host}:{int(port)}/act"
        self._timeout = float(timeout_seconds)

    def _open(self, req: Any) -> Any:
        from odyssey.runners.evals.molmoact2_transforms import json_numpy_loads

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json_numpy_loads(resp.read())
        except urllib.error.HTTPError as e:
            # The server returns {"error": ...} bodies on 4xx/5xx — surface them.
            try:
                body = json_numpy_loads(e.read())
            except Exception:
                body = None
            if isinstance(body, dict) and "error" in body:
                raise RuntimeError(
                    f"MolmoAct2 policy server error ({e.code}): {body['error']}"
                ) from e
            raise
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"MolmoAct2 policy server unreachable at {self._url}: {e}. " + _SERVE_HINT
            ) from e

    def info(self) -> dict[str, Any]:
        return dict(self._open(self._url))

    def infer(self, request: dict[str, Any]) -> Any:
        from odyssey.runners.evals.molmoact2_transforms import json_numpy_dumps

        req = urllib.request.Request(
            self._url,
            data=json_numpy_dumps(request).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._open(req)


def make_molmoact2_pilot(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    n_action_steps: int | None = None,
    num_steps: int | None = None,
    invert_gripper: bool = False,
    translation_only: bool = False,
    client: Any = None,
) -> Any:
    """Build a chunk-aware MolmoAct2 ``PilotRuntime`` over a running ``/act`` server.

    Wires the MolmoAct2 wire transforms into the shared ``ChunkPilotAdapter``:

      * ``predict_chunk`` = ``client.infer`` (``POST /act`` -> action chunk);
      * ``observation_builder`` = ``build_molmoact2_libero_request`` (agentview +
        wrist frames, 8-D state, instruction, flow ``num_steps``);
      * ``action_decoder`` = ``molmoact2_action_to_libero`` (7-D passthrough).

    ``n_action_steps`` is how many chunk steps are replayed before the next
    query; leave it ``None`` to adopt the server's ``action_horizon`` from
    ``GET /act`` (falling back to 10 when the endpoint is unavailable).
    ``num_steps`` is the flow-matching solver step count (server default 10).
    """
    from odyssey.runners.agents.chunk_pilot import ChunkPilotAdapter
    from odyssey.runners.evals.molmoact2_transforms import (
        MOLMOACT2_DEFAULT_CHUNK_SIZE,
        MOLMOACT2_DEFAULT_FLOW_STEPS,
        build_molmoact2_libero_request,
        molmoact2_action_to_libero,
        molmoact2_chunk_from_response,
    )

    policy = client if client is not None else MolmoAct2HttpClient(host=host, port=port)
    flow_steps = int(num_steps) if num_steps else MOLMOACT2_DEFAULT_FLOW_STEPS

    if n_action_steps is None:
        try:
            n_action_steps = int(policy.info().get(
                "action_horizon", MOLMOACT2_DEFAULT_CHUNK_SIZE
            ))
        except Exception as e:  # GET /act unreachable -> default, fail late in act()
            n_action_steps = MOLMOACT2_DEFAULT_CHUNK_SIZE
            logger.warning(
                "MolmoAct2 GET /act failed (%s); assuming action_horizon=%d. "
                "Pass n_action_steps explicitly to override.",
                e,
                n_action_steps,
            )
    steps = int(n_action_steps)

    def predict_chunk(request: Any) -> Any:
        chunk = molmoact2_chunk_from_response(policy.infer(request))
        if chunk.shape[0] < steps:
            raise ValueError(
                f"MolmoAct2 returned a {chunk.shape[0]}-step chunk but "
                f"n_action_steps={steps}; lower n_action_steps."
            )
        return chunk

    def observation_builder(raw_obs: Any, instruction: str) -> Any:
        # raw_obs is the kwargs dict the eval recipe shapes from the env obs
        # (image / wrist_image / eef_pos / eef_quat_xyzw / gripper_qpos).
        return build_molmoact2_libero_request(
            instruction=instruction, num_steps=flow_steps, **raw_obs
        )

    def action_decoder(chunk: Any, k: int) -> Any:
        return molmoact2_action_to_libero(
            chunk, k, invert_gripper=invert_gripper, translation_only=translation_only,
        )

    return ChunkPilotAdapter(
        predict_chunk=predict_chunk,
        action_decoder=action_decoder,
        observation_builder=observation_builder,
        n_action_steps=steps,
    )
