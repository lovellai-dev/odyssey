"""WALL-OSS (X Square Robot) pilot — chunk-emitting VLA behind the shared adapter.

WALL-OSS (4B, Qwen2.5-VL-3B backbone + flow-matching action expert,
Apache-2.0, https://github.com/X-Square-Robot/wall-x) runs as an
**out-of-process policy server** — the wall-x repo's WebSocket server — exactly
the π0.5 / Cosmos 3 serving posture: the user pre-starts it in the wall-x env
(torch + flash-attn + GPU) ::

    bash scripts/run_serving.sh --checkpoint-path <ckpt> \
        --train-config-path <train.yml> --env LIBERO --port 32195 \
        -- --model-config.norm-key libero_all

and this process holds only a light WebSocket client and the sim. One query
returns a whole action **chunk**, so the pilot reuses the pilot-agnostic
``ChunkPilotAdapter`` (buffer + replay + flush-on-instruction-change,
``runners/agents/chunk_pilot.py``).

The client speaks the server's own protocol (msgpack frames; metadata frame on
connect; ``msgpack-numpy`` ext-dict arrays in the response), decoded by
``wall_oss_transforms`` — no ``wall_x`` / ``msgpack_numpy`` / ``openpi_client``
import here. ``websockets`` + ``msgpack`` are imported lazily (``[wall-oss]``
extra) and the client is injectable, so this module unit-tests on a CPU box.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_SERVE_HINT = (
    "Start it first in the wall-x env, e.g. bash scripts/run_serving.sh "
    "--checkpoint-path <ckpt> --train-config-path <train.yml> --env LIBERO "
    "--port <port> -- --model-config.norm-key libero_all"
)


class WallOSSWebsocketClient:
    """Minimal synchronous client for the wall-x WebSocket policy server.

    Connects lazily on first use, consumes the metadata frame the server sends
    on connect, then does one request/response round-trip per ``infer``.
    Keepalive pings are disabled client-side: the server's first inference can
    block for a long time (model warm-up / CUDA graph capture), which would
    otherwise trip the default 20 s ping timeout (the π0.5 lesson,
    ``models/pi05.py``).
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 32195,
                 open_timeout: float = 300.0) -> None:
        self._uri = f"ws://{host}:{int(port)}"
        self._open_timeout = float(open_timeout)
        self._ws: Any = None
        self._metadata: dict[Any, Any] = {}

    def _modules(self) -> tuple[Any, Any]:
        try:
            import msgpack
            import websockets.sync.client as wsc
        except ImportError as e:
            raise NotImplementedError(
                "The WALL-OSS pilot needs the client deps 'websockets' and "
                "'msgpack' (pip install 'lovell-odyssey[wall-oss]'). " + _SERVE_HINT
            ) from e
        return msgpack, wsc

    def _unpack(self, frame: Any) -> Any:
        from odyssey.runners.evals.wall_oss_transforms import decode_msgpack_numpy

        msgpack, _ = self._modules()
        if isinstance(frame, str):  # server error frame: a traceback string
            raise RuntimeError(f"WALL-OSS policy server error:\n{frame}")
        return msgpack.unpackb(frame, object_hook=decode_msgpack_numpy, strict_map_key=False)

    def connect(self) -> None:
        if self._ws is not None:
            return
        _, wsc = self._modules()
        try:
            self._ws = wsc.connect(
                self._uri,
                compression=None,
                max_size=None,
                ping_interval=None,
                open_timeout=self._open_timeout,
            )
        except OSError as e:
            raise RuntimeError(
                f"WALL-OSS policy server unreachable at {self._uri}: {e}. " + _SERVE_HINT
            ) from e
        meta = self._unpack(self._ws.recv())
        self._metadata = dict(meta) if isinstance(meta, dict) else {}
        logger.info("connected to WALL-OSS server %s (metadata=%s)", self._uri, self._metadata)

    @property
    def metadata(self) -> dict[Any, Any]:
        self.connect()
        return self._metadata

    def infer(self, request: dict[str, Any]) -> Any:
        msgpack, _ = self._modules()
        self.connect()
        self._ws.send(msgpack.packb(request, use_bin_type=True))
        return self._unpack(self._ws.recv())

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None


def make_wall_oss_pilot(
    *,
    host: str = "127.0.0.1",
    port: int = 32195,
    n_action_steps: int = 10,
    jpeg_quality: int = 95,
    binarize_gripper: bool = False,
    invert_gripper: bool = False,
    translation_only: bool = False,
    client: Any = None,
) -> Any:
    """Build a chunk-aware WALL-OSS ``PilotRuntime`` over a running wall-x server.

    Wires the WALL-OSS wire transforms into the shared ``ChunkPilotAdapter``:

      * ``predict_chunk`` = ``client.infer`` (one msgpack round-trip -> chunk);
      * ``observation_builder`` = ``build_wall_oss_libero_request`` (follow2_pos
        state + base64-JPEG face/wrist views + instruction);
      * ``action_decoder`` = ``wall_oss_action_to_libero`` (26-wide row -> 7-D).

    ``n_action_steps`` (chunk steps replayed before re-querying) must not exceed
    the server's ``ACTION_HORIZON`` (10 for ``--env LIBERO``). ``client`` may be
    injected (tests / a pre-built client); otherwise a WebSocket client is
    created against ``host:port`` and connects on the first ``act``.
    """
    from odyssey.runners.agents.chunk_pilot import ChunkPilotAdapter
    from odyssey.runners.evals.wall_oss_transforms import (
        build_wall_oss_libero_request,
        wall_oss_action_to_libero,
        wall_oss_chunk_from_response,
    )

    policy = client if client is not None else WallOSSWebsocketClient(host=host, port=port)

    def predict_chunk(request: Any) -> Any:
        chunk = wall_oss_chunk_from_response(policy.infer(request))
        if chunk.shape[0] < n_action_steps:
            raise ValueError(
                f"WALL-OSS returned a {chunk.shape[0]}-step chunk but n_action_steps="
                f"{n_action_steps}; lower n_action_steps or raise the server's "
                "ACTION_HORIZON."
            )
        return chunk

    def observation_builder(raw_obs: Any, instruction: str) -> Any:
        # raw_obs is the kwargs dict the eval recipe shapes from the env obs
        # (image / wrist_image / eef_pos / eef_quat_xyzw / gripper_qpos).
        return build_wall_oss_libero_request(
            instruction=instruction, jpeg_quality=jpeg_quality, **raw_obs
        )

    def action_decoder(chunk: Any, k: int) -> Any:
        return wall_oss_action_to_libero(
            chunk, k,
            binarize_gripper=binarize_gripper,
            invert_gripper=invert_gripper,
            translation_only=translation_only,
        )

    return ChunkPilotAdapter(
        predict_chunk=predict_chunk,
        action_decoder=action_decoder,
        observation_builder=observation_builder,
        n_action_steps=n_action_steps,
    )
