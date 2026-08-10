"""Cosmos 3 (WAM) pilot — chunk-emitting world-action model behind the shared adapter.

Cosmos 3 policy checkpoints (any family member: Edge 4B / Nano 16B / Super 64B,
DROID- or LIBERO-SFT'd) run as an **out-of-process policy server** —
cosmos-framework's ``action_policy_server_libero`` (HTTP) — exactly the π0.5
serving posture: the server is pre-started by the user (it needs the
cosmos-framework env + GPU), this process holds only a stdlib HTTP client and
the sim. One ``POST /predict`` returns a whole action **chunk** (plus the WAM's
predicted rollout video, which the eval ignores), so the pilot reuses the
pilot-agnostic ``ChunkPilotAdapter`` for buffering + replay +
flush-on-instruction-change gating (``runners/agents/chunk_pilot.py``) — no
bespoke replay loop.

Family-wide by construction: the checkpoint lives server-side (start the server
with any ``--checkpoint-path``), and everything variant-specific reaching this
client — ``domain_name``, ``image_size``, chunk length — is a parameter. When
``n_action_steps`` is omitted the factory asks the server's ``GET /info`` for
its resolved ``action_chunk_size``, so one mission YAML serves Edge and Nano
alike.

The wire glue is Cosmos3-specific but thin (``cosmos3_transforms``): a
``concat_view`` base64-PNG request packer and a rot6d -> axis-angle chunk
decoder. The client is ``urllib`` from the stdlib — no new dependency — and is
injectable for tests, so this module imports and unit-tests on a CPU box.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class Cosmos3HttpClient:
    """Minimal stdlib client for the cosmos-framework action policy server.

    ``infer`` posts one policy request to ``/predict`` and returns the decoded
    JSON response; ``info`` reads ``GET /info`` (model/runtime metadata,
    including the server's resolved ``action_chunk_size``). The first WAM
    inference can take a while (diffusion sampling + model warm-up), hence the
    generous default timeout.
    """

    def __init__(self, *, host: str = "127.0.0.1", port: int = 8000,
                 timeout_seconds: float = 600.0) -> None:
        self._base = f"http://{host}:{int(port)}"
        self._timeout = float(timeout_seconds)

    def info(self) -> dict[str, Any]:
        with urllib.request.urlopen(
            f"{self._base}/info", timeout=self._timeout
        ) as resp:
            return dict(json.loads(resp.read().decode("utf-8")))

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self._base}/predict",
            data=json.dumps(request).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return dict(json.loads(resp.read().decode("utf-8")))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cosmos3 policy server unreachable at {self._base}/predict: {e}. "
                "Start it first, e.g. python -m "
                "cosmos_framework.scripts.action_policy_server_libero "
                "--checkpoint-path <hf-id-or-export> --port <port>."
            ) from e


def make_cosmos3_pilot(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    n_action_steps: int | None = None,
    domain_name: str | None = None,
    image_size: int | None = None,
    client: Any = None,
    translation_only: bool = False,
) -> Any:
    """Build a chunk-aware Cosmos 3 ``PilotRuntime`` over a running policy server.

    Wires the Cosmos3 wire transforms into the shared ``ChunkPilotAdapter``:

      * ``predict_chunk`` = ``client.infer`` (``POST /predict`` -> action chunk
        + predicted video; the decoder keeps only the actions);
      * ``observation_builder`` = ``build_cosmos3_predict_request`` (concat_view
        base64 PNG + prompt + domain);
      * ``action_decoder`` = ``cosmos3_action_to_libero`` (rot6d -> axis-angle).

    ``n_action_steps`` is how many chunk steps are replayed before the next
    query; leave it ``None`` to adopt the server's own ``action_chunk_size``
    from ``GET /info`` (falling back to the family default when the endpoint
    is unavailable). ``client`` may be injected (tests / a pre-built client);
    otherwise a stdlib HTTP client is created against ``host:port``.
    """
    from odyssey.runners.agents.chunk_pilot import ChunkPilotAdapter
    from odyssey.runners.evals.cosmos3_transforms import (
        COSMOS3_DEFAULT_CHUNK_SIZE,
        COSMOS3_DEFAULT_DOMAIN,
        COSMOS3_DEFAULT_IMAGE_SIZE,
        build_cosmos3_predict_request,
        cosmos3_action_to_libero,
    )

    policy = client if client is not None else Cosmos3HttpClient(host=host, port=port)
    domain = domain_name if domain_name is not None else COSMOS3_DEFAULT_DOMAIN
    size = int(image_size) if image_size is not None else COSMOS3_DEFAULT_IMAGE_SIZE

    if n_action_steps is None:
        try:
            n_action_steps = int(policy.info().get(
                "action_chunk_size", COSMOS3_DEFAULT_CHUNK_SIZE
            ))
        except Exception:  # /info unreachable -> family default, fail late in act()
            n_action_steps = COSMOS3_DEFAULT_CHUNK_SIZE

    def observation_builder(raw_obs: Any, instruction: str) -> Any:
        # raw_obs is the kwargs dict shaped by the eval recipe from the env
        # observation (image / wrist_image); no proprio state on this wire —
        # the WAM is video-conditioned.
        return build_cosmos3_predict_request(
            instruction=instruction, domain_name=domain, image_size=size, **raw_obs
        )

    def action_decoder(chunk: Any, k: int) -> Any:
        return cosmos3_action_to_libero(chunk, k, translation_only=translation_only)

    return ChunkPilotAdapter(
        predict_chunk=policy.infer,
        action_decoder=action_decoder,
        observation_builder=observation_builder,
        n_action_steps=n_action_steps,
    )
