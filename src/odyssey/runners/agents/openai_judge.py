"""OpenAI-compatible VLM completion judge — a ``CompletionDetector`` over HTTP.

Issue #78's finding: the co-resident Gemma int4 ``check_done`` judge answers NO
essentially always — even after the zoom fix made the gripper clearly visible —
so multi-agent phases advance by step-cap instead of by completion. The fix
direction is a **stronger judge**, and the serving landscape makes one judge
surface cover them all: Cosmos 3's Reasoner (Edge 4B / Nano 16B / Super 64B),
vLLM- or NIM-served VLMs, and hosted endpoints all speak the OpenAI
``/chat/completions`` protocol with image content parts.

``OpenAICompatCompletionJudge`` satisfies the ``CompletionDetector`` protocol
(``runners/agents/runtime.py``) structurally, so it drops into the
``ChunkCompletionGate`` wherever the Gemma judge does today — the gate neither
knows nor cares that the judge moved out-of-process. Family-/vendor-wide by
construction: ``base_url`` + ``model`` are the only required knobs.

The judge frames the question as a strict YES/NO visual check and parses the
first YES/NO token of the reply; an unparseable reply counts as NO (not done) —
the conservative direction, since a spurious YES would skip a phase mid-task.

stdlib-only transport (``urllib``) with an injectable ``transport`` callable, so
unit tests stub the endpoint without a server and this module stays
strict-mypy-clean (contrast the exempted SDK-mirror glue in ``models/``).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import urllib.request
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_TEMPLATE = (
    "You are a strict visual judge for a robot manipulation task. "
    "Look at the image and answer whether the robot has ALREADY fully "
    "completed this sub-instruction: {instruction!r}. "
    "Answer with exactly one word: YES or NO."
)

_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _encode_data_uri(observation: Any) -> str:
    """Encode an observation image (HWC uint8 ndarray or PIL Image) as a data URI."""
    from PIL import Image  # lazy: pillow is a dev/eval dep, not a core one

    if not isinstance(observation, Image.Image):
        import numpy as np

        observation = Image.fromarray(
            np.ascontiguousarray(np.asarray(observation, dtype=np.uint8))
        )
    buf = io.BytesIO()
    observation.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def parse_yes_no(text: str) -> bool:
    """``True`` iff the reply's first YES/NO token is YES; unparseable -> ``False``.

    Conservative by design: a judge that rambles without committing keeps the
    phase running (the step cap still bounds it), whereas a false YES would
    skip a phase mid-task.
    """
    match = _YES_NO_RE.search(text or "")
    if match is None:
        logger.warning("completion judge reply had no YES/NO token: %r", text)
        return False
    return match.group(1).lower() == "yes"


class OpenAICompatCompletionJudge:
    """Judge sub-instruction completion via an OpenAI-compatible chat endpoint.

    Parameters
    ----------
    base_url:
        Endpoint base, e.g. ``http://127.0.0.1:8001/v1`` (vLLM/NIM serving a
        Cosmos 3 Reasoner) or a hosted provider. ``/chat/completions`` is
        appended.
    model:
        The served model id, e.g. ``nvidia/Cosmos3-Edge``.
    api_key / api_key_env:
        Bearer token, given directly or named via an env var (never hardcode
        secrets in missions; ``api_key_env`` is the YAML-friendly form).
    prompt_template:
        ``str.format``-style template receiving ``instruction``.
    transport:
        Injectable ``payload -> response-dict`` callable (tests / custom HTTP
        stacks). Defaults to a stdlib ``urllib`` POST.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        max_tokens: int = 8,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._template = prompt_template
        self._max_tokens = int(max_tokens)
        self._temperature = float(temperature)
        self._timeout = float(timeout_seconds)
        self._transport = transport
        key = api_key or (os.getenv(api_key_env) if api_key_env else None)
        self._headers = {"Content-Type": "application/json"}
        if key:
            self._headers["Authorization"] = f"Bearer {key}"

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(payload)
        req = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return dict(json.loads(resp.read().decode("utf-8")))

    def build_payload(self, observation: Any, instruction: str) -> dict[str, Any]:
        """The chat-completions request for one judgement (exposed for tests)."""
        return {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": _encode_data_uri(observation)},
                        },
                        {
                            "type": "text",
                            "text": self._template.format(instruction=instruction),
                        },
                    ],
                }
            ],
        }

    # -- CompletionDetector surface -------------------------------------------

    def is_complete(self, observation: Any, instruction: str) -> bool:
        """``True`` when the VLM judges ``instruction`` complete in ``observation``.

        Endpoint/parse failures are logged and reported as NO (not done) rather
        than raised — a flaky judge must not abort the episode; the phase's step
        cap remains the fallback, exactly as with the in-process judge.
        """
        try:
            response = self._post(self.build_payload(observation, instruction))
            content = response["choices"][0]["message"]["content"]
        except Exception:
            logger.warning(
                "completion judge call failed (%s); judging NOT complete", self._url,
                exc_info=True,
            )
            return False
        return parse_yes_no(str(content))
