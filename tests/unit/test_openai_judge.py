"""Tests for the OpenAI-compatible VLM completion judge (issue #78 direction).

``OpenAICompatCompletionJudge`` is the out-of-process judge surface (a served
Cosmos 3 Reasoner, any vLLM/NIM VLM, or a hosted endpoint): a
``CompletionDetector`` over ``/chat/completions``. Everything is exercised with
an injected ``transport`` stub — no server, no network, no GPU:

  * YES/NO reply parsing (first committed token wins; unparseable -> NO);
  * the chat payload (model id, base64 PNG data URI, instruction in the text);
  * failure semantics: endpoint/parse errors judge NOT complete, never raise
    (a flaky judge must not abort an episode — the step cap is the fallback);
  * auth header wiring via ``api_key`` / ``api_key_env``;
  * it satisfies ``CompletionDetector`` structurally and drops into the
    ``ChunkCompletionGate`` exactly like the in-process Gemma judge.

All tests are named ``test_openai_judge_*`` (the ``-k openai_judge`` gate).
"""

from __future__ import annotations

from typing import Any

import pytest

from odyssey.runners.agents.completion_gate import ChunkCompletionGate
from odyssey.runners.agents.openai_judge import (
    OpenAICompatCompletionJudge,
    parse_yes_no,
)
from odyssey.runners.agents.runtime import CompletionDetector

BASE_URL = "http://127.0.0.1:8001/v1"


def _reply(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


def _judge(transport, **kwargs: Any) -> OpenAICompatCompletionJudge:
    return OpenAICompatCompletionJudge(
        base_url=BASE_URL, model="nvidia/Cosmos3-Edge", transport=transport, **kwargs
    )


def _image():
    np = pytest.importorskip("numpy")
    return np.zeros((4, 4, 3), np.uint8)


# ---------------------------------------------------------------------------
# Reply parsing.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("YES", True),
        ("yes.", True),
        ("Yes, the banana is in the bowl.", True),
        ("NO", False),
        ("No — the gripper is still empty.", False),
        # First committed token wins over later hedging.
        ("No. Well, possibly yes.", False),
        # Unparseable rambling -> conservative NO.
        ("The scene shows a robot arm above a table.", False),
        ("", False),
    ],
)
def test_openai_judge_parse_yes_no(text: str, expected: bool) -> None:
    assert parse_yes_no(text) is expected


# ---------------------------------------------------------------------------
# Payload shape.
# ---------------------------------------------------------------------------

def test_openai_judge_payload_carries_image_and_instruction() -> None:
    judge = _judge(transport=lambda p: _reply("NO"))
    payload = judge.build_payload(_image(), "place the cup on the shelf")

    assert payload["model"] == "nvidia/Cosmos3-Edge"
    assert payload["temperature"] == 0.0
    (message,) = payload["messages"]
    assert message["role"] == "user"
    image_part, text_part = message["content"]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert text_part["type"] == "text"
    assert "place the cup on the shelf" in text_part["text"]
    assert "YES or NO" in text_part["text"]


def test_openai_judge_custom_prompt_template() -> None:
    judge = _judge(transport=lambda p: _reply("NO"),
                   prompt_template="Done with {instruction}? YES/NO")
    payload = judge.build_payload(_image(), "grasp")
    assert payload["messages"][0]["content"][1]["text"] == "Done with grasp? YES/NO"


# ---------------------------------------------------------------------------
# is_complete — verdicts + failure semantics.
# ---------------------------------------------------------------------------

def test_openai_judge_yes_verdict_is_complete() -> None:
    calls: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(payload)
        return _reply("YES")

    judge = _judge(transport=transport)
    assert judge.is_complete(_image(), "grasp the bowl") is True
    assert len(calls) == 1


def test_openai_judge_no_verdict_is_not_complete() -> None:
    judge = _judge(transport=lambda p: _reply("NO"))
    assert judge.is_complete(_image(), "grasp the bowl") is False


def test_openai_judge_transport_failure_judges_not_complete() -> None:
    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        raise ConnectionError("endpoint down")

    judge = _judge(transport=transport)
    # Never raises — a flaky judge must not abort the episode.
    assert judge.is_complete(_image(), "grasp") is False


def test_openai_judge_malformed_response_judges_not_complete() -> None:
    judge = _judge(transport=lambda p: {"unexpected": "shape"})
    assert judge.is_complete(_image(), "grasp") is False


# ---------------------------------------------------------------------------
# Auth wiring.
# ---------------------------------------------------------------------------

def test_openai_judge_api_key_env_sets_bearer_header(monkeypatch) -> None:
    monkeypatch.setenv("JUDGE_API_KEY", "sk-test-123")
    judge = _judge(transport=lambda p: _reply("NO"), api_key_env="JUDGE_API_KEY")
    assert judge._headers["Authorization"] == "Bearer sk-test-123"


def test_openai_judge_no_key_means_no_auth_header() -> None:
    judge = _judge(transport=lambda p: _reply("NO"))
    assert "Authorization" not in judge._headers


# ---------------------------------------------------------------------------
# Protocol + gate integration.
# ---------------------------------------------------------------------------

def test_openai_judge_satisfies_completion_detector_protocol() -> None:
    judge = _judge(transport=lambda p: _reply("YES"))
    assert isinstance(judge, CompletionDetector)


def test_openai_judge_drops_into_chunk_completion_gate() -> None:
    verdicts = iter(["NO", "YES"])
    judge = _judge(transport=lambda p: _reply(next(verdicts)))
    gate = ChunkCompletionGate(detector=judge, n_action_steps=2)

    image = _image()
    # Steps 1-2: first boundary -> judge says NO -> keep going.
    assert gate.update(image, "grasp") is False
    assert gate.update(image, "grasp") is False
    # Steps 3-4: second boundary -> judge says YES -> hand back.
    assert gate.update(image, "grasp") is False  # mid-chunk, no judge call
    assert gate.update(image, "grasp") is True


def test_openai_judge_extra_body_merges_into_payload() -> None:
    """extra_body keys ride top-level on every request (e.g. vLLM-Omni's
    ``modalities: ["text"]`` routing knob) without clobbering the core fields."""
    judge = _judge(
        transport=lambda p: _reply("YES"),
        extra_body={"modalities": ["text"], "model": "should-not-win"},
    )
    payload = judge.build_payload(_image(), "grasp the capsule")
    assert payload["modalities"] == ["text"]
    # Core fields always win over extra_body on collision.
    assert payload["model"] != "should-not-win"
    assert payload["messages"]
