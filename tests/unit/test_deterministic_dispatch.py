"""Unit tests for the deterministic dispatch (concept lexicon, no LLM)."""

from __future__ import annotations

import pytest

from odyssey.runners.agents.brain.dispatch import DispatchCandidate, select_specialists


class FakeCandidate:
    """Minimal dispatch candidate: name + routing vocab + vision capability."""

    def __init__(self, name: str, vocab: set[str], *, is_vision: bool = False) -> None:
        self._name = name
        self._vocab = set(vocab)
        self._is_vision = is_vision

    @property
    def name(self) -> str:
        return self._name

    @property
    def vocab(self) -> set[str]:
        return self._vocab

    @property
    def is_vision(self) -> bool:
        return self._is_vision


def test_fake_candidate_satisfies_protocol() -> None:
    assert isinstance(FakeCandidate("x", set()), DispatchCandidate)


def test_exact_token_overlap_engages() -> None:
    sd = FakeCandidate("inspector", {"inspection"})
    [d] = select_specialists("inspect the equipment", [sd], has_image=False)
    assert d.engaged and "matches" in d.reason


def test_synonym_via_concept_lexicon_engages() -> None:
    # "examine"->inspect, "machinery"->equipment: concept overlap, not token overlap.
    sd = FakeCandidate("auditor", {"equipment", "inspect"})
    [d] = select_specialists("examine the machinery", [sd], has_image=False)
    assert d.engaged


def test_fuzzy_fallback_engages_on_typo() -> None:
    # "cartograpy" (typo) is outside the concept lexicon, so only the fuzzy
    # fallback can match it to the specialist's "cartography" vocab.
    sd = FakeCandidate("mapper", {"cartography"})
    [d] = select_specialists("do a cartograpy", [sd], has_image=False)
    assert d.engaged and "fuzzy" in d.reason


def test_no_signal_no_image_skips_in_selective_mode() -> None:
    sd = FakeCandidate("mapper", {"cartography"})
    [d] = select_specialists("say hello", [sd], has_image=False)
    assert not d.engaged and d.reason == "no signal match"


def test_image_plus_vision_engages_without_text_signal() -> None:
    sd = FakeCandidate("scene", {"cartography"}, is_vision=True)
    [d] = select_specialists("say hello", [sd], has_image=True)
    assert d.engaged and "image present" in d.reason


def test_image_without_vision_does_not_engage() -> None:
    sd = FakeCandidate("mapper", {"cartography"}, is_vision=False)
    [d] = select_specialists("say hello", [sd], has_image=True)
    assert not d.engaged


def test_engage_all_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ODYSSEY_DISPATCH", "all")
    sds = [FakeCandidate("a", set()), FakeCandidate("b", set())]
    decisions = select_specialists("anything", sds, has_image=False)
    assert all(d.engaged for d in decisions)


def test_empty_specialists_returns_empty() -> None:
    assert select_specialists("inspect", [], has_image=True) == []
