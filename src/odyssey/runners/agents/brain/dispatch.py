"""Deterministic dispatch — decide which specialists to engage, no LLM.

The DISPATCH stage of dispatch -> compose -> converge. Engagement is decided
from explicit, auditable signals: semantic term overlap between the request and
each specialist's vocabulary (both sides expanded through a small concept
lexicon + a crude stemmer so synonyms/paraphrases match), with a fuzzy fallback
for typos/morphology, and an image-present + vision-capable rule.

Module philosophy: *a readable rule table beats an opaque score.* Extend
``_CONCEPT_LEXICON`` rather than reaching for an embedding model — matching
stays deterministic, local, and testable without any model dependency
(stdlib ``re`` + ``difflib`` only).
"""

from __future__ import annotations

import difflib
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_DISPATCH_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "onto",
    "your", "you", "are", "was", "will", "would", "should", "could", "have",
    "has", "had", "then", "than", "them", "they", "there", "here", "what",
    "when", "where", "which", "while", "about", "over", "under", "please",
}


def _dispatch_terms(text: str) -> set[str]:
    """Significant lowercase tokens for dispatch matching (len>=4, non-stopword)."""
    return {
        t
        for t in re.split(r"[^a-z0-9]+", (text or "").lower())
        if len(t) >= 4 and t not in _DISPATCH_STOPWORDS
    }


# Small, transparent concept lexicon so engagement matches on MEANING, not just
# identical tokens. Each canonical concept lists surface forms that resolve to
# it; both the request and each specialist's vocabulary are expanded through
# this map before overlap is computed. EXTEND this table rather than reaching
# for an embedding model.
_CONCEPT_LEXICON: dict[str, set[str]] = {
    "inspect": {"inspect", "inspection", "examine", "check", "monitor", "survey",
                "audit", "assess", "evaluate", "review", "scan", "look"},
    "equipment": {"equipment", "machinery", "machine", "device", "asset",
                  "hardware", "apparatus", "unit", "component", "gear",
                  "instrument", "motor", "pump", "valve", "pipe", "panel"},
    "anomaly": {"anomaly", "anomalies", "abnormal", "irregular", "fault",
                "defect", "malfunction", "issue", "problem", "fail", "failure"},
    "overheat": {"overheat", "overheating", "hotspot", "hot", "heat", "thermal",
                 "temperature", "warm", "burning"},
    "leak": {"leak", "leaking", "leakage", "spill", "drip", "seepage", "seep"},
    "damage": {"damage", "crack", "cracked", "broken", "break", "dent", "wear",
               "corrosion", "rust", "worn", "degraded"},
    "patrol": {"patrol", "route", "rounds", "tour", "sweep", "stop", "waypoint"},
    "navigate": {"navigate", "navigation", "drive", "move", "travel", "path",
                 "traverse", "roam", "wander"},
    "grasp": {"grasp", "grip", "pick", "place", "manipulate", "handle", "grab",
              "lift", "assemble", "assembly"},
    "obstacle": {"obstacle", "collision", "avoid", "avoidance", "clearance",
                 "blocked", "hazard"},
    "inventory": {"inventory", "stock", "count", "counting", "barcode", "shelf",
                  "warehouse"},
    "safety": {"safety", "hazard", "danger", "risk", "unsafe", "emergency"},
    "quality": {"quality", "flaw", "tolerance", "conformance", "reject",
                "grade", "finish"},
    "report": {"report", "reporting", "findings", "record", "document",
               "alert", "notify", "flag"},
    "measure": {"measure", "measurement", "gauge", "reading", "sensor",
                "sensing", "telemetry"},
    "scene": {"scene", "object", "objects", "color", "colour", "shape",
              "identify", "detect", "locate", "recognize", "recognise",
              "describe", "vision", "visual", "see"},
}


def _stem(token: str) -> str:
    """Very small suffix-stripping stemmer — deterministic, no dependencies.

    Collapses common inflections so ``inspecting``/``inspected``/``inspects``
    all map to the same base as ``inspect``. Intentionally crude; it only needs
    to normalize morphology enough for concept-lexicon and fuzzy matching.
    """
    for suf in ("ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "s"):
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            base = token[: -len(suf)]
            if suf == "ies":
                base += "y"
            return base
    return token


# Reverse index: stemmed surface form -> concept tags. Built once at import.
_SURFACE_TO_CONCEPTS: dict[str, set[str]] = {}
for _concept, _surfaces in _CONCEPT_LEXICON.items():
    for _surface in _surfaces:
        _SURFACE_TO_CONCEPTS.setdefault(_stem(_surface), set()).add(_concept)


def _concept_tags(term: str) -> set[str]:
    """Concept tags a term resolves to (namespaced so they never collide with a
    literal stemmed token)."""
    return {f"concept:{c}" for c in _SURFACE_TO_CONCEPTS.get(_stem(term), set())}


def _expand_terms(terms: set[str]) -> set[str]:
    """Expand raw tokens into stems + concept tags for semantic overlap.

    Literal exact matches are preserved (the stem of an unchanged token is the
    token itself), so this is strictly a superset of plain token overlap.
    """
    expanded: set[str] = set()
    for t in terms:
        expanded.add(_stem(t))
        expanded |= _concept_tags(t)
    return expanded


def _humanize_concepts(overlap: set[str]) -> set[str]:
    """Strip the ``concept:`` namespace prefix for readable audit reasons."""
    return {
        t.split("concept:", 1)[1] if t.startswith("concept:") else t
        for t in overlap
    }


def _fuzzy_overlap(
    a_terms: set[str], b_terms: set[str], threshold: float = 0.86
) -> set[tuple[str, str]]:
    """Near-identical token pairs across two sets (typos / morphology the
    stemmer and concept map miss). Deterministic; term sets are small."""
    hits: set[tuple[str, str]] = set()
    for a in a_terms:
        for b in b_terms:
            if a == b:
                continue
            if difflib.SequenceMatcher(None, a, b).ratio() >= threshold:
                hits.add((a, b))
    return hits


@runtime_checkable
class DispatchCandidate(Protocol):
    """The dispatch surface a specialist must expose (structural).

    ``vocab`` is the specialist's authored routing vocabulary (tokens from its
    name/id, and any configured keywords); ``is_vision`` marks a multimodal
    specialist that can consume the camera image.
    """

    @property
    def name(self) -> str: ...

    @property
    def vocab(self) -> set[str]: ...

    @property
    def is_vision(self) -> bool: ...


@dataclass
class Decision:
    """One dispatch decision — auditable per candidate, engaged or skipped."""

    specialist: DispatchCandidate
    engaged: bool
    reason: str
    is_vision: bool


def select_specialists(
    instruction: str,
    specialists: Sequence[DispatchCandidate],
    has_image: bool,
) -> list[Decision]:
    """Decide which specialists to engage for a request — deterministically.

    Cascade: engage-all mode -> concept overlap -> fuzzy overlap ->
    image-present + vision-capable -> skip.

    Modes (env ``ODYSSEY_DISPATCH``):
      * ``selective`` (default): engage only matching specialists; a request
        with no signal engages NONE (pure Pilot).
      * ``all``: engage every specialist (escape hatch).

    Returns one ``Decision`` per candidate so the full table is auditable.
    """
    if not specialists:
        return []

    mode = os.getenv("ODYSSEY_DISPATCH", "selective").lower()
    msg_terms = _dispatch_terms(instruction)
    msg_expanded = _expand_terms(msg_terms)

    decisions: list[Decision] = []
    for sd in specialists:
        vocab = sd.vocab
        concept_overlap = msg_expanded & _expand_terms(vocab)
        is_vision = sd.is_vision

        fuzzy: set[tuple[str, str]] = set()
        if not concept_overlap and mode != "all":
            fuzzy = _fuzzy_overlap(msg_terms, vocab)

        if mode == "all":
            engaged, reason = True, "engage-all dispatch (ODYSSEY_DISPATCH=all)"
        elif concept_overlap:
            matched = sorted(_humanize_concepts(concept_overlap))[:4]
            engaged, reason = True, f"request matches {matched}"
        elif fuzzy:
            pairs = sorted({f"{a}~{b}" for a, b in fuzzy})[:3]
            engaged, reason = True, f"request fuzzy-matches {pairs}"
        elif has_image and is_vision:
            engaged, reason = True, "image present + vision-capable specialist"
        else:
            engaged, reason = False, "no signal match"

        decisions.append(
            Decision(specialist=sd, engaged=engaged, reason=reason, is_vision=is_vision)
        )

    return decisions
