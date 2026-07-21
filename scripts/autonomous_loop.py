"""Autonomous multi-agent loop — the SPECIALIST decision core (pure, testable).

Operates over the OBSERVER-measured funnel abstraction, so it is task-agnostic:
the same logic drives any pick-place-shaped task. Given the current funnel (with
Wilson CIs) and the ladder state, it (a) finds the dominant failure transition,
(b) selects the next applicable lever, (c) emits a human-readable rationale, and
(d) decides promotion from a gate result. No torch / GPU / task specifics here —
that all lives in the levers' drivers.

See examples/ur5e-drugsort/AUTONOMOUS_LOOP.md for the design.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

# Funnel stages in order — each task's TaskSpec supplies these; this is the
# pick-place default. A "transition" is stage[i] -> stage[i+1].
FUNNEL_STAGES = ("reached", "centered", "closed", "lifted", "transported", "seated")

# Which lever attacks which failing transition (see the ladder table). Ordered
# by preference within a transition (cheapest / highest-leverage first).
TRANSITION_LEVERS: dict[str, list[str]] = {
    "start->reached":     ["L0_base"],                      # arm doesn't reach
    "reached->centered":  ["L1_steering", "L0_base"],       # approaches, not centered
    "centered->closed":   ["L2_selection", "L5_servo"],     # centered, won't close/capture
    "closed->lifted":     ["L3_dagger", "L5_servo", "L4_distill"],  # closes, no capture
    "lifted->transported":["L6_place"],
    "transported->seated":["L6_place"],
}

# Lever preconditions: a lever is applicable only if the ladder has reached it.
LEVER_ORDER = ["L0_base", "L1_steering", "L2_selection", "L3_dagger",
               "L4_distill", "L5_servo", "L6_place"]

# A lever is RETIRED after this many gate failures (falsifiability).
MAX_LEVER_FAILURES = 2


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(rate, lo, hi) Wilson 95% interval; (0,0,0) for n==0."""
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


@dataclass
class LadderState:
    reached_lever_idx: int = 0        # highest lever the ladder has unlocked
    failures: dict[str, int] = field(default_factory=dict)   # lever -> gate-fail count
    best_seated_rate: float = 0.0
    iteration: int = 0

    def retired(self, lever: str) -> bool:
        return self.failures.get(lever, 0) >= MAX_LEVER_FAILURES

    def unlocked(self, lever: str) -> bool:
        return LEVER_ORDER.index(lever) <= self.reached_lever_idx


def funnel_transitions(funnel: dict[str, dict[str, Any]]) -> list[tuple[str, float]]:
    """Per-transition survival drop = rate[i] - rate[i+1] (fraction of episodes
    lost at that step). Higher drop = bigger bottleneck. Uses point rates."""
    stages = ["start"] + list(FUNNEL_STAGES)
    rates = {"start": 1.0}
    for s in FUNNEL_STAGES:
        rates[s] = float(funnel.get(s, {}).get("rate", 0.0))
    out = []
    for i in range(len(stages) - 1):
        a, b = stages[i], stages[i + 1]
        out.append((f"{a}->{b}", rates[a] - rates[b]))
    return out


def diagnose(funnel: dict[str, dict[str, Any]], ladder: LadderState,
             wired: set[str] | None = None) -> dict[str, Any]:
    """SPECIALIST: pick the next lever + rationale from the funnel + ladder.

    Strategy: attack the LARGEST funnel drop with the highest-preference lever
    FOR THAT TRANSITION that is not retired and — when ``wired`` is given — is
    actually BUILT. If the chosen lever isn't unlocked yet, jump the ladder
    straight to it (unlock everything up to and including it; unbuilt rungs in
    between are simply skipped — the funnel, not the rung order, dictates what to
    run). Only if a transition's levers are all retired/unbuilt do we fall
    through to the next-largest drop; if none remain, escalate.
    """
    trans = sorted(funnel_transitions(funnel), key=lambda t: -t[1])
    considered = []
    for name, drop in trans:
        for lev in TRANSITION_LEVERS.get(name, []):
            considered.append((name, drop, lev))
            if ladder.retired(lev):
                continue
            if wired is not None and lev not in wired:
                continue  # can't run a lever with no driver — try the next pref
            unlock_to = LEVER_ORDER.index(lev)
            if not ladder.unlocked(lev):
                return {
                    "bottleneck": name, "drop": round(drop, 3),
                    "lever": lev, "action": "unlock+run", "unlock_to": unlock_to,
                    "rationale": (f"Largest funnel loss is {name} ({drop:.0%} of episodes). "
                                  f"{lev} is the built lever that attacks it; jumping the ladder "
                                  f"to it (skipping unbuilt intermediate rungs)."),
                }
            return {
                "bottleneck": name, "drop": round(drop, 3),
                "lever": lev, "action": "run", "unlock_to": unlock_to,
                "rationale": (f"Largest funnel loss is {name} ({drop:.0%} of episodes lost there). "
                              f"{lev} directly attacks that transition; unlocked and not retired."),
            }
    # every applicable lever is retired or unbuilt -> escalate
    return {
        "bottleneck": trans[0][0] if trans else None,
        "drop": round(trans[0][1], 3) if trans else 0.0,
        "lever": None, "action": "escalate",
        "rationale": ("No BUILT, non-retired lever remains for the bottleneck transitions "
                      "(unbuilt levers can't run; retired ones failed their gate twice). "
                      "Escalate to a human/architectural decision with the funnel as evidence."),
        "considered": [f"{n}:{l}" for n, _, l in considered],
    }


def gate_result(lever: str, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Decide promotion. Generic gate: the metric the lever targets must improve
    with non-overlapping-or-better CI vs the incumbent. `before`/`after` carry
    {metric: {"wilson": (p,lo,hi), "k":, "n":}} for the lever's target metric.

    Promotion rule: after.lo >= before.point (the new lower bound clears the old
    point estimate) OR after.point > before.point AND after.lo > before.lo
    (dominant shift). Conservative on purpose — no promotion on noise.
    """
    tgt = LEVER_TARGET_METRIC.get(lever, "seated")
    b = before.get(tgt, {}).get("wilson", (0.0, 0.0, 0.0))
    a = after.get(tgt, {}).get("wilson", (0.0, 0.0, 0.0))
    bp, blo, _ = b
    ap, alo, _ = a
    # Conservative: promote only if the new CI LOWER bound clears the incumbent's
    # POINT estimate (a genuinely significant shift). No promotion on noise.
    # Special-case a first non-zero result over a zero incumbent.
    promote = bool(alo >= bp - 1e-9) if bp > 1e-9 else bool(alo > 1e-9)
    return {
        "target_metric": tgt,
        "before": {"rate": round(bp, 3), "lo": round(blo, 3)},
        "after": {"rate": round(ap, 3), "lo": round(alo, 3)},
        "promote": promote,
        "rationale": (f"{tgt}: {bp:.1%} -> {ap:.1%} (CI-lo {blo:.1%} -> {alo:.1%}). "
                      + ("PROMOTE: new CI lower bound clears the incumbent point." if promote
                         else "REJECT: improvement within noise; record as tried branch.")),
    }


LEVER_TARGET_METRIC: dict[str, str] = {
    "L0_base": "reached", "L1_steering": "centered", "L2_selection": "closed",
    "L3_dagger": "lifted", "L4_distill": "seated", "L5_servo": "closed",
    "L6_place": "seated",
}


def target_met(funnel: dict[str, dict[str, Any]], seated_lo_target: float = 0.5) -> bool:
    """Loop stops when the seated-rate Wilson LOWER bound clears the target."""
    _, lo, _ = tuple(funnel.get("seated", {}).get("wilson", (0.0, 0.0, 0.0)))
    return lo >= seated_lo_target
