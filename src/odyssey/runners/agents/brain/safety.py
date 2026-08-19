"""SafetyBoundary — deterministic, monitor-only action checks.

A small brain-side gate the action traverses before it leaves the runtime.
MONITOR-only by default: it reports (and can expose a clamped copy of) the
action, but does NOT mutate the action stream — enforcement is a later,
separate concern at the actuator boundary. It is plain runtime code (numpy
only), never inside a framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class SafetyFinding:
    """One check result: ``severity`` in {"violation", "warning"}."""

    check: str
    severity: str
    detail: str


@dataclass
class SafetyResult:
    """Outcome of one safety pass over an action vector."""

    ok: bool
    mode: str
    findings: list[SafetyFinding]
    clamped: NDArray[np.floating[Any]]

    @property
    def violation_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "violation")


class SafetyBoundary:
    """Structural action checks (finite / shape / range), monitor-only.

    Parameters
    ----------
    expected_dim:
        Expected action dimensionality (e.g. 7 for a 7-DoF end-effector delta).
        ``None`` skips the shape check.
    abs_limit:
        Symmetric magnitude bound each component is checked (and clamped, in the
        returned copy) against. The clamp is reported, NOT applied to the stream
        while ``mode == "monitor"``.
    mode:
        ``"monitor"`` (default) reports only; ``"enforce"`` returns the clamped
        action. Enforce is provided for completeness but off by default.
    """

    def __init__(
        self,
        *,
        expected_dim: int | None = 7,
        abs_limit: float = 1.0,
        mode: str = "monitor",
    ) -> None:
        self._expected_dim = expected_dim
        self._abs_limit = abs_limit
        self._mode = mode

    def evaluate(self, action: NDArray[np.floating[Any]]) -> SafetyResult:
        """Run the checks and return a structured result (does not log)."""
        findings: list[SafetyFinding] = []
        arr = np.asarray(action, dtype=np.float64)

        if not np.all(np.isfinite(arr)):
            findings.append(
                SafetyFinding("finite", "violation", "action contains NaN/Inf")
            )
            arr = np.nan_to_num(arr, nan=0.0, posinf=self._abs_limit, neginf=-self._abs_limit)

        if self._expected_dim is not None and arr.shape[-1] != self._expected_dim:
            findings.append(
                SafetyFinding(
                    "shape",
                    "warning",
                    f"expected dim {self._expected_dim}, got {arr.shape[-1]}",
                )
            )

        out_of_range = int(np.count_nonzero(np.abs(arr) > self._abs_limit))
        clamped = np.clip(arr, -self._abs_limit, self._abs_limit)
        if out_of_range:
            findings.append(
                SafetyFinding(
                    "range",
                    "warning",
                    f"{out_of_range} component(s) exceed |{self._abs_limit}| (clamped copy exposed)",
                )
            )

        ok = not any(f.severity == "violation" for f in findings)
        return SafetyResult(ok=ok, mode=self._mode, findings=findings, clamped=clamped)

    def check(self, action: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Evaluate + log; return the action (monitor) or the clamped one (enforce).

        MONITOR-only by default: the original action is returned unchanged so the
        boundary never silently alters behavior; findings surface as log lines.
        """
        result = self.evaluate(action)
        if result.findings:
            logger.warning(
                "safety [%s]: %d finding(s) (action %s)",
                self._mode,
                len(result.findings),
                "clamped" if self._mode == "enforce" else "NOT modified",
            )
            for f in result.findings:
                logger.warning("safety   %s: %s — %s", f.severity, f.check, f.detail)
        if self._mode == "enforce":
            return result.clamped
        return np.asarray(action, dtype=np.float64)
