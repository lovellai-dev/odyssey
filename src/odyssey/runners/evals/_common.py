"""Shared evaluation runner utilities and results contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from odyssey.runners.base import TaskContext


def grade(success_rate: float) -> str:
    """Map a success rate to the A-F letter grade. Single source of truth."""
    if success_rate >= 0.9:
        return "A"
    if success_rate >= 0.8:
        return "B"
    if success_rate >= 0.7:
        return "C"
    if success_rate >= 0.6:
        return "D"
    return "F"


def build_eval_summary(
    *,
    num_episodes: int,
    successes: int,
    episode_returns: list[float],
    benchmark_name: str,
    checkpoint_path: Path | str,
    metrics: dict[str, Any] | None = None,
    success_rate: float | None = None,
    performance_score: float | None = None,
) -> dict[str, Any]:
    """Build the success_rate / performance_score / letter_grade / passed / metrics

    dictionary that the trainer expects. One place to evolve the contract.
    """
    if success_rate is None:
        success_rate = successes / num_episodes if num_episodes else 0.0
    if performance_score is None:
        performance_score = (
            sum(episode_returns) / num_episodes if num_episodes else 0.0
        )

    out_metrics = dict(metrics or {})
    out_metrics.setdefault("successes", successes)
    out_metrics.setdefault(
        "episode_returns", [round(r, 4) for r in episode_returns]
    )
    out_metrics.setdefault("benchmark", benchmark_name)
    out_metrics.setdefault("checkpoint_path", str(checkpoint_path))

    return {
        "num_episodes": num_episodes,
        "success_rate": round(success_rate, 4),
        "performance_score": round(performance_score, 4),
        "letter_grade": grade(success_rate),
        "passed": success_rate >= 0.5,
        "metrics": out_metrics,
    }


def resolve_eval_checkpoint(context: TaskContext) -> Path:
    """Find the PILOT checkpoint the evaluation should run.

    Uses ``context.agent_checkpoints`` (populated by the engine) to find
    the first PILOT agent with a trained checkpoint. Multi-agent loadouts
    (PILOT + SPECIALIST) are supported — the SPECIALIST doesn't need a
    checkpoint (it uses its base model for inference only).
    """
    from odyssey.spec.agents import AgentRole

    for agent in context.agents or context.mission.spec.robot.agents:
        if agent.role != AgentRole.PILOT:
            continue
        checkpoint = (context.agent_checkpoints or {}).get(agent.id)
        if not checkpoint:
            checkpoint = context.mission.latest_checkpoint_for(agent.id)
        if checkpoint:
            return Path(checkpoint)

    raise ValueError(
        "No completed training task produced a checkpoint for any PILOT "
        "agent on this mission — cannot evaluate."
    )
