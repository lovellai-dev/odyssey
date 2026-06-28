"""CPUMockRunner — always-works fallback for CI and the engine smoke path.

Emits a small number of progress events and returns a plausible result
summary. Does no actual training or evaluation. Honors cooperative
cancellation between progress steps.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from odyssey.runners.base import WILDCARD_TYPE, Runner, TaskContext
from odyssey.spec.tasks import EvaluationTask, TaskKind, TrainingTask


class CPUMockRunner(Runner):
    """No-op runner that exercises the engine's lifecycle without touching
    a GPU, a dataset, or the network. Registered for both kinds, all types."""

    def __init__(self, *, step_delay_seconds: float = 0.0):
        self._step_delay = step_delay_seconds

    @property
    def name(self) -> str:
        return "cpu_mock"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.TRAINING, TaskKind.EVALUATION}

    @property
    def supported_types(self) -> set[str]:
        return {WILDCARD_TYPE}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        spec = context.task.spec
        await context.emit_progress("model_loading", step="load_target")
        await self._sleep(context)
        if context.cancelled():
            return {"_mock_cancelled": True}

        if isinstance(spec, TrainingTask):
            return await self._run_training(context, spec)
        if isinstance(spec, EvaluationTask):
            return await self._run_evaluation(context, spec)
        raise TypeError(f"CPUMockRunner: unsupported task type {type(spec).__name__}")

    async def _run_training(
        self, context: TaskContext, spec: TrainingTask
    ) -> dict[str, Any]:
        steps = 3
        for i in range(1, steps + 1):
            if context.cancelled():
                return {"_mock_cancelled_at_step": i}
            await context.emit_progress(
                "executing",
                step="training_step",
                step_index=i,
                step_total=steps,
                step_label=f"mock step {i}/{steps}",
            )
            await self._sleep(context)

        await context.emit_progress("checkpoint_saving", step="finalize")
        return {
            "_mock": True,
            "checkpoint_path": f"mock://{context.task.id}/final",
            "training_type": spec.training_type.value,
            "steps": steps,
        }

    async def _run_evaluation(
        self, context: TaskContext, spec: EvaluationTask
    ) -> dict[str, Any]:
        episodes = min(spec.num_episodes, 3)
        successes = 0
        for ep in range(1, episodes + 1):
            if context.cancelled():
                return {"_mock_cancelled_at_episode": ep}
            await context.emit_progress(
                "executing",
                step="episode_complete",
                step_index=ep,
                step_total=episodes,
                step_label=f"mock episode {ep}/{episodes}",
            )
            successes += 1
            await self._sleep(context)

        success_rate = successes / episodes if episodes else 1.0
        from odyssey.runners.evals._common import grade
        return {
            "_mock": True,
            "num_episodes": episodes,
            "success_rate": success_rate,
            "performance_score": success_rate,
            "letter_grade": grade(success_rate),
            "passed": success_rate >= 0.5,
            "benchmark_name": spec.benchmark_name,
        }

    async def _sleep(self, context: TaskContext) -> None:
        if self._step_delay <= 0:
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                context.cancel_event.wait(), timeout=self._step_delay
            )
