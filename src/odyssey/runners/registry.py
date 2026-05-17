"""RunnerRegistry — dispatches a TaskSpec to the right Runner.

Routing key: ``(TaskKind, type-string)`` where type-string is the task's
``training_type`` or ``evaluation_type`` value. The registry also accepts
wildcard registrations (type ``"*"``) so a single runner can serve every
type within its kinds.

Selection order:
  1. ``override`` argument matches a registered runner's ``name``.
  2. Specific ``(kind, type)`` match — first registered wins.
  3. Wildcard ``(kind, "*")`` match — first registered wins.
  4. ``NoRunnerForTaskError``.
"""

from __future__ import annotations

from odyssey.engine.errors import NoRunnerForTaskError
from odyssey.runners.base import WILDCARD_TYPE, Runner
from odyssey.spec.tasks import EvaluationTask, TaskKind, TaskSpec, TrainingTask


class RunnerRegistry:
    def __init__(self) -> None:
        self._by_key: dict[tuple[TaskKind, str], list[Runner]] = {}
        self._by_name: dict[str, Runner] = {}

    def register(self, runner: Runner) -> None:
        self._by_name[runner.name] = runner
        for kind in runner.supported_kinds:
            for type_value in runner.supported_types:
                self._by_key.setdefault((kind, type_value), []).append(runner)

    def select(self, task: TaskSpec, *, override: str | None = None) -> Runner:
        if override is not None:
            if override in self._by_name:
                return self._by_name[override]
            raise NoRunnerForTaskError(task.kind, f"override={override!r}")

        kind = TaskKind(task.kind)
        type_value = _type_value(task)

        specific = self._by_key.get((kind, type_value))
        if specific:
            return specific[0]
        wildcard = self._by_key.get((kind, WILDCARD_TYPE))
        if wildcard:
            return wildcard[0]
        raise NoRunnerForTaskError(task.kind, type_value)


def _type_value(task: TaskSpec) -> str:
    if isinstance(task, TrainingTask):
        return task.training_type.value
    if isinstance(task, EvaluationTask):
        return task.evaluation_type.value
    raise TypeError(f"Unrecognized task kind: {type(task).__name__}")
