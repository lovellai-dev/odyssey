"""Tests for the providers → engine → TaskContext threading.

Covers:
  * MissionEngine without providers still works (back-compat with
    Week-2 tests).
  * MissionEngine with providers exposes them on TaskContext.
  * OpenVLA runner consults ctx.providers when an HFModelRef is set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from odyssey.engine import MissionEngine
from odyssey.persistence import InMemoryPersistence
from odyssey.providers import (
    ProviderRegistry,
    ResolvedModel,
    ResolvedRobot,
    RobotProvider,
)
from odyssey.providers.huggingface import HFModelProvider
from odyssey.runners import (
    WILDCARD_TYPE,
    Runner,
    RunnerRegistry,
    TaskContext,
)
from odyssey.spec import (
    AgentRole,
    AgentSpec,
    EvaluationTask,
    EvaluationType,
    HFModelRef,
    Mission,
    MissionMetadata,
    RobotSpec,
    TaskKind,
    TrainingTask,
    TrainingType,
)
from odyssey.telemetry import EventPublisher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NullPublisher(EventPublisher):
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        return


class _CapturingRunner(Runner):
    """Records the providers ref on the TaskContext it receives."""

    def __init__(self) -> None:
        self.seen_providers: ProviderRegistry | None = None

    @property
    def name(self) -> str:
        return "capturing"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.TRAINING, TaskKind.EVALUATION}

    @property
    def supported_types(self) -> set[str]:
        return {WILDCARD_TYPE}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        self.seen_providers = context.providers
        return {"ok": True}


class _StubRobotProvider(RobotProvider):
    @property
    def name(self) -> str:
        return "local"

    async def resolve(self, spec: object) -> ResolvedRobot:  # type: ignore[override]
        return ResolvedRobot(provider="local", name="x")


def _spec() -> Mission:
    return Mission(
        metadata=MissionMetadata(name="ctx-test"),
        objective="o",
        acceptance_criteria="a",
        robot=RobotSpec(
            embodiment="franka_panda",
            agents=[
                AgentSpec(
                    id="pilot",
                    role=AgentRole.PILOT,
                    model=HFModelRef(base="openvla/openvla-7b"),
                ),
            ],
        ),
        tasks=[
            TrainingTask(
                name="train",
                training_type=TrainingType.DEMONSTRATION,
                agent_id="pilot",
            ),
            EvaluationTask(
                name="eval",
                evaluation_type=EvaluationType.ROBOSUITE,
                benchmark_name="Lift",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Engine ↔ TaskContext threading
# ---------------------------------------------------------------------------

async def test_engine_without_providers_passes_none() -> None:
    capturing = _CapturingRunner()
    runners = RunnerRegistry()
    runners.register(capturing)
    engine = MissionEngine(
        persistence=InMemoryPersistence(),
        runners=runners,
        event_publisher=_NullPublisher(),
    )
    await engine.initialize()
    run = await engine.create_mission(_spec())
    await engine.start_mission(run.id)
    assert capturing.seen_providers is None


async def test_engine_with_providers_passes_registry() -> None:
    capturing = _CapturingRunner()
    runners = RunnerRegistry()
    runners.register(capturing)

    providers = ProviderRegistry()
    providers.register_robot(_StubRobotProvider(), handles="local")

    engine = MissionEngine(
        persistence=InMemoryPersistence(),
        runners=runners,
        event_publisher=_NullPublisher(),
        providers=providers,
    )
    await engine.initialize()
    run = await engine.create_mission(_spec())
    await engine.start_mission(run.id)
    assert capturing.seen_providers is providers


# ---------------------------------------------------------------------------
# OpenVLA runner uses providers when set
# ---------------------------------------------------------------------------

class _FakeHfApi:
    def model_info(self, repo_id: str, revision: str | None = None) -> object:
        from types import SimpleNamespace
        return SimpleNamespace(sha="resolved-sha-abc")


async def test_openvla_runner_substitutes_provider_fetched_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ctx.providers is set and model is HFModelRef, the runner
    should call provider.resolve + provider.fetch and use the resulting
    path as vla_path — overriding env-var fallbacks."""
    from odyssey.runners.openvla import _resolve_and_fetch_hf_model

    fetched_path = tmp_path / "fetched-model"
    fetched_path.mkdir()

    class _FakeProvider(HFModelProvider):
        def __init__(self) -> None:
            super().__init__(api=_FakeHfApi())
            self.resolve_calls = 0
            self.fetch_calls = 0

        async def fetch(self, resolved: ResolvedModel, dest: Path) -> Path:
            self.fetch_calls += 1
            return fetched_path

    provider = _FakeProvider()
    providers = ProviderRegistry()
    providers.register_model(provider)

    from odyssey.engine.records import MissionRun
    mission = MissionRun.from_spec(_spec())
    train = mission.tasks[0]
    ctx = TaskContext(
        task=train,
        mission=mission,
        publisher=_NullPublisher(),
        output_dir=tmp_path,
        providers=providers,
    )
    ref = HFModelRef(base="openvla/openvla-7b")
    resolved = await _resolve_and_fetch_hf_model(ctx, ref, tmp_path / "model")
    assert resolved == fetched_path
    assert provider.fetch_calls == 1


async def test_openvla_runner_works_without_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Back-compat: when providers is None, the runner relies on
    build_openvla_argv's existing env/config/HF-id fallback against
    the agent's model base."""
    from odyssey.runners.openvla import build_openvla_argv

    monkeypatch.delenv("OPENVLA_OPENVLA_7B_PATH", raising=False)
    task = TrainingTask(
        name="ft",
        training_type=TrainingType.DEMONSTRATION,
        agent_id="pilot",
    )
    argv = build_openvla_argv(
        task=task,
        agent_model_base="openvla/openvla-7b",
        output_dir=tmp_path,
        run_id="r",
    )
    idx = argv.index("--vla_path")
    # Fallback: HF base id itself.
    assert argv[idx + 1] == "openvla/openvla-7b"
