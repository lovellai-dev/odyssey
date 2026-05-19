"""Engine-level robot resolution at create_mission time.

When a ProviderRegistry is wired, ``MissionEngine.create_mission``
resolves the robot up front so unknown embodiments / missing URDFs fail
before the mission persists. Without providers (the engine unit-test
default), resolution is skipped and ``resolved_robot`` stays None.
"""

from __future__ import annotations

from typing import Any

import pytest

from odyssey.engine import MissionEngine
from odyssey.persistence import InMemoryPersistence
from odyssey.providers import ProviderRegistry
from odyssey.providers.local import LocalRobotProvider
from odyssey.runners import CPUMockRunner, RunnerRegistry
from odyssey.spec import (
    AgentRole,
    AgentSpec,
    EvaluationTask,
    EvaluationType,
    HFModelRef,
    Mission,
    MissionMetadata,
    RobotSpec,
    TrainingTask,
    TrainingType,
)
from odyssey.telemetry import EventPublisher


class _NullPublisher(EventPublisher):
    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        return


def _spec(*, embodiment: str = "franka_panda") -> Mission:
    return Mission(
        metadata=MissionMetadata(name="rr"),
        objective="o",
        acceptance_criteria="a",
        robot=RobotSpec(
            embodiment=embodiment,
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


async def _engine(*, with_providers: bool) -> MissionEngine:
    runners = RunnerRegistry()
    runners.register(CPUMockRunner())
    providers = None
    if with_providers:
        providers = ProviderRegistry()
        providers.register_robot(LocalRobotProvider(), handles="local")
    engine = MissionEngine(
        persistence=InMemoryPersistence(),
        runners=runners,
        event_publisher=_NullPublisher(),
        providers=providers,
    )
    await engine.initialize()
    return engine


async def test_create_mission_stashes_resolved_robot_when_providers_set() -> None:
    engine = await _engine(with_providers=True)
    run = await engine.create_mission(_spec(embodiment="ur5e"))
    assert run.resolved_robot is not None
    assert run.resolved_robot.provider == "local"
    assert run.resolved_robot.embodiment == "ur5e"


async def test_create_mission_skips_resolve_without_providers() -> None:
    engine = await _engine(with_providers=False)
    run = await engine.create_mission(_spec(embodiment="franka_panda"))
    assert run.resolved_robot is None


async def test_create_mission_fails_fast_on_unknown_embodiment() -> None:
    engine = await _engine(with_providers=True)
    # Bypass the spec-level allowlist check (none enforced at spec layer
    # — only the provider holds it) by constructing the robot directly.
    spec = _spec(embodiment="franka_panda")
    spec.robot = RobotSpec.model_construct(
        embodiment="made_up_robot_99",
        agents=spec.robot.agents,
    )
    with pytest.raises(ValueError, match="Unknown embodiment"):
        await engine.create_mission(spec)


async def test_resolved_robot_round_trips_through_persistence() -> None:
    engine = await _engine(with_providers=True)
    run = await engine.create_mission(_spec(embodiment="sawyer"))
    fetched = await engine.get_mission(run.id)
    assert fetched.resolved_robot is not None
    assert fetched.resolved_robot.embodiment == "sawyer"
    assert fetched.resolved_robot.provider == "local"
