"""Integration tests for the multi-agent evaluation pipeline.

End-to-end lifecycle: multi-agent mission spec → engine → training
(mock) → evaluation with PlannedEvalRuntime composing a mock planner
+ mock pilot. Verifies the full chain without GPU dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from odyssey.engine import MissionEngine, MissionStatus, TaskStatus
from odyssey.persistence import InMemoryPersistence
from odyssey.runners import WILDCARD_TYPE, CPUMockRunner, Runner, RunnerRegistry, TaskContext
from odyssey.runners.agents.planned import PhaseConfig, PlannedEvalRuntime
from odyssey.runners.agents.runtime import PilotRuntime, PlannerRuntime
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
    load_mission,
)
from odyssey.telemetry import EventPublisher

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MISSION = REPO_ROOT / "examples" / "multiagent-openvla-gemma" / "mission.yaml"


# ---------------------------------------------------------------------------
# Mock agent runtimes for CI
# ---------------------------------------------------------------------------

class MockPilot:
    """Returns zero actions and records calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, str]] = []

    def act(self, image: Any, instruction: str) -> np.ndarray:
        self.calls.append((image, instruction))
        return np.zeros(7, dtype=np.float64)


class MockPlanner:
    """Returns a fixed plan for any instruction."""

    def __init__(self, steps: list[str] | None = None) -> None:
        self._steps = steps or ["reach for object", "grasp", "lift"]
        self.call_count = 0

    def plan(self, task_instruction: str) -> list[str]:
        self.call_count += 1
        return list(self._steps)


# ---------------------------------------------------------------------------
# Mock runner that uses PlannedEvalRuntime for eval tasks
# ---------------------------------------------------------------------------

class MultiAgentMockRunner(Runner):
    """Exercises the full PlannedEvalRuntime flow during eval.

    Training tasks: returns a mock checkpoint (same as CPUMockRunner).
    Eval tasks: constructs a PlannedEvalRuntime from mock agents and
    runs a short simulated episode loop.
    """

    def __init__(self) -> None:
        self.pilot = MockPilot()
        self.planner = MockPlanner()
        self.eval_contexts: list[TaskContext] = []

    @property
    def name(self) -> str:
        return "multiagent_mock"

    @property
    def supported_kinds(self) -> set[TaskKind]:
        return {TaskKind.TRAINING, TaskKind.EVALUATION}

    @property
    def supported_types(self) -> set[str]:
        return {WILDCARD_TYPE}

    async def run(self, context: TaskContext) -> dict[str, Any]:
        spec = context.task.spec

        if isinstance(spec, TrainingTask):
            await context.emit_progress("training", step="mock_train")
            return {
                "checkpoint_path": f"/tmp/mock-ckpt-{spec.agent_id}",
                "_mock": True,
            }

        if isinstance(spec, EvaluationTask):
            self.eval_contexts.append(context)
            return await self._run_eval(context, spec)

        raise TypeError(f"Unexpected task type: {type(spec).__name__}")

    async def _run_eval(
        self, context: TaskContext, spec: EvaluationTask
    ) -> dict[str, Any]:
        # Build PlannedEvalRuntime from mock agents
        phase_config = PhaseConfig(steps_per_phase=5)
        runtime = PlannedEvalRuntime(
            self.pilot, self.planner, phase_config=phase_config
        )

        num_episodes = min(spec.num_episodes, 2)
        successes = 0
        fake_image = np.zeros((256, 256, 3), dtype=np.uint8)

        for ep in range(1, num_episodes + 1):
            if context.cancelled():
                break

            task_instruction = f"complete the {spec.benchmark_name} task"
            plan = runtime.begin_episode(task_instruction)

            await context.emit_progress(
                "executing",
                step="episode_start",
                step_index=ep,
                step_total=num_episodes,
                metadata={"plan": plan},
            )

            # Simulate 15 steps (3 phases x 5 steps_per_phase)
            for _step in range(15):
                if context.cancelled():
                    break
                runtime.get_action(fake_image)

            successes += 1
            await context.emit_progress(
                "executing",
                step="episode_complete",
                step_index=ep,
                step_total=num_episodes,
                step_label=f"episode {ep}: PASS",
            )

        success_rate = successes / max(num_episodes, 1)
        return {
            "num_episodes": num_episodes,
            "success_rate": success_rate,
            "performance_score": success_rate,
            "letter_grade": "A",
            "passed": True,
            "_mock": True,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CapturingPublisher(EventPublisher):
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))

    @property
    def event_types(self) -> list[str]:
        return [e[0] for e in self.events]


def _multiagent_spec() -> Mission:
    return Mission(
        metadata=MissionMetadata(name="msn-multiagent-test"),
        objective="test multi-agent eval",
        acceptance_criteria="acceptance",
        robot=RobotSpec(
            embodiment="franka_panda",
            agents=[
                AgentSpec(
                    id="pilot",
                    role=AgentRole.PILOT,
                    model=HFModelRef(base="openvla/openvla-7b"),
                ),
                AgentSpec(
                    id="task-planner",
                    role=AgentRole.SPECIALIST,
                    model=HFModelRef(
                        base="google/gemma-3-4b-it", quantization="int4"
                    ),
                ),
            ],
        ),
        tasks=[
            TrainingTask(
                name="train-pilot",
                training_type=TrainingType.DEMONSTRATION,
                agent_id="pilot",
            ),
            EvaluationTask(
                name="eval-lift",
                evaluation_type=EvaluationType.ROBOSUITE,
                benchmark_name="Lift",
                num_episodes=4,
            ),
        ],
    )


async def _make_engine(
    runner: Runner,
) -> tuple[MissionEngine, CapturingPublisher]:
    persistence = InMemoryPersistence()
    runners = RunnerRegistry()
    runners.register(runner)
    publisher = CapturingPublisher()
    engine = MissionEngine(
        persistence=persistence, runners=runners, event_publisher=publisher
    )
    await engine.initialize()
    return engine, publisher


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_multiagent_mission_completes_end_to_end() -> None:
    """A multi-agent mission (PILOT + SPECIALIST) runs to completion
    with mock agents and the PlannedEvalRuntime driving eval."""
    runner = MultiAgentMockRunner()
    engine, pub = await _make_engine(runner)
    spec = _multiagent_spec()

    run = await engine.create_mission(spec)
    assert run.status == MissionStatus.DRAFT
    assert len(run.tasks) == 2

    final = await engine.start_mission(run.id)
    assert final.status == MissionStatus.COMPLETED
    assert all(t.status == TaskStatus.COMPLETED for t in final.tasks)

    # Training task produced a checkpoint
    train_task = final.tasks[0]
    assert train_task.result_summary["checkpoint_path"] == "/tmp/mock-ckpt-pilot"

    # Eval task produced results
    eval_task = final.tasks[1]
    assert eval_task.result_summary["success_rate"] == 1.0
    assert eval_task.result_summary["passed"] is True

    # Overall grade computed from eval scores
    assert final.overall_grade is not None

    # Lifecycle events fired in order
    assert pub.event_types[0] == "mission.created"
    assert "mission.completed" in pub.event_types
    assert "task.completed" in pub.event_types


async def test_eval_context_has_all_agents_and_checkpoints() -> None:
    """The engine passes all agents and checkpoint mappings to the eval runner."""
    runner = MultiAgentMockRunner()
    engine, _ = await _make_engine(runner)

    run = await engine.create_mission(_multiagent_spec())
    await engine.start_mission(run.id)

    # The runner captured the eval TaskContext
    assert len(runner.eval_contexts) == 1
    ctx = runner.eval_contexts[0]

    # All agents present
    assert len(ctx.agents) == 2
    agent_ids = {a.id for a in ctx.agents}
    assert agent_ids == {"pilot", "task-planner"}

    # Pilot was trained — has checkpoint
    assert ctx.agent_checkpoints["pilot"] == "/tmp/mock-ckpt-pilot"
    # Specialist not trained — None
    assert ctx.agent_checkpoints["task-planner"] is None


async def test_planner_called_per_episode() -> None:
    """The PlannedEvalRuntime calls the planner once per episode."""
    runner = MultiAgentMockRunner()
    engine, _ = await _make_engine(runner)

    run = await engine.create_mission(_multiagent_spec())
    await engine.start_mission(run.id)

    # Runner ran 2 episodes (capped from spec's 4)
    assert runner.planner.call_count == 2


async def test_pilot_receives_phased_instructions() -> None:
    """The pilot receives different instructions as phases advance."""
    runner = MultiAgentMockRunner()
    engine, _ = await _make_engine(runner)

    run = await engine.create_mission(_multiagent_spec())
    await engine.start_mission(run.id)

    # Pilot was called 30 times (2 episodes x 15 steps)
    assert len(runner.pilot.calls) == 30

    # Check that different sub-instructions were used across phases.
    # With 3 phases and 5 steps_per_phase, instructions should change.
    instructions = [call[1] for call in runner.pilot.calls]
    # First episode: steps 0-14 → phases [reach, grasp, lift]
    ep1_instructions = instructions[:15]
    # First 5 steps should use "reach for object"
    assert all(i == "reach for object" for i in ep1_instructions[:5])
    # Steps 5-9 should use "grasp"
    assert all(i == "grasp" for i in ep1_instructions[5:10])
    # Steps 10-14 should use "lift"
    assert all(i == "lift" for i in ep1_instructions[10:15])


async def test_multiagent_example_yaml_loads() -> None:
    """The multi-agent example YAML parses as a valid Mission spec."""
    spec = load_mission(EXAMPLE_MISSION)
    assert spec.metadata.name == "multiagent-lift"
    assert len(spec.robot.agents) == 2
    pilots = [a for a in spec.robot.agents if a.role == AgentRole.PILOT]
    specialists = [a for a in spec.robot.agents if a.role == AgentRole.SPECIALIST]
    assert len(pilots) == 1
    assert len(specialists) == 1
    assert specialists[0].model.quantization == "int4"  # type: ignore[union-attr]


async def test_multiagent_example_runs_with_cpu_mock() -> None:
    """The multi-agent example YAML drives cleanly through the engine
    with CPUMockRunner (no GPU required)."""
    engine, _ = await _make_engine(CPUMockRunner())
    spec = load_mission(EXAMPLE_MISSION)
    run = await engine.create_mission(spec)
    final = await engine.start_mission(run.id)
    assert final.status == MissionStatus.COMPLETED


async def test_mock_pilot_satisfies_protocol() -> None:
    assert isinstance(MockPilot(), PilotRuntime)


async def test_mock_planner_satisfies_protocol() -> None:
    assert isinstance(MockPlanner(), PlannerRuntime)
