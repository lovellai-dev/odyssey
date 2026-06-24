# Concepts

## Missions

A **mission** is the unit of work in Odyssey: a single, reproducible recipe
that trains a multi-agent robot brain by fine-tuning its underlying models on a dataset
and benchmarking the result. You describe one in a `mission.yaml`; the framework loads it, drives it through a
lifecycle (`DRAFT → QUEUED → ACTIVE → COMPLETED | FAILED | CANCELLED`),
persists every status transition to `~/.odyssey/missions.db`, and emits one
JSON event per state change to stdout.

Every mission has four required pieces:

1. **An objective** — prose stating what you're trying to achieve.
2. **Acceptance criteria** — prose stating how success is judged.
3. **A robot** — the embodiment plus a loadout of agents (see below).
4. **A list of tasks** — *at least one training task, exactly one
   evaluation task, and the evaluation task must be the last entry.*
   Each training task updates one agent on the robot; the evaluation
   task runs the robot after every training task has completed.

Training tasks chain implicitly through the agent: when multiple
training tasks target the same `agent_id`, each one starts from the
previous one's checkpoint. No explicit `from_task` reference is needed,
because the model lives on the agent and tasks update it. When every
task reaches `COMPLETED`, the mission's `overall_grade` is set to the
average of the evaluation scores.

Shape of a minimal mission:

```yaml
odysseyVersion: "0.1"
kind: Mission
metadata:
  name: my-mission
objective: |
  Fine-tune OpenVLA so it can pick up a cube in Robosuite Lift.
acceptance_criteria: |
  At least one successful lift across 10 evaluation episodes.
robot:
  embodiment: franka_panda
  agents:
    - id: pilot
      role: PILOT
      model: { source: huggingface, base: openvla/openvla-7b }
tasks:
  - name: finetune
    kind: training
    training_type: demonstration
    agent_id: pilot              # updates the pilot agent's model
    config: { method: peft_lora, lora_rank: 8, epochs: 1 }
  - name: bench
    kind: evaluation
    evaluation_type: robosuite
    benchmark_name: Lift
    num_episodes: 10
    # no model / agent_id — the eval runs the robot
```

`objective` and `acceptance_criteria` are required prose fields. They aren't
parsed by anything today, but in later releases the Mission Materializer
will extract structured artifacts from them (evaluation predicates,
deadlines, instruction prefixes injected into VLA prompts). Write them like
you mean it — future-you will reread them in leaderboard submissions and
graph queries.

## Robots and agents

In the Lovell AI architecture, a **robot** is more than an embodiment —
it's a composition of an embodiment with a **loadout of agents**. Every
robot has exactly one **PILOT** (running a Vision-Language-Action model
with physical authority over the actuators) and zero or more
**SPECIALISTs** (running language models for delegated reasoning — map
queries, calculations, lookups). Each agent runs against a pinned model
checkpoint.

Odyssey's spec models this hierarchy directly. A `robot:` block
declares an embodiment and an inline loadout of agents:

```yaml
robot:
  embodiment: franka_panda
  agents:
    - id: pilot
      role: PILOT
      model: { source: huggingface, base: openvla/openvla-7b }
```

Each agent owns its model — the `model:` field lives on the agent, not
on a task. Training tasks reference an agent by id (`agent_id: pilot`),
and the framework looks up the model from the agent. When several
training tasks target the same agent, each one starts from the previous
one's checkpoint. The evaluation task takes no model reference at all:
it runs the robot — that is, it composes the current checkpoints of
every agent in the loadout.

**Today, Odyssey fine-tunes the underlying model for one of the
robot's agents at a time.** v0.0.x enforces exactly one agent on the
robot (the implicit PILOT), and the eval composes a single policy from
that single agent. Multi-agent loadouts (PILOT + one or more
SPECIALISTs) and the multi-agent execution that goes with them ship
when the multi-agent runtime lands.

## Robot specs in v0.0.x

The `robot:` block names a robot's embodiment in one of three forms.
The spec validator enforces that exactly one embodiment form is set
and that `agents` contains exactly one agent.

| Form | Example | What it does today |
|---|---|---|
| `embodiment:` | `franka_panda`, `ur5e`, `sawyer` | Names a built-in catalog embodiment. 8 names accepted: `franka_panda`, `panda`, `sawyer`, `iiwa`, `jaco`, `kinova_gen3`, `ur5e`, `baxter` — the arms Robosuite's built-in robot models cover. Resolved at mission-creation by `LocalRobotProvider`; passed through to `robosuite.make(robots=...)` at evaluation. |
| `urdf:` | `./arms/my_arm.urdf` | Names a local URDF/xacro path. Existence-checked at mission-creation. No robot pass-through to Robosuite — falls back to the env's default robot. |
| `id:` | `rob_01HQR...` | Reserved for a robot registered in your Lovell account. When the Lovell provider ships, this will fetch the loadout from the account rather than requiring an inline `agents:` block. Requires `odyssey login`, not yet shipped. |

Two missions with the same model and benchmark but different
embodiments produce genuinely different eval runs (Robosuite simulates
the named robot, not its per-env default). The embodiment is what
categorizes leaderboard submissions when the leaderboard backend ships
— a Franka Panda result and a Sawyer result are different categories.
Multi-agent comparison — same loadout, different model checkpoint in
one agent — will become possible when the agent cap lifts.
