# Lovell Odyssey

[![CI](https://github.com/lovellai-dev/odyssey/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lovellai-dev/odyssey/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

Open-source framework for defining, running, and benchmarking robot training missions.

> **Status: pre-alpha (v0.0.x).** Not yet on PyPI. The first public alpha is
> targeted at `v0.1.0-alpha`. The API, CLI, schemas, and wire protocols are
> still subject to change without notice. See `docs/` for the design refs.

## What it is

You describe a mission in YAML — a robot, a model, a dataset to train on, an
evaluation benchmark to score against — and `odyssey run` walks it through the
full lifecycle: load → validate → execute training tasks → execute the
evaluation task → persist results. Local-mode by default; the hosted Lovell
services (leaderboard, learning graph, hosted runners) are optional layers
that land in later releases.

## Concepts

### Missions

A **mission** is the unit of work in Odyssey: a single, reproducible recipe
that fine-tunes a model on a dataset and benchmarks the result. You describe
one in a `mission.yaml`; the framework loads it, drives it through a
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

### Robots and agents

In the Lovell AI architecture, a **robot** is more than an embodiment —
it's a composition of an embodiment with a **loadout of agents**. Every
robot has exactly one **PILOT** (running a Vision-Language-Action model
with physical authority over the actuators) and zero or more
**SPECIALISTs** (running language models for delegated reasoning — map
queries, calculations, lookups). Each agent is authored with a persona,
goals, and success criteria, and runs against a pinned model checkpoint;
its behavior is conditioned at runtime by materialized artifacts
produced from that prose. The fuller picture is described in the Lovell
AI robot-brain paper.

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
when the multi-agent runtime lands. What v0.0.x does *not* model from
the brain paper — agent persona / goals / success criteria,
materialized artifacts that condition runtime behavior, the
deterministic safety stack, conduct-rule enforcement, the deployment
contract — is on the roadmap. Where each of those ultimately lives
(in `src/odyssey/` versus in a hosted Lovell service) is a strategic
decision still being worked out.

### Robot specs in v0.0.x

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

## Install

**Linux only** — install build dependencies before proceeding (needed by `.[all]`):

```bash
sudo apt update && sudo apt install build-essential python3-dev -y
```

```bash
git clone https://github.com/lovellai-dev/odyssey.git
cd odyssey
python3 -m venv .venv
source .venv/bin/activate
pip install -e .              # CLI, validate, mock runs (lightweight)
pip install -e ".[all]"       # real training + evaluation (torch, robosuite…)
pip install -e ".[all,dev]"   # + pytest, ruff, mypy
```

The base install pulls in pydantic, click, pyyaml, and aiosqlite — enough to
run `validate`, `list`, `status`, and `run --use-mock-runner` against any
mission spec without a GPU. `.[all]` adds everything needed for real training
and evaluation runs.

## 60-second smoke test (no GPU, no network)

```bash
# Validate the mission spec
$ odyssey validate examples/quickstart-openvla/mission.yaml
OK  examples/quickstart-openvla/mission.yaml
  spec version : 0.1
  mission name : openvla-bridge-lift
  robot        : embodiment=franka_panda
  tasks        : 1 training, 1 evaluation

# Run the full mission with a CPU mock (no GPU needed)
$ odyssey run examples/quickstart-openvla/mission.yaml --use-mock-runner
{"ts": "...", "event": "mission.created", ...}
{"ts": "...", "event": "mission.queued", ...}
{"ts": "...", "event": "mission.started", ...}
...
{"ts": "...", "event": "mission.completed", "overall_grade": 1.0}

COMPLETED  c1756bad855e45cc9a95b5b0566c948b
  overall_grade : 1.000

# List all missions from the local DB
$ odyssey list
c1756bad855e  COMPLETED   openvla-bridge-lift   2026-05-17T23:18:34+00:00  grade=1.000

# Show details for a specific mission (prefix match)
$ odyssey status c1756bad
COMPLETED  c1756bad855e45cc9a95b5b0566c948b
  name         : openvla-bridge-lift
  ...
  tasks:
    COMPLETED    training    finetune-openvla
    COMPLETED    evaluation  eval-on-robosuite-lift
```

`--use-mock-runner` swaps in the CPU mock for every task, so this works on a
laptop without a GPU. State is persisted to `~/.odyssey/missions.db`;
artifacts under `~/.odyssey/runs/<mission-id>/<task-id>/`.

## Real run (OpenVLA + Robosuite)

This works only after the extras are installed AND the upstream
[openvla](https://github.com/openvla/openvla) repo is cloned somewhere
findable (`$OPENVLA_REPO_PATH` or `/srv/openvla`):

```bash
pip install -e ".[huggingface,openvla,robosuite]"
git clone https://github.com/openvla/openvla.git /srv/openvla

odyssey run examples/quickstart-openvla/mission.yaml
```

Hardware: 24 GB GPU (RTX 4090-class or better) for the OpenVLA fine-tune.

**Known gap for v0.1.0-alpha:** the Robosuite evaluation runner ships with
the lifecycle plumbing wired but no built-in OpenVLA→robosuite-action
adapter. Real eval numbers require supplying a `policy_factory` to
`RobosuiteRunner` — see the docstring in `src/odyssey/runners/robosuite.py`.
The built-in adapter is a v0.2.x line item.

## CLI reference

| Command | What it does |
|---|---|
| `odyssey init [DIR]` | Scaffold a new mission directory. `--template openvla\|cpu_mock`. |
| `odyssey validate <mission.yaml>` | Parse + validate a spec. Exits 0 if clean. |
| `odyssey run <mission.yaml>` | Execute end-to-end. `--use-mock-runner` for no-GPU smoke. |
| `odyssey list` | Recent missions from the local SQLite DB. `--status` to filter. |
| `odyssey status <mission_id>` | One mission's detail. Accepts an id prefix. |

All commands respect `--db` and `--working-dir` to override the
`~/.odyssey/` defaults.

## Project layout

```
src/odyssey/
  spec/         Pydantic schemas for mission.yaml
  engine/       MissionEngine + lifecycle + runtime records
  runners/      Runner ABC, registry, CPU mock, subprocess infra,
                OpenVLA training, Robosuite evaluation
  providers/    Provider ABCs + registry, local/ + huggingface/
  persistence/  Persistence ABC + InMemory + SQLite
  telemetry/    Event vocabulary + stdout publisher
  cli/          Click-based `odyssey` command + subcommands
  utils/        ~/.odyssey/ path management
```

## Status snapshot (v0.0.x)

| Area | Done | Deferred |
|---|---|---|
| Spec + validate | ✓ | — |
| Engine + lifecycle | ✓ | watchdog timers, materialized profiles |
| In-memory + SQLite persistence | ✓ | — |
| Provider ABCs + Local + HF | ✓ | OXE, Lovell-mode |
| CPU mock runner | ✓ | — |
| OpenVLA training runner | skeleton + tests | end-to-end smoke with real OpenVLA |
| Robosuite eval runner | skeleton + tests | built-in OpenVLA→action adapter |
| `odyssey init / run / list / status / validate` | ✓ | `logs`, `publish` |
| Leaderboard publish, Learning Graph, Anonymizer, Auth | — | post-v0.1.0-alpha |

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). DCO sign-off required on every
commit. Open an issue before non-trivial PRs — the API surface is moving
weekly until v0.1.0-alpha freezes.
