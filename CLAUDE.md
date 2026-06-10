# Odyssey — Claude Code Instructions

## What is this project

Odyssey is a mission-based framework for orchestrating robot training pipelines.
CLI: `odyssey run <mission.yaml>` → engine → tasks (training, evaluation).
Version: v0.1.0-alpha. Package name: `lovell-odyssey`.

## Architecture

```
odyssey run mission.yaml
  → cli/commands/run.py        # parses YAML, builds engine + runners + providers
  → engine/mission_engine.py   # orchestrator: DRAFT → QUEUED → ACTIVE → tasks → COMPLETED/FAILED
  → runners/registry.py        # dispatches task to runner by (TaskKind, type)
  → runners/openvla.py         # training: launches finetune.py via subprocess
  → runners/robosuite.py       # evaluation: rollouts in robosuite sim (policy_factory NOT implemented)
  → runners/cpu_mock.py        # fallback: fake runner for smoke tests
```

Tasks execute sequentially in YAML order. Each runner gets a `TaskContext` with
output_dir, agent, providers, checkpoint info, cancel event.

## Key directories

- `src/odyssey/spec/` — Pydantic schemas for mission.yaml (mission, tasks, agents, refs)
- `src/odyssey/engine/` — MissionEngine, lifecycle state machine, records
- `src/odyssey/runners/` — Runner ABC, registry, OpenVLA, Robosuite, CPU mock, subprocess
- `src/odyssey/providers/` — Provider ABCs + registry, local + HuggingFace (models, datasets, robots)
- `src/odyssey/persistence/` — Persistence ABC + InMemory + SQLite
- `src/odyssey/telemetry/` — Event types + stdout publisher
- `src/odyssey/cli/` — Click CLI: init, validate, run, list, status
- `examples/quickstart-openvla/` — reference mission.yaml
- `tests/` — pytest + pytest-asyncio, `asyncio_mode = "auto"`

## Git workflow

- `main` = stable, `develop` = integration branch
- PRs go to `develop`, not `main`
- Branch naming: `feat/`, `fix/`, `docs/`
- DCO sign-off required on commits

## Build and test

```bash
pip install -e ".[dev]"       # base + test/lint tools
pytest                        # runs all tests
ruff check src/ tests/        # lint
mypy                          # type check (strict mode)
```

Optional extras: `huggingface`, `openvla`, `robosuite`, `all`, `dev`.

## Code conventions

- Python 3.10+, strict mypy, ruff with bugbear + simplify rules
- Line length: 100
- Async-first: engine and runners are async (asyncio)
- No hardcoded env vars in runners — user sets them in shell
- `subprocess.py` uses `read(8192)` not `readline()` for tqdm compatibility
- Runners return `dict[str, Any]` as `result_summary`
- Pydantic v2 models for all specs

## Known gaps (v0.1.0-alpha)

- Evaluation pipeline: `RobosuiteRunner` has no built-in `policy_factory` — crashes
  with `NotImplementedError`. Training works end-to-end, eval does not. See issue #5.
- No watchdog timers or materialized profiles yet
- Tasks execute sequentially only (no parallelism)
- Leaderboard publish, learning graph, anonymizer, auth — all deferred post-v0.1.0-alpha

## Mission YAML structure

```yaml
odysseyVersion: "0.1"
kind: Mission
metadata: { name, description, tags }
objective: "..."
acceptance_criteria: "..."
robot:
  embodiment: franka_panda
  agents:
    - id: pilot
      role: PILOT
      model: { source: huggingface, base: openvla/openvla-7b }
tasks:
  - name: finetune-openvla
    kind: training
    training_type: demonstration
    agent_id: pilot
    dataset: { source: huggingface, ref: ..., format: ... }
    config: { method, lora_rank, batch_size, learning_rate, epochs }
  - name: eval-on-robosuite-lift
    kind: evaluation
    evaluation_type: robosuite
    benchmark_name: Lift
    num_episodes: 10
```

## Environment for real training runs

- GPU: 24GB+ (L4, RTX 4090 class)
- Needs OpenVLA repo cloned + `OPENVLA_REPO_PATH` env var
- Needs RLDS dataset on disk (Bridge V2 ~124GB)
- GCP single-GPU: `export NCCL_NET=Socket`
