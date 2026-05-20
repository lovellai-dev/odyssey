<h1 align="center">Lovell Odyssey</h1>

<p align="center">
  <a href="https://github.com/lovellai-dev/odyssey/actions/workflows/ci.yml"><img src="https://github.com/lovellai-dev/odyssey/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/status-pre--alpha-orange.svg" alt="Status: Pre-Alpha">
</p>

Open-source framework for defining, running, and benchmarking robot training missions.

## Install

> [!IMPORTANT]
> **Linux only** — install build dependencies before proceeding (needed by `.[all]`):
> ```bash
> sudo apt update && sudo apt install build-essential python3-dev -y
> ```

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

## Quick start (no GPU, no network)

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

## What it is

You describe a mission in YAML — a robot, a model, a dataset to train on, an
evaluation benchmark to score against — and `odyssey run` walks it through the
full lifecycle: load → validate → execute training tasks → execute the
evaluation task → persist results. Local-mode by default; the hosted Lovell
services (leaderboard, learning graph, hosted runners) are optional layers
that land in later releases.

Odyssey organizes work around **missions**, **robots**, and **agents**.
See [docs/concepts.md](docs/concepts.md) for the full architecture.

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

> [!NOTE]
> **Known gap for v0.1.0-alpha:** the Robosuite evaluation runner ships with
> the lifecycle plumbing wired but no built-in OpenVLA→robosuite-action
> adapter. Real eval numbers require supplying a `policy_factory` to
> `RobosuiteRunner` — see the docstring in `src/odyssey/runners/robosuite.py`.
> The built-in adapter is a v0.2.x line item.

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
