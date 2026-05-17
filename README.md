# Lovell Odyssey

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

## Install

```bash
git clone https://github.com/lovell/odyssey.git
cd odyssey
pip install -e .
```

The base install pulls in pydantic, click, pyyaml, and aiosqlite — enough to
run `validate`, `list`, `status`, and `run --use-mock-runner` against any
mission spec. Real training and evaluation runners need their own extras:

```bash
pip install -e ".[huggingface]"   # HF model + dataset providers
pip install -e ".[openvla]"       # OpenVLA training runner deps
pip install -e ".[robosuite]"     # Robosuite evaluation runner deps
pip install -e ".[dev]"           # pytest, ruff, mypy
```

## 60-second smoke test (no GPU, no network)

```bash
$ odyssey validate examples/quickstart-openvla/mission.yaml
OK  examples/quickstart-openvla/mission.yaml
  spec version : 0.1
  mission name : openvla-bridge-lift
  robot        : embodiment=franka_panda
  tasks        : 1 training, 1 evaluation

$ odyssey run examples/quickstart-openvla/mission.yaml --use-mock-runner
{"ts": "...", "event": "mission.created", ...}
{"ts": "...", "event": "mission.queued", ...}
{"ts": "...", "event": "mission.started", ...}
...
{"ts": "...", "event": "mission.completed", "overall_grade": 1.0}

COMPLETED  c1756bad855e45cc9a95b5b0566c948b
  overall_grade : 1.000

$ odyssey list
c1756bad855e  COMPLETED   openvla-bridge-lift   2026-05-17T23:18:34+00:00  grade=1.000

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
| `odyssey run / list / status / validate` | ✓ | `init`, `logs`, `publish` |
| Leaderboard publish, Learning Graph, Anonymizer, Auth | — | post-v0.1.0-alpha |

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). DCO sign-off required on every
commit. Open an issue before non-trivial PRs — the API surface is moving
weekly until v0.1.0-alpha freezes.
