# Odyssey Overview

An **open-source framework for defining, running, and benchmarking robot training missions**. Built by Lovell AI, currently in pre-alpha (v0.0.1).

You describe a mission in YAML — specifying a robot, a model, a dataset, and an evaluation benchmark — and `odyssey run` orchestrates the full lifecycle: load → validate → train → evaluate → persist results.

## Key Concepts

- **Missions** — YAML specs that define what to train, how to evaluate, and on which robot
- **Runners** — Pluggable backends that execute tasks (CPU mock for testing, OpenVLA for real training, Robosuite for evaluation)
- **Models** — The neural network being trained (e.g. OpenVLA). A mission specifies which model to fine-tune and how
- **Agents** — The autonomous entity resulting from training a mission. It becomes part of the robot as a PILOT or SPECIALIST
- **Providers** — Abstractions for data/model sources (local filesystem, HuggingFace)

## Codebase Structure (`src/odyssey/`)

| Module | Purpose |
|---|---|
| `spec/` | Pydantic schemas for `mission.yaml` validation |
| `engine/` | MissionEngine — lifecycle orchestration & runtime records |
| `runners/` | Runner ABC + registry, CPU mock, OpenVLA training, Robosuite eval |
| `providers/` | Provider ABCs + local & HuggingFace implementations |
| `persistence/` | Storage layer — InMemory + SQLite backends |
| `telemetry/` | Event vocabulary + stdout publisher |
| `cli/` | Click-based CLI (`odyssey validate/run/list/status/init`) |
| `materializer/` | Handles materializing assets |
| `utils/` | `~/.odyssey/` path management |

## Tech Stack

- **Python 3.10+**, Apache 2.0 licensed
- Core deps: pydantic, click, pyyaml, aiosqlite
- Optional extras: `torch` + `transformers` + `peft` (OpenVLA training), `robosuite` + `mujoco` (evaluation), `huggingface-hub` + `datasets`
- Tooling: hatchling build, ruff linter, mypy strict, pytest

## CLI

`odyssey init`, `validate`, `run`, `list`, `status` — with `--use-mock-runner` for GPU-free testing on a laptop.

## Focus

The pre-alpha version supports **VLA (Vision-Language-Action) model** fine-tuning only, with OpenVLA as the first supported model and Robosuite as the first eval environment. Future releases will add multi-agent training and VLM support.
