# Odyssey Overview

An **open-source framework for defining, running, and benchmarking robot training missions**. Built by Lovell AI, currently in alpha (v0.1.0-alpha.1).

You train a robot with a multi-agent brain by describing a mission in YAML — specifying an embodiment, agents and model, tasks with a dataset, and an evaluation benchmark — and `odyssey run` orchestrates the full lifecycle: load → validate → train → evaluate → persist results.

## Key Concepts

- **Missions** — YAML specs that define what to train, how to evaluate, and on which robot
- **Runners** — Pluggable backends that execute tasks (CPU mock for testing, OpenVLA for real training, Robosuite for evaluation)
- **Models** — The neural network being trained (e.g. OpenVLA). A mission specifies which model to fine-tune and how
- **Agents** — The autonomous entities that define the models to use and provide instructions for models to follow.
- **Robots** — The embodiment of the physical robot and a collection of agents purposely collaborating to operate the robot.
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

The alpha version supports **VLA (Vision-Language-Action) model** fine-tuning only, with OpenVLA as the first supported model and Robosuite as the first eval environment. Future releases will add multi-agent training and VLM support.
