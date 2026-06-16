"""Runner ABC, registry, and built-in runners.

Package layout
--------------
The package root holds the **infrastructure / contract layer** — modules that
define the framework itself rather than any concrete robot/model integration:

- ``base.py``       — Runner ABC, TaskContext, WILDCARD_TYPE (imported by all)
- ``registry.py``   — RunnerRegistry, the (TaskKind, type) dispatch table
- ``subprocess.py`` — model-agnostic training-subprocess helper (LineParser,
                      TrainingProcessSpec, run_training_subprocess)
- ``cpu_mock.py``   — CPUMockRunner, the universal fallback runner

Concrete implementations live in subpackages, grouped by concern:

- ``models/`` — model loading + training runners (OpenVLA, Gemma)
- ``evals/``  — evaluation runners (Robosuite)
- ``agents/`` — multi-agent orchestration (no model loading)

Subpackages import "up" from the root contracts; the root never imports an
implementation except to re-export it below.
"""

from odyssey.runners.base import WILDCARD_TYPE, Runner, TaskContext
from odyssey.runners.cpu_mock import CPUMockRunner
from odyssey.runners.evals.robosuite import RobosuiteRunner
from odyssey.runners.models.openvla import OpenVLARunner, build_openvla_argv, parse_openvla_line
from odyssey.runners.registry import RunnerRegistry
from odyssey.runners.subprocess import (
    LineParser,
    TrainingProcessSpec,
    run_training_subprocess,
)

__all__ = [
    "WILDCARD_TYPE",
    "CPUMockRunner",
    "LineParser",
    "OpenVLARunner",
    "RobosuiteRunner",
    "Runner",
    "RunnerRegistry",
    "TaskContext",
    "TrainingProcessSpec",
    "build_openvla_argv",
    "parse_openvla_line",
    "run_training_subprocess",
]
