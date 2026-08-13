"""Model loaders — training runners and inference models."""

from odyssey.runners.models.gr00t_train import (
    GR00TRunner,
    build_gr00t_argv,
    parse_gr00t_line,
)
from odyssey.runners.models.openvla import (
    make_openvla_policy,
)
from odyssey.runners.models.openvla_train import (
    OpenVLARunner,
    build_openvla_argv,
    parse_openvla_line,
)

__all__ = [
    "GR00TRunner",
    "GemmaVLMGenerator",
    "OpenVLARunner",
    "VLARuntime",
    "build_gr00t_argv",
    "build_openvla_argv",
    "make_openvla_policy",
    "parse_gr00t_line",
    "parse_openvla_line",
]


def __getattr__(name: str) -> object:
    """Lazy-import heavy implementations to avoid pulling in torch at import."""
    if name == "GemmaVLMGenerator":
        from odyssey.runners.models.gemma_vlm import GemmaVLMGenerator

        return GemmaVLMGenerator
    if name == "VLARuntime":
        from odyssey.runners.models.openvla import VLARuntime

        return VLARuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
