"""Model loaders — training runners and inference models."""

from odyssey.runners.models.openvla import (
    OpenVLARunner,
    build_openvla_argv,
    make_openvla_policy,
    parse_openvla_line,
)

__all__ = [
    "GemmaTextGenerator",
    "OpenVLARunner",
    "build_openvla_argv",
    "make_openvla_policy",
    "parse_openvla_line",
]


def __getattr__(name: str) -> object:
    """Lazy-import heavy implementations to avoid pulling in torch at import."""
    if name == "GemmaTextGenerator":
        from odyssey.runners.models.gemma import GemmaTextGenerator

        return GemmaTextGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
