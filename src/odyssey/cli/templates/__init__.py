"""Mission YAML templates shipped with the CLI.

Each subdirectory is one template name (``openvla``, ``cpu_mock``, ...)
containing a ``mission.yaml`` with ``{{ name }}`` placeholders. The
``odyssey init`` command reads them via ``importlib.resources``.
"""
