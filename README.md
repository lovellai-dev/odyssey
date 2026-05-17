# Lovell Odyssey

Open-source framework for defining, running, and benchmarking robot training missions.

> **Status: pre-alpha (v0.0.x).** This repository is being bootstrapped. There is
> no usable public release yet. The first usable alpha is targeted at `v0.1.0-alpha`.
> Until then, the API, CLI, schemas, and wire protocols are all subject to change
> without notice. See `docs/` for the design references that shape what is being
> built.

## What this will be

Odyssey is an installable Python package and CLI that lets you write a short YAML
file describing a robot, a model, a dataset, and an evaluation — then runs the
training and the eval locally and reports a score. The hosted Lovell services
(leaderboard, learning graph, hosted runners) are optional integrations layered on
top, not requirements.

```bash
pip install lovell-odyssey
odyssey init my-mission
odyssey run my-mission/mission.yaml
```

## What is in this repository today

- `pyproject.toml` — package metadata, dependencies, tool config.
- `src/odyssey/` — empty package skeleton, ready for the engine + spec + runners.
- `tests/`, `examples/` — placeholders.
- `LICENSE`, `NOTICE` — Apache 2.0.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` — community files.
- `.github/workflows/ci.yml` — drafted, not yet wired to a remote.

## License

Apache License 2.0. See [LICENSE](LICENSE).
