# Contributing to Odyssey

Odyssey is an open-source project from Lovell AI. We welcome contributions —
bug reports, feature ideas, documentation fixes, and patches. This file
explains what you need to know before sending one.

## Pre-alpha disclaimer

This project is in pre-alpha (`v0.0.x`). The schema, CLI, ABCs, and wire
protocols are all in flux. If you are thinking about a non-trivial
contribution, please open a GitHub issue first to check that it fits the
direction we are heading. Small fixes (typos, docstrings, obvious bugs) can
go straight to a pull request.

## Developer Certificate of Origin (DCO)

Every commit must be signed off under the
[Developer Certificate of Origin](https://developercertificate.org/). The
sign-off is a single line at the end of the commit message:

```
Signed-off-by: Jane Smith <jane@example.com>
```

Use the `-s` flag to add it automatically:

```bash
git commit -s -m "Fix typo in CLI help text"
```

The DCO is checked by CI on every pull request. We do not require a CLA.

## Development setup

```bash
git clone https://github.com/femtechie/lovell-odyssey.git
cd lovell-odyssey
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Recommended:

```bash
ruff check .
ruff format .
mypy src/odyssey
```

CI runs all three on every PR.

## Filing issues

When opening a bug, please include:

- Odyssey version (`odyssey --version` once the CLI lands; for now, the commit
  hash you are on).
- Python version, OS.
- The mission YAML (or the smallest fragment that reproduces the problem).
- The full error / traceback.

## Code style

- Type hints required on all new public functions.
- One-line docstrings for any non-obvious helper.
- No multi-paragraph comment blocks. If a comment is longer than a sentence,
  it usually belongs in a design doc, not the source.
- Match the existing module layout; do not introduce new top-level packages
  without an issue first.

## Where to find the design

`docs/` in this repo mirrors the internal design docs that shape the
framework. Two starting points:

- `docs/architecture/odyssey-design.md` — what the framework is and how it
  is shaped (Pydantic schemas, ABCs, wire protocols, anonymizer).
- `docs/architecture/odyssey-publication-plan.md` — what is being built
  for the first publish, in what order.
