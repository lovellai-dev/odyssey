# GR00T + Odyssey + Isaac Lab eval stack (managed with uv)

Running a closed-loop GR00T policy eval in Isaac Lab spans **three Python
environments that cannot share an interpreter** — their `torch` / CUDA / Python
ABIs conflict. This is the same pattern Odyssey already uses for the OpenVLA vs
SPECIALIST split: **separate venvs, opt-in extras, `constraints/*-known-good.txt`
pins, and an interpreter selected by an env var.** We manage all of it with
[uv](https://docs.astral.sh/uv/).

## The three environments

| Env | Interpreter (env var) | Python | Holds | Managed by |
|---|---|---|---|---|
| **Odyssey core** | `.venv` (this repo) | 3.10–3.12 | mission engine, CLI, training runner | `uv venv` + `.[dev,huggingface]` |
| **GR00T server** | `GR00T_VENV_PYTHON` | 3.10 | GR00T model, `torch 2.7.1+cu128`, flash-attn | `uv venv` in `$ISAAC_GR00T_DIR`, `constraints/gr00t-server-known-good.txt` |
| **Isaac Lab eval** | `ISAAC_PYTHON` | 3.11 | Isaac Sim 5.1.0, `isaaclab`, the GR00T **client** transport | Isaac's python + `constraints/isaac-eval-known-good.txt` (additive) |

They talk over **ZMQ**: `IsaacLabRunner` (Odyssey core) spawns the eval recipe
under `ISAAC_PYTHON`; the recipe runs the Isaac Sim rollout and drives the GR00T
policy as a **client** of the server running under `GR00T_VENV_PYTHON`. The
`make_gr00t_policy` factory is a thin ZMQ client for exactly this reason — it
never imports the GR00T model in-process.

## Setup

```bash
bash scripts/setup-eval-stack.sh        # idempotent; builds all 3 envs with uv
. ./odyssey-eval-env.sh                 # the interpreter map it writes
```

Overridable: `ODYSSEY_DIR`, `ISAAC_GR00T_DIR`, `ISAACLAB_PATH`, `ISAAC_PYTHON`, `HF_HOME`.

## Hard rules (why it's split this way)

- **Never co-install across envs.** GR00T (`torch 2.7.1`, `transformers 4.57`)
  vs Isaac Sim's bundled torch vs Odyssey core are mutually incompatible —
  exactly like `[openvla]` (transformers 4.40.1) and `[specialist]` (4.49+),
  which Odyssey already keeps in separate venvs.
- **Isaac Sim / Isaac Lab are a binary install, not a pip dependency.** The
  `constraints/isaac-eval-known-good.txt` file adds only the small client
  transport into that env; it does not (and can't) pin `isaacsim`.
- **flash-attn is a CUDA/arch-specific compiled wheel** (cu128 x86_64; Blackwell
  needs the cu13 path) — installed per Isaac-GR00T's `pyproject`, after torch.
- **Launch with `env -u PYTHONPATH -u VIRTUAL_ENV …`** so a sourced parent venv
  or the ROS Jazzy `PYTHONPATH` leak doesn't shadow Isaac's bundled python.
- The gated models (`GR00T-N1.7-3B`, `Cosmos-Reason2-2B`) are HF downloads under
  `$HF_HOME`, not Python dependencies.

## Why uv (and not a single resolver / Poetry)

The hard part here is *inter-environment* (incompatible stacks), not resolving
one graph — so no single lockfile can span it. uv gives fast, reproducible
**per-env** installs, handles the PyTorch `--extra-index-url` cleanly, and is
already what Isaac-GR00T uses. Pins stay in `constraints/*-known-good.txt`
(flexible ranges in `pyproject`, exact pins per venv) — the same discipline as
the rest of the repo.
