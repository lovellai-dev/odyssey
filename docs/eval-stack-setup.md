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

## Provisioning Isaac Sim / Isaac Lab (the binary precondition)

Isaac Sim is a large NVIDIA Omniverse binary with its own bundled Python — it is
installed by NVIDIA's tooling, **not** by uv/pip resolution, and
`setup-eval-stack.sh` treats it as a precondition (env #3 is SKIPped with a note
if `ISAAC_PYTHON` is missing). Provision it once into a dedicated py3.11 conda env:

```bash
# 1) Dedicated py3.11 env for Isaac
conda create -n isaaclab python=3.11 -y && conda activate isaaclab
pip install --upgrade pip

# 2) Isaac Sim 5.1.0 (the extscache extras pull the Omniverse kit runtime)
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

# 3) Isaac Lab on top — its installer wires the conda env
git clone https://github.com/isaac-sim/IsaacLab.git "$HOME/IsaacLab"
cd "$HOME/IsaacLab" && git checkout v2.3.2 && ./isaaclab.sh --install

# 4) Verify (find_spec checks the install without importing Omniverse / a GPU)
python -c "import importlib.util as u, sys; sys.exit(0 if u.find_spec('isaaclab') else 1)" \
  && echo "isaaclab OK"
```

Then point the stack at it (these are the defaults in `setup-eval-stack.sh`):

```bash
export ISAAC_PYTHON="$HOME/miniconda3/envs/isaaclab/bin/python"
export ISAACLAB_PATH="$HOME/IsaacLab"
export OMNI_KIT_ACCEPT_EULA=YES        # non-interactive EULA for headless / CI
```

Requirements & gotchas:

- **GPU + driver:** the eval *renders*, so it needs an NVIDIA GPU with a
  CUDA-12.x driver and a working Vulkan ICD (`find_spec` above only checks the
  install). Verified on the H100 VM (Isaac Sim 5.1.0.0, driver 580.x).
- **Headless hosts:** set `OMNI_KIT_ACCEPT_EULA=YES`; EGL rendering works with no
  display.
- **Blackwell (RTX PRO 6000):** needs the cu13 wheel path (see Isaac-GR00T's
  `pyproject`); the GR00T server pins here are cu128 (pre-Blackwell, x86_64).
- Keep the Isaac Lab checkout on the version matching Isaac Sim 5.1.0 (here
  `v2.3.2`) so the APIs line up.

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
