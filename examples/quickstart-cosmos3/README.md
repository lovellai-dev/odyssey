# Cosmos 3 (WAM) × LIBERO quickstart (eval-only, `pilot: cosmos3`)

Scores a Cosmos 3 world-action model on the LIBERO object suite through the
same chunk-aware LIBERO bridge GR00T and π0.5 use. Externally-served (π0.5
posture): cosmos-framework's HTTP `action_policy_server_libero` holds the
weights; Odyssey's client side is stdlib-only.

## Setup

```bash
bash examples/quickstart-cosmos3/setup.sh            # both halves
bash examples/quickstart-cosmos3/setup.sh --help     # knobs (checkpoint, CUDA 12, …)
```

Sets up (1) an Odyssey client venv with the LIBERO/MuJoCo stack and (2) the
cosmos-framework server env (its own uv env; NGC container also works). Nano
16B needs ≥32 GB VRAM in bf16 (H100 box) — Edge 4B fits smaller GPUs. NVIDIA
checkpoints are gated: `export HF_TOKEN=...`.

## Run (two terminals)

```bash
# T1 — serve (any family member: HF id or an export_model dir)
uv run --project ~/cosmos-framework python -m \
    cosmos_framework.scripts.action_policy_server_libero \
    --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8000

# T2 — eval
source env_pilot_cosmos3/bin/activate
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
odyssey run examples/quickstart-cosmos3/mission.yaml
```

`n_action_steps` is auto-adopted from the server's `GET /info`
(`curl :8000/info` to inspect), so the same mission serves Edge and Nano.

## ⚠️ Checkpoint reality

Only DROID policies are published — on LIBERO they are **out-of-distribution**
(episodes must complete; success is not expected). A real LIBERO score needs
your own SFT export (training runner = follow-up PR; recipe:
`nvidia/LIBERO_LeRobot_v3`, LeRobot v3 format).

First-smoke watch-list: /predict wire, concat_view order, rot6d axes +
gripper polarity (no fix-up). See `docs/migration/cosmos3-wam-integration.md`.
For the published DROID checkpoints on their native benchmark, use
`examples/cosmos3-robolab/` instead.
