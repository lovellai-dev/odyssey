# MolmoAct2 on LIBERO (eval-only)

This example scores Ai2's [MolmoAct2](https://github.com/allenai/molmoact2)
on LIBERO with `pilot: molmoact2`. The checkpoint is
[`allenai/MolmoAct2-LIBERO`](https://huggingface.co/allenai/MolmoAct2-LIBERO),
which Ai2 publishes fine-tuned on the full LIBERO mix (Spatial + Object + Goal
+ Long). MolmoAct2 is a Molmo2-ER reasoning backbone with a flow-matching
action expert, licensed Apache-2.0.

**Status: the odyssey side is wired and passes CPU tests. The GPU smoke test is
still pending.**

## How it runs

```
odyssey (LIBERO env)                                     MolmoAct2 env (GPU)
LiberoRunner ─► molmoact2_libero_eval.py ──HTTP json_numpy──► host_server_libero.py
                  ChunkPilotAdapter                         model.predict_action(norm_tag="libero")
```

Ai2 ships `/act` servers for DROID and YAM, but not for LIBERO. Their LIBERO
numbers come from the LeRobot stack. So this folder includes
`host_server_libero.py`, which is Ai2's DROID server adapted to the LIBERO
schema.

The LIBERO convention is the same one π0.5 and GR00T use. It was checked
against the checkpoint's `norm_stats.json` and LeRobot's `LiberoProcessorStep`:

- **Cameras:** `[agentview, wrist]`, in that order, both flipped 180°.
- **State (8-D):** eef position + axis-angle + the 2 gripper finger positions.
- **Action (7-D):** delta EEF, already de-normalized server-side. The gripper is
  already in LIBERO's `[-1, 1]` range.
- **Chunk length:** `action_horizon` = 10. The client reads it from `GET /act`.

## 1. Start the server (MolmoAct2 env, GPU)

```bash
git clone https://github.com/allenai/molmoact2 && cd molmoact2
uv sync                                   # torch (CUDA 12.8), transformers, fastapi, json-numpy
uv run python /path/to/odyssey/examples/quickstart-molmoact2/host_server_libero.py \
    --repo-id allenai/MolmoAct2-LIBERO --port 8000      # add --dtype float32 for the reference precision
curl -s localhost:8000/act                # {"status": "ok", "action_horizon": 10, ...}
```

## 2. Run the mission (LIBERO / odyssey env)

The client only needs the standard library and numpy, so there is no extra to
install.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    odyssey run examples/quickstart-molmoact2/mission.yaml
```

## Knobs (`task.config`)

| key | default | meaning |
|---|---|---|
| `host` / `port` | `127.0.0.1` / `8000` | `/act` server address |
| `n_action_steps` | server `action_horizon` (10) | chunk steps replayed before re-querying |
| `num_steps` | `10` | flow-matching solver steps per query |
| `invert_gripper` | `false` | flip gripper polarity (escape hatch) |
| `translation_only` | `false` | de-risk: zero rotation, gripper forced open |

**Sanity check:** Ai2's reference numbers come from
[`allenai/lerobot:molmoact2-hf-inference`](https://github.com/allenai/lerobot/tree/molmoact2-hf-inference).
If odyssey's success rate on the same suite is far off, the transforms probably
don't match: look at image flip, gripper sign and dtype first.
