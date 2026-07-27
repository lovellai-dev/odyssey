# quickstart-pi05 — Physical Intelligence π0.5 pilot on LIBERO (eval-only)

Score a **π0.5 (pi05)** checkpoint on the LIBERO object suite (Franka Panda,
sim). π0.5 is a **chunk-emitting generalist pilot**: one policy query returns a
whole action chunk that the env replays open-loop, so the pilot is queried ~an
order of magnitude fewer times per episode than the single-step OpenVLA pilot.
This is the "fewer calls/episode" latency win tracked in [issue #74][74].

This directory is the **eval half only** — there is no training task. π0.5 runs
out-of-process in an **openpi `WebsocketPolicyServer`** (openpi/JAX, a separate
env); odyssey holds only the lightweight websocket client + the LIBERO env.

[74]: https://github.com/lovellai-dev/odyssey/issues/74

---

## ⚠️ Status — read this first

The odyssey-side wiring is **complete and CPU-tested**, but **no end-to-end
LIBERO rollout has been run on real hardware yet.** Treat this as *ready to
attempt* an inference run, not *proven*. Specifically:

- **Not GPU-verified.** The CI tests pin the argv/protocol/import contracts only,
  not a real rollout. First smoke on an L4 is the acceptance gate.
- **Gripper polarity is unverified.** π0.5 applies **no** gripper fix-up (unlike
  GR00T, which normalizes+inverts) because its checkpoint is assumed baked to
  LIBERO's convention. If the gripper opens when it should close on the first
  rollout, flip it (see *Troubleshooting*).
- **openpi `infer` wire shape is assumed.** The recipe expects the server to
  return `{"actions": <ndarray>}` (openpi `LiberoOutputs`, already 7-D). Confirm
  against your openpi version's `serve_policy.py`.
- **VRAM footprint is an analytical estimate**, not a measurement
  (`docs/pi05-scoping.md` → "VRAM footprint"). Single-agent on a 24 GB L4 is the
  target; co-residence with a SPECIALIST for the multi-agent arms is unconfirmed.
- **FAST is NOT on the inference path here.** π0.5 is flow-matching → it emits
  **continuous** action chunks, so FAST detokenization does not run at inference
  (it only matters in π0.5 *training* or for an autoregressive π0-FAST model).
  The `pi05_fast.py` codec is reserved scaffolding, not wired in.

---

## Prerequisites

1. **A running openpi π0.5 policy server.** openpi lives in its own environment
   (JAX). Install it per [Physical-Intelligence/openpi][openpi] and serve a
   `pi05_libero` checkpoint, e.g. (check your openpi version for exact flags):

   ```bash
   # in the openpi repo/env — serves a WebsocketPolicyServer on 0.0.0.0:8000
   uv run scripts/serve_policy.py --env LIBERO
   #   (or: policy:checkpoint --policy.config=pi05_libero \
   #        --policy.dir=gs://openpi-assets/checkpoints/pi05_libero)
   ```

   The checkpoint is Apache 2.0 (`docs/pi05-scoping.md` → "License and weights").

2. **The openpi client + LIBERO in the odyssey env.** The eval recipe imports
   `openpi_client` (the lightweight websocket client) and the `libero` package:

   ```bash
   pip install openpi-client            # the client only, not the JAX server stack
   # LIBERO + robosuite/MuJoCo: see examples/franka-libero/setup.sh
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl   # headless VM
   ```

[openpi]: https://github.com/Physical-Intelligence/openpi

---

## Run

Point the mission at your server's `host`/`port` (defaults `127.0.0.1:8000` in
`mission.yaml`), then:

```bash
odyssey validate examples/quickstart-pi05/mission.yaml
odyssey run      examples/quickstart-pi05/mission.yaml
```

The runner spawns `src/odyssey/runners/evals/pi05_libero_eval.py`, which:

1. builds the LIBERO env for `benchmark_name` / `task_id`;
2. connects to the openpi server via `make_pi05_pilot(host, port, …)`, wrapping
   the client in the shared `ChunkPilotAdapter` (buffer + replay + flush-on-
   instruction-change);
3. per env step calls `pilot.act(...)`, which re-queries the server only when the
   chunk drains (every `n_action_steps`);
4. streams the `ODYSSEY_EPISODE` / `ODYSSEY_RESULT` protocol the runner scores.

Videos (one MP4 per episode) land under the task's output dir when
`capture_video: true`.

---

## What this covers vs [issue #74][74]

| Acceptance item (#74) | State here |
| --- | --- |
| Confirm π0.5 weights + FAST availability/license | ✅ `docs/pi05-scoping.md` (Apache 2.0; FAST reusable via openpi/HF) |
| Shared chunk-aware `PilotRuntime` abstraction (GR00T + π0.5) | ✅ `ChunkPilotAdapter` (pilot-agnostic, reused, not forked) |
| π0.5 pilot runs a LIBERO eval end-to-end (single-agent) on the VM | 🚧 **wiring done, GPU smoke pending** (this mission is the vehicle) |
| Measure calls/episode + wall-clock vs OpenVLA baseline | ⬜ after the smoke runs |
| Docs: pilot line-up + chunk-aware runtime notes | ✅ scoping doc + this README |

Open questions from #74 and where they stand: **weights/license** (Q1) and **FAST
availability** (Q2) are resolved in the scoping doc; **VRAM on a single L4** (Q3)
and **sim embodiment/action mapping** (Q4) are implemented per the scoping
analysis but **await the GPU smoke** for confirmation.

---

## Config keys (`tasks[].config`)

| key | meaning |
| --- | --- |
| `pilot: pi05` | routes the eval to `pi05_libero_eval.py` + the openpi client |
| `checkpoint` | recorded in the summary; the **server** loads the real weights |
| `task_id` | which task within the suite (0..9) |
| `host` / `port` | the pre-started openpi `WebsocketPolicyServer` address |
| `api_key` | optional, only if your server requires one |
| `n_action_steps` | steps replayed per chunk before re-query (10 for `pi05_libero`) |
| `translation_only` | de-risk knob: zero rotation + force gripper open |
| `capture_video` | one MP4 per episode |
| `task_instruction` / `strict_instruction` | optional instruction-drift alignment contract |

Keys the runner consumes itself (`pilot`, `checkpoint`, `runner`,
`capture_video`, `video_fps`, `video_format`) are **not** forwarded to the eval
script as flags.

---

## Troubleshooting

- **Connection refused / timeout** — the openpi server isn't up (or wrong
  `host`/`port`). This mission is open-loop: start the server first.
- **`NotImplementedError: … requires the openpi client ('openpi_client')`** —
  install `openpi-client` in the odyssey env.
- **Gripper does the opposite of what it should** — the "no fix-up" assumption is
  wrong for your checkpoint. As a stopgap, try `translation_only: true` to
  isolate the arm, then port GR00T's gripper fix-up into `pi05_action_to_libero`
  (`src/odyssey/runners/evals/pi05_transforms.py`) if confirmed.
- **`'libero' is not a package`** — the recipe drops its own directory from
  `sys.path` to avoid shadowing the real LIBERO namespace; if you see this,
  you're likely importing the script wrong (run via `odyssey run`, not directly).
- **Black/upside-down video** — LIBERO's offscreen frames are stored 180°-rotated;
  `flip_images` defaults on. Leave it unless your frames are already upright.
