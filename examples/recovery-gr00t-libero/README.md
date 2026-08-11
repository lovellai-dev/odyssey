# Closed-loop recovery — GR00T pilot + Reasoner SPECIALIST on LIBERO

The recovery experiment's LIBERO harness
(`docs/experiment-recovery-design-vla.md`): a tiered stuck detector runs
inside the eval loop of a chunk-emitting pilot; on trigger it truncates the
stale chunk ([VLA-Corrector](https://arxiv.org/abs/2607.01804)'s
event-triggered truncation via `ChunkPilotAdapter.flush()`) and — on a
SPECIALIST verdict — rolls the arm back to the last chunk boundary whose
verdicts were clean. The SPECIALIST is polled **asynchronously** at chunk
boundaries (`SpecialistGate`): the sim never blocks on the VLM.

The shipped `mission.yaml` is **arm B (shadow)**: detect + log everything,
intervene never. Run it first — it is behavior-identical to the plain GR00T
eval and yields the detector-quality data the live arms depend on.

## Experiment arms

| Arm | config | measures |
| --- | --- | --- |
| A. baseline | drop every recovery/shadow/specialist key | today's success rate |
| B. shadow (shipped) | `shadow_mode: true` | detector precision/recall, zero risk |
| C. kinematic-only | `recovery: true`, no `specialist_base_url` | do physics tiers alone help? |
| D. full delegation | `recovery: true` + `specialist_base_url` | the headline |

Run the same `task_id` + `num_episodes` across arms; success rates and the
`recovery` block inside `result_summary.metrics` are directly comparable.

## Prerequisites

1. **GR00T server deps** — same as `examples/franka-libero/mission-gr00t.yaml`
   (auto-serve; `server_python` if gr00t lives in another venv).
2. **Specialist endpoint** (arms B/D; drop `specialist_base_url` otherwise) —
   any OpenAI-compatible server. The default judge is `nvidia/Cosmos3-Nano`
   via vLLM-Omni:

   ```bash
   vllm serve nvidia/Cosmos3-Nano --host 0.0.0.0 --port 8002 --max-model-len 8192
   ```

   Calibrate it FIRST on your rollout videos with `examples/reasoner-probe/`
   (its `retry` question is exactly the loop's stuck prompt). The issue #78
   lesson applies: a judge that always answers NO silently degrades arm D
   into arm C — the probe catches that before you burn GPU-days.

## Outputs (under the task's output dir)

- `recovery/recovery_events.jsonl` — one line per trigger/outcome: step,
  cause (`kinematic`/`continuity`/`specialist`), kind (`flush`/`rollback`),
  `applied` (always `false` in shadow), rollback target, controller outcome.
- `recovery/rollouts/episode_NN_{PASS,FAIL}.npz` — per-step
  `frames uint8[T,H,W,3]` + `actions float32[T,7]` (`log_actions: true`).
  This is the training corpus for the VLA-Corrector LVM tier (PR 2).
  Footprint: a 520-step episode at 256×256 is ~100 MB in RAM before
  compression — budget disk accordingly or lower `camera_height/width`.
- `result_summary.metrics.recovery` — counters (shadow_triggers, flushes,
  recoveries_triggered, post_recovery_successes, specialist_polls, …).

## Flag reference (all optional, all default-off)

`recovery` · `shadow_mode` · `recovery_mode: command|teleport` ·
`max_recoveries` · `recovery_steps` · `recovery_settle_steps` ·
`stuck_window_steps` · `stuck_eps_m` · `stuck_min_commanded` ·
`continuity_eps` (0 = tier 2 off) · `poll_every_chunks` ·
`specialist_base_url` · `specialist_model` · `specialist_max_tokens` ·
`specialist_timeout_s` · `specialist_text_modality` ·
`specialist_api_key_env` · `log_actions` · `recovery_dir`
(runner-resolved to `<output_dir>/recovery` when any recovery flag is set).

`teleport` mode restores MuJoCo sim state via `env.set_init_state` when the
env exposes a state getter; otherwise the rollback degrades to a logged
`teleport_unavailable` event (command mode is unaffected).

## Notes

- The SPECIALIST entry under `robot.agents` documents the loadout; the wiring
  is `config.specialist_base_url` (the `_has_specialist` multi-agent path
  only affects the in-process OpenVLA pilot, which `pilot: gr00t` bypasses).
- π0.5 runs the identical recovery surface: swap to
  `pilot: pi05` + the openpi server keys (`examples/franka-libero`'s π0.5
  mission) — the flags are shared verbatim (`evals/recovery_wiring.py`).
- With every recovery key removed this mission is byte-identical to the plain
  GR00T eval — the flags are inert by construction.
