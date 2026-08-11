# Experiment: closed-loop VLA recovery (GR00T / π0.5) with VLA-Corrector integration

**Branch**: `experiment-vla-recovery-closed-loop` (off `develop`)
**Status**: design final + implementation plan agreed; implementation not started.
**Paper being integrated**: [VLA-Corrector — Lightweight Detect-and-Correct Inference
for Adaptive Action Horizon](https://arxiv.org/abs/2607.01804)
**Code being integrated**: <https://github.com/ZJU-OmniAI/vla-corrector>
(Apache-2.0; LeRobot fork whose self-contained `src/siglip_dynamics/` module is the
vendorable piece)

> The full design history (RoboLab/WAM arm, prior-art survey, code audits, scoping
> decisions) lives in this file's git history (commit `49a739a`). This version is the
> clean statement of the final experiment.

## 1. The experiment

A SPECIALIST agent (a served VLM reasoner) monitors the simulation **while** the PILOT
executes action chunks. When the arm is stuck or failing, the system **truncates** the
stale chunk (VLA-Corrector's mechanism) and/or **rolls back** the arm to the last good
chunk boundary, then re-queries the policy. The question: does
detection → truncate/rollback → re-query convert failures into successes — and what
does VLM *semantics* add over progressively cheaper signals?

- **Pilots**: GR00T-N1.7-LIBERO (primary — in-domain, 10/10 on `libero_object`, and
  the existing perturbation harness induces failures on demand) and π0.5 (secondary —
  demonstrates pilot-agnosticism). Both are chunk-emitting and run through the shared
  `ChunkPilotAdapter`.
- **Specialist**: any OpenAI-compatible served reasoner via
  `OpenAICompatCompletionJudge` with a stuck/RETRY prompt. Default model:
  `nvidia/Cosmos3-Nano` on vLLM (pure config, not a code dependency).
- **Eval**: LIBERO, through our own recipes (`gr00t_libero_eval.py`,
  `pi05_libero_eval.py`) — we own the loop; MuJoCo state restore enables the
  `teleport` ablation.
- **Experiment arms** (same tasks, same init states):
  A. baseline (recovery off) · B. shadow (detect + log, never intervene) ·
  C. kinematic-only recovery · D. full delegation (specialist in the loop).
- **Metrics**: success rate per arm; detector precision/recall (from shadow verdicts
  vs post-hoc labels); recoveries per episode; post-recovery success rate; specialist
  latency distribution; wall-clock overhead.

## 2. What we take from VLA-Corrector, and when

The paper has two halves — a **detector** and a **response**:

| | VLA-Corrector (paper) | This experiment |
| --- | --- | --- |
| Detection | trained LVM (~40M params) over latent visual dynamics | tier 1 kinematic + tier 2 inter-chunk (training-free) + tier 4 specialist VLM (semantic) — **+ the LVM as tier 3 in PR 2** |
| Response | truncate stale chunk + re-query (+ OGG gradient guidance) | truncate (same mechanism) **+ rollback** to the last-good snapshot (not in the paper) |

**Adopted now (PR 1)**: event-triggered **truncation** → a new
`ChunkPilotAdapter.flush()`. The paper's own ablation shows truncation alone carries
most of the gain (MetaWorld 48.70 → 60.35 of the full 64.35). Plus per-step
**(frame, action) logging** — this is exactly the training corpus the LVM needs.

**Adopted next (PR 2)**: the **LVM detector**, as the *independent-encoder variant*:
our pilots are served over HTTP/WebSocket, so the paper's faithful read of the VLA's
internal SigLIP features is unavailable — instead a client-side SigLIP encoder feeds
the same corrector architecture (vendorable from `src/siglip_dynamics/`,
Apache-2.0), trained on the rollouts PR 1 logs, plugged in as one more detector tier
behind the same trigger interface. The ordering is a hard data dependency: the LVM
cannot be trained before rollouts are logged.

**Never**: **OGG** (online gradient guidance) — needs gradients through the policy,
impossible with served pilots.

**Finetuned checkpoints are fully compatible**: the paper keeps the VLA frozen (it
evaluates finetuned π0.5 on LIBERO: 94.0 → 97.8) and our external variant never
touches policy weights. Finetuned GR00T/π0.5 checkpoints are in fact what makes
corpus generation cheap. GR00T is not evaluated in the paper (its backbone is not
SigLIP-based) — another reason the independent-encoder variant is the right fit.

Related work informing the design (details in git history): EWAM
(<https://arxiv.org/abs/2606.12690>, rollback on a Cosmos3 backbone — our published
baseline), Rewind-IL (<https://arxiv.org/abs/2604.16683>, command-back respawn +
inter-chunk TIDE signal), SV-VLA (<https://arxiv.org/abs/2604.02965>, concurrent
verifier pattern; no code reuse — OpenVLA-fork entangled).

## 3. Architecture

New strict-mypy modules in `src/odyssey/runners/agents/`:

- **`recovery.py`** (stdlib-only math): `BoundarySnapshot` + `ChunkLedger`
  (boundary snapshots: EE pos/quat, gripper, optional sim state, per-tier verdicts;
  `last_good()` defines "last successful chunk"), `StuckMonitor` (tier 1: commanded
  motion vs EE displacement over a window; tier 2: new-chunk head vs previous-chunk
  tail continuity), `RecoveryController` (EE-space P-controller emitting 7-D OSC
  delta actions back to a snapshot; outcomes idle/driving/arrived/exhausted),
  `RecoveryPolicy` (orchestrator: tiers 1–2 → **flush**, specialist verdict →
  **rollback**; `shadow_mode`; `max_recoveries` budget; staleness rules; jsonl
  events + metrics), `RolloutLog` (npz per episode: `frames uint8[T,H,W,3]`,
  `actions float32[T,7]`).
- **`specialist_gate.py`**: `STUCK_PROMPT_TEMPLATE` (ported RETRY wording from the
  reasoner probe) + `SpecialistGate` — background single-worker thread + mailbox,
  at-most-one call in flight, skip-never-queue (judge slower than a chunk degrades
  to skipped polls), (episode, chunk) staleness tags, injectable executor for
  deterministic tests. `specialist_max_tokens` defaults 256 (reasoner CoT truncates
  at the judge's default 8) + `extra_body={"modalities": ["text"]}`.

Changes to existing code:

- `ChunkPilotAdapter.flush()` — drop the buffered chunk so the next `act()`
  re-queries (`steps_remaining == 0` already flags boundaries incl. post-flush).
- **GR00T recipe migrated onto `ChunkPilotAdapter`** (it still inlines its chunk
  loop today); equivalence pinned by test (identical actions + query cadence vs the
  old loop). π0.5 already uses the adapter.
- Recovery wiring in both recipes behind default-off flags (`recovery`,
  `shadow_mode`, `recovery_mode command|teleport`, stuck thresholds,
  `poll_every_chunks`, `specialist_base_url/model`, `log_actions`, …) — flags off ⇒
  byte-identical behavior to today. `teleport` hasattr-guards `get_sim_state` and
  degrades to `command`.
- `LiberoRunner`: runner-resolved `--recovery_dir` (the `--video_dir` pattern);
  recovery counts flow through `ODYSSEY_RESULT` metrics into `result_summary`
  verbatim; `recovery_events.jsonl` + npz attached as artifacts.
- Examples: `examples/reasoner-probe/` (ported pilot-neutral from
  `cosmos3-integration~1`) and `examples/recovery-gr00t-libero/` (PILOT GR00T +
  SPECIALIST Cosmos3-Nano mission, arms A–D documented).

## 4. Implementation plan (PR 1 — draft PR → develop, 9 commits, each green)

1. `feat(agents): ChunkPilotAdapter.flush()` + tests
2. `feat(recovery): pure recovery core — ledger, tiered monitor, EE controller, policy` + tests
3. `feat(recovery): async specialist stuck-gate over the OpenAI-compat judge` + tests
4. `refactor(gr00t-libero): migrate inline chunk loop onto ChunkPilotAdapter` + equivalence tests
5. `feat(recovery): wire recovery/specialist/rollout-logging into both LIBERO recipes` + tests
6. `feat(libero): recovery_dir plumbing + recovery artifacts in LiberoRunner` + tests
7. `feat(examples): port reasoner-probe (pilot-neutral)`
8. `feat(examples): recovery-gr00t-libero mission`
9. `docs(recovery): implementation status`

Detailed file-by-file plan (APIs, flags, test list):
`~/.claude/plans/clever-giggling-harbor.md`.

**Top risks**: command-mode controller vs OSC gains/frames (shadow first; exhausted
outcome is data; teleport ablation isolates it) · GR00T migration regression
(equivalence test; per-episode `pilot.reset()`) · gate thread lifecycle/staleness
(at-most-one in flight, judge-side timeout, injectable executor) · `get_sim_state`
unverified (guarded degrade) · npz memory (~100 MB/episode pre-compression;
per-episode save, default-off).

## 5. Roadmap

- **PR 1 (this)**: recovery skeleton + VLA-Corrector truncation + rollout logging.
- **PR 2**: VLA-Corrector LVM tier — vendor/adapt `siglip_dynamics`, train the
  corrector on the logged corpus, calibrate thresholds conformal-style, plug in as
  tier 3.
- **GPU phases (after PR 1 merges)**: 0) specialist calibration with
  `examples/reasoner-probe/` on our rollout videos · 2) shadow runs (arm B data) ·
  3) live arms A/C/D.
- **Iteration 2**: port to Cosmos3-WAM × RoboLab (entry-script design preserved in
  this file's git history).
