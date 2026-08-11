# Closed-loop recovery experiment — VLA pilots (GR00T / π0.5) + Reasoner specialist

**Branch**: `feat/vla-recovery-closedloop` (off `develop`)
**Status**: design draft — iteration 1 scoped VLA-first, see §13 (which
supersedes the RoboLab/WAM framing of §§1–6 for this iteration; that design
is kept as the iteration-2 target).

> History: this doc started as the Cosmos3-WAM × RoboLab design
> (`docs/cosmos3-recovery-experiment-design.md` on the discarded
> `feat/cosmos3-recovery-closedloop` branch). §§10–13 record how analysis,
> prior art, and the code audit re-scoped iteration 1 onto the VLAs we
> already operate end-to-end. The only cosmos3-branch *code* this experiment
> needs — the generic `OpenAICompatCompletionJudge`
> (`runners/agents/openai_judge.py` + its unit test, 19 tests green on this
> base) — has been ported onto this branch; the specialist model
> (Cosmos3-Nano Reasoner) is pure served config, not a code dependency.

## 1. The experiment

Can a SPECIALIST agent, watching the sim **while** the PILOT executes action
chunks, detect that the arm is stuck and trigger a recovery that returns the
arm to the last *successful* action-chunk boundary — and does that recovery
improve episode success rate?

- **Pilot**: `nvidia/Cosmos3-Nano-Policy-DROID` (chunk-emitting WAM, served by
  cosmos-framework's `action_policy_server_robolab`, WebSocket).
- **Specialist**: `nvidia/Cosmos3-Nano` Reasoner (vLLM-Omni, OpenAI-compatible
  endpoint) — the exact judge `examples/cosmos3-reasoner-probe/` already
  exercises, including a `RETRY_TEMPLATE` ("is the robot FAILED or STUCK such
  that it should abort and retry?") that is the seed of the stuck check.
- **Eval**: RoboLab (NVlabs' Isaac Lab twin of the DROID rig), already a
  first-class `evaluation_type: robolab` (`runners/evals/robolab.py`).
- **Coordination style**: delegation — the specialist authors no plan; it is
  asked perception questions on demand while actions are produced
  (the PR #68 delegation spirit, moved from post-hoc video probing to
  real-time in-the-loop).

This is the step from *post-hoc* analysis (reasoner probe on rollout MP4s) to
*concurrent* analysis (verdicts consumed inside the control loop).

## 2. What exists today (inventory)

| Piece | Where | State |
| --- | --- | --- |
| Cosmos3 pilot factory (HTTP client + chunk adapter) | `runners/models/cosmos3.py` (`make_cosmos3_pilot`, lines 75–138) | done (LIBERO wire) |
| Chunk buffer/replay + flush-on-instruction-change | `runners/agents/chunk_pilot.py` (`ChunkPilotAdapter`) | done, pilot-agnostic |
| Chunk-boundary VLM polling (cost model: poll per chunk, not per step) | `runners/agents/completion_gate.py` (`ChunkCompletionGate`) | done, **not wired** into any default flow |
| OpenAI-compat YES/NO judge (vLLM, `modalities:["text"]` fix, zoom/upscale lesson from issue #78) | `runners/agents/openai_judge.py` + `examples/cosmos3-reasoner-probe/` | done, validated post-hoc |
| RoboLab runner (subprocess, `episode_results.jsonl` scoring, docker bring-up) | `runners/evals/robolab.py` + `examples/cosmos3-robolab/` | done, H100-smoked |
| `entry_script` mission knob (select RoboLab policy backend) | `robolab.py` handled config keys; README: "a future backend is a mission edit, not a new runner" | done — **our hook** |
| Multi-agent runtimes (planner arm) | `runners/agents/{runtime,planner,planned,remote_planner}.py` | done (planning), delegation arm lives on `feat/multiagent-delegation-grounding`, **not merged here** |
| SPECIALIST role in mission spec | `spec/agents.py` (`AgentRole.SPECIALIST`) | done |

What does **not** exist: any per-step observation access or mid-episode state
manipulation for RoboLab from Odyssey. `RobolabRunner` launches
`policies/cosmos3/run.py`, waits, and reads `episode_results.jsonl`. The sim
loop, the policy WebSocket client, and all robot state live inside the Isaac
Sim docker process.

## 3. The architectural gap and the placement decision

The closed loop needs three capabilities RoboLab's stock `run.py` doesn't
expose: (a) frames at chunk boundaries for the specialist, (b) a decision
point between chunks, (c) a way to move the arm back to a saved configuration.

### Options considered

**A. Policy-proxy shim (interpose on the WebSocket).** Odyssey starts a proxy
that RoboLab connects to as if it were the policy server; the proxy forwards
obs → real server, sees every observation and chunk, and can substitute a
"recovery chunk" (reverse-replay of executed deltas). No RoboLab fork.
*Rejected as primary*: no access to proprio/joint state (the DROID wire is
video-conditioned), recovery must be approximated by negating action deltas
(fragile: controller dynamics, gripper state), and the RoboLab client wire
protocol has to be reverse-engineered and pinned. Keep in the back pocket.

**B. Custom RoboLab entry script (recommended).** A recovery-aware sibling of
`policies/cosmos3/run.py`, selected via the existing `config.entry_script`
mission knob, running **inside** the Isaac process where robot state is
first-class. It reuses RoboLab's own policy client and env setup and wraps the
step loop with our recovery logic. The *brains* (stuck gate, chunk ledger,
recovery trajectory) live in Odyssey as pure numpy/stdlib modules — same
pattern as `ChunkPilotAdapter`/`ChunkCompletionGate`: unit-testable without
GPU/Isaac, imported by the entry script via the dual bind-mount +
`sys.path.insert(repo/src)` trick the example bridges already use.

**C. Do it on LIBERO instead.** In-process env, `set_init_state`, full loop
control — much easier plumbing. *Rejected as the headline* (the experiment is
DROID-rig + RoboLab by design) but noted as a de-risking fallback if RoboLab
iteration cost becomes the bottleneck.

### Decision

**Option B**: Odyssey-owned recovery brain + thin Isaac-side entry script.
Rationale: real joint access makes "return to last good chunk" a *commanded*
motion (transferable to a real arm) instead of an action-algebra
approximation; RoboLab's client code is reused rather than re-implemented; and
the mission-level surface is just `entry_script:` + config keys — no new
runner, minor `RobolabRunner` extensions for artifacts.

## 4. Proposed architecture

```
┌────────────────────────────── Isaac Sim docker ──────────────────────────────┐
│  recovery_run.py  (entry_script — sibling of policies/cosmos3/run.py)        │
│                                                                              │
│   sim step loop ──► chunk boundary ──► ChunkLedger.snapshot(joints, frame)   │
│        │                   │                                                 │
│        │                   ├─► StuckMonitor (kinematic, cheap, every step)   │
│        │                   │                                                 │
│        │                   └─► SpecialistGate ──(bg thread)──► vLLM Reasoner │
│        │                              │      verdict mailbox   (Cosmos3-Nano)│
│        ▼                              ▼                                      │
│   policy WebSocket client      RecoveryController                            │
│   (RoboLab's own, unchanged)   (joint-interp back to last-good snapshot,     │
│        │                        then flush pilot chunk state + re-query)     │
│        ▼                                                                     │
│   action_policy_server_robolab (Cosmos3-Nano-Policy-DROID)                   │
│                                                                              │
│   emits: episode_results.jsonl (+ per-episode recovery events)               │
└──────────────────────────────────────────────────────────────────────────────┘
   Odyssey side: RobolabRunner (unchanged launch) + recovery_events artifact
                 + recovery metrics folded into result_summary
```

### 4.1 Components (new code)

1. **`ChunkLedger`** (`src/odyssey/runners/agents/chunk_ledger.py`, pure).
   At every chunk boundary records: chunk index, joint positions + gripper
   state, EE pose if available, boundary frame (downscaled), and later the
   verdicts. Exposes `last_good()` — the most recent boundary whose verdicts
   were clean. This *defines* "último action chunk exitoso": a chunk after
   which neither the kinematic monitor nor the specialist flagged trouble.

2. **`StuckMonitor`** (same module or sibling, pure). Cheap kinematic
   first-tier: commanded motion ≠ 0 but EE/joint displacement < ε over the
   last K steps ⇒ *stuck candidate*. Runs every step, costs nothing, and
   gates the expensive VLM call — this is the false-positive/latency
   flywheel: the specialist is only consulted when physics already looks
   suspicious (plus a low-rate background poll, `poll_every_chunks`, to catch
   semantic failure modes the kinematics can't see, e.g. dropped object).

3. **`SpecialistGate`** (evolution of `ChunkCompletionGate`). Wraps the
   existing `OpenAICompatCompletionJudge` with the `RETRY_TEMPLATE`-style
   stuck prompt, but **asynchronously**: the judge call (~1–2 s) runs in a
   background thread; the sim never blocks. Verdicts land in a mailbox and
   are consumed at the *next* chunk boundary. A chunk of 16 steps at DROID
   control rate gives comfortable headroom; a verdict arriving one boundary
   late is acceptable because "stuck" is a persistent state, not an edge.
   Includes timeout + in-flight-call cancellation on episode end.

4. **`RecoveryController`** (pure numpy). Given current joints and the
   `last_good()` snapshot, emits a joint-space interpolated trajectory
   (M steps, capped velocity) driving the arm back, gripper commanded to the
   snapshot's state. After arrival: flush the pilot's chunk state (the
   WebSocket client just re-queries — the server is stateless per call) and
   resume normal control from the restored pose. Budget: `max_recoveries`
   per episode; exceeding it ends the episode as a scored failure with
   reason `recovery_budget_exhausted`.

5. **`recovery_run.py`** (`examples/cosmos3-recovery-robolab/`, Isaac-side).
   Forked from RoboLab's `policies/cosmos3/run.py`; wraps its loop with 1–4.
   Writes `recovery_events.jsonl` (one line per trigger: step, cause
   kinematic|specialist, verdict latency, rollback target chunk, outcome)
   next to `episode_results.jsonl`. Example-bridge conventions: repo-`src`
   path injection, argparse passthrough, not in strict-mypy scope.

6. **`RobolabRunner` extensions** (small): copy `recovery_events.jsonl` as an
   artifact when present; fold counts (recoveries triggered / episodes with
   recovery / post-recovery successes) into `result_summary.metrics`.

7. **Mission surface**: SPECIALIST agent declared in `robot.agents`
   (`nvidia/Cosmos3-Nano`); eval task keys (all passthrough → argv, no spec
   change needed):

   ```yaml
   config:
     entry_script: examples/cosmos3-recovery-robolab/recovery_run.py  # host path, dual-mounted
     specialist_base_url: "http://<host>:8002/v1"
     specialist_model: nvidia/Cosmos3-Nano
     recovery: true            # false = baseline arm of the experiment
     shadow_mode: false        # true = detect + log, never intervene (Phase 2)
     stuck_eps_m: 0.005        # kinematic threshold
     stuck_window_steps: 12
     poll_every_chunks: 2
     max_recoveries: 3
     recovery_steps: 24
   ```

### 4.2 Concurrency model (the "fontanería")

Single sim thread, one background judge thread, mailbox in between — no
asyncio inside Isaac (their loop is synchronous; a thread + `queue.Queue` is
the entire concurrency story). Ordering guarantees:

- Snapshot happens *before* the chunk that follows it executes, so a rollback
  target is always a pose the arm actually held at a boundary.
- At most one specialist call in flight; if a boundary arrives while one is
  pending, we skip launching another (the pending verdict covers it).
- A stale verdict (episode ended, or a recovery already ran since the frame
  was captured) is discarded by tagging calls with (episode, chunk index).
- Recovery is exclusive: while the `RecoveryController` drives, the policy
  client is not queried and the monitors are muted until arrival + settle.

### 4.3 Recovery semantics — command-back vs teleport

Two modes, worth having both behind a flag:

- **`command` (default, headline)**: drive the arm back via the controller.
  Physically honest, transferable to a real robot, but can fail (truly wedged
  arm may not track the trajectory — that itself is a measurable outcome).
- **`teleport` (ablation/upper bound)**: write joint state directly into the
  articulation (Isaac API). Clean geometry, breaks physics realism (held
  objects, contacts). Useful to separate "detector was right" from "recovery
  motion failed".

Not proposed: full sim-state restore (objects included). It measures a
different (purely sim-side) capability and has no real-robot analogue.

## 5. Experiment protocol

**Arms** (same task set, same episode seeds/init states where RoboLab allows):

| Arm | recovery | specialist | purpose |
| --- | --- | --- | --- |
| A. baseline | off | off | today's success rate |
| B. shadow | off | on (log-only) | detector precision/recall + latency, zero intervention |
| C. kinematic-only | on | off | is the VLM adding anything over physics? |
| D. full delegation | on | on | the headline |

**Metrics**: episode success rate per arm; detector precision/recall (shadow
verdicts vs post-hoc labels from rollout videos — the reasoner-probe tooling
already produces per-frame verdict tables for exactly this kind of labeling);
recoveries per episode; post-recovery success rate (episodes that succeeded
after ≥1 recovery — the direct evidence recovery *causes* success); judge
latency distribution; wall-clock overhead per arm.

**Stuck induction**: start with the natural failure rate (the policy on
RoboLab is not saturated; failures exist — `episode_results.jsonl` already
carries failure reasons to mine). If natural stuck events are too rare for
statistical power, inject perturbations in the entry script (action-offset
window, transient gripper freeze) — the GR00T×LIBERO perturbation work
(`docs/gr00t-libero-perturbations-summary.md`) is the precedent. Decide after
Phase 2 data.

## 6. GPU topology

Three GPU consumers (H100-class box, per the existing bring-up):

1. Policy server (Nano-Policy-DROID, bf16 ≥32 GB) — port 8000.
2. vLLM Reasoner (Cosmos3-Nano) — port 8002, bounded `--gpu-memory-utilization`.
3. RoboLab Isaac docker (rendering + PhysX).

Mandatory: `disable_subtask: true` in the mission — RoboLab otherwise
auto-spawns its *own* ~50 GB vLLM judge (documented VRAM landmine in
`examples/cosmos3-robolab/README.md`). Our specialist replaces it. If one
card can't hold all three, split policy server vs (Isaac + reasoner) across
two — all links are already network-transparent.

## 7. Phasing (de-risk order)

- **Phase 0 — specialist calibration (no new code)**: run the existing
  reasoner probe against RoboLab rollout videos (from the smoke runs) with
  the retry/stuck question; verify the judge discriminates stuck vs nominal
  on *this* visual domain before building anything on it (issue #78 lesson:
  the judge, not the camera, was the problem last time).
- **Phase 1 — pure modules + tests (no GPU)**: `ChunkLedger`, `StuckMonitor`,
  `SpecialistGate` (fake judge), `RecoveryController`; unit tests pin
  ordering, mailbox staleness, budget, trajectory bounds. Strict-mypy clean.
- **Phase 2 — shadow mode on hardware**: entry script forked, specialist
  wired, `recovery: false, shadow_mode: true`. Validates the fontanería
  (mounts, imports, threading, latency) with zero behavioral risk, and
  yields the detector-quality data (arm B).
- **Phase 3 — recovery live**: arms A/C/D, headline comparison.

Phase boundaries are also natural PR boundaries (draft PR → develop, per
repo flow).

## 8. Risks

| Risk | Mitigation |
| --- | --- |
| RoboLab `run.py` internals differ from assumption (loop/client structure) — the fork inherits them | Read + pin the actual `run.py` from the checkout **first task of Phase 2**; keep the fork minimal (wrap, don't rewrite) |
| Judge quality on RoboLab frames (issue #78 déjà vu: always-NO/always-YES) | Phase 0 gate — no build-out until the probe discriminates; zoom/upscale knobs already exist |
| Judge latency > chunk duration | Async mailbox tolerates one-boundary lag; `poll_every_chunks` throttle; measure in Phase 2 |
| Commanded rollback fails when genuinely wedged | That outcome is itself data; `teleport` ablation isolates it; recovery budget bounds the cost |
| Odyssey code import inside the Isaac container | Already-solved pattern: dual bind-mount + `PYTHONPATH` + example-bridge `src` injection (see robolab README wrapper table) |
| Natural stuck events too rare | Perturbation injection fallback (§5); decide on Phase 2 data |
| VRAM contention (3 consumers) | `disable_subtask`, bounded vLLM util, two-card split if needed |

## 9. Open questions (to settle before plan mode)

1. **Recovery semantics v1**: geometric rollback only (return + re-query), or
   should the specialist also *re-instruct* the pilot on recovery (e.g. "lift
   the arm and retry") — true delegation? Proposal: v1 geometric (measurable,
   simple), v2 semantic re-instruction as follow-up.
2. **`command` vs `teleport`** as default recovery mode (proposal: command
   default, teleport as ablation flag).
3. **Task selection**: which RoboLab task(s) and how many episodes per arm
   for acceptable power? (Depends on natural failure rate from the smoke
   data.)
4. **"Last good" definition**: strictly the last boundary with clean verdicts,
   or last boundary with clean verdicts *and* forward progress (EE
   displacement toward target)? Proposal: start with clean-verdicts only.
5. **Where the delegation naming lands**: reuse the `coordination: delegation`
   key from the unmerged `feat/multiagent-delegation-grounding` arm for
   consistency, or keep this experiment's keys self-contained under
   `recovery:`? (The delegated runtime itself is not needed — this loop is
   its own runtime — but key naming should not collide later.)
6. **LIBERO as development harness** (§10): build and debug the full recovery
   loop on LIBERO first (cheap iteration, real state save/restore, in-domain
   frames for the judge) and port to RoboLab as the final step — or go
   RoboLab-direct?

## 10. Would the methodology change on LIBERO / robosuite? (backend portability)

Short answer: the **brain doesn't change; the plumbing collapses**. The entire
Option-A/B/C deliberation in §3 exists only because RoboLab's sim loop lives
inside an Isaac Sim docker process Odyssey can't reach into. That constraint
is RoboLab-specific.

### What is backend-agnostic by construction

`ChunkLedger`, `StuckMonitor`, `SpecialistGate`, `RecoveryController`, the
four experiment arms, the metrics, and the phasing were all deliberately
designed as pure numpy/stdlib modules with no Isaac or RoboLab imports. They
port verbatim. This is the same bet `ChunkPilotAdapter` already won across
GR00T/π0.5/Cosmos3.

### What changes per backend

| | RoboLab | LIBERO | Robosuite |
| --- | --- | --- | --- |
| Loop ownership | RoboLab's `run.py` (Isaac docker) — must fork via `entry_script` | **ours**: `cosmos3_libero_eval.py` is Odyssey's own recipe | ours: `RobosuiteRunner` steps in-process |
| Integration cost | docker mounts, PYTHONPATH, fork of external code | edit our own script — no external code touched | as LIBERO, but **no cosmos3 pilot wiring exists** (OpenVLA only) |
| Proprio for `StuckMonitor` | joints read Isaac-side | robosuite obs carry joint/EE state natively | same |
| `teleport` recovery | joint write via Isaac API (arm only) | **real sim-state restore** (`get_sim_state`/`set_init_state`, objects included) | same MuJoCo machinery |
| `command` recovery | joint-space trajectory | EE-space P-controller emitting the env's native 7-D delta actions toward the saved EE pose | same |
| Pilot domain fit | **in-domain** (Cosmos3-…-Policy-DROID on the DROID twin) | OOD until a Cosmos3-LIBERO SFT exists (quickstart-cosmos3 is OOD smoke) | no Cosmos3 path at all |
| Judge visual domain | must calibrate on RoboLab frames (Phase 0) | probe already ran on this class of frames | uncalibrated |

So on LIBERO the methodology *simplifies* rather than changes: no entry-script
fork, no docker plumbing — the recovery loop wraps the rollout loop of our own
recipe (`cosmos3_libero_eval.py`, `run_eval`), and mid-episode MuJoCo state
save/restore makes `teleport` a *true* restore (objects included), which
RoboLab can't offer. Robosuite is the weakest fit: it would first need the
cosmos3 pilot wiring that LIBERO already has.

### The trade-off is scientific, not architectural

LIBERO buys iteration speed and determinism but costs validity: the
cosmos3-droid pilot is OOD there, so failures are dominated by
"policy doesn't know this domain" rather than the recoverable-stuck events the
experiment is about, and "return to last good chunk" is less meaningful when
few chunks are good. RoboLab is the opposite: in-domain pilot, meaningful
stuck events, expensive plumbing.

**Implication for staging** (open question 6): the strongest sequencing may be
LIBERO as the *development harness* — Phases 1–2 (and mechanically Phase 3)
debugged there with real state restore as ground truth — then RoboLab as the
*headline experiment*, where the only new work left is the thin entry-script
glue of §4.1(5). That converts RoboLab risk into a port, not a build.

## 11. Prior art (web survey, 2026-08-11)

**Nobody has done closed-loop recovery/intervention on RoboLab publicly.**
All 16 issues on [NVlabs/RoboLab](https://github.com/NVlabs/RoboLab) (423★,
last push 2026-08-10) are reproducibility/VRAM/checkpoint-access/success-judge
questions — zero about intervention hooks, custom recovery backends, or
mid-episode state control. The niche is open.

The surrounding literature, though, is active and directly validates several
of our design choices:

| Work | What it does | Relevance to us |
| --- | --- | --- |
| [EWAM](https://arxiv.org/abs/2606.12690) (2026) | Closed-loop online adaptation on a **frozen Cosmos3 backbone**: internal anomaly-detection layer monitors predicted-vs-actual state divergence; a routing layer picks direct execution / conservative replanning / **rollback recovery**. Evaluated on **RoboLab and LIBERO** vs Cosmos3/GR2/π0. | The closest work — same backbone, same two benchmarks, includes rollback. Key difference: their detector/recovery is *learned internal layers*; ours is an **explicit multi-agent delegation** (separate Reasoner specialist, interpretable YES/NO verdicts, framework-orchestrated). EWAM is our natural published baseline; evaluating on both RoboLab *and* LIBERO matches their protocol (reinforces open question 6). |
| [Rewind-IL](https://arxiv.org/html/2604.16683v1) (2026) | Online failure detection via **TIDE** (temporal inter-chunk discrepancy — divergence between the current chunk and the one predicted a step earlier, policy-internal, 0.2 ms) + **state respawning by commanding the robot back** to a checkpointed pose (not sim-state restore). Checkpoints picked semi-semantically (offline VLM labels + online feature similarity). RoboCasa + real bimanual; +13.3 pp success, 76.7% vs 18.3% under disturbance. | Validates the **`command` recovery mode** as the physically-honest default, and validates chunk-boundary checkpointing. Two ideas worth adopting: (a) an inter-chunk-discrepancy tier in `StuckMonitor` — at each boundary compare the tail of the previous chunk with the head of the new one (free at our cadence, no extra server calls); (b) *semantic* checkpoints (e.g. post-grasp boundaries as preferred rollback targets) over "most recent clean boundary". |
| [FAR](https://arxiv.org/pdf/2607.01111) (2026) | Failure-aware retry for test-time recovery + continual improvement. | Retry-budget framing (our `max_recoveries`). |
| [VLA-Corrector](https://arxiv.org/pdf/2607.01804) (2026) | Lightweight detect-and-correct on chunked VLAs: monitors latent visual dynamics, **truncates stale chunk actions** when drift persists, biases next inference toward recovery. | Names our exact blind spot — open-loop chunk execution — and supports flush-and-requery as the resume mechanism. |
| [SV-VLA](https://arxiv.org/pdf/2604.02965) (2026) | Speculative verification: verifier monitors during macro-chunk execution, discards remaining chunk on deviation, replans. | Concurrent-verifier pattern ≈ our background `SpecialistGate`. |
| [FailSafe](https://arxiv.org/html/2510.01642), [SAFE](https://arxiv.org/abs/2506.09937) (NeurIPS), [AHA](https://arxiv.org/abs/2410.00371) (ICLR'25) | VLM-based failure detection/reasoning for manipulation. | Establishes VLM-judge failure detection as a recognized approach; AHA-style failure *reasoning* is a v2 direction for the specialist (explain, not just flag). |

**Positioning**: our contribution is not "recovery exists" (EWAM/Rewind-IL
cover that) but (a) recovery as **explicit agent delegation** — a served
Reasoner specialist with interpretable verdicts, composed by an orchestration
framework, swappable per mission YAML — vs learned internal machinery; and
(b) doing it on RoboLab's in-domain DROID policy, where no public work has
put a specialist in the loop.

**Design updates adopted from the survey**: add the inter-chunk-discrepancy
signal as a third `StuckMonitor` tier (cheap, policy-internal, complements
kinematics + VLM); keep `command` as default recovery (Rewind-IL evidence);
consider semantic rollback targets (post-grasp) as a Phase 3 option; report
EWAM as baseline context in the write-up.

## 12. Code-level compatibility: VLA-Corrector and SV-VLA (repo audit, 2026-08-11)

Both papers released code. Audited for direct integration into Odyssey's
served-pilot architecture (the pilot is an HTTP/WebSocket server: frames in,
action chunks out — **no policy latents, no gradients on the wire**).

### [VLA-Corrector](https://github.com/ZJU-OmniAI/vla-corrector) — Apache-2.0, 66★, active (2026-07)

**What the repo is**: a LeRobot fork plus a self-contained
`src/siglip_dynamics/` module — the ~40M-param latent dynamics corrector
(MLP/DiT variants), latent-cache extraction (`extract.py`), training
(`train.py`), and an inference surface with promisingly clean names
(`inference/circuit_breaker.py`, `safety_module.py`, `guidance_injector.py`).
Evaluated on π0.5/SmolVLA/X-VLA over MetaWorld + **LIBERO** + real AgileX.

**Mechanism → portability, piece by piece**:

| Piece | What it needs | Fits served pilot? |
| --- | --- | --- |
| Event-triggered **truncation** (discard stale queued actions on drift) | a flush hook on the action queue | **Yes, trivially** — maps 1:1 onto a new `ChunkPilotAdapter.flush()` (we already flush on instruction change; this adds flush-on-trigger). Their ablation: truncation *alone* gives +11.65 of the +15.65 pp on MetaWorld — **most of the win is the black-box-compatible part**. |
| **LVM** (latent-space vision monitor): learned dynamics model predicting visual-feature evolution from features + executed actions; drift = predicted vs observed mismatch | a visual encoder + executed actions + trained ~40M corrector | **Adaptable**: they read the frozen VLA's own SigLIP features, which our wire doesn't expose — but client-side we *do* have every frame and every executed action, so an **independent SigLIP encoder** feeding the same corrector architecture is a faithful variant. Requires training the corrector on our rollout videos (capture already exists). The `siglip_dynamics` module is importable/vendorable (Apache-2.0). |
| **OGG** (online gradient guidance for the recovery replan) | gradients through the policy | **No** — impossible through a server. Skip, citing their own truncation-only ablation as the justification. |

### [SV-VLA](https://github.com/edsad122/SV-VLA) — MIT, 10★, OpenVLA fork

**What the repo is**: a prismatic/OpenVLA fork. The "lightweight verifier" is
a **temporal-fusion head grafted onto the same OpenVLA backbone** (extra tiny
ViT via timm, `vla_scripts/temporal_fusion_utils.py` loads
`OpenVLAForActionPrediction` with fusion modules), trained on RLDS tfrecords
(`train_from_pruning_tfrecord.py`). Replan triggers when the deviation between
the executing macro-chunk (64) and the verifier's high-frequency reference
actions (chunk 8) exceeds `controller_deviation_threshold`.

**Portability verdict**: **no direct code reuse.** The verifier is not
observation-only — it shares the OpenVLA backbone in-process and needs
training on that backbone; none of that transfers to a served Cosmos3 WAM.
The *pattern* (cheap high-frequency reference vs executing chunk, deviation
threshold) is already covered training-free by the Rewind-IL-style
inter-chunk-discrepancy tier adopted in §11. SV-VLA's headline is efficiency
(2.17× speed-up), which is orthogonal to our recovery goal. Cite as related
work only.

### Resulting detector ensemble (supersedes the two-tier sketch in §4.1)

`StuckMonitor` becomes a tiered, ablatable ensemble — each tier is a config
flag, giving the experiment its ablation arms for free:

1. **Kinematic** (free, every step): commanded motion vs actual displacement.
2. **Inter-chunk discrepancy** (free, per boundary): previous chunk tail vs
   new chunk head (Rewind-IL TIDE-style, training-free).
3. **Latent-dynamics drift** (optional, trained): VLA-Corrector-style LVM on
   an independent SigLIP encoder — *stretch goal / Phase 3+*, needs a
   corrector training run on our rollouts.
4. **VLM specialist retry trigger** (per boundary, async): Cosmos3-Nano
   Reasoner — the delegation headline, and the only tier that understands
   *semantics* (dropped object, wrong object, unreachable target).

Trigger action for tiers 1–3 = truncate + re-query (VLA-Corrector semantics,
via `ChunkPilotAdapter.flush()`); tier 4 additionally decides **rollback** to
the ledger's last-good snapshot. Licenses (Apache-2.0, MIT) are both
compatible with vendoring if we lift the corrector architecture later.

## 13. Iteration scoping decision (2026-08-11): VLA-first

**Decision**: iteration 1 runs the recovery mental model on the VLAs we
already operate end-to-end — **GR00T (primary) and π0.5 (secondary) on
LIBERO** — and defers the WAM/RoboLab arm (which requires the entry-script
fork of §3) to iteration 2 as a port. This resolves open questions 3 and 6.

### Why this is the better experiment, not just the safer one

Recovery research needs **recoverable failures**: a policy competent enough
that "return to the last good chunk" lands it back in known-good territory.
Ranking the substrates:

| Substrate | Pilot domain fit | Failure supply | Plumbing cost |
| --- | --- | --- | --- |
| GR00T × LIBERO | **in-domain** (10/10 on `libero_object` t0; t1 5/5) | **controlled** — the existing GR00T×LIBERO perturbation harness induces failures on demand (`docs/gr00t-libero-perturbations-summary.md`) | lowest: our own recipes, in-process auto-serve, known-good VMs |
| π0.5 × LIBERO | in-domain (fine-tune path exists, PR #90) | natural + perturbable | low: openpi server pre-started, recipe is ours |
| Cosmos3-droid × RoboLab | in-domain | natural only, uncontrolled | highest: RoboLab fork + Isaac docker |
| Cosmos3-droid × LIBERO | **OOD** (no LIBERO SFT) | failures are domain mismatch, few good chunks exist — wrong failure *type* | low |

In-domain policy + controlled perturbation = the cleanest causal read on
whether detection→rollback→re-query converts failures into successes. Even
the RoboLab fork wouldn't buy that control.

### What changes and what doesn't

**Unchanged**: the four pure modules (§4.1), the detector ensemble (§12), the
four arms + metrics (§5), the phasing (§7), and — importantly — the
**specialist**: Cosmos3-Nano Reasoner via vLLM stays the delegation headline
(it is pilot-independent; the reasoner-probe Phase 0 calibration applies
as-is, and the cosmos3 branch work on `openai_judge` / probe stays
load-bearing).

**Changed**:

- Pilots: `pilot: gr00t` (primary; in-domain + perturbation harness) and
  `pilot: pi05` (secondary; demonstrates pilot-agnosticism across an
  auto-served and a pre-started-server pilot).
- Harness: LIBERO recipes we own — no external code forked. `teleport`
  recovery = real MuJoCo state restore (§10), `command` = EE-delta
  P-controller.
- Stuck induction: the perturbation harness is now a *first-class* part of
  the protocol (perturb-at-step-k, measure recovery), not a fallback.
- **Migration note**: `gr00t_libero_eval.py` still inlines its own chunk loop
  and does not use `ChunkPilotAdapter`; moving it onto the adapter is a
  prerequisite (and a wanted consolidation anyway) so `flush()`/ledger hooks
  are shared across all pilots.

**Deferred to iteration 2+**: RoboLab entry-script fork + Cosmos3 WAM pilot
(the §3–§4 design stands, becomes a port per §10); LVM trained tier (§12
tier 3); semantic re-instruction on recovery (open question 1, v2);
FlowDAgger-style *steering* recovery (GR00T-only: instead of rolling back,
steer the frozen flow policy's noise — a third recovery mode candidate,
connects to the drugsort L1 work).

### The LVM tier across GR00T and π0.5 — pilot-agnostic by construction

Decision (2026-08-11): the VLA-Corrector-style tier targets **both pilots
with one design**, which is only possible because §12 already replaced the
paper's faithful variant (reading the VLA's own SigLIP features) with an
**independent client-side SigLIP encoder**. The faithful variant cannot be
pilot-agnostic here: π0.5 sits behind openpi's WebsocketPolicyServer (no
internals on the wire) and GR00T auto-serves in-process (internals reachable
but pilot-coupled). The external variant needs only what both pilots already
expose identically in our eval loops: the frames and the decoded 7-D env
actions passed to `env.step()`.

Consequences:

- **One corrector can serve both pilots**: it models *environment* dynamics
  conditioned on env-level actions — nothing about the policy enters the
  model. Same env, same action space ⇒ shareable weights; only the trigger
  thresholds are calibrated per pilot (chunk sizes differ), conformal-style
  on successful-rollout discrepancy distributions (the Rewind-IL recipe:
  ~99.9th percentile).
- **Finetuned checkpoints are an asset, not a problem**: the corrector never
  touches policy weights, so base-vs-finetuned is irrelevant to
  compatibility — and finetuned checkpoints are exactly what makes training
  data cheap. GR00T-LIBERO at 10/10 with ~30 s episodes can mass-produce
  successful rollouts today. π0.5's corrector quality is gated on its
  finetune (PR #90) producing enough successful rollouts — GPU smoke still
  pending there.
- **New instrumentation requirement**: current capture saves MP4s but *not
  actions*. The recipes must log per-step (frame, action) pairs (e.g. an
  `.npz` next to each rollout video) — this is the corrector's training
  corpus, and it costs nothing to start collecting from the first Phase 2
  runs even though the LVM tier itself is Phase 3+.
- OGG stays excluded for both pilots (gradients through a server — see §12).

### Branch note

Iteration 1 lives on **`feat/vla-recovery-closedloop`, cut from `develop`**
(the cosmos3-integration base was dropped once the WAM/RoboLab arm moved to
iteration 2 — nothing cosmos3-specific remains in scope). Two pieces were
ported over from `cosmos3-integration` because they had not reached develop
yet: `src/odyssey/runners/agents/openai_judge.py` and
`tests/unit/test_openai_judge.py` (self-contained; suite green on this base).
Still on the cosmos3 branch, to port when Phase 0 starts: the reasoner-probe
example (`examples/cosmos3-reasoner-probe/`) — it is pilot-agnostic (probes
rollout MP4s through the judge) and would be renamed to a neutral
`examples/reasoner-probe/` here.
