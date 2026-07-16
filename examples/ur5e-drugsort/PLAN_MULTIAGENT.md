# Multi-Agent Plan — UR5e Vial Pick-and-Place (Drug Sorting, Lovell AI Robot Playground)

Status: adopted 2026-07-16. Produced by a 5-way adversarially-judged design panel
(perception-led servoing / residual RL / specialist-orchestrated / data-alignment /
systems-pragmatist), synthesized on the systems-pragmatist backbone with grafts from
the others. Gate 0 (env fidelity) resolved **Outcome A** before adoption — see
`browser_capture/env_fidelity_out/env_fidelity.json`.

## Backbone: prune before you pay

Order every intervention by information-gained-per-GPU-hour; gate the expensive
levers (SFT retrains, residual RL) behind ~zero-cost probes. Gate 0 is already
resolved: the eval action path is FAITHFUL (scripted expert 5/5 native vs 5/5
through the eval's zero-order-hold path), so the in-browser 0/15 is a genuine
policy-side failure. The decisive cheap experiment therefore moves up one layer to
the observation/inference path. Residual RL — the committed heavy lever — is reached
last, on a base first lifted to a near-miss regime: a bounded residual polishes a
near-miss; it cannot resurrect a home-collapsed arm.

Two facts from Gate 0 reshape everything downstream:

- **The env is ~40x faster than assumed.** 23,700 physics steps ran in ~2.1 s
  in-browser; the 85 s/episode in GR00T evals is almost entirely inference +
  SSH-tunnel latency (~0.75 s/query). RL sample budgets are inference-bound, not
  env-bound.
- **The render-gap story was never statistically sound.** Headless 2/20 (10%) vs
  browser 0/15: a true-10% policy yields 0/15 ~21% of the time. One weak policy
  everywhere — not a good policy destroyed by the browser. Stop buying
  domain-transfer machinery; buy grasp competence.

## Architecture — three models, three timescales, one actuator authority

- **PILOT — GR00T N1.7-3B (control-rate, 20 Hz).** Sole actuator authority.
  8-step absolute-joint-target + grip chunks (~113 queries per 900-tick episode).
  Observation: 6 joints + **measured** grip + Observer grasp-target xyz (10-dim
  state) + exterior/wrist browser RGB. Every correction (residual delta, retry
  offset) is composed INTO this one action stream before application — never a
  second actuator. Flow-matching DiT head ⇒ no tractable log-prob ⇒ no policy
  gradients through it; all learning is SFT or gradient-free.
- **OBSERVER — RT-DETR + DINOv2 (control-rate injection + episodic reward;
  perception service, not a registered agent).** Control-rate: injects grasp-target
  xyz (0.617 cm median on Three.js frames) into the Pilot's context every query;
  the distilled phase/success classifier (100% held-out) emits {phase,
  success_prob} per tick. Episodic: dense + event reward for the residual
  optimizer, demo success verifier, auto-retry gate. The trustworthy
  deployment-domain anchor.
- **SPECIALIST — Cosmos Reason 2 (mission-level, async, off the control loop).**
  Pre-episode scene sanity check; post-episode failure diagnosis that ROUTES
  (a) targeted DAgger collection during training and (b) the best-of-N retry
  decision in the shipped demo — emitted only as an edit to the Pilot's next
  instruction/target-offset (a bounded modification of the Pilot channel).
  Periodic distillation teacher refreshing the fast classifier. Never actuates,
  never in-loop (19% zero-shot accuracy, 1.5 s/frame).

Data flow: Observer → Pilot context + reward + verify/retry gate. Pilot → sim →
browser frames → Observer + Specialist. Specialist → DAgger routing (train) and
next-instruction/target-offset (ship).

## Phases

**Phase 0 — Gate 0: env fidelity. DONE (Outcome A).**
Expert 5/5 through the eval action path; harness faithful; 0/15 is policy-side.

**Phase 1 — Inference-input integrity probe. (0 GPU-h, ~0.5 browser-h)**
`probe_inference_path.py`: per tick of a live browser eval, dump the exact 10-dim
state + image hashes GR00T receives and the raw action chunk returned. Check:
(a) obs variance across ticks and across conditioned-vs-dummy runs; (b) action
variance with obs; (c) different-obs→same-action determinism; (d) returned action
chunk **magnitude/range vs the scripted-expert action distribution** — Gate 0
replayed *recorded* actions, so it could not catch a de-normalization/decoding bug
in GR00T's serving path. Note: gripMax varied between the conditioned and ablation
runs, so outputs are not fully constant — the probe targets the ARM dims.
*Exit fork:* wiring/scale bug found → fix + re-eval (may close most of the gap for
free). Obs fine but arm actions degenerate → mode collapse → Phase 3 mandatory.
*Kill:* obs varies AND actions vary at correct scale AND still fails → capability
ceiling; skip to Phase 3 knowing cheap fixes are dead.

**Phase 1b — Grip state-semantics fix. (0 GPU-h, ~0.5 browser-h)**
Training states record MEASURED grip; eval feeds COMMANDED. Patch
`eval_browser_groot.js` (+ conditioner) to feed measured grip. Re-run headless then
15 browser episodes. *Exit:* headless must not regress below 2/20; report browser
delta. *Kill/revert:* headless regresses (the skew was compensating).

**Phase 2 — Same-checkpoint headless↔browser side-by-side. (~2 GPU-h, ~2 browser-h)**
With 1/1b applied, run ONE checkpoint in both renderers with byte-verified
identical state and matched initial poses; diff per-tick actions. Quantifies what
remains of the visual gap on the same policy. *Kill:* unattributable divergence →
MuJoCo-WASM contact/solver parity audit before further spend.

**Phase 3 — Lift the base to a near-miss regime. (~24–30 GPU-h serial, ~4 browser-h)**
Gated in only if Phase 1 = mode-collapse or Phase 2 = capability gap. On the fully
corrected pipeline (measured grip, verified-wired Observer xyz, browser frames):
(i) clean `launch_finetune` re-run, 2–3 seeds; (ii) **phase-labeled segmented
SFT** — relabel all demos with the FSM's exact phase boundaries into per-phase
instructions, plus **reset-pose regrasp demos** so recovery is a learned GR00T
subgoal; (iii) instruction-attention ablation (dummy vs real instruction must
change behavior) so conditioning-blindness cannot recur silently.
*Exit:* ≥30% browser success (best seed, Observer-classifier-scored) + ablation
passes. *Kill:* best-of-3 ≤10% browser → BC has plateaued; proceed to Phase 4 only
if the base is a near-miss (arm reaches the vial neighborhood), else Phase 5.

**Phase 4 — Gradient-free residual RL (the committed heavy lever). (~6 GPU-h, ~3 h wall browser-parallel)**
CEM optimizes a small **state-augmented residual head** — inputs: a_base from
GR00T + low-dim physical state (qpos, measured grip, TCP pose, Observer target,
target−TCP delta); render-gap-immune by construction (no pixels). Output: bounded
±Δ in **TCP/end-effector error frame, executed through the expert's existing
damped-least-squares Jacobian**, **phase-gated** by the classifier (zero authority
in transport), composed into GR00T's action before application. Reward:
potential-based Observer-distance shaping + sparse classifier events
(grasp/lift/seat/success). Popsize ~16, ~8 iters, ~5 evals per candidate,
Chrome-parallel workers.
*Mandatory pre-tuning gates:* residual-causality (nonzero residual must change
executed TCP pose) and bounds→0 no-op sanity (must reproduce the base exactly).
*Exit:* ≥2x success lift over the base AND the **attribution/dominance gate**:
zeroing/shuffling a_base must sharply drop success — GR00T must own the majority
of task-relevant motion. This is the authenticity ship-block: if the residual is
the de-facto pilot, we do NOT ship it. *Kill:* reward flat at iter 4, or dominance
gate fails → ship best-of-N without residual.

**Phase 5 — Ship the authentic demo. (~2 GPU-h, ~2 browser-h)**
`demo_bestof_n.js`: curated demo-friendly tray pose; GR00T pilots live (+ residual
only if it passed dominance). Observer classifier verifies each rollout and
auto-retries on failure; Specialist diagnoses each failure and routes the retry's
next instruction/target-offset; `authenticity_logger.py` records per-tick action
provenance + GR00T query count + residual-attribution. *Exit:* a live
browser episode — lift >2 cm + seated — Observer-verified, GR00T-driven per the
log, within one or two attempts. *Kill:* no GR00T-driven success in 30 curated
attempts → the base ceiling is below demo-viability; report honestly.

## Compute discipline

One H100 80 GB; serving and finetuning cannot overlap → GPU work is serial.
Worst case ~40–55 H100-h (hard ceiling 55). Browser ~24 h, parallelized on the
proven headless-Chrome workers to ~6–8 h wall. Phases 1/1b/2 cost ~0 GPU and can
delete Phase 3's 24–30 h outright if Phase 1 finds a wire/decode bug.
TDD; Conventional Commits; DCO sign-off; no Co-Authored-By.

## Single biggest risk

If Phase 1 shows obs varies, actions vary at correct scale, and it still fails,
then GR00T's genuine ceiling from 151–303 scripted demos is ~its headless 10%, and
there may be no near-miss base for the residual to rescue — a bounded corrector
cannot move a home-collapsed arm ~0.4 m to a vial. Segmented SFT is the one cheap
shot at converting that ceiling into a near-miss; if it fails, the authentic
deliverable narrows to best-of-N-with-retry on a curated pose. Everything
downstream multiplies or polishes; nothing resurrects. Phase 1 is the whole
ballgame — and it costs ~0 GPU and runs first.
