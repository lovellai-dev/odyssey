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

## Status updates

**2026-07-16 — Phase 1 executed (commit 753c55a): verdict CAPABILITY_CEILING,
with the mechanism identified as input-attention collapse.** All plumbing clean:
obs alive (hash uniqueness 1.0, target tracks scene), action scale 100% inside
the expert range, bridge≡native (0.0008 rad), tracking OK. The sensitivity
battery found the policy is **wrist-camera-only**: blacking the wrist view moves
actions 1.47 rad (SNR 9.5) while blacking the exterior view (SNR 0.6), shifting
proprio (SNR 0.4) and shifting the injected grasp target ±5 cm (SNR 0.4–1.1) all
do nothing — mechanistic confirmation that the Step-2 conditioning was
learned-ignored, not miswired. Behavior: approach to within 6.1–11.6 cm of the
vial, blind grip fire, miss. Consequences: **near-miss base confirmed** (Phase-4
precondition met); **Phase 1b closed** (grip skew 0.06 mean, below flag);
**Phase 2 devalued** (exterior-renderer differences cannot drive a policy that
ignores the exterior view). The probe battery is retained as a mandatory
per-checkpoint attention test for every future retrain.

**2026-07-16 — Phase 4 re-based on FlowDAgger (Microsoft, latent-space DAgger
for flow-matching policies; deep-read + port assessment complete).** A small
steering network predicts the flow sampler's INITIAL NOISE, trained by BC on
expert corrections inverted through the sampling ODE; base weights frozen.
Fit: steering inputs are OUR choice (Observer grasp-target xyz + proprio +
phase one-hot) — bypasses the collapsed attention, render-gap-immune, no
log-prob needed, and strictly stronger on authenticity than an action-space
residual (GR00T's frozen flow head remains the sole emitter of actuator
values). Verified Isaac-GR00T seams: `get_action_with_features` already takes
`options` with an RTC noise-override precedent (~4-line `init_noise` patch),
`Gr00tPolicy._get_action` needs 2 lines, ZMQ server kwarg-splats client data
(zero server changes). Port cautions: GR00T's time convention is the reverse of
pi0.5's; N=4 Euler steps with bucketized timesteps {0,250,500,750}; recon MSE
scored on dims [:7] only (pad dims untrained). CEM survives as **noise-space
CEM** (authority probe + optional deploy-time fallback). Action-space residual
retired.

**Gate before the port (Step 0/1, ~2 days, no upstream patch):**
`probe_flow_inversion_groot.py` on the H100 — (a) perstep_fp inversion
round-trip on expert chunks (pass: recon MSE < 1e-3 on dims [:7], p99|w*| < ~3);
(b) steering-authority probe at recorded miss states: invert the IK-FSM
corrective chunk (on/off-manifold verdict) + decoded end-effector spread over
64 random noise seeds via FK (must cover the 6–12 cm correction). Both pass →
full port (~2–3 weeks). Either fails → exactly ONE Phase-3 DAgger finetune
round to put corrective descent on-manifold, then port steering on the new
frozen checkpoint.

**2026-07-16 — Step 0/1 probe executed on the H100 (checkpoint-12000,
`scripts/probe_flow_inversion_groot.py`): verdict PORT_GO — both gates PASS.**
The `perstep_fp` inverter (per-step fixed point over the frozen 4-step forward-Euler
flow sampler; buckets {0,250,500,750}; backbone bf16 cached once, action-head chain
run fp32; recon scored on the real dims `[:16,:7]` of the padded `40×132` chunk) is
TDD'd against synthetic constant/contractive-affine fields
(`tests/unit/test_flow_inversion_math.py`, 5/5 green in the GR00T venv).

- **STEP 0 (inversion round-trip) — PASS.** 16 expert chunks sampled across
  episodes and phases (reach/descent/grasp/transport). Reconstruction MSE on real
  dims: **mean 9.1e-5, p95 2.2e-4, worst 2.3e-4** (gate < 1e-3, ~10× margin).
  Recovered noise `|w*|` (pooled real dims): mean 0.88, **p99 2.79** (gate < ~3),
  max 4.67. `fp_per_step=16`; per-step FP residuals `[2e-8, 2e-8, 4e-5, 1.8e-3]`
  (only the t=0 step is loose, well inside tolerance); Adam fallback never needed.
  → The frozen sampler is invertible and expert chunks sit on its manifold at
  in-distribution noise. Latent-noise steering is mechanically viable.
- **STEP 1 (steering authority at near-grasp states) — PASS.** The obscond
  checkpoint is visually OUT-OF-DISTRIBUTION in the MuJoCo aseptipack render (a
  fresh sim rollout base-joint-runs-away; a discriminator confirmed the served
  policy reproduces recorded expert actions to <0.1 rad on real dataset frames but
  diverges on sim frames), so the 18 miss states were harvested from the
  IN-DISTRIBUTION dataset in the 4–15 cm near-grasp band (grip open, mean needed
  correction 11.7 cm); MuJoCo is used only for gr_pinch FK. (a) Inverting the
  recorded expert corrective chunk: **recon MSE mean 1.1e-4, 100% below 1e-3** —
  every corrective chunk is on-manifold. (b) 64 random-noise seeds decoded per
  state give an end-effector spread **mean 10.2 cm** (median 4.9; per-state
  1.7–24.5 cm — the envelope scales with the needed correction: 18–24 cm at the
  14 cm states, 2–5 cm at the 6.6 cm states) and the corrective EE point lies
  inside the reachable noise cloud (within 2 cm) at **100% of states**.
  → Latent noise has real actuator authority over the EE across the 6–12 cm
  correction band, and the corrective action is reachable from noise.

Decision: STEP0 pass + STEP1 pass → **PORT_GO** — proceed to the full FlowDAgger
port; no Phase-3 finetune-first detour required. Honest caveat: deploy inference is
bf16 (the probe ran the action-head chain in fp32 for clean inversion), and Step 1
authority was measured on in-distribution dataset states because the checkpoint is
render-OOD in the MuJoCo cell — the steering net must key on the Observer
grasp-target + proprio (not sim pixels), consistent with the Phase-1
attention-collapse finding. Result JSON: `flowdagger_probe_result.json`.

**2026-07-16 (evening) — steering v0→v0.2 iteration arc + Stage C launched.**
Stage-B v0 A/B: steered 0/15 (13 home-freezes; 2 sub-cm approaches) — root cause
68%-static expert chunks → time-free net collapsed to "stay" (absorbing fixed
point). v0.1 (t_norm + arm-motion weights): freezes 0/15, all approaches,
median pad 7.6 cm — but grip ≤0.04 everywhere; grip-authority probe: oracle
noise closes at 100% of windows, random seeds 96%, v0.1-steered 8% → v0.1's
ARM-only motion weights had floored the (arm-static) closing windows. v0.2
(grip-inclusive motion + 3D dwell promotion): **STEERED_PROGRESS_NO_SUCCESS —
0/15 but median pad 4.4 cm (stock 8.1), zero freezes, first steered full grip
closures (2, mistimed at 4–6 cm)**; 76% of ticks hover in promoted-GRASP at
~4 cm — a state band only the steered policy visits (the expert descends
through it). Textbook DAgger condition → **Stage C (noise-space DAgger)
launched**: steered rollouts (fresh seed-4242 plans) → IK-expert relabel →
invert → aggregate (dagger-boosted) → retrain → gate → A/B, mini-round smoke
first.

**2026-07-17 — FIRST LIFT (best-of-N R2, commit fd17e2e..5dab103):** the
selection-first pivot (16 candidates around the v0.2 steering mean, CBF filter,
CLF rank, HOLD fallback) produced the program's first grasp-and-lift: ep001
lifted 11.4 cm (expert height) after a 0.2 cm approach. Full eval 0/15 success
but 1 lift, **13/15 full grip closures** (v0.2: 2 mistimed), all pads <=4.3 cm,
holds ~3%. Three telemetry-driven CBF calibration cycles preceded it (caps ->
1.5x expert p99; executed-horizon safety; decay only on state-distance
barriers; workspace box MEASURED from the expert pinch envelope — the guessed
box bisected the transport arc). New dominant failure: contact-without-capture
(6 vial strikes during closing). Next: capture refinement — graded near-vial
speed barrier, precision-weighted GRASP CLF, adaptive K, stateless-relabel
DAgger. Also: v3 stuckness feature falsified as the corrective-fit fix
(dagger-source val MSE floor ~0.09 across 6 configs -> stateful-relabeler
target noise is the standing hypothesis).

**2026-07-17 (later) — POWERED VERDICT (45 eps, 3 fixed-seed blocks, commit
769f380..b0d8594): closure 40/45 = 88.9% CI [76.5, 95.2]; lift 0/45, CI upper
bound 7.9%; pad median 4.4 cm.** The selection stack solves approach+closure
robustly across the task distribution; capture is a sub-8% tail event at
current candidate quality. Eight steering configs pinned at the ~0.08-0.09
corrective-fit floor -> surviving diagnosis: NOISE TARGETS ARE IMAGE-
CONDITIONED (the inverted w* depends on frames the 15-d steering input cannot
see; base targets fit at 0.008 because the image-conditioned field already
flows toward expert actions there). Shaping variants (hard/graded centering,
graded contact cap, CEM-at-grasp) were statistically indistinguishable at
n=15 and are now config pins. NEXT: image-conditioned steering head v2
(pooled backbone features + low-dim -> noise; features already computed
in-process by the best-of-N service; ~1-2 GPU-h re-encoding pass), then the
same 3-block powered protocol; escalation if the lift CI stays on zero = the
roadmap's SFT decision gate.

**2026-07-17 (v4 image-conditioned head) — FIRST CI-BACKED SUCCESSES (45-ep
powered, 3 blocks, commit 7da91eb..8c2cef9):** success 2/45 = 4.4% CI
[1.2, 14.8]; lift 6/45 = 13.3% CI [6.3, 26.2]; pad median 2.1 cm (was 4.4).
Both CIs clear zero vs the v0.2-mean baseline (0/45 lift, CI upper 7.9%) —
the FIRST full pick-and-place (grasp+transport+seat) in the program, and it
generalizes across all 3 pose blocks incl. untuned 8888/9999. The v4 head
(pooled backbone features + low-15 -> noise) broke the corrective-fit floor
by intervention (dagger-source MSE 0.087 -> 0.041, base EE 0.86 cm, first
aggregated net to PASS the base gate). Server-head deploy: sidecar ships
steer_low15, best-of-N service computes mean = head(low15 + pool_backbone)
in-process (identical pooling to train); response carries steer_head:v4.
Honest gaps: closures dropped to 47%, holds rose to 32% (1636/5086) — CBF
envelope now over-conservative for the sharper v4 candidates. NEXT: (1)
re-tune CBF caps to the v4 distribution (cheap, holds are leaving good
candidates on the table); (2) DAgger round 2 on v4-visited states (lifts now
happen -> first POST-GRASP transport/place states to harvest); (3)
transport/place refinement (2 successes already exercise it). Then the demo
HUD (Observer marker, safety-certificate ticker, Specialist panel).

**2026-07-17 (CBF re-tune, commit f779d3d) — HOLDS+CLOSURES FIXED, SUCCESS
UNCHANGED (honest).** The v4 run's 32% HOLD rate traced to two over-firing
barriers (vial_protect 30k: flat 0.022 cap throttled the whole 10cm zone;
table_clear:decay 16k: criminalized the legitimate fast descent). Measured
fix: vial_protect TAPERS from the global cap (0.16) at the edge to a 0.02
contact floor (calibrated to the v4 near-vial profile — v4 mean contact max
0.015); table_clear dropped from DECAY_BARRIERS. Re-tuned 45-ep powered:
holds 32%->1.1%, closures 47%->88.9% [76.5,95.2] — both decisively fixed. BUT
success 4.4% [1.2,14.8] IDENTICAL to pre-tune, lift 8.9% [3.5,20.7] statistically
unchanged (overlapping CIs). LESSON: the 32% holds were a real inefficiency but
NOT the success bottleneck — my "unblock->more success" hypothesis FALSIFIED.
The funnel is now 89% closure -> 9% lift -> 4% seat: the loss is CAPTURE
SECURITY (close != secure) and lift->seat. This routes the next work with
evidence: DAgger round 2 on v4-VISITED states (running — relabels v4 grasp-slip
states in the image-conditioned space + first post-grasp data) attacks capture
security; transport/place refine attacks lift->seat. The tapered CBF is the
config of record going forward (removes the hold/closure confound).

## Single biggest risk

If Phase 1 shows obs varies, actions vary at correct scale, and it still fails,
then GR00T's genuine ceiling from 151–303 scripted demos is ~its headless 10%, and
there may be no near-miss base for the residual to rescue — a bounded corrector
cannot move a home-collapsed arm ~0.4 m to a vial. Segmented SFT is the one cheap
shot at converting that ceiling into a near-miss; if it fails, the authentic
deliverable narrows to best-of-N-with-retry on a curated pose. Everything
downstream multiplies or polishes; nothing resurrects. Phase 1 is the whole
ballgame — and it costs ~0 GPU and runs first.
