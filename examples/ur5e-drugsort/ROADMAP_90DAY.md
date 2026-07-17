<!-- Produced by a 5-stance adversarially-judged design panel (simulator /
il-rl-ladder / few-shot / multi-agent contracts / pragmatist), synthesized
2026-07-17. Companion to PLAN_MULTIAGENT.md; supersedes neither the role law
nor the running Stage-C DAgger round. -->

# Unified 90-Day Roadmap — Playground Simulator, IL+RL Brain, Few-Shot Adaptation

Synthesis note: the backbone is the pragmatist's frozen-base, information-per-browser-episode sequencing; the simulator layer keeps StateVault/RandFX/Arena unconditionally and makes the capture rebuild spike-gated; the ladder adopts il-rl-ladder's dominance gates and grip floor; brain contracts adopt multiagent-contracts' M1/M2 with the feasibility oracle demoted to a validated-first hypothesis; the few-shot layer keeps only the honest "few human inputs" claim.

## Ground rules (measured, non-negotiable)
1. **Browser-rendered frames are the only valid training/scoring pixels** (39% MuJoCo render gap killed perception AND policy). Every capture-path change gates on pixel parity (`env_fidelity.js`, SSIM ≥0.999 / MSE <1e-6). SwiftShader banned unless it passes the same gate.
2. **No PPO/GRPO through GR00T's flow head** (no tractable log-prob). All "RL" = noise-space search + reward-filtered imitation; reward comes from sim-GT state and Observer heads — never from GR00T likelihood.
3. **GR00T is sole actuator** (PILOT). Specialist (Cosmos Reason, 1.5 s/frame, 19% zero-shot acc) is episodic and advisory only — never in the control loop, never a load-bearing verifier (19% acc cannot audit anything).
4. **Frame capture (4–7 min/ep) is THE scarce currency; GPU is cheap.** Default scoring path is dynamics-only WASM replay (2 s/ep, bit-faithful); pixels only for perception training and final gates. Live closed-loop GR00T (85 s+/ep) is eval-only.
5. **Steering/inversion are checkpoint-specific.** Every SFT triggers an invalidation tax: re-inversion (~1.4 GPU-h full-scale, ~0.1–0.2 GPU-h per 50-ep task), steering retrain (35 s), and loss of browser-bought DAgger correctives. All artifacts carry a checkpoint hash; mismatch forces HOLD until refit.

## Promotion protocol (fixes the n=15 noise problem — applies to every gate below)
- Eval suite = **45 episodes (3×15 fixed-seed blocks)**, Wilson 95% CI reported. Promote only when the CI lower bound clears the gate, or the candidate dominates the incumbent A/B with non-overlapping CIs.
- **Dominance gate everywhere**: motion-stratified offline consistency AND browser success. Never median-pad-only (the v0.2 trap: 4.4 cm median, 0/15).
- **Grip-authority floor**: promoted checkpoints close the gripper in ≥60% of grip windows (measured ladder: oracle 100% / random-16 96% / steered 8%).

## Week-1 spikes (cheap; go/no-go for everything downstream)
- **S1 Capture profiling** (~2 eng-days): split the 4–7 min/ep into render vs readPixels vs compositing vs rAF throttle. GO for the capture rebuild only if plumbing ≥70% of cost.
- **S2 Feasibility-signal ROC** (~0.5 GPU-h, existing logs): does inversion recon-MSE (measured ~1e-4 everywhere) actually discriminate realized vs failed subgoals, motion-stratified? AUROC ≥0.7 → build the oracle gate; else drop it and use Observer-predicate promotion only.
- **S3 H100 headless GPU-WebGL check** (1 eng-day): `--use-gl=egl` pixel parity vs desktop render. If it fails and SwiftShader fails parity, the fleet stays on current gen_parallel.sh shards.
- **S4 Freeze the 45-ep protocol + Arena regression fixtures** (oracle 100 / random-16 96 / steered 8 / stock pad 8.1 cm).
- **Continuity**: finish the running noise-space DAgger round on the current frozen checkpoint before any SFT (it is already paid for and SFT would invalidate it).

## Layer 1 — SIMULATOR
1. **StateVault** (wks 1–3, ~1 eng-week, ~0 GPU; unconditional — highest-value grounded component). `get_state/set_state/reset(seed)/branch` over WASM mjData including act/warmstart/contact; heap views re-acquired after every write (known detach bug). Gates: round-trip qpos/qvel MSE=0; **mid-episode** restore replay 5/5; branch ≥16 forks. Substrate for render-free best-of-N and <10-min checkpoint eval.
2. **RandFX velocity-profile DR** (wks 2–4; decoupled from the throughput rebuild; runs on existing shards). Regenerate 300–600 demos with per-episode velocity/standoff/dwell/waypoint jitter (dexmimicgen diversity idea, native implementation). Run as a **hypothesis test** with an honest gate: retrained base ≥3/15-equivalent (CI excludes 0) OR exterior-cam attention mass ≥2× baseline. Cost: 20–70 shard-hours capture; SFT itself is a Phase-2 decision, not automatic.
3. **CaptureCore/CaptureFleet** (conditional on S1/S3; ~2 eng-weeks). Offscreen-canvas WebGL2 async readback, chunk-cadence capture. Gates: <45 s/ep single shard (honest 8× from the 6-min midpoint), pixel parity, replay 5/5; fleet: 500 eps/h sustained, zero crashes over 1000 per-episode sessions, PID hygiene preserved.
4. **CellForge-lite** (wks 7–10, ~1 eng-week): task.yaml → auto-emitted Observer GT labels + compiled success predicates + parametrized DLS-IK expert. Defer the "zero bespoke code for novel objects" claim; gate = second cell variant stood up with <200 LOC bespoke, expert ≥95%, retrained heads <1 cm grasp.
5. **Arena** (wks 8–12, ~1 eng-week): batched eval service wrapping StateVault; must reproduce the S4 fixtures within noise; <10 min/checkpoint replacing the 85 s+/ep SSH loop.

## Layer 2 — LEARNING LADDER (frozen base first; SFT gated, never default)
- **R1 (wks 1–2)**: fold finished DAgger correctives → steering v0.3 (grip-inclusive weights). Gate: freeze rate <20%, pad < 8.1 cm, motion-stratified offline gate pass.
- **R2 (wks 2–4) Render-free best-of-16** (~4 eng-days, cents of GPU/ep): sample seeds around the steering mean; decode batched on H100; score **without pixels** — gripper command read directly off the decoded 40×132 chunk, ee-target distance via FK, forward dynamics via StateVault branch (2 s/ep). Observer browser frames only at the final 45-ep gate. Feasibility pruning only if S2 passed. Gates: executed grip closure ≥40% (from 8%); premature-GRASP hover <20% (from 76%); success CI excludes 0. **KILL: grip <40% → frozen-base bet failing → pull the SFT decision forward.**
- **R3 (wks 5–8) Reward-filtered self-imitation**: winners (sim-GT success + grip closure, bit-faithful replay verified) become steering targets; retrain 35 s/round; 3 rounds × 30–50 browser eps ≈ 6–18 shard-hours. Reward-hacking audit = human spot-check + sim-GT cross-check (not Cosmos). Gate: ≥40% success CI-backed, grip ≥60% floor held.
- **R4 (wks 9–12, stretch)**: CEM over a **low-dim residual (≤32-d) on the steering output** — never the raw 5280-d noise. Population 64/iter, dynamics-scored, elites-only browser confirmation. Gate: dominates R3 in A/B.
- **SFT decision gate (wk 5)**: run one FSM-DR SFT iff (a) RandFX hypothesis test showed lift, or (b) the R2 kill fired. Budget the invalidation tax explicitly; re-derive (not carry over) all steering thresholds on the new checkpoint.
- **Distillation (≥wk 11, stretch)**: SFT-distill winning chunks into the base; ship to Playground only if it dominates stock unassisted.

## Layer 3 — BRAIN CONTRACTS
- **C1 (wks 3–5, ~1 eng-week)**: versioned Pydantic contracts (Percept / Subgoal / action-chunk / EpisodeReward / Diagnosis), checkpoint-hash tagging, enforced invariants: single actuator; Specialist air-gap; browser-only perception scoring; deadline→escalation (no silent retry); HOLD on checkpoint mismatch. Gate: current vial task re-expressed as a TaskSpec and reproduces the v0.3 baseline through the new plumbing.
- **C2 (wks 5–8, ~1.5 eng-weeks) Planner replaces the phase FSM**: promotion driven by Observer predicates (100% phase/success classifier) — a subgoal promotes only when its predecessor's predicate is *realized*, never on a clock. Feasibility oracle included only if S2 validated it. Gate: hover <15% measured with the identical per-tick definition pre/post refactor; first robust demo = **≥70% success CI-backed, end-to-end in the Playground** (kill: <8/15-equivalent after two steering iterations → SFT path).
- **C3 (wks 8–12)**: Specialist episodic loop — post-episode diagnosis proposing config/curriculum edits (e.g. new DR axes), adopted only after an Observer-gated A/B. Advisory forever.

## Layer 4 — FEW-SHOT (the claim we can actually make)
**Honest definition**: few-shot = **few human inputs** — 1 NL spec + ≤10 human demos anchoring goal poses and success predicates. Compute, scripted-expert rollouts, and browser capture are NOT shots and are budgeted separately. Per-task budget (correctly scaled): expert ~1 eng-day; 50–100 eps capture = 4–12 shard-hours; inversion ~0.1–0.2 GPU-h; steering 35 s; heads from sim GT ~free.
- **F1 (wks 9–11) Second task, 5-day time-box, base frozen**: expert → capture → inversion → steering → selector. Gate: ≥60% CI-backed within 5 working days, no SFT. **The time-box is the deliverable.** Kill: needs SFT → report the broken <1-week boundary explicitly.
- **F2 (wks 11–12) Held-out variation**: ≤10 new human demos + steering retrain + short DAgger round. Gate: ≥60% CI-backed, no base retrain. **SFT-escalation fraction is tracked and reported (target ≤1/3 of tasks)** — an ungated escape hatch does not count as few-shot success.
- **Not claimed**: novel-object arbitrary tasks, 1-day onboarding, 12/15 few-shot, or steering-weight transfer (only the spec layer — subgoal graph, predicates, Observer targets, expert recipe — transfers; steering is re-derived per checkpoint).

## 90-day backbone
- **Wk 1**: S1–S4 spikes; finish DAgger round; StateVault start.
- **Wks 2–4**: R1–R2 grip authority; RandFX regen; StateVault done; CaptureCore if S1 GO.
- **Wk 5**: SFT decision gate; C1 contracts.
- **Wks 5–8**: C2 planner; R3 self-imitation; CaptureFleet if GO; robust-demo gate.
- **Wks 8–12**: Arena; CellForge-lite; F1 second task; F2 variation; R4 + distillation (stretch); C3.

## Kill criteria (summary)
- S1 plumbing <70% → no capture rebuild; live on shards + StateVault economics.
- S2 AUROC <0.7 → no feasibility oracle; Observer-predicate promotion only.
- R2 grip <40% → schedule SFT with full invalidation tax; drop R4/F2 stretch and re-cut the timeline.
- F1 blows the 5-day box → narrow the few-shot claim to "steering-first with reported escalation rate"; publish the boundary, don't hide it.

## DO NOT BUILD YET
PPO/GRPO through the flow head (impossible); Cosmos in the control loop or as a load-bearing auditor; MuJoCo-rendered training data; Q(s,w) or bandits over 5280-d noise; task-conditioned steering nets before per-task steering works on ≥2 tasks; CL2A fleet wiring before F1 ships; dexmimicgen code port; learned world models; "zero-code novel-object" asset synthesis; a third task or multi-cell generalization before the F1 pipeline is proven.
