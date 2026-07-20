# Autonomous Multi-Agent Loop — UR5e Manipulation

A self-improving research loop that closes the Observer / Specialist / Pilot
brain around the task and drives it to competence autonomously. Designed to be
**resumable** (survives session/login death — every prior manual driver died
with its session), **generalizable** (task-parameterized, no vial-specific
logic in the loop), and **explainable** (every decision is a human-readable
record with the evidence that drove it).

## The three agents, as loop roles

- **OBSERVER (perception)** — RT-DETR/DINOv2 grasp-target head (0.6 cm on browser
  frames) + the distilled phase/success classifier (100% held-out). In the loop
  it is the *measurement instrument*: it turns each episode into a **funnel**
  — reached → centered → closed → lifted → transported → seated — the abstraction
  every decision is made over. Perception is what makes the loop task-agnostic:
  the funnel is the same for any pick-place-shaped task.
- **SPECIALIST (diagnosis + planner)** — reads the funnel + its Wilson CIs,
  identifies the **dominant failure transition** (the biggest drop in the
  funnel), and selects the next improvement **lever** from a decision policy,
  emitting a natural-language rationale. This is `autonomous_loop.py::diagnose`.
  Cosmos-Reason is the heavy realization of this role for open-ended tasks; the
  loop's default is the fast, auditable rule policy so it can run unattended.
- **PILOT (GR00T)** — the frozen/fine-tuned VLA being improved. Sole actuator
  authority throughout; every lever either improves its weights (SFT/distill) or
  improves the selection/steering *around* its own sampled behaviors — never a
  second actuator.

## The improvement ladder (levers)

Each lever names a precondition, an action (reusing the committed drivers), and
a gate. The Specialist picks among *applicable* levers by which funnel
transition is bottlenecking.

| # | Lever | Attacks (funnel transition) | Action | Gate |
|---|---|---|---|---|
| L0 | **competent base** | reached≈0 / everything | diverse-DR SFT (`run_base_fix.sh`) | bare-base browser ≥ prev |
| L1 | **steering mean** | reached→centered | image-cond head (`run_v4head.sh`) | base-gate pass, dagger MSE↓ |
| L2 | **selection** | centered→closed | best-of-N + CBF/CLF (`run_bestofn_ab`) | closure↑ CI |
| L3 | **capture DAgger** | closed→lifted | v4-visited DAgger (`run_dagger2_v4.sh`) | lift↑ CI |
| L4 | **flywheel distill** | all (compounding) | SFT GR00T on best-of-N wins | bare-base ≥ prev + dominance |
| L5 | **final-cm servo** | centered→closed (precision) | Observer-error DLS correction | capture↑, GR00T dominance |
| L6 | **place refine** | lifted→seated | transport/place CLF+guard | seat↑ CI |

The loop always evaluates with the **45-episode powered protocol** (3×15
fixed-seed blocks, Wilson 95% CI) so promotion decisions are statistical, never
single-run luck. A lever's output is *promoted* to "current best" only if its
gate clears; otherwise it is recorded as a tried-and-rejected branch.

## The loop

```
while not (target_met or budget_exhausted):
    deploy(current_best)                       # PILOT + OBSERVER live
    funnel = powered_eval(45)                  # OBSERVER measures
    record = diagnose(funnel, ladder_state)    # SPECIALIST decides + rationale
    result = execute(record.lever)             # run the chosen driver
    if gate(result): promote(result)           # statistical promotion
    append(research_log, record ⊕ result)      # EXPLAINABILITY
    checkpoint(state)                           # RESUMABILITY
```

`target_met` default: seated-rate Wilson lower bound ≥ 0.5. `budget`: GPU-hours
and iteration count. Every stage is a resumable checkpoint keyed by a content
hash of its inputs — a killed loop re-attaches at the first incomplete stage.

## Generalization

Nothing in the loop or the Specialist policy is vial-specific. A new task is a
`TaskSpec` (`task.yaml`): success predicate, the expert recipe (or human demos),
which Observer heads to (auto-)train, the funnel stages, and per-task CBF
constraints. Swapping the spec swaps the task; the loop, ladder, gates, and
brain contracts are unchanged. This is the platform's few-shot story: a task
stands up from its spec, the loop drives it to competence, and the research log
is the audit trail.

## Explainability

The loop is a glass box by construction:
- **research_log.jsonl** — one record per iteration: funnel + CIs, the diagnosed
  bottleneck transition, the chosen lever, the Specialist's rationale, the gate
  math, and promote/reject. Reads like a lab notebook.
- **per-decision provenance** — every executed action already emits its own
  certificate (best-of-N: candidates/CBF-rejections/CLF-choice; SFT: data
  provenance + dominance).
- **falsifiable** — each lever carries a kill-criterion; a lever that fails its
  gate twice is retired with the evidence, and the loop escalates. (This is how
  we learned the steering-regression floor was image-conditioned, and that holds
  were never the success bottleneck — the loop encodes that discipline.)

## Honest scope

The loop cannot exceed what its levers can do. If L0–L6 all clear their gates
and seated-rate still caps below target, the loop's final record is the
evidence for an architectural decision (e.g. IK owns fine sub-cm manipulation,
GR00T owns high-level) rather than a silent plateau — a shippable conclusion,
not a failure.
