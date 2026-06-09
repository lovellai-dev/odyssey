# Telemetry Alignment: Odyssey <-> lai-trainer

> Status: analysis complete, zero-cost items implemented.
> Date: 2026-06-09
> Branch: `fix/oxe-dataset-ref`

## Context

Both Odyssey (local CLI runner) and lai-trainer (hosted mission service) publish
events during mission execution. This document maps the principles used in
lai-trainer's job-orchestration design to Odyssey's telemetry layer, identifies
gaps, and records what was aligned.

## Architecture Comparison

### Event Flow

```
lai-trainer (hosted)                    Odyssey (local)
─────────────────────                   ─────────────────
Inference VM                            subprocess (finetune.py)
  │                                       │
  ├─ webhook ──► trainer API              ├─ stdout bytes ──► terminal (raw tqdm)
  │  (coarse state, at-least-once)        │  (real-time, read(8192) chunks)
  │                                       │
  └─ Pub/Sub ──► trainer API              └─ emit_progress() ──► EventPublisher
     (fine progress, best-effort)            (structured events, best-effort)
                    │                                     │
                    ▼                                     ▼
          task_progress_events (DB)            StdoutEventPublisher (JSON lines)
          command-events topic (Pub/Sub)       [future: SQLite, remote]
```

### Shared Principles (already aligned)

| Principle | lai-trainer | Odyssey |
|---|---|---|
| Lifecycle vs progress separation | Engine publishes state events; inference VM publishes progress | Engine `_transition_mission()` publishes state; runners call `emit_progress()` |
| Publisher abstraction | `EventPublisher` ABC → `PubSubEventPublisher` | `EventPublisher` ABC → `StdoutEventPublisher` |
| Best-effort progress | Pub/Sub is loss-tolerant; failures don't kill the job | `emit_progress` wrapped in try/except; runner continues |
| Event type vocabulary | `MissionEventType`, `TaskEventType` enums | Same enums, same string values |
| Progress field set | stage, step, step_index, step_total, step_label, metadata | Same fields in `emit_progress()` signature |

### Gaps Identified

| Gap | lai-trainer | Odyssey (before) | Priority |
|---|---|---|---|
| Typed progress model | `ProgressEvent` Pydantic model | Ad-hoc dict in `emit_progress()` | **P0 — implemented** |
| Sequence numbers | Monotonic `seq` per job, used for ordering + polling | None | **P0 — implemented** |
| Progress persistence | `task_progress_events` SQLite table | Transient stdout only | P1 — deferred to v0.2.x |
| Polling endpoint | `GET /progress?since=<seq>` | None | P2 — deferred |
| Heartbeat / stale detection | `last_heartbeat_at` + sweeper | Direct subprocess handle | N/A — local mode |
| Idempotency keys | `{job_id}:{state}:{seq}` dedup | Not needed locally | N/A |

## What Was Implemented

### 1. `ProgressEvent` Pydantic model (`telemetry/events.py`)

Formalizes the contract between runners and the publisher. Validates field
types at emission time. Serializes to the same JSON shape lai-trainer consumes.

```python
class ProgressEvent(BaseModel):
    mission_id: str
    task_id: str
    task_name: str
    stage: str
    seq: int
    step: str | None = None
    step_index: int | None = None
    step_total: int | None = None
    step_label: str | None = None
    metadata: dict[str, Any] | None = None
```

### 2. Monotonic `seq` counter on `TaskContext`

Each `TaskContext` maintains a `_progress_seq` counter incremented on every
`emit_progress()` call. Enables ordered replay and future polling with
`since=<seq>`.

## Subprocess Verbosity (branch history)

Two commits on this branch changed how subprocess output reaches the developer:

### `442caba` — Buffer for tqdm

- **Problem**: tqdm writes `\r` without `\n`; asyncio's `readline()` accumulates
  data until the 64 KB buffer overflows (`LimitOverrunError`).
- **Fix**: Increased `limit` from 64 KB to 10 MB in `create_subprocess_exec()`.

### `17d7372` — Real-time streaming

- **Problem**: Terminal appeared frozen during training — output only visible
  after task completion.
- **Fix**: Switched from `readline()` to `read(8192)` (8 KB chunks). Added
  `sys.stdout.buffer.write(raw)` for immediate terminal output. Internal line
  buffer accumulates partial chunks for structured parsing.

### Current output pipeline

```
subprocess stdout
  │
  ├──► read(8192) raw bytes
  │       │
  │       ├──► sys.stdout.buffer.write(raw)   # developer sees tqdm live
  │       │
  │       └──► line buffer accumulates
  │               │
  │               └──► complete lines parsed by line_parser
  │                       │
  │                       └──► emit_progress(stage, step, ...)
  │                               │
  │                               └──► ProgressEvent validated
  │                                       │
  │                                       └──► publisher.publish() → JSON stdout
  │
  └──► asyncio limit = 10 MB (safety net for \r-heavy output)
```

## Deferred Work

### Progress persistence (v0.2.x)

Add a `task_progress_events` table to Odyssey's SQLite store, mirroring
lai-trainer's schema. Enables:

- Crash recovery ("training was at step 450 when VM was preempted")
- Post-mortem debugging without log archaeology
- Future UI with same polling pattern (`GET /progress?since=<seq>`)

### Shared event contract (post-v0.2.x)

Consider extracting `ProgressEvent` and the event enums into a shared
`lovell-events` package consumed by both repos. Low priority until both
repos stabilize their event shapes.
