# Browser closed-loop eval harness

`eval_browser_groot.js` scores a pick-and-place policy in the deployment
renderer rather than in a native physics viewer, because the two do not agree:
policies score differently the moment the pixels come from a different
renderer, so the eval has to run where the policy will run.

It drives a headless Chrome against the served scene, replays a frozen seeded
fold of object start poses, queries the policy over HTTP, and writes a summary
plus per-episode telemetry.

## Prerequisites

This harness is the scoring half of a stack it does not contain:

| Needs | Provided by |
|---|---|
| The scene served over HTTP on `$PORT` | the deployment renderer app (external) |
| `plans_eval.json` — the frozen fold | `precompute_plans.py --seed <fold>` |
| A policy endpoint | your serving stack |
| `puppeteer-core` + a Chrome binary | `npm install`; `$CHROME` |

Run `npm install` in this directory — `node_modules/` is gitignored and must
never be committed.

## Configuration

All via environment: `PLANS`, `OUT`, `PORT`, `AGENTS` (`groot` |
`groot-observer`), `N`, `N_ACTION_STEPS`, `MAX_TICKS`, `CHROME`, `ARM`,
`RAW_DUMP=1` to dump policy frames and ground truth for observer retraining.

## Placement metrics — read this before quoting a number

The harness emits three placement criteria per episode. They are not
interchangeable, and the difference is large enough to invert a conclusion.

| Field | Criterion | Use |
|---|---|---|
| `success` | lifted, then released near the target | coarse |
| `seated` | within 5 cm laterally | **legacy — loose** |
| `in_well` | within 4 mm laterally, base below the rim, upright | **the task** |

`seated` was the historical criterion and it overstates placement badly: a
5 cm tolerance passes an object resting on the rack surface next to the hole.
Re-scoring one campaign against `in_well` moved it from 62.5% to 17.5%. Quote
`in_well`; `seated` is retained only so older results remain comparable.

`reltilt` is measured **at release**, not at rest — an object that lands
upright after toppling off a rim is not a successful placement, and measuring
at rest hides the cause. `tilt_ok` compares it against the geometric budget:
past that angle the object's silhouette is wider than the opening and entry is
impossible regardless of aim.

## Physics overrides

The harness mutates the physics at runtime — it disables collision on the arm
and gripper geoms except the pads, and raises the solver's no-slip iterations.
Neither is recorded in the scene file, so the scene on disk describes a
different world than the numbers came from (issue #89).

Every summary now carries a `physics_overrides` block naming exactly what was
changed, also logged as `PHYSICS_OVERRIDES` on stdout. This makes the current
behaviour honest; it does not remove the override. The eval still cannot
observe arm self-collision or arm-environment contact, so a defect in the
scene's arm collision geometry is invisible here and will only show up in
native physics. Removing the override is a cross-repo change and #89 stays
open for it.
