# RoboBrain 2.5 retry-specialist probe

The RoboBrain arm of the **Specialist Model Map v0.5** bake-off for the
**retry strategy** role. RoboBrain 2.5's native benchmark family is mistake
existence / classification / recovery — "did the grasp fail, why, and how do I
recover?" is its training distribution, which is why the map names it the
primary specialist for retry, with the Cosmos3-Nano Reasoner as challenger.

This example is a deliberate copy of `examples/cosmos3-reasoner-probe/` with
only the serving glue changed. The four YES/NO templates (`control`, `grasp`,
`completion`, `retry`) are verbatim copies and must stay that way: both arms
answer the same questions over the same frames, so any verdict delta is the
model. `retry` is the headline question for this arm (it was diagnostic-only
for the Cosmos grasp probe).

## Serving

Vanilla vLLM >= 0.11 (Qwen3-VL support — RoboBrain 2.5 is Qwen3-VL-8B-based),
in its **own venv**; do not touch the vllm-omni env that serves Cosmos:

```bash
vllm serve BAAI/RoboBrain2.5-8B-NV --host 0.0.0.0 --port 8002 \
    --gpu-memory-utilization 0.60 --max-model-len 32768
```

Gotchas:

* The checkpoint is **gated on HF** — accept the conditions on the model page
  and export `HF_TOKEN`; the first download must run with `HF_HUB_OFFLINE=0`.
* First smoke before any mission: one `curl` to `/v1/chat/completions` with a
  frame + the retry template, to confirm the checkpoint loads under vanilla
  vLLM and answers with a parseable YES/NO. If vLLM rejects the architecture
  name, retry with `--trust-remote-code`.
* No `extra_body` / `modalities` knob — that is a vLLM-Omni (Cosmos) quirk.
* ~18 GB bf16: fits an H100 next to the pilot with headroom.

## Running

1. Serve the model (above) and check `GET /v1/models` answers.
2. Point `config.videos_dir` at the rollout MP4s — the committed mission uses
   the same successful drug-sort rollouts as the Cosmos arm (Runs 1–2), so the
   first read is apples-to-apples, matching Cosmos Run 2's `view: wrist`,
   `upscale: 3`.
3. `odyssey run examples/specialist-retry-probe/mission.yaml`
4. Metrics land in the task's `out.json`: per-question YES rates, latency,
   per-frame verdicts with raw-reply excerpts.

## Watching it live

`utils/visualize_probe.py` (this example's copy of the Cosmos probe viewer,
already pointed at RoboBrain defaults) replays the judging loop over one MP4
and renders the streaming HTML report — playable slow-motion video, live
answer badges, per-question timeline bands with a synced playhead, and the
full Q/A table (it adds the `empty` degeneracy-detector question on top of
the mission's four):

```bash
python examples/specialist-retry-probe/utils/visualize_probe.py \
    --video ~/cosmos3_probe_videos_success/rollout_ep001_success.mp4 \
    --instruction "pick up the red capsule and place it in the blue tray" \
    --view wrist --upscale 3 --stride 5 --out /tmp/robobrain_report.html
open /tmp/robobrain_report.html   # rows stream in without reloading
```

Tunnel with `ssh -L 8002:127.0.0.1:8002` if the model is served on the H100;
`--fake` exercises the viewer with no server at all.

Retry's YES polarity (flip on failures) needs a failed-rollout dir; the
mission ships that task commented out until one is collected. That run is the
map's C3b offline-oracle test.

## Iteration 2 — retry as a progress curve (`robobrain_value_probe.py`)

Run 1 finding: an *eventless* failure (robot moves nominally, never
progresses — `rollout_ep000_fail.mp4`) is invisible to single-frame YES/NO
retry judging; every frame honestly looks like a nominal in-progress attempt.
The failure lives in the sequence, so Iteration 2 asks RoboBrain's native
strength instead (card: Temporal Value Estimation / Dense Progress
Prediction): a task-progress percentage per sampled frame, reading the
*curve* — a stalled curve is the retry signal, a threshold on a continuous
quantity instead of a binary opinion.

```bash
python examples/specialist-retry-probe/robobrain_value_probe.py \
    --videos_dir ~/cosmos3_probe_videos_success \
    --instruction "pick up the red capsule and place it in the blue tray" \
    --view side --upscale 2 --stride 10 --out-json /tmp/value_probe.json
```

`utils/visualize_value_probe.py` is the Iteration-2 companion of the frame-
by-frame viewer (same streaming pattern — one MP4, sidecar `_data.js`, page
never reloads): playable video with a synced playhead over the live progress
curve, opening/peak/verdict badges computed with the same stall rule, and the
per-frame table with raw replies. `--fake` exercises it with no server.

```bash
python examples/specialist-retry-probe/utils/visualize_value_probe.py \
    --video ~/cosmos3_probe_videos_success/rollout_ep000_fail.mp4 \
    --instruction "pick up the red capsule and place it in the blue tray" \
    --view side --upscale 2 --stride 5 --out /tmp/fail_value_report.html
# copy BOTH the .html and its _data.js sidecar if viewing on another machine
```

First live read (H100, 2026-08-11/12, `view: side`, stride 5): the smoothed
stall rule separates 3/3 — the failed rollout's spikes are single isolated
samples that the median removes (rise 0 → STALLED), while both successes hold
an early two-sample plateau that survives it (rise 35 → progressing). `wrist`
view is noisy (the crop hides the trays), confirming the Specialist Map's
camera hypothesis: retry wants workspace geometry, not the gripper close-up.

Two hard-won rules about the signal:

* **The verdict is computed on the median-of-3 smoothed curve** (see
  `stall_verdict` in `robobrain_value_probe.py`, imported by both the batch
  probe and the viewer — single source of truth). Raw per-frame estimates
  jitter enough that one isolated spike can cross `MIN_RISE` and flip the
  verdict with the raw rule.
* **Sample densely: stride ≤ 5.** The median needs a real rise to span ≥2
  samples; at stride 10 the successes' early plateau is one sample wide and
  gets erased along with the noise.

Caveats: n=3 rollouts (one failure); thresholds (`MIN_RISE`, 60% checkpoint)
are calibrated on that tiny set and must be re-fit when more failures exist.
