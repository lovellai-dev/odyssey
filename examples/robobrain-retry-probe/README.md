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
3. `odyssey run examples/robobrain-retry-probe/mission.yaml`
4. Metrics land in the task's `out.json`: per-question YES rates, latency,
   per-frame verdicts with raw-reply excerpts. The visualizer from the Cosmos
   probe (`examples/cosmos3-reasoner-probe/utils/visualize_probe.py`) reads
   this format unchanged.

Retry's YES polarity (flip on failures) needs a failed-rollout dir; the
mission ships that task commented out until one is collected. That run is the
map's C3b offline-oracle test.
