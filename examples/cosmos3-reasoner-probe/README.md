# Experiment: Cosmos 3 Reasoner as a grasp-verification SPECIALIST

Branch: `experiment-specialist-grasp-verification` (off `cosmos3-integration`,
PR #95). Status: **probe complete, verdict below; multi-agent wiring not yet
done.** Motivating issue: #78 (the co-resident Gemma int4 `check_done` judge
answered NO essentially always, so multi-agent phases advanced by step-cap
instead of by completion — the fix direction was a *stronger judge*).

## Question under test

Can a served `nvidia/Cosmos3-Nano` — the **Reasoner** surface of NVIDIA's
Cosmos 3 family (16B omnimodal, *not* the `Cosmos3-Nano-Policy-DROID` action
model) — act as a SPECIALIST that verifies **grasp state** from camera frames,
in the *delegation* posture of the planner-vs-delegation experiment (PR #68,
closed unmerged): the SPECIALIST authors no plan, it answers on-demand
perception questions.

## What actually runs (and what does not)

- **No simulator runs.** The probe extracts frames from pre-recorded rollout
  MP4s and queries the served model over HTTP. `evaluation_type: custom`
  (subprocess, sim-agnostic).
- The **SPECIALIST is the only exercised agent**: `nvidia/Cosmos3-Nano` served
  by vLLM, driven through `OpenAICompatCompletionJudge`
  (`src/odyssey/runners/agents/openai_judge.py` — framework code, a
  `CompletionDetector` that drops into `ChunkCompletionGate`; it gained the
  `extra_body` knob for this experiment).
- The **PILOT records provenance, not execution**: the judged rollouts were
  driven on the H100 by the GR00T + FlowDAgger drug-sort policy (a frozen
  GR00T flow pilot finetuned from `nvidia/GR00T-N1.7`, FlowDAgger-steered) —
  that is what the loadout's PILOT (`gr00t-flowdagger-pilot`) names. This
  mission never loads it; the spec simply requires a PILOT.
- **Video provenance**: drug-sort DAgger evaluation rollouts from the UR-arm
  drugsort campaign (the H100 scripts label the campaign **UR10e** DAgger
  while reusing `ur5e-drugsort` tooling paths — episodes idle at the tick cap
  after early success). No Cosmos model drove these robots.

## Serving recipe (H100-validated, 2026-08-10)

```bash
sudo docker run --rm --gpus all --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-omni:v0.26.0 \
    vllm serve nvidia/Cosmos3-Nano --host 0.0.0.0 --port 8002 \
        --gpu-memory-utilization 0.60 --max-model-len 32768
```

**Plain `vllm serve`, NOT `--omni`.** `Cosmos3ForConditionalGeneration` is
registered in plain vLLM and that is the reasoning path — image+text in, text
out, 0.05–0.2 s per YES/NO reply. Under `--omni` the identical chat request
routes to the *diffusion* pipeline: 50 denoise steps, an **image** reply, so
no YES/NO text ever comes back and every verdict parses as the conservative
NO (in the `:cosmos3` image tag the half-built text path 500s / OOMs the
diffusion worker). `--omni --no-guardrails` remains the recipe for
*generation* serving only (the gated `nvidia/Cosmos-1.0-Guardrail` otherwise
401s at startup). Box gotchas: the NVIDIA container runtime needs
`nvidia-persistenced` running; the AR surface loads ≈ 40 GB VRAM.

## Method

One eval-only mission, `mission.yaml`, over **successful** drug-sort episodes
(expected: grasp flips YES while held, completion YES at the end). An earlier
companion mission covered the NO polarity on failed LIBERO rollouts; it was
dropped from this branch for clarity — it lives in the git history and on
`cosmos3-integration`, and its numbers are reported below as Runs 1–2.

`reasoner_probe.py` samples frames at 0/50/75/100 % of each MP4 and asks four
strict YES/NO questions per frame through the judge: `control` (arm visible?
— detects the #78 always-NO degenerate mode), `grasp`, `completion` (stock
template), `retry`. Knobs: `view: wrist|side` crops one half of a 2:1
concat_view frame, `upscale: N` LANCZOS-resizes (the #78 zoom lesson).
Metric-only out-json: rates, latency, per-frame verdicts with reply excerpts.

## Results (H100, 2026-08-10)

**Run 1 — full 256 px frames** and **Run 2 — `view: wrist`, `upscale: 3`**
(both missions COMPLETED, 80/80 calls parsed, latency 0.05–0.17 s mean):

| question | result (both runs) |
|---|---|
| control | **100 % YES** — not Gemma-degenerate |
| grasp / completion / retry | 100 % NO |

**Run 3 — dense paired diagnostic sweep** (every 10th frame of a successful
episode, wrist crop ×3, paired opposite questions *"holding an object?"* /
*"gripper empty?"*):

- Pairs answered **consistently** (empty → NO/YES) on almost every frame.
- **holding=YES / empty=NO exactly inside the true grasp window** (frame 10 of
  138; verified against the actual frame).
- Ground truth explains runs 1–2: the drug-sort pick lands in the **first
  ~15 % of the episode** and the episode idles at the tick cap after early
  success — so the fixed 0/50/75/100 % samples fall entirely post-place and
  the all-NO grasp column was *honest*, *not* the #78 degeneracy.

Where it still fails: open-ended task-completion phrasings bias NO even after
visible success, and fine object-location questions on full 256 px concat
frames are inconsistent (it answered YES to two mutually exclusive location
questions on the same frame).

## Verdict

`Cosmos3-Nano` is **usable as a gripper-state oracle on wrist-camera crops
with concrete perception questions** — consistent paired answers, correct
inside the real grasp window, and 25–100× faster per judgement than the
co-resident Gemma judge. It is **not yet trustworthy** for open-ended
"is the sub-task done" judgements or fine spatial grounding at low
resolution. For SPECIALIST wiring, prefer perception-style delegation
questions (the PR #68 posture) over completion-style ones.

## Next steps

1. Mission-YAML plumbing to select `OpenAICompatCompletionJudge` as the
   `completion`-strategy detector in the multi-agent runtimes (nothing merged
   wires a detector from config today).
2. Add the paired *empty?* question and denser sampling to the scripted
   probe, so the mission itself captures what the diagnostic sweep showed.
3. Re-run against confirmed-UR10e rollouts with the campaign's real task
   instruction; fix the `embodiment` label accordingly.

## Watch it happen: the probe visualizer

The model never sees the video — it sees isolated frames, one independent
image+text call per (frame, question). `utils/visualize_probe.py` makes that
loop visible: it replays it over one rollout MP4, streams a table row per
answer to the terminal, and writes a self-contained HTML report — the
playable video, the judged (cropped/upscaled) thumbnails, and the growing
Q/A table with the *full prompt sent* per row; clicking a row seeks the
video to that frame. The page auto-refreshes while the run is live. It also
asks the paired `empty` question (the degeneracy detector from Run 3).

```bash
python examples/cosmos3-reasoner-probe/utils/visualize_probe.py \
    --video ~/videos/rollout_ep001_success.mp4 \
    --instruction "pick up the red capsule and place it in the blue tray" \
    --view wrist --upscale 3 --stride 10 --out /tmp/probe_report.html
open /tmp/probe_report.html
```

Tunnel first (`ssh -L 8002:127.0.0.1:8002 <gpu-box>`) when the model is
served remotely; `--fake` exercises the viewer without any server.

## Reproduce

1. Serve the model (recipe above; any Edge/Nano/Super Reasoner id works —
   family-wide by construction).
2. Point the mission's `config.videos_dir` at a directory of rollout MP4s
   and set `config.instruction` to what those rollouts attempted; set
   `config.eval_python` to a venv with `imageio` + `pillow` (e.g. the
   `env_pilot_cosmos3` venv that `quickstart-cosmos3/setup.sh` builds on the
   `cosmos3-integration` branch — this branch keeps only this experiment).
3. `odyssey run examples/cosmos3-reasoner-probe/mission.yaml`. Metrics land
   in the task's `custom_eval_metrics.json`; per-frame verdicts under
   `metrics.verdicts`.

## Molmo 2 arms — the map's PRIMARY enters the bake-off

The Specialist Model Map v0.5 names Molmo 2 the primary for grasp
verification. It enters as TWO pre-registered directions, measured apart:

* **Direction A — `mission-molmo2.yaml`**: Molmo2-8B as a single-frame VQA
  judge, same script (`reasoner_probe.py`) and VERBATIM prompts as the
  Cosmos arm — apples-to-apples; the only variable is the model. Uses the
  new `--extra_body` knob (`'{}'` — no vLLM-Omni modalities routing; the
  default keeps the Cosmos mission byte-identical).
* **Direction B — `mission-molmo2-tracking.yaml`** (`molmo2_tracking_probe.py`):
  Molmo 2's NATIVE modality — K chronological frames in one multi-image
  request, per rollout: `carry` ("does the object travel WITH the gripper
  during the lift?") and `grasp_frame` localization (scored against the known
  early pick window). Deliberately NOT comparable with direction A: it
  measures whether sequence-native questioning recovers the signal that
  single-frame judges miss (the same lesson the retry bake-off learned with
  its value probe).

Serving recipe (vanilla vLLM from the same image, port 8003; the
`--limit-mm-per-prompt` allowance is required by direction B) lives in the
`mission-molmo2.yaml` header. Both Cosmos (:8002) and Molmo2 (:8003) fit the
H100 together next to the GR00T pilot.

### Comparative viewer

`utils/visualize_probe.py` now takes a repeatable `--arm` JSON flag to judge
the SAME frames with several models side by side — one timeline band per
question x arm, an arm column in the table, badges grouped per arm (with no
`--arm`, the old single-model flags still work unchanged):

```bash
python examples/cosmos3-reasoner-probe/utils/visualize_probe.py \
    --video ~/videos/rollout_ep001_success.mp4 \
    --instruction "pick up the red capsule and place it in the blue tray" \
    --view wrist --upscale 3 --stride 10 --out /tmp/compare_report.html \
    --arm '{"label": "cosmos3", "model": "nvidia/Cosmos3-Nano", "base_url": "http://127.0.0.1:8002/v1", "extra_body": {"modalities": ["text"]}}' \
    --arm '{"label": "molmo2", "model": "allenai/Molmo2-8B", "base_url": "http://127.0.0.1:8003/v1"}'
```

RoboBrain 2.5 (`BAAI/RoboBrain2.5-8B-NV`, serve recipe on the
`experiment-specialist-retry` branch) can join as a third arm labelled
out-of-role — the map does not list it for grasp, but it feeds the
specialist-vs-multiplexed-generalist question.
