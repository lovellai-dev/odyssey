# Object-verification specialist probe — RoboBrain 2.5 vs SAM 3.1

The **object-verification** role of the **Specialist Model Map v0.5** bake-off:
show a model frames of a simulated manipulation rollout and ask *which of these
named objects do you actually see?* Two arms answer the same paper over the same
drug-sort rollout frames and the same object set:

* **RoboBrain 2.5** (`mission-robobrain.yaml`, `robobrain_probe.py`) — a VLM; it
  answers a YES/NO presence question per object through the
  `OpenAICompatCompletionJudge` surface (the exact `CompletionDetector` the
  multi-agent runtimes gate on), served by **vanilla vLLM** on port 8002.
* **SAM 3.1** (`mission-sam.yaml`, `sam_probe.py` + `sam_server.py`) — promptable
  **concept segmentation**; it returns masks/scores, and a concept counts
  PRESENT when its best score clears a threshold. Served by the repo's reference
  `sam_server.py` on port 8003.

## The interface asymmetry (read this before comparing arms)

Unlike the retry / grasp probes — where both arms shared a *verbatim* prompt so
"any delta is the model" — these two arms **cannot share a prompt surface**. SAM
is not a chat model; it does not emit YES/NO. So what is held constant across the
bake-off is:

* the **frames** (same rollout MP4s, same `view`/`upscale` crop), and
* the **object set** (`objects` = present, `distractors` = absent),

not the wording. Each arm turns its native output into the same two headline
metrics, so a single downstream read compares them:

* **`present_recall`** — mean YES / PRESENT rate over `objects` (want high: the
  model recognises what is really there).
* **`distractor_fpr`** — mean YES / PRESENT rate over `distractors` (want low: it
  does not hallucinate objects that are absent — an always-YES model is as
  useless as issue #78's always-NO judge).

Plus a `control` (RoboBrain: "arm + tabletop visible?"; SAM: segment
`"robot arm"`) as the always-empty degeneracy detector, and per-object rates +
per-frame verdicts for the discriminative reading.

## Serving

Two independent endpoints, each in its **own venv** (do not cross the deps):

**RoboBrain 2.5** — vanilla vLLM ≥ 0.11 (Qwen3-VL-8B base), gated on HF (accept
the conditions, export `HF_TOKEN`, first download online):

```bash
vllm serve BAAI/RoboBrain2.5-8B-NV --host 0.0.0.0 --port 8002 \
    --gpu-memory-utilization 0.60 --max-model-len 32768
```

**SAM 3.1** — the reference server here, symmetric with the vLLM one:

```bash
python examples/specialist-objectverification-probe/sam_server.py \
    --host 0.0.0.0 --port 8003 --model facebook/sam3.1
# no weights / no GPU, for wiring the whole path end-to-end:
SAM_SERVER_FAKE=1 python .../sam_server.py --port 8003
```

The exact SAM 3.1 concept-segmentation call is isolated in
`SamBackend.segment` — adapt that one method to the released API (its header
documents the contract). The HTTP contract the probe/viewer depend on:

```
POST /segment  {"image": "<data-uri>", "concept": "red capsule"}
  -> {"detections": [{"score": 0..1, "box": [x0,y0,x1,y1]}...], "image_size": [W,H]}
GET  /info     -> {"model": "...", "device": "...", "fake": <bool>}
```

## Running the missions

1. Serve the relevant model (above); check `GET /v1/models` (RoboBrain) or
   `GET /info` (SAM) answers.
2. Point `config.videos_dir` at the rollout MP4s (the committed missions use the
   same successful drug-sort rollouts as the retry arm, `view: wrist`,
   `upscale: 3`).
3. `odyssey run examples/specialist-objectverification-probe/mission-robobrain.yaml`
   and `.../mission-sam.yaml`.
4. Metrics land in each task's `out.json`: `present_recall`, `distractor_fpr`,
   per-object rates, latency, per-frame verdicts.

## Watching it live

Each arm has a streaming HTML viewer (Serene Ocean theme, sidecar `*_data.js`
polled with no page reload, playable slow-motion video, per-object timeline
bands, synced playhead). Copy BOTH the `.html` and its `_data.js` sidecar when
viewing on another machine. Tunnel with `ssh -L 8002:...` / `-L 8003:...` if the
model is served on the H100; `--fake` exercises each viewer with no server.

**RoboBrain arm** — `utils/visualize_probe.py` (YES/NO answer badges):

```bash
python examples/specialist-objectverification-probe/utils/visualize_probe.py \
    --video ~/cosmos3_probe_videos_success/rollout_ep001_success.mp4 \
    --instruction "pick up the red capsule and place it in the blue tray" \
    --objects "red capsule, blue tray" --distractors "green bottle, yellow block" \
    --view wrist --upscale 3 --stride 5 --out /tmp/robobrain_objverif.html
```

**SAM arm** — `utils/visualize_sam_probe.py`, the **variation**: instead of a
YES/NO badge it **overlays the returned detection boxes** on each judged frame
and reads the best mask score, so you see what SAM latched onto:

```bash
python examples/specialist-objectverification-probe/utils/visualize_sam_probe.py \
    --video ~/cosmos3_probe_videos_success/rollout_ep001_success.mp4 \
    --instruction "pick up the red capsule and place it in the blue tray" \
    --objects "red capsule, blue tray" --distractors "green bottle, yellow block" \
    --score_threshold 0.5 --view wrist --upscale 3 --stride 5 \
    --out /tmp/sam_objverif.html
```

Distractor polarity (a good specialist answers ABSENT / low-score on the
distractors) is the specificity half of the map's object-verification test; the
present-object recall is the other half.

## Reading the bake-off result — the scoreboard

The two viewers above are for *inspecting* one rollout. To *read the benchmark
verdict at a glance* — which objects each model recognises over the whole run,
no playback — use `utils/scoreboard.py`. It builds a static HTML from the probe
result JSONs: rows are objects, columns are models (RoboBrain and SAM side by
side), each cell a recall / false-positive bar plus a per-frame heat-strip.
Colour is **correctness**, so the board reads green = good (present→YES,
distractor→NO, control→YES). Models with no data yet are a `pending` column.

```bash
python examples/specialist-objectverification-probe/utils/scoreboard.py \
    --result "RoboBrain·side=/tmp/rb_side.json" \
    --result "RoboBrain·wrist=/tmp/rb_objverif.json" \
    --pending "SAM 3.1" \
    --out /tmp/objverif_scoreboard.html
```

Two RoboBrain columns (`side` vs `wrist`) also make the camera effect legible:
`red capsule` recall jumps from ~8% in the gripper close-up to ~67% in the
workspace view — small objects need the side plane.
