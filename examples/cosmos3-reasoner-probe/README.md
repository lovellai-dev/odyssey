# Cosmos 3 Reasoner SPECIALIST probe — grasp verification

Eval-only mission that answers, **before any multi-agent wiring**: can a served
`nvidia/Cosmos3-Nano` — the Reasoner surface of the Cosmos 3 family, *not* the
`-Policy-DROID` action model — work as a **grasp-verification SPECIALIST**?

It follows the *delegation* posture of the planner-vs-delegation experiment
(PR #68, closed unmerged): the SPECIALIST authors no plan; it answers on-demand
perception questions. The probe drives `OpenAICompatCompletionJudge` (the
`CompletionDetector` the multi-agent runtimes gate on — issue #78's "stronger
judge" direction) over frames sampled at 0% / 50% / 75% / 100% of rollout MP4s,
asking four YES/NO questions per frame:

| question   | failed rollout             | successful rollout                  |
|------------|----------------------------|-------------------------------------|
| control    | YES everywhere (arm visible) | YES everywhere                    |
| grasp      | NO everywhere              | **YES while the object is held**    |
| completion | NO everywhere              | YES at the end                      |
| retry      | YES late in the episode    | NO everywhere                       |

`control` detects the Gemma-judge degenerate mode (always NO). The grasp
column is the discrimination a delegation SPECIALIST needs. Metric-only: no
success_rate is fabricated.

## 1. Serve the Reasoner (GPU box; the AR surface loads ≈ 40 GB)

```bash
sudo docker run --rm --gpus all --network host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-omni:v0.26.0 \
    vllm serve nvidia/Cosmos3-Nano --host 0.0.0.0 --port 8002 \
        --gpu-memory-utilization 0.60 --max-model-len 32768
```

**Plain `vllm serve`, NOT `--omni`** (validated on H100, 2026-08-10):
`Cosmos3ForConditionalGeneration` is registered in plain vLLM and that is the
reasoning path — image+text in, text out, sub-second replies. Under `--omni`
the identical chat request routes to the *diffusion* pipeline: 50 denoise
steps and an image reply, so no YES/NO text ever comes back and every verdict
parses as the conservative NO (and in the `:cosmos3` image tag the half-built
text path 500s / can OOM the diffusion worker). `--omni --no-guardrails`
remains the recipe for *generation* serving only (the gated
`nvidia/Cosmos-1.0-Guardrail` would otherwise 401 at startup). Trim
`--gpu-memory-utilization` to co-exist with other jobs on the GPU.

## 2. Stage probe frames

Two directories of rollout MP4s, one per polarity:

- `videos_dir` of task 1 → **failed** rollouts, e.g. the `videos/` dir of a
  `quickstart-cosmos3` OOD smoke under `~/.odyssey/runs/<mission>/<task>/videos/`.
- `videos_dir` of task 2 → **successful** rollouts (e.g. FlowDAgger UR5e
  drug-sort evals). Set each task's `config.instruction` to what its rollouts
  attempted.

## 3. Run

```bash
odyssey run examples/cosmos3-reasoner-probe/mission.yaml
```

The eval env needs `imageio` + `pillow` (the `env_pilot_cosmos3` from
`../quickstart-cosmos3/setup.sh` has both — name it via `config.eval_python`).

## Reading the result

`metrics.verdicts` carries one row per (question, frame) with the answer,
latency and a reply excerpt. Interpretation:

- `control_yes_rate` < 1.0 → the judge can't even confirm the scene; distrust
  the rest (this is what #78's Gemma judge failed).
- `grasp` YES concentrated on held-object frames of successful rollouts, NO on
  failed rollouts → a real grasp-verification SPECIALIST candidate.
- `latency_s_mean` bounds how often you could afford to gate (the multi-agent
  runtimes judge at chunk boundaries, ~every 16 env steps).
