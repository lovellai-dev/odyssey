# Cosmos 3 × RoboLab (eval-only, first-class `evaluation_type: robolab`)

Scores a published Cosmos 3 DROID world-action policy on
[NVlabs/RoboLab](https://github.com/NVlabs/RoboLab) — the **Isaac Lab-based
simulation twin of the DROID rig** (the real-robot cell in NVIDIA's Cosmos
videos: Franka + two mounted cameras + tabletop tasks). The `RobolabRunner`
launches RoboLab's `policies/cosmos3/run.py` headless and scores its
`episode_results.jsonl` into the standard eval summary (per-task rates +
failure reasons in `metrics`).

## Prerequisites (two external processes)

1. **Policy server** (cosmos-framework env, GPU box — Nano 16B needs ≥32 GB
   VRAM in bf16; Edge 4B fits smaller GPUs):

   ```bash
   # --checkpoint-path takes a LOCAL dir — stage it first (setup.sh does this):
   hf download nvidia/Cosmos3-Nano-Policy-DROID --local-dir ~/checkpoints/Cosmos3-Nano-Policy-DROID
   python -m cosmos_framework.scripts.action_policy_server_robolab \
       --checkpoint-path ~/checkpoints/Cosmos3-Nano-Policy-DROID --port 8000
   # Edge variant: same, plus --format-prompt-as-json True
   ```

   Guardrails are gated (`nvidia/Cosmos-Guardrail1`) — see setup.sh for the
   HF-access / disable options.

2. **RoboLab checkout with its Isaac Sim env** (client side; the docker
   images work):

   ```bash
   git clone https://github.com/NVlabs/RoboLab.git
   cd RoboLab && ./docker/build_docker.sh latest
   ```

## Run

Edit `mission.yaml` (`robolab_root`, `eval_python` for the Isaac interpreter,
`remote_host`/`remote_port` for the server), then:

```bash
odyssey run examples/cosmos3-robolab/mission.yaml
```

Contract details (pinned from RoboLab source): `benchmark_name` → `--task`;
`num_episodes` → `--num-runs ceil(episodes / num_envs)`; results are read from
`<robolab_root>/output/odyssey_<task-id>/episode_results.jsonl` and the raw
jsonl is copied into the task's output dir as an artifact. Passthrough config
keys reach `run.py` with underscores dashed (`remote_port` → `--remote-port`).

## Running inside docker: the wrapper (and why)

RoboLab's own `docker/run_docker.sh` is interactive-only and assumes a legacy
docker setup. `setup.sh` therefore generates `~/robolab_python.sh` — a
non-interactive wrapper you point `config.eval_python` at. Every flag in it
was a real failure on the first hardware bring-up (H100, docker 27 /
driver 580):

| Symptom | Cause | Fix (in the wrapper / setup) |
| --- | --- | --- |
| exit **125**, `unknown or invalid runtime name: nvidia` | modern daemons don't register the legacy `nvidia` runtime | `--gpus all` |
| exit **127** (command not found) | the image ships no bare `python` on PATH | `--entrypoint /isaac-sim/python.sh` |
| Isaac boots then **hangs**; log shows Warp `cuDeviceGetUuid` errors, PhysX `no suitable CUDA GPU`, renderer `invalid device` | docker injects only `compute,utility` driver capabilities — no graphics/Vulkan | `-e NVIDIA_DRIVER_CAPABILITIES=all` (injects host libGLX/libEGL + Vulkan ICD) |
| `carb::graphics::createInstance failed` / `Failed to create any GPU devices` **persisting** after the caps fix; the Warp/PhysX lines above are downstream noise (`cuInit` actually works — verified) | the image ships **no system `libvulkan.so.1`** (the Vulkan *loader* is an OS package, not part of the driver injection; only stale copies exist in XR extension caches) | build the derived image: `FROM robolab:latest` + `apt-get install libvulkan1` → `robolab:odyssey` (setup.sh does this) |
| same, after bind-mounting the **host's** `libvulkan.so.1` | host glibc (e.g. 2.38+) newer than the container's (22.04 = 2.35) — the mounted `.so` won't load | don't mount host userland libs across glibc versions; install inside the image instead |
| GPU **VRAM exhausted** mid-boot; a surprise `vllm serve nvidia/Cosmos3-Nano` container (~50 GB, `--gpu-memory-utilization 0.60`) appears | RoboLab's subtask progress checker auto-spawns a Cosmos3-Nano Reasoner judge via vLLM | on shared GPUs set `disable_subtask: true` in the mission config (bare store_false flag; episode success still comes from the sim's termination conditions — only per-subtask scores are lost) |
| `ModuleNotFoundError: robolab`/`policies` | script-mode sys.path doesn't include the repo root | `-e PYTHONPATH=/workspace/robolab` + `-w /workspace/robolab` |
| runner's host-absolute script path not found in container | container only mounts `/workspace/robolab` | dual mount: checkout also bind-mounted at its **host path** |

Verify before running a mission (both must pass, else Isaac hangs silently):

```bash
sudo docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
    --entrypoint /bin/bash robolab:odyssey -c \
    "nvidia-smi --query-gpu=name --format=csv,noheader \
     && ls /etc/vulkan/icd.d/nvidia_icd.json \
     && ldconfig -p | grep libvulkan.so.1"
```

If the Vulkan ICD check fails, the **host** needs the driver's graphics
userspace (`libnvidia-gl-<version>` + `/etc/vulkan/icd.d/nvidia_icd.json`) and
a current nvidia-container-toolkit. First Isaac boot compiles shaders —
expect several minutes before episode 1; the `.cache/ov` and `.cache/kit`
mounts persist them across runs.

## Family notes

Any Cosmos 3 policy member works — the server loads the weights; the mission
only records the id. `config.entry_script` selects the RoboLab policy backend
(default `policies/cosmos3/run.py`) — a future backend is a mission edit, not
a new runner. For LIBERO evals of the family use `examples/quickstart-cosmos3/`
(`pilot: cosmos3`); for the no-sim decode check use
`examples/cosmos3-droid-openloop/`.
