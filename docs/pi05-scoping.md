# pi0.5 scoping (issue #74)

## License and weights

### Executive summary

The **π0.5 (pi05)** checkpoints published by Physical Intelligence in the
official `openpi` repository are distributed under the **Apache 2.0 license**. Apache
2.0 is a permissive license that **explicitly permits commercial use**,
modification, distribution, and use in private/industrial deployments,
royalty-free, provided the copyright notice, license text, and `NOTICE` file (if
present) are retained in redistributions. **There is no "research-only"
clause and no commercial-use restriction.**

→ For the intended use (**commercial industrial deployment**), the license **is
not a blocker**.

### Openly available checkpoints

Physical Intelligence releases π0.5 weights through two equivalent channels: Google
Cloud Storage (`gs://openpi-assets/...`, the source consumed by the `openpi` repo)
and the Hugging Face Hub (mirror under the `lerobot` organization).

| Checkpoint | Path / repo | Intended use |
| --- | --- | --- |
| **π0.5 base** | `gs://openpi-assets/checkpoints/pi05_base` · HF: [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) | Fine-tuning (generalist model pre-trained on 10k+ h of robot data) |
| **π0.5-LIBERO** | `gs://openpi-assets/checkpoints/pi05_libero` · HF: [`lerobot/pi05_libero`](https://huggingface.co/lerobot/pi05_libero) | Inference / LIBERO benchmark (SOTA) |
| **π0.5-DROID** | `gs://openpi-assets/checkpoints/pi05_droid` | Inference / fine-tuning (fine-tuned on DROID with *knowledge insulation*, fast inference and language following) |

Notes:
- π0.5 is the evolution of π0 oriented toward **open-world generalization** (published
  in September 2025).
- The `openpi` repo code is likewise under Apache 2.0; the LeRobot
  implementation (`policy.type=pi05`) derives from the same repository and declares the same
  Apache 2.0 license for the model.
- π0.5 default `chunk_size` = 50 (relevant to the pilot's *chunking*
  gateway).

### Recommended diligence before production

Although the license is permissive, for a commercial deployment it is advisable to:
1. **Retain openpi's `LICENSE`/`NOTICE`** in any redistributed artifact
   or container image that includes the weights.
2. Verify each checkpoint's **model card** on the Hub at download
   time: Apache 2.0 applies today, but a specific card could add
   dataset terms (e.g. DROID) or *acceptable use* terms — review before
   pinning.
3. Check the license of the **datasets** used if you do your own fine-tuning
   (the base weights are Apache 2.0, but the fine-tuning data the project
   provides has its own provenance).

### Sources

- Physical-Intelligence/openpi — repository and checkpoints (LICENSE = Apache 2.0):
  https://github.com/Physical-Intelligence/openpi
- LICENSE (Apache License, Version 2.0):
  https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/LICENSE
- Physical Intelligence — "Open Sourcing π0" (blog):
  https://www.pi.website/blog/openpi
- Physical Intelligence — π0.5 (open-world generalization):
  https://www.physicalintelligence.company/blog/pi05
- LeRobot docs — π0.5 (Pi05) Policy, "This model follows the Apache 2.0 License":
  https://huggingface.co/docs/lerobot/pi05
- HF checkpoints: [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base),
  [`lerobot/pi05_libero`](https://huggingface.co/lerobot/pi05_libero)

**Conclusion:** π0.5 is published under **Apache 2.0**, which authorizes
commercial and industrial use. **No BLOCKER line is added** — the license permits the
intended use.

## FAST tokenizer

### Executive summary

**FAST (Frequency-space Action Sequence Tokenization) is available as a
reusable library — it does NOT require reimplementation.** Physical Intelligence
publishes it in two complementary forms, both under **Apache 2.0**:

1. **As a HuggingFace `AutoProcessor`** (checkpoint `physical-intelligence/fast`),
   consumable in 3 lines via `transformers` + `scipy`, with no openpi dependency.
2. **`FASTTokenizer` wrapper** inside `openpi`
   (`src/openpi/models/tokenizer.py`), which in turn loads the same `AutoProcessor`
   underneath. It is the path used by π0-FAST / π0.5 itself in training.

→ For the π0.5 pilot, action tokenization **is reused as-is**; the
only work is integration (normalization to `[-1, 1]`, chunk wiring), not
reimplementing the algorithm (DCT + JPEG-style quantization).

### Path A — universal library via `transformers` (independent of openpi)

Requirements: `pip install transformers scipy`. Checkpoint: `physical-intelligence/fast`
(universal **FAST+** tokenizer, trained on 1M real action sequences).

```python
from transformers import AutoProcessor
import numpy as np

# trust_remote_code=True: the FAST algorithm travels as code in the Hub repo
tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)

action_data = np.random.rand(256, 50, 14)  # (batch, time_horizon, action_dim), normalized to [-1, 1]
tokens = tokenizer(action_data)             # -> list[int]  (encoding with compression)
decoded = tokenizer.decode(tokens)          # -> reconstructed actions
```

Usage notes:
- Recommended for ~1 s *chunks* **pre-normalized to `[-1, 1]`**.
- Encode/decode support **batched inference**.
- You can **train your own tokenizer** for a specific dataset with
  `tokenizer.fit(action_data)`, followed by `save_pretrained(...)` /
  `push_to_hub(...)`. Useful if the target embodiment's action statistics
  differ significantly from FAST+.

### Path B — `FASTTokenizer` wrapper in openpi (the one π0.5 uses)

In `openpi`, the `FASTTokenizer` class (`src/openpi/models/tokenizer.py`) wraps
the same `AutoProcessor` and adds the assembly with the VLM's language tokenizer:

```python
# openpi (abridged)
self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
#   fast_tokenizer_path default = "physical-intelligence/fast"

action_tokens = self._fast_tokenizer(actions[None])[0]                    # tokenize()
... = self._fast_tokenizer.decode([action_tokens.tolist()],              # extract_actions()
                                  time_horizon=action_horizon, action_dim=action_dim)[0]
```

Implication for the pilot: if you rely on the `openpi` stack for π0.5,
tokenization comes **free** with the repo (same Hub checkpoint underneath);
there is no need to bring an extra dependency or reimplement anything.

### How it works (why "frequency-space")

FAST compresses each action sequence via: (1) normalization, (2) **DCT
(Discrete Cosine Transform)** per action dimension, and (3) quantization that
rounds/discards insignificant coefficients — the same compression principle
as JPEG (image) or MP3 (audio). This allows training **autoregressive** VLAs
on high-frequency/dexterous actions, reaching the dexterity of
flow-matching/diffusion with ~5× less training time.

### Sources

- Universal model/tokenizer on the Hub (Apache 2.0), with usage example and `.fit()`:
  https://huggingface.co/physical-intelligence/fast
  · README: https://huggingface.co/physical-intelligence/fast/blob/main/README.md
- `FASTTokenizer` wrapper in openpi:
  https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py
- Paper "FAST: Efficient Action Tokenization for Vision-Language-Action Models":
  https://arxiv.org/abs/2501.09747 · PDF: https://www.pi.website/download/fast.pdf
- Research page (summary + figures): https://www.pi.website/research/fast
- LeRobot integration (π0-FAST policy, same Apache 2.0 license):
  https://huggingface.co/docs/lerobot/pi0fast
  · tokenizer mirror: https://huggingface.co/lerobot/fast-action-tokenizer

**Conclusion:** FAST **is reusable as a library** (`AutoProcessor` from the Hub or the
`FASTTokenizer` wrapper in openpi), Apache 2.0. **No
reimplementation is required** — only integration (normalization + chunk wiring). **No
BLOCKER line is added.**

## VRAM footprint (estimate)

> ⚠️ **NOT VERIFIED ON HARDWARE.** All figures in this section are an
> **analytical** estimate (parameters × precision + KV cache + runtime
> overhead). They do **not** come from a real GPU measurement (`nvidia-smi`, memory
> profiling, or observed OOM). They serve to size co-residence *a priori*;
> they must be confirmed on a real L4 before fixing any
> deployment decision.

### Objective

Do **π0.5 (pilot)** and a **Gemma-type Specialist** (judge/grounding,
int4-quantized) fit **co-resident** on a single **24 GB NVIDIA L4**?

### Starting assumptions

| Parameter | Assumed value | Assumption basis |
| --- | --- | --- |
| π0.5 architecture | PaliGemma-3B (SigLIP ViT ~0.4B + Gemma-2B) + *action expert* ~0.3B | Same family as π0/π0.5 in openpi; ≈ **3.3B parameters** total |
| π0.5 inference precision | **bf16** (2 bytes/param) | openpi/JAX runs in bf16 by default |
| Context (prefix tokens) | ~1–2k tokens (256 tok/image × ~2–3 cameras + language) | Typical multi-camera π0.5 config |
| `chunk_size` | 50 | Documented above (π0.5 default) |
| Specialist | Gemma **int4** (≈4B) VLM judge | Memory note: `check_done` = "Gemma int4" |
| Specialist precision | int4 (~0.5 byte/param effective + scales) | GPTQ/AWQ-style quantization |

### Analytical breakdown

**1. π0.5 weights (bf16)**
- 3.3B × 2 bytes ≈ **6.6 GB**
- (If run in fp32 it would double to ~13.2 GB → co-residence would no longer be
  comfortable; **using bf16/fp16 is a requirement**, not an optimization.)

**2. π0.5 KV cache** — *negligible*
- Gemma-2B uses MQA (≈1 KV head, head_dim 256, 18 layers).
- Per token: `2 (K,V) × 18 layers × 1 KV head × 256 × 2 bytes ≈ 18 KB/token`.
- With ~2k context tokens: **~0.04 GB**. Even with generous margins, < 0.1 GB.
- Reason: unlike a chat LLM, the context is short and fixed (image
  prefix + instruction); it does not grow with long decoding.

**3. Activations / inference workspace (batch 1)**
- Vision encoder (SigLIP) + backbone attention + *action expert* sampling
  (several denoising steps that reuse buffers): **~1–2 GB**.

**4. Runtime overhead (CUDA context + XLA/cuDNN)**
- CUDA context, kernels, cuDNN/cuBLAS workspaces of **two** co-resident
  frameworks (JAX for π0.5 + PyTorch for the Specialist): **~1–1.5 GB**.

**5. Specialist Gemma int4 (≈4B)**
- Weights: 4B × 0.5 byte ≈ 2.0 GB + int4 scheme scales/zeros (~+10–15%) ≈
  **~2.3 GB**.
- KV cache + activations + (if VLM) image tokens: **~0.5–0.7 GB**.
- Specialist subtotal: **~3.0 GB**.
- (If the Specialist were a Gemma-2B int4, it would drop to ~1.2 GB of weights → ~1.7 GB
  total.)

### Estimated total

| Component | VRAM (GB) |
| --- | --- |
| π0.5 weights (bf16) | 6.6 |
| π0.5 KV cache | ~0.05 |
| π0.5 activations/workspace | 1.5 |
| Runtime overhead (CUDA + XLA/cuDNN, 2 frameworks) | 1.5 |
| Specialist Gemma int4 (~4B) | 3.0 |
| **Total** | **≈ 12.6 GB** |

**On L4 (24 GB): ~12.6 GB used → ~11 GB of headroom (~52% free).**

Range with uncertainty (worst case of activations/overhead and 4B Specialist):
**~11–15 GB**. Co-residence **fits with a comfortable margin** in the estimate; the
risk is not the static *size* but **allocation management** (see below).

### Critical caveats for co-residence (operational risks, not size risks)

1. **JAX/XLA preallocation (risk #1).** By default JAX reserves **75–90%
   of VRAM** at startup, which would **kill the PyTorch Specialist by OOM**
   even if the real footprint fits. It is **mandatory** to set
   `XLA_PYTHON_CLIENT_PREALLOCATE=false` or
   `XLA_PYTHON_CLIENT_MEM_FRACTION=~0.5` to bound π0.5.
2. **Fragmentation between two allocators.** JAX and PyTorch maintain
   independent memory pools; the nominal sum may fit but fragmentation
   reduces the maximum contiguous allocatable block. Consider
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
3. **Precision.** The calculation assumes bf16 for π0.5; in fp32 it would not fit with headroom.
4. **Actual Specialist size unconfirmed.** "Gemma int4" does not fix the variant
   (2B vs 4B vs 3-4B VLM) nor the quantization scheme — it changes the subtotal by
   ±1.3 GB. Confirm the exact checkpoint.
5. **Transient peaks at load.** Loading checkpoints (int4 dequant,
   dtype conversion) may demand a temporary peak above the steady
   state; sequence the loads (do not load both models in parallel).

### How to verify it on hardware (pending)

- `nvidia-smi --query-gpu=memory.used --format=csv -l 1` during a co-resident
  rollout.
- `torch.cuda.max_memory_allocated()` for the Specialist and JAX's *memory profiler*
  (`jax.profiler` / `XLA_FLAGS=--xla_dump...`) for π0.5.
- Stress test: full rollout (chunk 50) with both models active and
  peak measurement, not average.

## Action space mapping

### Executive summary

The operational question is: **what is π0.5's `unnorm_key`?** Short answer:
π0.5 **does not expose an `unnorm_key` in the mission config**. The role that the
string `unnorm_key` plays in OpenVLA (selecting, at inference time, which
de-normalization statistics to apply to the action) is covered in `openpi` by a
**`norm_stats`** file that comes **packaged inside the checkpoint itself**
(under `assets/<asset_id>/`) and is selected by the **`repo_id`/`asset_id`** it
was trained with — not by a key the user passes in the config. For
`pi05_libero` that `asset_id` is `physical-intelligence/libero`.

→ Practical consequence for the pilot: where OpenVLA carries
`config.unnorm_key: libero_object`, **π0.5-LIBERO carries no
equivalent key** in the YAML. The correct normalization travels with the weights. The
analogous "knob" only exists in *training* (choosing the embodiment's
`repo_id`/`norm_stats`), not in *evaluation*.

### The exact analog of `unnorm_key`

| | OpenVLA (`predict_action`) | π0.5 / `openpi` |
| --- | --- | --- |
| What it normalizes | The output action (7-D), from normalized space → robot physical units | Input state **and** output action (chunk), via `Normalize`/`Unnormalize` |
| Where the statistics live | `dataset_statistics` inside the checkpoint's `config.json` (multiple datasets at once) | `norm_stats.json` under the checkpoint's `assets/<asset_id>/` (one per embodiment) |
| How it is selected | `unnorm_key` string passed at inference (`predict_action(..., unnorm_key=...)`) | Implicit: `asset_id = assets.asset_id or repo_id`, fixed at training |
| Mission surface | `config.unnorm_key` (required, must match suite/checkpoint) | **none** — the checkpoint is self-contained |
| Typical failure | wrong `unnorm_key` → actions at the wrong scale, erratic arm | (n/a in eval) wrong checkpoint/repo_id at training |

Reference in the repo on the OpenVLA side: `src/odyssey/runners/models/openvla.py:481`
(`unnorm_key = cfg.get("unnorm_key", "bridge_orig")`) and its passing to
`model.predict_action(..., unnorm_key=self._unnorm_key)`
(`openvla.py:674`).

### π0.5-LIBERO observation space

`openpi` normalizes the LIBERO observation to the same Franka Panda *embodiment* that
the rest of the stack consumes (robosuite/LIBERO), through `LiberoInputs`:

- **Images** (3 model inputs, PaliGemma/Pi0 family):
  - `base_0_rgb` ← third-person overhead view (`observation/image`, `agentview`).
  - `left_wrist_0_rgb` ← wrist camera (`observation/wrist_image`).
  - `right_wrist_0_rgb` ← **zero-filled** (LIBERO is single-arm; masked
    with `image_mask`).
  - Format normalized to `uint8 (H, W, C)`.
- **Proprioceptive state, 8-D** (identical to the robosuite/LIBERO Franka Panda
  convention):

  ```python
  state = np.concatenate([
      obs["robot0_eef_pos"],                    # (3) EEF position x,y,z
      quat2axisangle(obs["robot0_eef_quat"]),   # (3) EEF orientation: quat xyzw -> axis-angle
      obs["robot0_gripper_qpos"],               # (2) qpos of the two gripper phalanges
  ])                                            # -> 8-D
  ```

  This is exactly the same state vector that NVIDIA's LIBERO eval builds for
  GR00T (`quat_xyzw_to_axis_angle` in
  `src/odyssey/runners/evals/gr00t_transforms.py:172`) and that OpenVLA assumes
  implicitly. The key convention conversion is
  **quaternion `xyzw` (robosuite) → axis-angle** via `quat2axisangle`.
- **Padding to the model dimension:** the 8-D state is zero-padded up to
  the model's `action_dim` (**32** in the Pi0 family) via `pad_to_dim` /
  `PadStatesAndActions`, and then **normalized** with `norm_stats`. The language
  *prompt* is taken from the task (`prompt_from_task=True`).

### π0.5-LIBERO action space

The model emits a **chunk** of normalized actions of shape
`(action_horizon, 32)`; `LiberoOutputs` de-normalizes it (with the same
`norm_stats`) and **crops the 32 dims to LIBERO's 7-DoF**:

```python
# openpi LiberoOutputs
return {"actions": np.asarray(data["actions"][..., :7])}   # the rest is padding
```

The 7 dims are LIBERO/Franka Panda's native **OSC_POSE** action:
`[dx, dy, dz, droll, dpitch, dyaw, gripper]` (EEF pose delta + gripper), which is
applied **directly** to `env.step(action.tolist())`.

**Critical difference from OpenVLA — the gripper is NOT re-processed.** In OpenVLA
odyssey's eval applies `_libero_action` (`src/odyssey/runners/evals/libero.py:159`):
it re-scales the gripper `[0,1]→[-1,1]`, **binarizes** it and **inverts** it to match
LIBERO's sign. In π0.5/`openpi` there is **no** such fix-up in evaluation: the
checkpoint was trained on data already in LIBERO's gripper convention and the
`norm_stats` fix the scale, so the output is passed as-is to `env.step`
(same criterion as openpi's `examples/libero/main.py`, without `binarize`/`invert`).
It also contrasts with GR00T-N1.7-LIBERO, which **does** apply
`normalize_gripper_action` + `invert_gripper_action`
(`gr00t_transforms.py:218-228`) because its checkpoint emits the gripper in `[0,1]`.

> ⚠️ **Verify gripper polarity on the first GPU rollout.** Although the
> theory says "no fix-up", an inverted gripper sign is a classic silent
> failure (the arm moves and approaches but never grasps). It is the same footgun
> noted for GR00T; confirm it against the π0.5 server before fixing the config.

### Comparative table of the three pilots on LIBERO/Franka Panda

| | OpenVLA-7B | GR00T-N1.7-LIBERO | **π0.5-LIBERO** |
| --- | --- | --- | --- |
| Input state | (implicit) | 8-D: eef_pos + quat→axis-angle + 2 gripper qpos | 8-D: eef_pos + quat→axis-angle + 2 gripper qpos |
| Emitted action | 1 step 7-D | chunk 7-D (`action.*`, absolute) | **chunk** `(H, 32)` → 7-D crop |
| Action space | OSC_POSE 7-DoF | OSC_POSE 7-DoF | OSC_POSE 7-DoF `[dx,dy,dz,droll,dpitch,dyaw,g]` |
| Norm-stats selection | `unnorm_key` in config | (baked in checkpoint) | `norm_stats`/`asset_id` baked (`physical-intelligence/libero`) |
| Gripper fix-up in eval | binarize + invert (`_libero_action`) | normalize + invert | **none** (baked in data + norm_stats) |
| `action_horizon` (chunk) | 1 | 40 (`ACTION_HORIZON`) | 10 (`pi05_libero`), 50 π0.5 default |

Note on the horizon: π0.5's general default is `chunk_size = 50` (see
license/weights section), but openpi's `pi05_libero` `TrainConfig` uses
`action_horizon=10`. Relevant to the pilot's *chunking* gateway: the number
of actions consumed per pilot query depends on the specific checkpoint, not
on a global constant.

### Implications for odyssey integration

1. **Do not add `unnorm_key` to the π0.5 YAML.** The `pi05_libero` checkpoint is
   self-contained; de-normalization travels in `assets/`. Any normalization
   key in a π0.5 mission config would be inert or misleading.
2. **The π0.5 action adapter crops to 7-D and applies without gripper fix-up**
   — unlike OpenVLA's `VLARuntime`/`_libero_action`. The
   `ChunkPilotAdapter` (see multiagent note) must treat π0.5 as
   *chunk-emitting* and **not** reintroduce gripper binarization/inversion unless
   the GPU smoke demonstrates inverted polarity.
3. **The state convention is the same Franka Panda** (8-D, quat xyzw→axis-angle,
   2 gripper qpos), so the observation construction can be shared with the
   GR00T-LIBERO path (`build_gr00t_libero_obs`) by reordering to the keys that
   `openpi` expects (`observation/image`, `observation/wrist_image`,
   `observation/state`), without re-deriving the kinematics.

### Sources

- `openpi` — LIBERO transforms (`LiberoInputs`/`LiberoOutputs`, 8-D state, 7-D crop):
  https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/libero_policy.py
- `openpi` — LIBERO eval (state construction from robosuite, `env.step` without
  gripper inversion):
  https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py
- `openpi` — training config (`LeRobotLiberoDataConfig`,
  `asset_id = assets.asset_id or repo_id`, `norm_stats`, `pi05_libero`):
  https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py
- Franka Panda / gripper convention in the repo: `src/odyssey/runners/evals/libero.py:159`
  (`_libero_action`), `src/odyssey/runners/evals/gr00t_transforms.py:172,218-270`
  (LIBERO_PANDA state and action), `src/odyssey/runners/models/openvla.py:481,674`
  (`unnorm_key`).

**Conclusion:** the analog of `unnorm_key` in π0.5 is the
`asset_id`/`norm_stats` pair **packaged in the checkpoint** — not a mission
key. The observation (8-D Franka Panda: eef_pos + quat→axis-angle + 2 gripper
qpos) and the action (7-DoF OSC_POSE, cropping the model's 32 dims) match the
LIBERO/robosuite convention that OpenVLA and GR00T already use; the only
operational difference is that π0.5 **requires no gripper fix-up in evaluation** (pending
polarity confirmation on GPU). **No BLOCKER line is added.**
