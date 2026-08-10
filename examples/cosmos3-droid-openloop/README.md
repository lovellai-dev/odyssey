# Cosmos 3 open-loop GT eval (no simulator)

The cheapest **in-distribution** check of the Cosmos 3 serving + decode path:
real DROID frames in → predicted action chunk out → per-dim MAE + gripper
agreement against the actions the robot actually took. Runs through
`evaluation_type: custom` (metric-only — no fabricated success_rate).

Use it to de-risk action semantics (dims, scales, gripper) before RoboLab or
a LIBERO SFT: a policy far from the `zero`/`hold` baselines on joint dims
points at a decode/prompt-format problem, not at the model.

## Data: the tiny cookbook episode (1 episode, ~300 MB)

```bash
BASE=https://raw.githubusercontent.com/NVIDIA/cosmos/main/cookbooks/cosmos3/generator/action/assets/droid_lerobot_example
mkdir -p ~/datasets/droid_lerobot_example/{meta/episodes/chunk-000,data/chunk-000,videos/observation.image.exterior_image_1_left/chunk-000,videos/observation.image.wrist_image_left/chunk-000}
cd ~/datasets/droid_lerobot_example
curl -sL -o meta/info.json      "$BASE/meta/info.json"
curl -sL -o meta/tasks.parquet  "$BASE/meta/tasks.parquet"
curl -sL -o meta/episodes/chunk-000/file-000.parquet "$BASE/meta/episodes/chunk-000/file-000.parquet"
curl -sL -o data/chunk-000/file-000.parquet          "$BASE/data/chunk-000/file-000.parquet"
curl -sL -o videos/observation.image.exterior_image_1_left/chunk-000/file-000.mp4 "$BASE/videos/observation.image.exterior_image_1_left/chunk-000/file-000.mp4"
curl -sL -o videos/observation.image.wrist_image_left/chunk-000/file-000.mp4      "$BASE/videos/observation.image.wrist_image_left/chunk-000/file-000.mp4"
```

Any LeRobot v3 dataset with DROID-style features works
(`action.joint_position` [7] + `action.gripper_position` [1] + exterior/wrist
videos) — e.g. a slice of `nvidia/Cosmos3-DROID`.

## Run

Serve the policy (see `../quickstart-cosmos3/setup.sh`, including the
gated-guardrail note), add the script deps to the client venv, then:

```bash
uv pip install --python env_pilot_cosmos3/bin/python pandas pyarrow
# edit mission.yaml: dataset_root, host/port, eval_python
odyssey run examples/cosmos3-droid-openloop/mission.yaml
```

## Reading the numbers

- `mae` — mean |pred − GT| over sampled chunks (joint radians + gripper).
- `baseline_zero_mae` / `baseline_hold_mae` — predicting zeros / persisting
  the last action. In-distribution the policy should clearly beat `zero`;
  `hold` is a very strong baseline at 15–20 fps (positions barely move in one
  chunk), so don't over-read it.
- `mae_per_dim` — per action dim, in dataset units.
- `gripper_agreement` — binarized (threshold 0.5) match rate (null when the
  server's row width doesn't cover the gripper column).
- `pred_action_width` — the server's embodiment-defined row width; compared
  over the common prefix with the dataset's `action_dim`.

## Measured (H100, 2026-08-10, Edge-Policy-DROID served)

| dataset (domain) | mae | zero | hold | width | note |
|---|---|---|---|---|---|
| DROID cookbook ep. (`droid_lerobot`) | 0.385 | 0.946 | 0.070 | 7/8 | in-dist: beats zero 2.5×; empty task annotation → generic prompt |
| UR drugsort ep. (`robomind-ur`) | 1.409 | 1.333 | 0.033 | 7/7 | OOD for this checkpoint (DROID-SFT): ≈noise, as expected — needs UR SFT |

`--format-prompt-as-json True` on the server (Edge's training format) shifted
DROID mae by <1% — prompt format is not the driver at this granularity.

**Embodiment support (from the server's own registry, seen live)**: valid
`domain_name` values include `droid_lerobot`, `libero`, `robomind-ur`,
`robomind-franka`, `bridge_orig_lerobot`, `fractal`, `agibotworld`, `umi`, …
— `robomind-ur` accepted our UR requests and answered the correct 7-D
(6 joints + gripper) rows, so the UR wiring is sound; performance on a
specific UR cell needs SFT on that cell's data.
