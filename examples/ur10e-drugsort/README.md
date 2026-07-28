# UR10e Drug-Sort — replication of the UR5e AseptiPack program

Parallel **UR10e** replica of `examples/ur5e-drugsort` (which stays untouched — its
datasets, checkpoints, and eval history are the UR5e lineage). Everything runs
through the *same* pipeline scripts in `../ur5e-drugsort/browser_capture/`; the
robot is selected by data, not by code forks:

| Piece | UR5e | UR10e |
|---|---|---|
| Cell MJCF | `lai-agent-multiagent/src/embodiments/urdf/aseptipack_description/aseptipack.xml` | `.../aseptipack_ur10e_description/aseptipack.xml` |
| Playground page | `robot-playground.html?demo=drugsorting` | `...&arm=ur10e` |
| Driver scripts (`browser_harness.js`, `dagger_rollout_browser.js`, `eval_browser_groot.js`) | default | `ARM=ur10e` env |
| Plans precompute | default XML | `ASEPTIPACK_XML=<ur10e xml>` |
| GR00T modality config (`vm_train_dagger.sh`) | default | `MODALITY_CONFIG=examples/UR10e_DrugSort/ur10e_config.py` |

The UR10e cell was built by grafting the MuJoCo Menagerie `universal_robots_ur10e`
model (defaults, body tree, inertials, collision capsules, OBJ meshes) into the
deployment cell. Joint, body, actuator, and site **names are identical** to the
UR5e cell, and the Robotiq 2F-85 mounts at the same flange offset (y=0.1), so the
IK expert, browser harness, GR00T bridge, and observer stack transfer unchanged.

Verified (2026-07-28, on the H100 VM):

- cell loads, home pose holds (0.009 rad drift), vial stable on worktop;
- `precompute_plans.py` converges sub-millimetre on all episodes (adaptive);
- headless closed-loop expert seats the vial at parity with the UR5e cell under
  identical deployment semantics (`~/exec_plans_headless.py`).

**Important:** all UR5e datasets/checkpoints are embodiment-specific. The UR10e
lineage needs its own capture → observer training → GR00T finetune → eval; none
of the UR5e weights transfer.

## Quickstart (any laptop, no GPU, no VM)

```bash
python -m venv .venv && . .venv/bin/activate
pip install "mujoco==3.10.0" numpy jupyter
jupyter lab UR10e_DrugSort_Pipeline.ipynb   # open from THIS directory, then Run All
```

The notebook opens with a **visual tour** — a pipeline diagram plus real videos
from the UR10e verification capture (`media/`): an expert episode seen from both
policy cameras, and an augmentation before/after. A Tier A cell also renders
your own physics rollout video locally when offscreen GL is available.

The notebook's Tier A smoke suite (≈3 min) proves the codebase on your machine:
scene loads and holds, the adaptive-IK expert plans converge, and the closed-loop
expert picks the vial and seats it in the nest (2/2 with the pinned mujoco).
Heavy VM stages are guarded behind `RUN_HEAVY = False`, so Run All is safe —
they print their commands instead of executing. mujoco is pinned to 3.10.0 to
match the VM training env (3.11 changes contact behaviour at the pocket well).

`aseptipack_ur10e_description/` is vendored here (scene + meshes, ~35 MB) so the
example is self-contained; the UR10e arm model and meshes come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`universal_robots_ur10e`, BSD-3-Clause). The canonical deployment copy lives in
`lai-agent-multiagent/src/embodiments/urdf/aseptipack_ur10e_description/` —
change that one first, then re-vendor.

The full stage-by-stage runbook — data collection, augmentation, training,
finetuning, evaluation, and the multi-agent serving stack — is the notebook:

**`UR10e_DrugSort_Pipeline.ipynb`** (shareable; start there).
