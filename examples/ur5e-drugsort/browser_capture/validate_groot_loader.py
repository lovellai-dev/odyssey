#!/usr/bin/env python3
"""Validate the browser-frame dataset loads with the UPSTREAM GR00T LeRobot loader.

Mirrors the RUNBOOK check: importlib-load ur5e_config.py (registers NEW_EMBODIMENT),
then build a ShardedSingleStepDataset over the browser dataset. If it builds shards,
the browser dataset is byte-compatible with what
``gr00t.experiment.launch_finetune`` consumes — i.e. the Three.js frames drop into
the SAME training path as the headless MuJoCo frames.

Run in the Isaac-GR00T model venv (needs `gr00t`), e.g.::

    env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        $ISAAC_GR00T_REPO_PATH/.venv/bin/python validate_groot_loader.py \
        --dataset ./dataset_browser
"""
import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="dataset_browser")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "ur5e_config.py"),
                    help="ur5e_config.py that registers the NEW_EMBODIMENT modality config")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("ur5e_config", args.config)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # registers NEW_EMBODIMENT modality config

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset

    ds = ShardedSingleStepDataset(
        dataset_path=args.dataset,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        modality_configs=MODALITY_CONFIGS[EmbodimentTag.NEW_EMBODIMENT.value],
        shard_size=256, seed=0,
    )
    n_shards = len(ds)
    print(f"OK — GR00T loader built {n_shards} shards from {args.dataset}")
    return 0 if n_shards > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
