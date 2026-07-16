#!/usr/bin/env python3
"""Relabel browser DAgger rollouts with the IK expert, then aggregate onto a base.

This is the Python half of the **DAgger-on-browser** loop. ``dagger_rollout_browser.js``
rolls the CURRENT GR00T out *in the browser* and records, per visited (often
failing) state, the Three.js exterior+wrist frames + proprio + ground-truth
vial/pocket/pinch geometry. Here we turn each visited state into a corrective
``(observation -> expert absolute joint target)`` training pair using the SAME
adaptive damped-least-squares IK teacher + latching phase machine as the headless
``scripts/dagger_ur5e_drugsort.py`` (a perfect corrective oracle in sim), and
write them through the EXISTING ``odyssey.datasets.LeRobotDatasetWriter`` +
``ur5e_drugsort.embodiment`` — byte-identical schema to the demonstration set.
The ONLY difference from a demo episode is the pixels (deployment Three.js frames)
and the labels (expert relabel of the policy's OWN visited states).

Two sub-commands (driven by ``run_dagger_browser.sh``):

* ``relabel`` — raw ``<raw>/epNNN`` -> a LeRobot v2.1 relabel dataset.
* ``merge``   — aggregate a relabel dataset onto a base LeRobot dataset.

Only ``numpy``/``pyarrow``/``PIL``/``mujoco`` + the odyssey package are needed
(NOT ``msgpack``/``pyzmq``): the policy was already rolled out in the browser, so
this runs entirely in the lightweight ``.venv-ur5e`` — it does not import the ZMQ
policy client that ``scripts/dagger_ur5e_drugsort.py`` pulls at module load. The
``ExpertRelabeler`` and ``merge`` logic below are direct ports of that module.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # only the IK solver's scratch MjData; no rendering here
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling precompute_plans

from odyssey.datasets import LeRobotDatasetWriter  # noqa: E402
from odyssey.embodiments.ur5e_drugsort import embodiment as emb  # noqa: E402
from odyssey.embodiments.ur5e_drugsort.ik import (  # noqa: E402
    GRASP_PINCH_BELOW_VIAL,
    PLACE_PINCH_BELOW_POCKET,
    DampedLeastSquaresIK,
    build_adaptive_phases,
)
from precompute_plans import build_index, DEFAULT_XML  # noqa: E402

CHUNKS_SIZE = emb.CHUNKS_SIZE


# ---------------------------------------------------------------------------
# The corrective expert (verbatim port of scripts/dagger_ur5e_drugsort.py
# ExpertRelabeler): visited state geometry -> expert absolute joint target.
# ---------------------------------------------------------------------------
class ExpertRelabeler:
    """State -> expert absolute joint target for the drug-sort pick-place.

    Given the per-episode IK anchor targets and the live sim geometry, return the
    correct absolute joint target for the arm's current physical situation. The
    phase index only ever advances (latches), so once committed to the descent the
    label keeps pointing at the deep ``descend`` target even if the policy stalls
    or drifts short — exactly the recovery signal behaviour-cloning lacked.
    """

    def __init__(self, targets, grasp_pt, place_pt, z0: float) -> None:
        self.q = targets
        self.grasp_pt = np.asarray(grasp_pt, dtype=np.float64).reshape(3)
        self.place_pt = np.asarray(place_pt, dtype=np.float64).reshape(3)
        self.z0 = float(z0)
        self.seq = (
            ("approach", 0.0),   # 0: move over the vial
            ("descend", 0.0),    # 1: lower onto the vial (recovery target)
            ("descend", 1.0),    # 2: close on the vial at grasp depth
            ("lift", 1.0),       # 3: raise the grasped vial
            ("transport", 1.0),  # 4: carry over the pocket
            ("lower", 1.0),      # 5: lower into the nest
            ("lower", 0.0),      # 6: release
        )
        self.i = 0

    @staticmethod
    def _horiz(a, b) -> float:
        return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))

    @staticmethod
    def _dist(a, b) -> float:
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))

    def _advance(self, i: int, pinch, vial, grip: float) -> bool:
        if i == 0:
            return self._horiz(pinch, self.grasp_pt) < 0.06
        if i == 1:
            return self._dist(pinch, self.grasp_pt) < 0.03
        if i == 2:
            return grip > 0.4
        if i == 3:
            return (float(vial[2]) - self.z0) > 0.03 and grip > 0.3
        if i == 4:
            return self._horiz(pinch, self.place_pt) < 0.05
        if i == 5:
            return self._dist(pinch, self.place_pt) < 0.03
        return False

    def step(self, pinch, vial, grip: float):
        while self.i < len(self.seq) - 1 and self._advance(self.i, pinch, vial, grip):
            self.i += 1
        name, g = self.seq[self.i]
        return self.q[name], g


def load_frames(d: Path):
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in sorted(d.glob("f*.png"))]


# ---------------------------------------------------------------------------
# relabel: raw browser rollouts -> LeRobot relabel dataset
# ---------------------------------------------------------------------------
def cmd_relabel(args) -> int:
    import mujoco as mj

    xml = Path(args.xml)
    if not xml.is_file():
        print(f"ERROR: MJCF not found: {xml}", file=sys.stderr)
        return 2
    model = mj.MjModel.from_xml_path(str(xml))
    idx = build_index(mj, model)
    ik = DampedLeastSquaresIK(
        model=model, site_id=idx["pinch_site"],
        arm_qadr=tuple(idx["arm_qadr"]), arm_dofadr=tuple(idx["arm_dofadr"]),
    )

    raw, out = Path(args.raw), Path(args.out)
    ep_dirs = sorted([p for p in raw.iterdir() if p.is_dir() and p.name.startswith("ep")])
    if not ep_dirs:
        print(f"ERROR: no raw episodes under {raw}", file=sys.stderr)
        return 2
    if out.exists():
        shutil.rmtree(out)

    writer = LeRobotDatasetWriter(
        out, modality=emb.modality_json(), info_builder=emb.info_json,
        video_keys=emb.VIDEO_ORIGINAL_KEYS, fps=emb.DEFAULT_FPS,
        width=emb.DEFAULT_WIDTH, height=emb.DEFAULT_HEIGHT,
    )

    n_used = n_skip = roll_succ = roll_lift = 0
    for d in ep_dirs:
        meta = json.loads((d / "meta.json").read_text())
        vial_xyz = np.asarray(meta["vial_xyz"], dtype=np.float64)
        pocket_xyz = np.asarray(meta["pocket_xyz"], dtype=np.float64)
        home_q = tuple(float(x) for x in meta["home_q"])

        plan = build_adaptive_phases(ik, vial_xyz=vial_xyz, pocket_xyz=pocket_xyz, home_q=home_q)
        targets = {name: (plan.solves[name].q if name in plan.solves else home_q)
                   for name in ("approach", "descend", "lift", "transport", "lower")}
        grasp_pt = (float(vial_xyz[0]), float(vial_xyz[1]), float(vial_xyz[2] - GRASP_PINCH_BELOW_VIAL))
        place_pt = (float(pocket_xyz[0]), float(pocket_xyz[1]), float(pocket_xyz[2] - PLACE_PINCH_BELOW_POCKET))
        relab = ExpertRelabeler(targets, grasp_pt, place_pt, z0=float(vial_xyz[2]))

        # ExpertRelabeler is stateful/latching — step ONCE per frame, in order.
        gt = meta.get("gt", [])
        actions = []
        for f in gt:
            tq, tg = relab.step(f["pinch"], f["vial"], float(f["grip"]))
            actions.append([*tq, tg])

        states = np.asarray(meta["states"], dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        ext = load_frames(d / "exterior")
        wrist = load_frames(d / "wrist")
        n = min(states.shape[0], actions.shape[0], len(ext), len(wrist))
        if n < args.min_frames:
            n_skip += 1
            print(f"[relabel] {d.name}: SKIP (n={n} < min_frames={args.min_frames}) err={meta.get('error')}")
            continue
        states, actions, ext, wrist = states[:n], actions[:n], ext[:n], wrist[:n]
        assert states.shape[1] == emb.STATE_DIM and actions.shape[1] == emb.ACTION_DIM, \
            f"{d.name}: dim mismatch state={states.shape} action={actions.shape}"
        assert ext[0].shape == (emb.DEFAULT_HEIGHT, emb.DEFAULT_WIDTH, 3), f"{d.name}: frame shape {ext[0].shape}"

        writer.add_episode(states=states, actions=actions,
                           frames={"exterior": ext, "wrist": wrist}, task=args.instruction)
        # GT sidecar in the augment_state_grasp_target.py schema (meta/gt/
        # episode_NNNNNN.json with per-frame vial_pose) so the 7-D relabel
        # dataset can be lifted to the 10-D conditioned state (Stage C).
        gt_dir = out / "meta" / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        (gt_dir / f"episode_{n_used:06d}.json").write_text(json.dumps({
            "frames": [{"vial_pose": list(map(float, f["vial"]))} for f in gt[:n]],
        }))
        n_used += 1
        roll_succ += int(bool(meta.get("success")))
        roll_lift += int(bool(meta.get("lifted")))
        print(f"[relabel] {d.name}: T={n} expert_phase_reached={relab.i} "
              f"rollout_success={meta.get('success')} lifted={meta.get('lifted')} "
              f"lift={float(meta.get('lift_height', 0.0))*100:.1f}cm")

    if n_used == 0:
        print("ERROR: no usable relabel episodes", file=sys.stderr)
        return 3
    info = writer.finalize()
    summary = {
        "relabel_episodes": info["total_episodes"], "relabel_frames": info["total_frames"],
        "skipped": n_skip,
        "rollout_success": roll_succ, "rollout_lifted": roll_lift, "rollout_episodes": n_used,
    }
    (out / "dagger_relabel.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("=" * 72)
    print(f"[relabel] wrote {info['total_episodes']} episodes / {info['total_frames']} frames -> {out}")
    print(f"[relabel] rollout policy (this ckpt) success={roll_succ}/{n_used} lifted={roll_lift}/{n_used} skipped={n_skip}")
    print("=" * 72)
    return 0


# ---------------------------------------------------------------------------
# merge: aggregate a relabel dataset onto a base dataset (verbatim port of
# scripts/dagger_ur5e_drugsort.py cmd_merge; numpy/pyarrow only).
# ---------------------------------------------------------------------------
def _stats_for(matrix: np.ndarray):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return {
        "mean": matrix.mean(axis=0).tolist(), "std": matrix.std(axis=0).tolist(),
        "min": matrix.min(axis=0).tolist(), "max": matrix.max(axis=0).tolist(),
        "q01": np.quantile(matrix, 0.01, axis=0).tolist(),
        "q99": np.quantile(matrix, 0.99, axis=0).tolist(),
    }


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def cmd_merge(args) -> int:
    base, add, out = Path(args.base), Path(args.add), Path(args.out)
    for d in (base, add):
        if not (d / "meta" / "info.json").is_file():
            print(f"ERROR: not a LeRobot dataset: {d}", file=sys.stderr)
            return 2

    binfo = json.load((base / "meta" / "info.json").open())
    fps = int(binfo["fps"])
    feat = binfo["features"]["observation.images.exterior"]["shape"]
    height, width = int(feat[0]), int(feat[1])

    base_eps = _read_jsonl(base / "meta" / "episodes.jsonl")
    add_eps = _read_jsonl(add / "meta" / "episodes.jsonl")
    nb = len(base_eps)
    if nb + len(add_eps) > CHUNKS_SIZE:
        print(f"ERROR: merged episode count {nb + len(add_eps)} exceeds single chunk "
              f"size {CHUNKS_SIZE}; multi-chunk merge not implemented", file=sys.stderr)
        return 4

    base_tasks = _read_jsonl(base / "meta" / "tasks.jsonl")
    task_to_idx = {t["task"]: int(t["task_index"]) for t in base_tasks}
    add_tasks = {int(t["task_index"]): t["task"] for t in _read_jsonl(add / "meta" / "tasks.jsonl")}

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out)

    base_frames = sum(int(e["length"]) for e in base_eps)
    new_eps = list(base_eps)
    global_off = base_frames
    for j, e in enumerate(add_eps):
        new_i = nb + j
        old_i = int(e["episode_index"])
        src_pq = add / "data" / "chunk-000" / f"episode_{old_i:06d}.parquet"
        tbl = pq.read_table(src_pq)
        length = tbl.num_rows
        old_task_idx = int(tbl.column("task_index")[0].as_py())
        old_task = add_tasks.get(old_task_idx, e["tasks"][0])
        if old_task not in task_to_idx:
            task_to_idx[old_task] = len(task_to_idx)
        tidx = task_to_idx[old_task]
        sch = tbl.schema
        tbl = tbl.set_column(sch.get_field_index("episode_index"), "episode_index",
                             pa.array([new_i] * length, pa.int64()))
        tbl = tbl.set_column(sch.get_field_index("task_index"), "task_index",
                             pa.array([tidx] * length, pa.int64()))
        tbl = tbl.set_column(sch.get_field_index("index"), "index",
                             pa.array(list(range(global_off, global_off + length)), pa.int64()))
        global_off += length
        pq.write_table(tbl, out / "data" / "chunk-000" / f"episode_{new_i:06d}.parquet")
        for _key, orig in emb.VIDEO_ORIGINAL_KEYS.items():
            src_v = add / "videos" / "chunk-000" / orig / f"episode_{old_i:06d}.mp4"
            dst_v = out / "videos" / "chunk-000" / orig / f"episode_{new_i:06d}.mp4"
            dst_v.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_v, dst_v)
        new_eps.append({"episode_index": new_i, "tasks": [old_task], "length": length})

    total_ep = len(new_eps)
    total_frames = sum(int(e["length"]) for e in new_eps)
    total_tasks = len(task_to_idx)

    info = emb.info_json(total_episodes=total_ep, total_frames=total_frames,
                         total_tasks=total_tasks, fps=fps, width=width, height=height)
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    _write_jsonl(out / "meta" / "episodes.jsonl", new_eps)
    _write_jsonl(out / "meta" / "tasks.jsonl",
                 [{"task_index": i, "task": t} for t, i in sorted(task_to_idx.items(), key=lambda kv: kv[1])])

    all_state, all_action, all_ts = [], [], []
    for e in new_eps:
        tbl = pq.read_table(out / "data" / "chunk-000" / f"episode_{int(e['episode_index']):06d}.parquet",
                            columns=["observation.state", "action", "timestamp"])
        all_state.append(np.asarray(tbl.column("observation.state").to_pylist(), dtype=np.float64))
        all_action.append(np.asarray(tbl.column("action").to_pylist(), dtype=np.float64))
        all_ts.append(np.asarray(tbl.column("timestamp").to_pylist(), dtype=np.float64))
    stats = {
        "action": _stats_for(np.concatenate(all_action, axis=0)),
        "observation.state": _stats_for(np.concatenate(all_state, axis=0)),
        "timestamp": _stats_for(np.concatenate(all_ts, axis=0).reshape(-1, 1)),
    }
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=4) + "\n")
    print(f"[merge] {base.name} ({nb} eps) + {add.name} ({len(add_eps)} eps) -> {out} "
          f"({total_ep} eps / {total_frames} frames / {total_tasks} tasks)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    rl = sub.add_parser("relabel", help="raw browser rollouts -> LeRobot relabel dataset")
    rl.add_argument("--xml", default=DEFAULT_XML)
    rl.add_argument("--raw", required=True)
    rl.add_argument("--out", required=True)
    rl.add_argument("--min-frames", type=int, default=10)
    rl.add_argument("--instruction", default=emb.DEFAULT_INSTRUCTION)
    rl.set_defaults(func=cmd_relabel)

    mg = sub.add_parser("merge", help="aggregate a relabel dataset onto a base dataset")
    mg.add_argument("--base", required=True)
    mg.add_argument("--add", required=True)
    mg.add_argument("--out", required=True)
    mg.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
