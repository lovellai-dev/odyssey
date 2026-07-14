#!/usr/bin/env python3
"""DAgger (Dataset Aggregation) for the UR5e / Robotiq drug-sorting GR00T policy.

The closed-loop GR00T fine-tune reliably *approaches* the vial and fires the
gripper but closes ~7 cm short on the final descent (iteration-3 eval: 0/20,
every episode ``lifted=False``). That failure is an **off-distribution** one:
behaviour-cloning only ever saw the *expert's* clean descent, so the policy was
never taught how to recover from the mid-descent states it actually drifts into
at deploy time. DAgger fixes exactly this: roll out the current policy, and at
every state it *actually visits* ask the expert what to do, then fold those
corrective ``(observation -> expert action)`` pairs back into training.

The expert here is the adaptive damped-least-squares IK teacher
(:mod:`odyssey.embodiments.ur5e_drugsort.ik`), which scores 64/64 in open-loop
sim — a perfect corrective oracle. Because DAgger runs *in* the MuJoCo sim we can
read the ground-truth vial / pocket / pinch poses, so the expert label at any
visited state is well defined even far off the demonstration manifold.

This module has two sub-commands (the DAgger data step), both meant to be driven
by ``examples/ur5e-drugsort/run_dagger_pipeline.sh``:

* ``rollout`` — serve a GR00T checkpoint (already running on ZMQ), roll it out
  over many domain-randomised vial poses in the SAME headless MuJoCo cell the
  eval harness uses (:mod:`scripts.eval_ur5e_drugsort_groot`), and at **every
  control tick** record ``(exterior, wrist, state) -> expert absolute joint
  target``. The expert action is chosen by a small latching, purely geometric
  phase machine over the per-episode IK solution (approach -> descend -> close
  -> lift -> transport -> lower -> release): whatever off-distribution state the
  policy drifts into, the label is the correct **absolute joint target** for
  where the arm physically is — e.g. while the policy hovers short of the vial
  the label stays the deep ``descend`` target, teaching it to push through. The
  visited states are written as one LeRobot v2.1 episode each, byte-compatible
  with the demonstration set (same ``LeRobotDatasetWriter`` schema).

* ``merge`` — aggregate a relabel dataset onto a base LeRobot dataset (the
  original demos, or a previous aggregation) into a fresh dataset directory,
  re-indexing episodes/frames, unioning tasks and recomputing normalisation
  stats. The result feeds ``gr00t.experiment.launch_finetune`` unchanged.

Only ``numpy``/``mujoco``/``pyarrow``/``pyzmq``/``msgpack``/``msgpack_numpy``
are needed (no torch): the policy runs in a separate served process, queried over
ZMQ exactly as the eval harness does.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# Offscreen GL before mujoco is imported (EGL on a headless GPU host).
os.environ.setdefault("MUJOCO_GL", "egl")

# Make the package + sibling harnesses importable from a source checkout.
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS.parents[0] / "src"))
sys.path.insert(0, str(_SCRIPTS))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from odyssey.datasets import LeRobotDatasetWriter  # noqa: E402
from odyssey.embodiments.ur5e_drugsort import embodiment as emb  # noqa: E402
from odyssey.embodiments.ur5e_drugsort.ik import (  # noqa: E402
    APPROACH_CLEARANCE,
    GRASP_PINCH_BELOW_VIAL,
    PLACE_PINCH_BELOW_POCKET,
    DampedLeastSquaresIK,
    build_adaptive_phases,
)
from odyssey.embodiments.ur5e_drugsort.policy import (  # noqa: E402
    GRIP_CLOSE,
    GRIP_OPEN,
    SETTLE_STEPS,
)

# Reuse the eval harness's served-policy client + obs builder, and the data-gen
# harness's MuJoCo indexing / arm-collision freeing / grip normalisation so the
# DAgger scene is byte-for-byte the one the demos + eval run in.
from eval_ur5e_drugsort_groot import GrootPolicyClient, build_obs  # noqa: E402
from gen_ur5e_drugsort_demos import (  # noqa: E402
    GRIP_DRIVER_RANGE,
    _build_index,
    _free_arm_collision,
)

VIDEO_KEY_TO_CAMERA = dict(emb.VIDEO_KEY_TO_CAMERA)
CHUNKS_SIZE = emb.CHUNKS_SIZE  # 1000 — single-chunk guard for the merge


# ---------------------------------------------------------------------------
# The corrective expert: a latching geometric phase machine over the IK solution
# ---------------------------------------------------------------------------
class ExpertRelabeler:
    """State -> expert **absolute joint target** for the drug-sort pick-place.

    Given the per-episode IK solution (the absolute joint targets that put the
    Robotiq pinch over the *actual* randomised vial, above it, at the pocket,
    etc.) and the live sim geometry, return the correct absolute joint target for
    the arm's current physical situation. The phase index only ever advances
    (latches), so once the policy is committed to the descent the label keeps
    pointing at the deep ``descend`` target even if the policy stalls or drifts —
    exactly the recovery signal behaviour-cloning lacked.

    Sequence (name, grip): approach(open) -> descend(open) -> descend(close,
    "grasp") -> lift(close) -> transport(close) -> lower(close) -> lower(open,
    "release"). ``grip`` is the normalised gripper command in {0.0 open, 1.0
    closed}, matching the recorded action's 7th channel.
    """

    def __init__(self, targets: dict[str, tuple[float, ...]],
                 grasp_pt, place_pt, z0: float) -> None:
        self.q = targets
        self.grasp_pt = np.asarray(grasp_pt, dtype=np.float64).reshape(3)
        self.place_pt = np.asarray(place_pt, dtype=np.float64).reshape(3)
        self.z0 = float(z0)
        # (target-name, normalised grip command)
        self.seq: tuple[tuple[str, float], ...] = (
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
        """Should latched phase ``i`` advance to ``i+1`` given the geometry?"""
        if i == 0:   # over the vial -> begin descent
            return self._horiz(pinch, self.grasp_pt) < 0.06
        if i == 1:   # reached grasp depth -> close
            return self._dist(pinch, self.grasp_pt) < 0.03
        if i == 2:   # gripper has closed -> lift
            return grip > 0.4
        if i == 3:   # vial lifted clear -> transport
            return (float(vial[2]) - self.z0) > 0.03 and grip > 0.3
        if i == 4:   # over the pocket -> lower
            return self._horiz(pinch, self.place_pt) < 0.05
        if i == 5:   # seated -> release
            return self._dist(pinch, self.place_pt) < 0.03
        return False

    def step(self, pinch, vial, grip: float) -> tuple[tuple[float, ...], float]:
        while self.i < len(self.seq) - 1 and self._advance(self.i, pinch, vial, grip):
            self.i += 1
        name, g = self.seq[self.i]
        return self.q[name], g


def _randomise_vial(mj, model, data, idx, *, area_x, area_y, yaw_jitter, rng):
    """Domain-randomise the vial free-joint pose (matches gen / eval)."""
    vq = idx["vial_qadr"]
    if area_x > 0:
        data.qpos[vq + 0] += float(rng.uniform(-area_x, area_x))
    if area_y > 0:
        data.qpos[vq + 1] += float(rng.uniform(-area_y, area_y))
    if yaw_jitter > 0:
        yaw = float(rng.uniform(-yaw_jitter, yaw_jitter))
        data.qpos[vq + 3] = math.cos(yaw / 2.0)
        data.qpos[vq + 4] = 0.0
        data.qpos[vq + 5] = 0.0
        data.qpos[vq + 6] = math.sin(yaw / 2.0)
    mj.mj_forward(model, data)


def _build_expert(ik, mj, model, data, idx, home_q):
    """Solve the adaptive IK plan for the current vial/pocket and wrap the
    corrective phase machine around its absolute joint targets."""
    vial_xyz = np.array(data.xpos[idx["vial_body"]], dtype=np.float64)
    pocket_xyz = np.array(data.site_xpos[idx["pocket_site"]], dtype=np.float64)
    plan = build_adaptive_phases(ik, vial_xyz=vial_xyz, pocket_xyz=pocket_xyz,
                                 home_q=home_q)
    s = plan.solves
    # Grip-only phases (close/release) inherit descend/lower, so only the five
    # Cartesian anchors are solved; fall back to the scripted seed if an anchor
    # failed to solve (never observed in practice).
    targets = {name: (s[name].q if name in s else home_q)
               for name in ("approach", "descend", "lift", "transport", "lower")}
    grasp_z = float(vial_xyz[2] - GRASP_PINCH_BELOW_VIAL)
    grasp_pt = (float(vial_xyz[0]), float(vial_xyz[1]), grasp_z)
    place_z = float(pocket_xyz[2] - PLACE_PINCH_BELOW_POCKET)
    place_pt = (float(pocket_xyz[0]), float(pocket_xyz[1]), place_z)
    relab = ExpertRelabeler(targets, grasp_pt, place_pt, z0=float(vial_xyz[2]))
    return relab, plan


def run_relabel_episode(mj, model, data, renderer, idx, ik, client, *, fps,
                        area_x, area_y, yaw_jitter, rng, max_ticks,
                        n_action_steps, instruction):
    """Roll out the served policy for one episode, relabelling every visited
    state with the expert's absolute joint target.

    Returns the per-step ``states`` / ``actions`` / ``frames`` (ready for
    :meth:`LeRobotDatasetWriter.add_episode`) plus rollout diagnostics.
    """
    key = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "home")
    mj.mj_resetDataKeyframe(model, data, key)
    _randomise_vial(mj, model, data, idx, area_x=area_x, area_y=area_y,
                    yaw_jitter=yaw_jitter, rng=rng)

    arm_act, grip_act = idx["arm_act"], idx["grip_act"]
    arm_qadr, grip_qadr = idx["arm_qadr"], idx["grip_qadr"]

    def read_arm() -> np.ndarray:
        return np.array([data.qpos[a] for a in arm_qadr], dtype=np.float32)

    home_q = tuple(float(v) for v in read_arm())
    relab, _plan = _build_expert(ik, mj, model, data, idx, home_q)

    # Settle at home (gripper open) before handing control to GR00T.
    for a, q in zip(arm_act, read_arm(), strict=True):
        data.ctrl[a] = q
    data.ctrl[grip_act] = GRIP_OPEN
    for _ in range(SETTLE_STEPS):
        mj.mj_step(model, data)

    decim = max(1, round((1.0 / fps) / model.opt.timestep))
    z0 = float(data.xpos[idx["vial_body"], 2])
    z_max = z0

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    frames: dict[str, list[np.ndarray]] = {k: [] for k in VIDEO_KEY_TO_CAMERA}

    arm_chunk = None
    grip_chunk = None
    ci = 0
    chunk_len = 0
    n_queries = 0
    query_t = 0.0
    grip_max = 0.0

    tick = 0
    while tick < max_ticks:
        measured = read_arm()
        grip_meas = float(np.clip(data.qpos[grip_qadr] / GRIP_DRIVER_RANGE, 0.0, 1.0))
        # Render every video key the fine-tune consumes from its MJCF camera.
        view: dict[str, np.ndarray] = {}
        for vkey, cam in VIDEO_KEY_TO_CAMERA.items():
            renderer.update_scene(data, camera=cam)
            view[vkey] = renderer.render().copy()

        # --- Expert relabel for THIS visited state (absolute joint target) ----
        pinch = np.array(data.site_xpos[idx["pinch_site"]], dtype=np.float64)
        vial = np.array(data.xpos[idx["vial_body"]], dtype=np.float64)
        tgt_q, tgt_grip = relab.step(pinch, vial, grip_meas)
        states.append(np.array([*measured, grip_meas], dtype=np.float32))
        actions.append(np.array([*tgt_q, tgt_grip], dtype=np.float32))
        for vkey in VIDEO_KEY_TO_CAMERA:
            frames[vkey].append(view[vkey])

        # --- Advance the sim under the SERVED policy (re-query per chunk) ------
        if arm_chunk is None or ci >= chunk_len:
            obs = build_obs(measured, grip_meas, view, instruction)
            t0 = time.time()
            action = client.get_action(obs)
            query_t += time.time() - t0
            n_queries += 1
            arm_chunk = np.asarray(action["single_arm"], dtype=np.float32).reshape(-1, 6)
            grip_chunk = np.asarray(action["gripper"], dtype=np.float32).reshape(-1, 1)
            chunk_len = min(n_action_steps, arm_chunk.shape[0])
            ci = 0

        for a, q in zip(arm_act, arm_chunk[ci], strict=True):
            data.ctrl[a] = float(q)
        g = float(np.clip(grip_chunk[ci, 0], 0.0, 1.0))
        grip_max = max(grip_max, g)
        data.ctrl[grip_act] = g * GRIP_CLOSE
        for _ in range(decim):
            mj.mj_step(model, data)
        z_max = max(z_max, float(data.xpos[idx["vial_body"], 2]))
        ci += 1
        tick += 1

    vial_xyz = np.array(data.xpos[idx["vial_body"]], dtype=np.float64)
    pocket_xyz = np.array(data.site_xpos[idx["pocket_site"]], dtype=np.float64)
    place_dist = float(np.linalg.norm(vial_xyz[:2] - pocket_xyz[:2]))
    lifted = (z_max - z0) > 0.02
    seated = place_dist < 0.05 and vial_xyz[2] > 0.20
    success = bool(lifted and seated)
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "frames": frames,
        "num_frames": len(states),
        "success": success, "lifted": bool(lifted), "seated": bool(seated),
        "lift_height": float(z_max - z0), "place_dist": place_dist,
        "grip_max": grip_max, "n_queries": n_queries,
        "mean_query_s": query_t / max(1, n_queries),
        "expert_phase_reached": relab.i,
    }


def cmd_rollout(args) -> int:
    import mujoco as mj

    xml_path = Path(args.xml)
    if not xml_path.is_file():
        print(f"ERROR: MJCF not found: {xml_path}", file=sys.stderr)
        return 2
    model = mj.MjModel.from_xml_path(str(xml_path))
    if args.noslip_iterations > 0:
        model.opt.noslip_iterations = args.noslip_iterations
    data = mj.MjData(model)
    idx = _build_index(mj, model)
    freed = _free_arm_collision(mj, model)
    decim = max(1, round((1.0 / args.fps) / model.opt.timestep))
    print(f"[dagger] loaded {xml_path.name}: nq={model.nq} nu={model.nu} "
          f"ncam={model.ncam}; freed {freed} arm-collision geoms; "
          f"noslip_iters={model.opt.noslip_iterations}; decim={decim} fps={args.fps}",
          flush=True)

    renderer = mj.Renderer(model, args.height, args.width)
    ik = DampedLeastSquaresIK(
        model=model, site_id=idx["pinch_site"],
        arm_qadr=tuple(idx["arm_qadr"]), arm_dofadr=tuple(idx["arm_dofadr"]),
    )

    client = GrootPolicyClient(args.host, args.port, timeout_ms=args.timeout_ms)
    if not client.ping():
        print(f"ERROR: GR00T server not reachable at {args.host}:{args.port}",
              file=sys.stderr)
        return 3
    print(f"[dagger] connected to GR00T server {args.host}:{args.port}; "
          f"episodes={args.num_episodes} n_action_steps={args.n_action_steps} "
          f"max_ticks={args.max_ticks} instruction={args.instruction!r}", flush=True)

    writer = LeRobotDatasetWriter(
        args.out, modality=emb.modality_json(), info_builder=emb.info_json,
        video_keys=emb.VIDEO_ORIGINAL_KEYS, fps=args.fps,
        width=args.width, height=args.height,
    )

    rng = np.random.default_rng(args.seed)
    n_success = n_lifted = n_seated = grip_fired = 0
    try:
        for ep in range(args.num_episodes):
            r = run_relabel_episode(
                mj, model, data, renderer, idx, ik, client,
                fps=args.fps, area_x=args.area_x, area_y=args.area_y,
                yaw_jitter=args.yaw_jitter, rng=rng, max_ticks=args.max_ticks,
                n_action_steps=args.n_action_steps, instruction=args.instruction,
            )
            writer.add_episode(states=r["states"], actions=r["actions"],
                               frames=r["frames"], task=args.instruction)
            n_success += int(r["success"])
            n_lifted += int(r["lifted"])
            n_seated += int(r["seated"])
            grip_fired += int(r["grip_max"] >= 0.9)
            print(
                f"[dagger] ep {ep:03d}: {'SUCCESS' if r['success'] else 'FAIL   '} "
                f"frames={r['num_frames']:4d} lift={r['lift_height']*100:5.1f}cm "
                f"place_dist={r['place_dist']*100:5.1f}cm grip_max={r['grip_max']:.2f} "
                f"(lifted={r['lifted']} seated={r['seated']}) "
                f"expert_phase={r['expert_phase_reached']} "
                f"queries={r['n_queries']} mean_query={r['mean_query_s']*1000:.0f}ms",
                flush=True,
            )
    finally:
        renderer.close()

    info = writer.finalize()
    rate = n_success / args.num_episodes if args.num_episodes else 0.0
    summary = {
        "relabel_episodes": info["total_episodes"],
        "relabel_frames": info["total_frames"],
        "rollout_success_rate": rate,
        "rollout_n_success": n_success,
        "rollout_n_episodes": args.num_episodes,
        "rollout_n_lifted": n_lifted,
        "rollout_n_seated": n_seated,
        "rollout_grip_fired": grip_fired,
        "n_action_steps": args.n_action_steps,
        "max_ticks": args.max_ticks,
    }
    (Path(args.out) / "dagger_rollout.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("=" * 72)
    print(f"[dagger] RELABEL wrote {info['total_episodes']} episodes / "
          f"{info['total_frames']} frames to {args.out}")
    print(f"[dagger] rollout policy success (this checkpoint): {n_success}/"
          f"{args.num_episodes} lifted={n_lifted} seated={n_seated} "
          f"grip_fired={grip_fired}")
    print("=" * 72, flush=True)
    return 0


# ---------------------------------------------------------------------------
# merge: aggregate a relabel dataset onto a base LeRobot dataset
# ---------------------------------------------------------------------------
def _stats_for(matrix: np.ndarray) -> dict[str, list[float]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return {
        "mean": matrix.mean(axis=0).tolist(),
        "std": matrix.std(axis=0).tolist(),
        "min": matrix.min(axis=0).tolist(),
        "max": matrix.max(axis=0).tolist(),
        "q01": np.quantile(matrix, 0.01, axis=0).tolist(),
        "q99": np.quantile(matrix, 0.99, axis=0).tolist(),
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def cmd_merge(args) -> int:
    """Aggregate ``--add`` onto ``--base`` into ``--out`` (fresh dataset dir)."""
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
        print(f"ERROR: merged episode count {nb + len(add_eps)} exceeds single "
              f"chunk size {CHUNKS_SIZE}; multi-chunk merge not implemented",
              file=sys.stderr)
        return 4

    base_tasks = _read_jsonl(base / "meta" / "tasks.jsonl")
    task_to_idx: dict[str, int] = {t["task"]: int(t["task_index"]) for t in base_tasks}
    add_tasks = {int(t["task_index"]): t["task"] for t in _read_jsonl(add / "meta" / "tasks.jsonl")}

    if out.exists():
        shutil.rmtree(out)
    # Copy the base wholesale (meta + data + videos); indices 0..nb-1 preserved.
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
        # Re-index the scalar columns; the list<float32> action/state columns
        # (and per-episode frame_index) are preserved verbatim.
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
                 [{"task_index": i, "task": t}
                  for t, i in sorted(task_to_idx.items(), key=lambda kv: kv[1])])

    # Recompute normalisation stats across the merged set (fresh, no fingerprints).
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

    print(f"[merge] {base.name} ({nb} eps) + {add.name} ({len(add_eps)} eps) "
          f"-> {out} ({total_ep} eps / {total_frames} frames / {total_tasks} tasks)",
          flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ro = sub.add_parser("rollout", help="roll out a served GR00T ckpt + relabel with the IK expert")
    ro.add_argument("--xml", default="/home/ubuntu/aseptipack_description/aseptipack.xml")
    ro.add_argument("--host", default="127.0.0.1")
    ro.add_argument("--port", type=int, default=5555)
    ro.add_argument("--out", required=True, help="output relabel dataset dir")
    ro.add_argument("--num-episodes", type=int, default=30)
    ro.add_argument("--fps", type=int, default=emb.DEFAULT_FPS)
    ro.add_argument("--width", type=int, default=emb.DEFAULT_WIDTH)
    ro.add_argument("--height", type=int, default=emb.DEFAULT_HEIGHT)
    ro.add_argument("--area-x", type=float, default=0.05)
    ro.add_argument("--area-y", type=float, default=0.07)
    ro.add_argument("--yaw-jitter", type=float, default=0.08)
    ro.add_argument("--max-ticks", type=int, default=1000)
    ro.add_argument("--n-action-steps", type=int, default=8)
    ro.add_argument("--noslip-iterations", type=int, default=20)
    ro.add_argument("--seed", type=int, default=100)
    ro.add_argument("--instruction", default=emb.DEFAULT_INSTRUCTION)
    ro.add_argument("--timeout-ms", type=int, default=180000)
    ro.set_defaults(func=cmd_rollout)

    mg = sub.add_parser("merge", help="aggregate a relabel dataset onto a base dataset")
    mg.add_argument("--base", required=True)
    mg.add_argument("--add", required=True)
    mg.add_argument("--out", required=True)
    mg.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
