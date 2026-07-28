"""Convert Playground teleoperation recordings into a LeRobot v2.1 dataset.

Input: one or more `autonomy-recording/1.0` manifests (the JSON the Teleop
panel's Export button downloads, or the payload the recordings API stores).
Output: the exact dataset layout the GR00T finetune consumes — 7-D state/action
(6 arm joints + normalised grip), two 256x256 h264 videos per episode
(exterior + wrist), meta/{info,episodes,tasks,modality,stats}, and — when the
recording carries vial/pocket scene objects (the MJCF-cell recorder does) —
per-episode gt/ files in the harness schema, so teleop episodes can feed the
observer-conditioning and augmentation stages exactly like scripted captures.

Alignment notes:
- action_semantics is `absolute_next_joint_targets` (action[t] = state[t+1]),
  matching the scripted expert's ABSOLUTE convention.
- Recordings at any fps are resampled to --fps (default 20, the pipeline rate)
  by nearest-source-frame; record at 20 fps to avoid resampling entirely.
- Videos are piped frame-by-frame to ffmpeg; frame counts are asserted equal to
  parquet rows per episode.

Usage:
  teleop_to_lerobot.py --src rec1.json [rec2.json ...] --out <dataset_dir>
                       [--fps 20] [--min-frames 20] [--task-override "..."]
"""
import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
FEATURE_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow.pos",
                 "wrist_1.pos", "wrist_2.pos", "wrist_3.pos", "gripper.pos"]
GRIP_JOINT = "gr_right_driver_joint"
GRIP_RANGE = 0.8
VIDEO_KEYS = {"exterior": "observation.images.exterior", "wrist": "observation.images.wrist"}
SIZE = 256


def vec7(state: dict) -> np.ndarray:
    v = [float(state.get(j, 0.0)) for j in ARM_JOINTS]
    v.append(min(1.0, max(0.0, float(state.get(GRIP_JOINT, 0.0)) / GRIP_RANGE)))
    return np.asarray(v, dtype=np.float32)


def decode_image(data_url):
    import cv2
    if not data_url or "," not in data_url:
        return None
    img = cv2.imdecode(np.frombuffer(base64.b64decode(data_url.split(",", 1)[1]), np.uint8),
                       cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.shape[:2] != (SIZE, SIZE):
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return img


def encode_video(frames_bgr, dst: Path, fps: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{SIZE}x{SIZE}", "-framerate", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
         str(dst)],
        stdin=subprocess.PIPE)
    n = 0
    for f in frames_bgr:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
        n += 1
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {dst}")
    return n


def resample_indices(n_src: int, src_fps: float, dst_fps: float):
    if abs(src_fps - dst_fps) < 1e-6:
        return list(range(n_src))
    dur = n_src / src_fps
    n_dst = max(1, int(round(dur * dst_fps)))
    return [min(n_src - 1, int(round(i * src_fps / dst_fps))) for i in range(n_dst)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", nargs="+", required=True, help="recording manifest JSON file(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--min-frames", type=int, default=20,
                    help="drop episodes shorter than this (after resampling)")
    ap.add_argument("--task-override", default=None)
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(args.out)
    if out.exists():
        print(f"ERROR: {out} exists — refusing to clobber", file=sys.stderr)
        return 1
    (out / "data/chunk-000").mkdir(parents=True)
    (out / "meta/gt").mkdir(parents=True)

    episodes_meta, tasks, all_state, all_action, all_ts = [], {}, [], [], []
    ep_out = 0
    global_index = 0
    robot_type = None
    for src in args.src:
        man = json.load(open(src))
        assert man.get("schema_version", "").startswith("autonomy-recording"), \
            f"{src}: not an autonomy-recording manifest"
        src_fps = float(man.get("capture", {}).get("fps") or 10)
        sem = man.get("capture", {}).get("action_semantics", "absolute_next_joint_targets")
        assert sem == "absolute_next_joint_targets", f"unsupported action semantics: {sem}"
        robot_type = robot_type or (man.get("robot", {}) or {}).get("robot_id") or "aseptipack_teleop"
        for ep in man["episodes"]:
            frames = ep.get("frames", [])
            idxs = resample_indices(len(frames), src_fps, args.fps)
            if len(idxs) < args.min_frames:
                print(f"[teleop2lerobot] skip ep (only {len(idxs)} frames @ {args.fps}fps)")
                continue
            task = args.task_override or ep.get("task") or "teleoperated pick and place"
            t_idx = tasks.setdefault(task, len(tasks))

            states, actions, imgs = [], [], {"exterior": [], "wrist": []}
            gt_frames = []
            dual = True
            for i in idxs:
                fr = frames[i]
                obs = fr.get("observation", {})
                states.append(vec7(obs.get("state", {})))
                actions.append(vec7((fr.get("action", {}) or {}).get("target_state", {})))
                img = obs.get("image")
                if isinstance(img, dict):
                    for cam in ("exterior", "wrist"):
                        imgs[cam].append(decode_image(img.get(cam)))
                else:   # single-camera recording — duplicate (degraded but usable)
                    dual = False
                    d = decode_image(img)
                    imgs["exterior"].append(d)
                    imgs["wrist"].append(d)
                vial = pocket = None
                for so in obs.get("scene_objects") or []:
                    if so.get("name") == "vial_0":
                        vial = list(so.get("pos", [])) + list(so.get("quat", []))
                    if so.get("name") == "pocket_0":
                        pocket = list(so.get("pos", []))
                if vial:
                    gt_frames.append({"vial_pose": vial, "pocket_pos": pocket, "phase": "teleop"})
            if any(x is None for cam in imgs.values() for x in cam):
                print(f"[teleop2lerobot] skip ep {ep.get('episode_index')} — missing frames")
                continue
            if not dual:
                print(f"[teleop2lerobot] WARNING ep {ep_out}: single-camera recording; "
                      "wrist view duplicated from primary")

            n = len(states)
            for cam, key in VIDEO_KEYS.items():
                wrote = encode_video(imgs[cam], out / f"videos/chunk-000/{key}/episode_{ep_out:06d}.mp4", args.fps)
                assert wrote == n, f"video/parquet mismatch: {wrote} vs {n}"
            ts = np.arange(n, dtype=np.float32) / args.fps
            table = pa.table({
                "action": pa.array([a.tolist() for a in actions], type=pa.list_(pa.float32())),
                "observation.state": pa.array([s.tolist() for s in states], type=pa.list_(pa.float32())),
                "timestamp": pa.array(ts, type=pa.float32()),
                "frame_index": pa.array(np.arange(n), type=pa.int64()),
                "episode_index": pa.array(np.full(n, ep_out), type=pa.int64()),
                "index": pa.array(np.arange(global_index, global_index + n), type=pa.int64()),
                "task_index": pa.array(np.full(n, t_idx), type=pa.int64()),
            })
            pq.write_table(table, out / f"data/chunk-000/episode_{ep_out:06d}.parquet")
            if len(gt_frames) == n:
                zs = [g["vial_pose"][2] for g in gt_frames]
                pk = gt_frames[-1].get("pocket_pos") or [0, 0, 0]
                vx = gt_frames[-1]["vial_pose"]
                dxy = float(np.hypot(vx[0] - pk[0], vx[1] - pk[1]))
                json.dump({"episode_index": ep_out, "success": True,
                           "lifted": bool(max(zs) > zs[0] + 0.05),
                           "seated": bool(dxy < 0.025),
                           "lift_height_m": float(max(zs) - zs[0]),
                           "place_dist_m": dxy, "frames": gt_frames},
                          open(out / f"meta/gt/episode_{ep_out:06d}.json", "w"))
            episodes_meta.append({"episode_index": ep_out, "tasks": [task], "length": n})
            all_state.append(np.stack(states)); all_action.append(np.stack(actions)); all_ts.append(ts)
            global_index += n
            ep_out += 1
            print(f"[teleop2lerobot] ep {ep_out - 1}: {n} frames @ {args.fps}fps task='{task}'")

    if not ep_out:
        print("ERROR: no usable episodes", file=sys.stderr)
        return 1

    S, A, T = np.concatenate(all_state), np.concatenate(all_action), np.concatenate(all_ts)
    def _stats(x):
        return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
                "min": x.min(0).tolist(), "max": x.max(0).tolist()}
    json.dump({"action": _stats(A), "observation.state": _stats(S),
               "timestamp": _stats(T.reshape(-1, 1))},
              open(out / "meta/stats.json", "w"), indent=1)
    with open(out / "meta/episodes.jsonl", "w") as f:
        for e in episodes_meta:
            f.write(json.dumps(e) + "\n")
    with open(out / "meta/tasks.jsonl", "w") as f:
        for task, i in sorted(tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": i, "task": task}) + "\n")
    json.dump({
        "state": {"single_arm": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7}},
        "action": {"single_arm": {"start": 0, "end": 6}, "gripper": {"start": 6, "end": 7}},
        "video": {"exterior": {"original_key": "observation.images.exterior"},
                  "wrist": {"original_key": "observation.images.wrist"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }, open(out / "meta/modality.json", "w"), indent=1)
    vid_info = {"video.height": SIZE, "video.width": SIZE, "video.codec": "h264",
                "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                "video.fps": args.fps, "video.channels": 3, "has_audio": False}
    feat7 = {"dtype": "float32", "shape": [7], "names": FEATURE_NAMES}
    scalar = lambda dt: {"dtype": dt, "shape": [1], "names": None}
    json.dump({
        "codebase_version": "v2.1", "robot_type": robot_type,
        "total_episodes": ep_out, "total_frames": int(global_index),
        "total_tasks": len(tasks), "total_videos": ep_out * 2,
        "total_chunks": 1, "chunks_size": 1000, "fps": args.fps,
        "splits": {"train": f"0:{ep_out}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"action": feat7, "observation.state": feat7,
                     "observation.images.exterior": {"dtype": "video", "shape": [SIZE, SIZE, 3],
                                                     "names": ["height", "width", "channels"], "info": vid_info},
                     "observation.images.wrist": {"dtype": "video", "shape": [SIZE, SIZE, 3],
                                                  "names": ["height", "width", "channels"], "info": vid_info},
                     "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                     "frame_index": scalar("int64"), "episode_index": scalar("int64"),
                     "index": scalar("int64"), "task_index": scalar("int64")},
    }, open(out / "meta/info.json", "w"), indent=4)
    print(f"TELEOP2LEROBOT_DONE eps={ep_out} frames={global_index} out={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
