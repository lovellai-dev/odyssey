#!/usr/bin/env python3
"""Runs ON the GPU VM: same observation through the HTTP bridge vs a native ZMQ
query — isolates bridge payload-construction bugs (image decode/resize, state
packaging, key names) that Gate 0's recorded-action replay structurally could not
catch.

Inputs: the two frame .b64 dumps + state json scp'd from the probe run.
Output: vm_diff.json with per-path first-step action means and their disagreement.

Run in the GR00T venv (zmq, msgpack_numpy, PIL, numpy available)::

    ~/Isaac-GR00T/.venv/bin/python vm_probe_native_diff.py \
        --ext frame_q0_ext.b64 --wrist frame_q0_wrist.b64 --state frame_q0_state.json \
        --target=-0.42,-0.01,0.27 --zmq-port 5558 --bridge http://127.0.0.1:5596 \
        --k 5 --out vm_diff.json

(NB the ``--target=`` equals form is required: argparse misparses a bare leading-
dash comma list as an unknown option.)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq
from PIL import Image

INSTRUCTION = "pick up the vial and place it in the rack"


def decode_frame(path: str) -> np.ndarray:
    raw = open(path).read().strip().split(",")[-1]
    im = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
    if im.size != (256, 256):
        im = im.resize((256, 256))
    return np.asarray(im, dtype=np.uint8)


def native_query(port: int, ext: np.ndarray, wrist: np.ndarray,
                 state10: list[float], k: int) -> list[list[float]]:
    """Query the ZMQ policy server directly, mirroring the bridge's obs layout."""
    ctx = zmq.Context.instance()
    firsts = []
    for _ in range(k):
        # Bounded: a server-side exception on a hand-built obs leaves REQ with no
        # reply — 60s x k, not 180s x k, caps the worst case (model already warm).
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 60000)
        sock.setsockopt(zmq.SNDTIMEO, 60000)
        sock.connect(f"tcp://127.0.0.1:{port}")
        s = np.asarray(state10, dtype=np.float32)
        obs = {
            "video": {"exterior": ext[None, None], "wrist": wrist[None, None]},
            "state": {
                "single_arm": s[:6][None, None],
                "gripper": s[6:7][None, None],
                "grasp_target": s[7:10][None, None],
            },
            "language": {"annotation.human.task_description": [[INSTRUCTION]]},
        }
        sock.send(msgpack.packb({"endpoint": "get_action",
                                 "data": {"observation": obs, "options": None}},
                                default=mnp.encode))
        resp = msgpack.unpackb(sock.recv(), object_hook=mnp.decode, raw=False)
        sock.close()
        action = resp[0] if isinstance(resp, (list, tuple)) else resp
        arm = np.asarray(action["single_arm"], dtype=np.float32).reshape(-1, 6)
        firsts.append([float(x) for x in arm[0]])
        time.sleep(0.15)
    return firsts


def bridge_query(url: str, ext_b64: str, wrist_b64: str, state10: list[float],
                 k: int) -> list[list[float]]:
    firsts = []
    for _ in range(k):
        body = json.dumps({"image_b64": ext_b64, "image_b64_wrist": wrist_b64,
                           "state": state10, "instruction": INSTRUCTION}).encode()
        req = urllib.request.Request(f"{url}/get_action", data=body,
                                     headers={"content-type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=180) as r:
            res = json.loads(r.read().decode())
        if "q" not in res:
            raise SystemExit(f"bridge error: {json.dumps(res)[:200]}")
        firsts.append([float(x) for x in res["q"]])
        time.sleep(0.15)
    return firsts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ext", required=True)
    ap.add_argument("--wrist", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--target", required=True, help="x,y,z grasp target (base frame)")
    ap.add_argument("--zmq-port", type=int, default=5558)
    ap.add_argument("--bridge", default="http://127.0.0.1:5596")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="vm_diff.json")
    args = ap.parse_args()

    state7 = json.loads(open(args.state).read())[:7]
    tgt = [float(x) for x in args.target.split(",")]
    state10 = [float(x) for x in state7] + tgt
    ext_b64 = open(args.ext).read().strip()
    wrist_b64 = open(args.wrist).read().strip()

    nat = native_query(args.zmq_port, decode_frame(args.ext),
                       decode_frame(args.wrist), state10, args.k)
    brg = bridge_query(args.bridge, ext_b64, wrist_b64, state10, args.k)

    nm = np.mean(np.asarray(nat), axis=0)
    bm = np.mean(np.asarray(brg), axis=0)
    within = float(max(np.mean(np.std(np.asarray(nat), axis=0)),
                       np.mean(np.std(np.asarray(brg), axis=0))))
    diff = float(np.mean(np.abs(nm - bm)))
    out = {
        "k": args.k,
        "native_mean_first_step": [float(x) for x in nm],
        "bridge_mean_first_step": [float(x) for x in bm],
        "mean_first_step_abs_diff_rad": round(diff, 5),
        "max_first_step_abs_diff_rad": round(float(np.max(np.abs(nm - bm))), 5),
        "within_path_repeat_std_rad": round(within, 5),
        "materially_different": bool(diff > 0.02 and diff > 3 * max(within, 1e-6)),
    }
    open(args.out, "w").write(json.dumps(out, indent=2))
    print("[vm-diff]", json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
