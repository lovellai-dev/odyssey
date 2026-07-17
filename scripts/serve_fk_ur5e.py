#!/usr/bin/env python3
"""Tiny batch-FK HTTP service (gr_pinch from arm q6) for the best-of-N selector.

Runs in the mujoco-capable venv next to the best-of-N policy service (whose
GR00T venv lacks mujoco — deliberately unmutated). One POST scores a whole
candidate batch, so HTTP overhead is one round-trip per control query.

  POST /fk {"q": [[6]...]} -> {"pinch": [[3]...]}   (robot base frame)

Run:  ~/odyssey-eval-venv/bin/python serve_fk_ur5e.py --xml aseptipack.xml --port 5560
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


class FK:
    def __init__(self, xml: str) -> None:
        import mujoco as mj
        self.mj = mj
        self.model = mj.MjModel.from_xml_path(xml)
        self.data = mj.MjData(self.model)
        self.qadr = []
        for name in ARM_JOINTS:
            j = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            assert j >= 0, name
            self.qadr.append(int(self.model.jnt_qposadr[j]))
        self.site = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "gr_pinch")
        assert self.site >= 0
        hk = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, "home")
        if hk >= 0:
            mj.mj_resetDataKeyframe(self.model, self.data, hk)
        mj.mj_forward(self.model, self.data)
        base = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "base")
        self.bpos = np.array(self.data.xpos[base])
        self.bmat = np.array(self.data.xmat[base]).reshape(3, 3)

    def batch(self, qs: np.ndarray) -> np.ndarray:
        out = np.zeros((len(qs), 3))
        for i, q in enumerate(qs):
            for a, v in zip(self.qadr, q):
                self.data.qpos[a] = float(v)
            self.mj.mj_kinematics(self.model, self.data)
            p = np.array(self.data.site_xpos[self.site])
            out[i] = self.bmat.T @ (p - self.bpos)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--port", type=int, default=5560)
    args = ap.parse_args()
    fk = FK(args.xml)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send(200, {"ok": True}) if self.path.startswith("/health") \
                else self._send(404, {"error": "nf"})

        def do_POST(self):
            if not self.path.startswith("/fk"):
                self._send(404, {"error": "nf"})
                return
            try:
                n = int(self.headers.get("content-length", 0) or 0)
                req = json.loads(self.rfile.read(n))
                qs = np.asarray(req["q"], dtype=float).reshape(-1, 6)
                self._send(200, {"pinch": fk.batch(qs).tolist()})
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": repr(exc)})

    print(f"[fk] serving :{args.port} (batch gr_pinch FK)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
