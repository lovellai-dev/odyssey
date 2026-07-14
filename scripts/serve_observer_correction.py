#!/usr/bin/env python3
"""Deploy-time **Observer correction sidecar** for the served GR00T pilot.

Sits between the browser Playground (``lai-agent`` :8020, ``?agents=groot-observer``)
and the GR00T HTTP<->ZMQ bridge (:5599). Per control tick it:

1. forwards the browser observation *unchanged* to the GR00T bridge to get GR00T's
   coarse action chunk (GR00T keeps actuator authority);
2. runs the **Observer agent** (:class:`odyssey.embodiments.ur5e_drugsort.observer.
   ObserverAgent`) on the same two frames — the DINOv2 grasp-target head + the phase
   CNN — with a real damped-least-squares FK/IK on the AseptiPack MJCF;
3. returns GR00T's action **composed with the Observer's advisory correction**
   (visual servoing of the grasp/place descent + verify-and-re-grasp).

The correction is fully guarded: any Observer/IK/bridge hiccup falls through to the
raw GR00T action, so the pilot never freezes. This is a **dev/local-only** seam
(the GR00T serving path is GPU-local, never Cloud Run).

Run (uses a torch+transformers+mujoco interpreter; the Isaac-GR00T venv + the
odyssey src on the path)::

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OBSERVER_WEIGHTS=/path/to/percep_weights \
    ASEPTIPACK_XML=/path/to/aseptipack.xml \
    GROOT_BRIDGE_URL=http://127.0.0.1:5599 \
    PYTHONPATH=/home/daniel/LovellAI/odyssey-ur5e/src \
    /home/daniel/Isaac-GR00T/.venv/bin/python scripts/serve_observer_correction.py --port 5601
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

DEFAULT_XML = (
    "/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/"
    "aseptipack_description/aseptipack.xml"
)
ARM_ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")


def _decode_frame(data_url: str) -> Any:
    """A browser ``data:image/png;base64,...`` URL -> ``(H,W,3)`` uint8 RGB array."""
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.array(img, dtype=np.uint8)  # writable copy (PIL views are read-only)


class ObserverService:
    """Loads the models + MJCF once, then corrects one action chunk per request."""

    def __init__(self, *, xml: str, weights: str, bridge_url: str, device: str) -> None:
        import mujoco as mj

        from odyssey.embodiments.ur5e_drugsort.ik import DampedLeastSquaresIK
        from odyssey.embodiments.ur5e_drugsort.observer import (
            GraspTargetObserver,
            ObserverAgent,
        )
        from odyssey.embodiments.ur5e_drugsort.success_classifier import SuccessClassifier

        self.bridge_url = bridge_url.rstrip("/")
        self._lock = threading.Lock()
        self._mj = mj

        # --- MJCF + indices -------------------------------------------------
        model = mj.MjModel.from_xml_path(xml)
        self._model = model
        self._fk_data = mj.MjData(model)
        m2i, obj = mj.mj_name2id, mj.mjtObj
        arm_qadr, arm_dofadr = [], []
        for n in ARM_ACTUATORS:
            j = m2i(model, obj.mjOBJ_JOINT, f"{n}_joint")
            arm_qadr.append(int(model.jnt_qposadr[j]))
            arm_dofadr.append(int(model.jnt_dofadr[j]))
        self._arm_qadr = arm_qadr
        self._pinch_site = m2i(model, obj.mjOBJ_SITE, "gr_pinch")
        base_body = m2i(model, obj.mjOBJ_BODY, "base")
        pocket_site = m2i(model, obj.mjOBJ_SITE, "pocket_0")

        # --- static base transform + nest pocket (forward on the home key) --
        key = m2i(model, obj.mjOBJ_KEY, "home")
        if key >= 0:
            mj.mj_resetDataKeyframe(model, self._fk_data, key)
        mj.mj_forward(model, self._fk_data)
        base_pos = np.array(self._fk_data.xpos[base_body], dtype=np.float64)
        base_mat = np.array(self._fk_data.xmat[base_body], dtype=np.float64).reshape(3, 3)
        pocket_world = np.array(self._fk_data.site_xpos[pocket_site], dtype=np.float64)

        # --- FK / IK --------------------------------------------------------
        self._ik = DampedLeastSquaresIK(
            model=model, site_id=self._pinch_site,
            arm_qadr=tuple(arm_qadr), arm_dofadr=tuple(arm_dofadr),
        )

        # --- perception (torch) --------------------------------------------
        self._observer = GraspTargetObserver(device=device)
        self._observer.load_head(str(Path(weights) / "observer_head.pt"))
        self._classifier = SuccessClassifier(num_views=2, device=device)
        self._classifier.load_weights(str(Path(weights) / "success_cnn.pt"))
        # warm up the CUDA graphs so the first live tick is not slow.
        warm = (np.zeros((256, 256, 3), np.uint8), np.zeros((256, 256, 3), np.uint8))
        self._observer.predict(*warm)
        self._classifier.predict([warm[0], warm[1]])

        self._ObserverAgent = ObserverAgent
        self._base_pos, self._base_mat, self._pocket_world = base_pos, base_mat, pocket_world
        self._agents: dict[str, Any] = {}
        print(
            f"[observer-sidecar] ready | backbone={self._observer.backbone_id} device={device} "
            f"base_pos={np.round(base_pos, 3).tolist()} pocket={np.round(pocket_world, 3).tolist()} "
            f"bridge={self.bridge_url}",
            flush=True,
        )

    # -- mujoco FK closure (guarded by the request lock) --------------------
    def _fk(self, q6: Any) -> Any:
        d = self._fk_data
        for a, v in zip(self._arm_qadr, q6):
            d.qpos[a] = float(v)
        self._mj.mj_kinematics(self._model, d)
        self._mj.mj_comPos(self._model, d)
        return np.array(d.site_xpos[self._pinch_site], dtype=np.float64)

    def _ik_solve(self, target: Any, seed: Any) -> Any:
        return self._ik.solve(target, tuple(float(s) for s in seed)).q

    def _agent_for(self, sid: str) -> Any:
        ag = self._agents.get(sid)
        if ag is None:
            ag = self._ObserverAgent(
                ik_solve=self._ik_solve, fk=self._fk,
                base_pos=self._base_pos, base_mat=self._base_mat,
                pocket_world=self._pocket_world,
                perception=self._observer, verifier=self._classifier,
            )
            ag.reset()
            if len(self._agents) > 8:      # bound memory across stale sessions
                self._agents.clear()
            self._agents[sid] = ag
        return ag

    def _query_bridge(self, body: bytes) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.bridge_url}/get_action", data=body,
            headers={"content-type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=180.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_action_corrected(self, body: bytes) -> dict[str, Any]:
        payload = json.loads(body.decode("utf-8"))
        gr00t = self._query_bridge(body)                    # GR00T keeps action authority
        q = gr00t.get("q")
        grip = gr00t.get("grip")
        chunk_q = gr00t.get("chunk_q") or ([q] if q is not None else [])
        chunk_grip = gr00t.get("chunk_grip") or ([grip] if grip is not None else [])
        out = dict(gr00t)                                   # default: raw GR00T (fall-through)

        try:
            ext = _decode_frame(payload["image_b64"])
            wrist = _decode_frame(payload["image_b64_wrist"])
            state = payload["state"]
            arm_now = tuple(float(x) for x in state[:6])
            sid = str(payload.get("sid") or "default")
            pilot_action = {"chunk_q": chunk_q, "chunk_grip": chunk_grip}
            with self._lock:
                corrected = self._agent_for(sid).correct(
                    exterior=ext, wrist=wrist, arm_now=arm_now, pilot_action=pilot_action
                )
            out["chunk_q"] = [list(x) for x in corrected.chunk_q]
            out["chunk_grip"] = [float(g) for g in corrected.chunk_grip]
            out["q"] = list(corrected.chunk_q[0])
            out["grip"] = float(corrected.chunk_grip[0])
            out["observer"] = {
                "mode": corrected.mode, "corrected": corrected.corrected, **corrected.debug,
            }
        except Exception as exc:  # noqa: BLE001 — never freeze; fall through to raw GR00T
            out["observer"] = {"mode": "error", "corrected": False, "error": repr(exc)}
        return out


def _make_handler(service: ObserverService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:  # quiet
            return

        def _send(self, code: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/health":
                self._send(200, {"ready": True, "bridge": service.bridge_url})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/get_action_corrected":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("content-length", 0))
            body = self.rfile.read(length)
            try:
                self._send(200, service.get_action_corrected(body))
            except Exception as exc:  # noqa: BLE001
                self._send(502, {"error": f"observer sidecar failure: {exc!r}"})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("GROOT_OBSERVER_PORT", "5601")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--xml", default=os.environ.get("ASEPTIPACK_XML", DEFAULT_XML))
    ap.add_argument("--weights", default=os.environ.get("OBSERVER_WEIGHTS", ""))
    ap.add_argument("--bridge", default=os.environ.get("GROOT_BRIDGE_URL", "http://127.0.0.1:5599"))
    ap.add_argument("--device", default=os.environ.get("OBSERVER_DEVICE", "cuda"))
    args = ap.parse_args()
    if not args.weights:
        raise SystemExit("--weights (or OBSERVER_WEIGHTS) is required: dir with observer_head.pt + success_cnn.pt")

    service = ObserverService(
        xml=args.xml, weights=args.weights, bridge_url=args.bridge, device=args.device,
    )
    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(service))
    print(f"[observer-sidecar] serving on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
