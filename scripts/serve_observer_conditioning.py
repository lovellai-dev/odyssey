#!/usr/bin/env python3
"""Deploy-time **Observer state-conditioning sidecar** for the served GR00T pilot.

Roadmap Step 2 (Observer-conditioned GR00T). The render-gap close proved the
Observer localises the vial cap to sub-cm on the deployment (Three.js) frames, yet
the pixel-only browser GR00T pilot still scored 0/15 — it could not infer *descent
depth* from pixels. Step 2 retrains GR00T with the grasp target as an EXPLICIT
state input (state 7 -> 10). This sidecar is the deploy half of that contract: it
is a **drop-in for the GR00T HTTP<->ZMQ bridge** (same ``POST /get_action`` +
``GET /health``) that, per control tick:

1. runs the Observer's DINOv2 grasp-target head on the exterior+wrist frames the
   browser already sends -> the 3D vial cap in the robot base frame (``cap_base``);
2. appends that target to the pilot's 7-D proprio state -> the 10-D state the
   Observer-conditioned checkpoint was trained on;
3. forwards the (now 10-D) observation to the real GR00T bridge and returns its
   action unchanged (GR00T keeps full actuator authority).

Point the agent-service ``GROOT_BRIDGE_URL`` (and/or ``GROOT_STATE_CONDITIONER_URL``)
at this sidecar; it forwards to the true bridge (``--bridge`` / ``GROOT_BRIDGE_URL``
of THIS process). It is a **dev/local-only** seam — the GR00T serving path is
GPU-local, never Cloud Run.

Robust by construction: if the Observer (or a frame) errors, it falls back to the
last good target for that session, then to a fixed nominal target — it ALWAYS emits
a 10-D state, so the 10-D model never sees a missing key and the pilot never freezes.

Run (torch+transformers interpreter; odyssey src on the path)::

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      OBSERVER_WEIGHTS=/path/to/percep_weights_browser \
      CONDITION_BRIDGE_URL=http://127.0.0.1:5596 \
      PYTHONPATH=/home/daniel/LovellAI/odyssey-ur5e/src \
      /home/daniel/Isaac-GR00T/.venv/bin/python scripts/serve_observer_conditioning.py --port 5604
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import traceback
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

# Nominal grasp target (robot base frame, metres) used only if the Observer errors
# before any good reading exists — the dataset grasp_target mean (a safe centre of
# the vial workspace). Env-overridable.
NOMINAL_TARGET = np.array(
    [float(x) for x in os.environ.get("GRASP_TARGET_FALLBACK", "-0.425,-0.015,0.272").split(",")],
    dtype=np.float32,
)


def _decode_frame(data_url: str) -> np.ndarray:
    b64 = data_url.split(",", 1)[1] if "," in str(data_url) else data_url
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.array(img, dtype=np.uint8)


class ConditioningService:
    """Loads the Observer grasp-target head once; conditions one request per call."""

    def __init__(self, *, weights: str, bridge_url: str, device: str) -> None:
        from odyssey.embodiments.ur5e_drugsort.observer import GraspTargetObserver

        self.bridge_url = bridge_url.rstrip("/")
        self._lock = threading.Lock()
        self._last: dict[str, np.ndarray] = {}     # per-session last good target (hold-last fallback)
        self._n = 0
        self._n_fallback = 0
        # Ablation switch: when set, ignore the Observer and inject the fixed nominal
        # target every tick. Isolates the *live-target signal* vs a constant — if the
        # Observer-conditioned success collapses to the nominal-target success, the
        # per-frame Observer localisation is what carries the policy.
        self._freeze_nominal = os.environ.get("CONDITIONER_FREEZE_NOMINAL", "").strip() in ("1", "true", "True")
        self._observer = GraspTargetObserver(device=device)
        self._observer.load_head(str(Path(weights) / "observer_head.pt"))
        warm = np.zeros((256, 256, 3), np.uint8)
        self._observer.predict(warm, warm)         # warm the CUDA graphs
        print(
            f"[conditioner] ready | backbone={self._observer.backbone_id} device={device} "
            f"bridge={self.bridge_url} nominal_target={NOMINAL_TARGET.tolist()}",
            flush=True,
        )

    def _target(self, payload: dict, sid: str) -> tuple[np.ndarray, str]:
        """Observer grasp target (base frame) for this tick, with layered fallback."""
        if self._freeze_nominal:
            return NOMINAL_TARGET.copy(), "frozen-nominal"
        try:
            ext = _decode_frame(payload["image_b64"])
            wrist = _decode_frame(payload.get("image_b64_wrist") or payload["image_b64"])
            with self._lock:
                tgt = np.asarray(self._observer.predict(ext, wrist), dtype=np.float32).reshape(3)
            if not np.all(np.isfinite(tgt)):
                raise ValueError("non-finite observer target")
            self._last[sid] = tgt
            if len(self._last) > 32:
                self._last.pop(next(iter(self._last)))
            return tgt, "observer"
        except Exception as exc:  # noqa: BLE001 — never freeze; hold-last then nominal
            self._n_fallback += 1
            if sid in self._last:
                return self._last[sid], f"hold-last({exc!r})"
            return NOMINAL_TARGET.copy(), f"nominal({exc!r})"

    def _query_bridge(self, body: bytes) -> tuple[int, dict[str, Any]]:
        req = urllib.request.Request(
            f"{self.bridge_url}/get_action", data=body,
            headers={"content-type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=180.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def get_action(self, body: bytes) -> tuple[int, dict[str, Any]]:
        payload = json.loads(body.decode("utf-8"))
        sid = str(payload.get("sid") or "default")
        state = list(np.asarray(payload.get("state", []), dtype=np.float32).reshape(-1)[:7])
        while len(state) < 7:
            state.append(0.0)
        tgt, mode = self._target(payload, sid)
        payload["state"] = [float(x) for x in state] + [float(x) for x in tgt]   # 7 -> 10
        self._n += 1
        if self._n <= 3 or self._n % 200 == 0:
            print(f"[conditioner] tick={self._n} sid={sid} mode={mode} "
                  f"target={np.round(tgt, 4).tolist()} fallbacks={self._n_fallback}", flush=True)
        code, out = self._query_bridge(json.dumps(payload).encode("utf-8"))
        if isinstance(out, dict):
            out["grasp_target"] = [float(x) for x in tgt]
            out["conditioner"] = {"mode": mode}
        return code, out

    def health(self) -> dict[str, Any]:
        ok = False
        try:
            with urllib.request.urlopen(f"{self.bridge_url}/health", timeout=5.0) as resp:
                ok = bool(json.loads(resp.read().decode("utf-8")).get("ok"))
        except Exception:  # noqa: BLE001
            ok = False
        # `ok` mirrors the downstream bridge so the agent-service /api/groot/health
        # wraps it as {"bridge_health":{"ok":true}} exactly like the raw bridge.
        return {"ok": ok, "observer_ready": True, "bridge": self.bridge_url,
                "conditioned": True, "ticks": self._n, "fallbacks": self._n_fallback}


def _make_handler(service: ConditioningService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:  # quiet
            return

        def _send(self, code: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/health":
                self._send(200, service.health())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            # Accept BOTH the plain bridge path (?agents=groot -> /get_action) and the
            # observer path (?agents=groot-observer -> /get_action_corrected) so the
            # conditioner is a drop-in for either agent-service route.
            if self.path.rstrip("/") not in ("/get_action", "/get_action_corrected"):
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("content-length", 0) or 0)
            body = self.rfile.read(length)
            try:
                code, out = service.get_action(body)
                self._send(code, out)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._send(502, {"error": f"conditioner failure: {exc!r}"})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=int(os.environ.get("CONDITIONER_PORT", "5604")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--weights", default=os.environ.get("OBSERVER_WEIGHTS", ""))
    ap.add_argument("--bridge", default=os.environ.get("CONDITION_BRIDGE_URL",
                                                        os.environ.get("GROOT_BRIDGE_URL", "http://127.0.0.1:5596")))
    ap.add_argument("--device", default=os.environ.get("OBSERVER_DEVICE", "cuda"))
    args = ap.parse_args()
    if not args.weights:
        raise SystemExit("--weights (or OBSERVER_WEIGHTS) is required: dir with observer_head.pt")

    service = ConditioningService(weights=args.weights, bridge_url=args.bridge, device=args.device)
    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(service))
    print(f"[conditioner] serving on http://{args.host}:{args.port} "
          f"(POST /get_action|/get_action_corrected, GET /health) -> {args.bridge}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
