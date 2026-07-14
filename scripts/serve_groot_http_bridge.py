#!/usr/bin/env python3
"""HTTP↔ZMQ bridge that lets a browser drive the served GR00T policy.

The GR00T policy server (``gr00t.eval.run_gr00t_server``) speaks ZMQ/msgpack,
which a browser cannot. This tiny stdlib HTTP server sits next to it on the GPU
host and translates one browser request into one GR00T query:

    POST /get_action   {image_b64: <exterior png/jpeg dataURL or base64>,
                        image_b64_wrist: <wrist view, iteration 3>, state: [7],
                        instruction?: str}
      -> {q: [6] rad, grip: 0..1, chunk_q: [[6]...], chunk_grip: [...]}

The iteration-3 checkpoint is trained on TWO video keys (``exterior`` overhead +
``wrist`` eye-in-hand), so the browser must send both frames; ``image_b64_wrist``
is required for that checkpoint (a single-view request would be missing the key
the policy expects).

It returns the whole 16-step action chunk so the caller can replay N steps per
query (the closed-loop eval found N=4 best for this checkpoint). The Playground
reaches it **same-origin** through the agent-service proxy over an SSH tunnel, so
the browser's CSP (`connect-src 'self'`) is satisfied; permissive CORS here is
just belt-and-braces for direct/tunnelled use.

Run on the GPU host in the GR00T venv, next to the ZMQ server::

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python scripts/serve_groot_http_bridge.py --isaac-gr00t ~/Isaac-GR00T \
      --zmq-host 127.0.0.1 --zmq-port 5555 --http-port 5599
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

DEFAULT_INSTRUCTION = "pick up the vial and place it in the rack"


def make_handler(client, instruction_default: str):
    class Handler(BaseHTTPRequestHandler):
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

        def _json(self, code: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self._cors()
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):  # CORS preflight
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/health"):
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/get_action"):
                self._json(404, {"error": "not found"})
                return
            try:
                from PIL import Image

                def _decode(b64: str) -> np.ndarray:
                    raw = str(b64).split(",")[-1]  # tolerate data: URL prefix
                    im = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
                    if im.size != (256, 256):
                        im = im.resize((256, 256))
                    return np.asarray(im, dtype=np.uint8)

                n = int(self.headers.get("content-length", 0) or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                # Video keys: exterior (required) + wrist (iteration-3 eye-in-hand).
                # Accept an explicit ``images`` dict or the flat image_b64 fields.
                images = req.get("images") if isinstance(req.get("images"), dict) else {}
                video: dict = {}
                ext = images.get("exterior") or req.get("image_b64")
                if ext is None:
                    raise KeyError("image_b64 (exterior view) is required")
                video["exterior"] = _decode(ext)[None, None]                       # (1,1,256,256,3)
                wrist = images.get("wrist") or req.get("image_b64_wrist")
                if wrist is not None:
                    video["wrist"] = _decode(wrist)[None, None]
                state = np.asarray(req["state"], dtype=np.float32).reshape(-1)[:7]
                instr = req.get("instruction") or instruction_default
                obs = {
                    "video": video,
                    "state": {
                        "single_arm": state[:6][None, None],                        # (1,1,6)
                        "gripper": state[6:7][None, None],                          # (1,1,1)
                    },
                    "language": {"annotation.human.task_description": [[instr]]},
                }
                action, _info = client.get_action(obs)
                arm = np.asarray(action["single_arm"], dtype=np.float32).reshape(-1, 6)
                grip = np.asarray(action["gripper"], dtype=np.float32).reshape(-1, 1)
                self._json(200, {
                    "q": [float(x) for x in arm[0]],
                    "grip": float(np.clip(grip[0, 0], 0.0, 1.0)),
                    "chunk_q": [[float(x) for x in row] for row in arm],
                    "chunk_grip": [float(np.clip(g[0], 0.0, 1.0)) for g in grip],
                })
            except Exception as exc:  # noqa: BLE001  — surface to the browser
                traceback.print_exc()
                self._json(500, {"error": repr(exc)})

        def log_message(self, *a):  # quiet
            pass

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--isaac-gr00t", default="/home/ubuntu/Isaac-GR00T",
                    help="path to the Isaac-GR00T checkout (for gr00t.policy import).")
    ap.add_argument("--zmq-host", default="127.0.0.1")
    ap.add_argument("--zmq-port", type=int, default=5555)
    ap.add_argument("--http-host", default="127.0.0.1")
    ap.add_argument("--http-port", type=int, default=5599)
    ap.add_argument("--timeout-ms", type=int, default=180000)
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    args = ap.parse_args()

    sys.path.insert(0, args.isaac_gr00t)
    from gr00t.policy.server_client import PolicyClient

    client = PolicyClient(host=args.zmq_host, port=args.zmq_port, timeout_ms=args.timeout_ms)
    print(f"[bridge] ping GR00T {args.zmq_host}:{args.zmq_port} -> {client.ping()}", flush=True)

    handler = make_handler(client, args.instruction)
    httpd = ThreadingHTTPServer((args.http_host, args.http_port), handler)
    print(f"[bridge] HTTP↔ZMQ bridge listening on http://{args.http_host}:{args.http_port} "
          f"(POST /get_action)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[bridge] shutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
