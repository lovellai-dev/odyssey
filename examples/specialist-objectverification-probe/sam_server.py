"""Minimal reference server exposing SAM 3.1 concept segmentation over HTTP.

The SAM arm of the object-verification bake-off is SYMMETRIC with the RoboBrain
arm: RoboBrain is served by vanilla vLLM on port 8002, SAM 3.1 by this process
on port 8003, each in its own venv. ``sam_probe.py`` / ``utils/visualize_sam_
probe.py`` are the only clients; they speak this exact contract:

    POST /segment  {"image": "<data-uri or bare base64 PNG>", "concept": "red capsule"}
      -> {"detections": [{"score": <0..1>, "box": [x0, y0, x1, y1]}...],
          "image_size": [W, H]}          # box coords are PIXELS in the posted frame
    GET  /info     -> {"model": "<id>", "device": "<cuda|cpu>", "fake": <bool>}

stdlib-only HTTP (``http.server``) so the server adds NO dependency beyond the
SAM backend itself — matching the judge's urllib-only transport. The SAM
backend is lazy-loaded on first request.

The exact SAM 3.1 promptable-concept-segmentation call is isolated in
``SamBackend.segment`` — adapt that ONE method to the released API (it targets
the HF ``transformers`` concept-segmentation head; if the package differs, only
this method changes). Run with ``SAM_SERVER_FAKE=1`` to serve deterministic
detections with no weights at all — for exercising the probe, the tunnel and the
viewer end-to-end offline.

    python sam_server.py --host 0.0.0.0 --port 8003 --model facebook/sam3.1
    SAM_SERVER_FAKE=1 python sam_server.py          # no GPU, no weights
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class SamBackend:
    """Lazy wrapper around SAM 3.1 promptable concept segmentation."""

    def __init__(self, model_id: str, score_threshold: float, fake: bool) -> None:
        self.model_id = model_id
        self.score_threshold = score_threshold
        self.fake = fake
        self._model: Any = None
        self._processor: Any = None
        self.device = "cpu"

    def _ensure_loaded(self) -> None:
        if self.fake or self._model is not None:
            return
        # --- ADAPT HERE to the released SAM 3.1 API -------------------------
        # Targets the HF transformers concept-segmentation head. If SAM 3.1
        # ships its own package, swap these two lines and `segment` below;
        # nothing else in the server needs to change.
        import torch  # noqa: F401
        from transformers import AutoModelForMaskGeneration, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForMaskGeneration.from_pretrained(self.model_id)
        self.device = "cuda" if _cuda_available() else "cpu"
        self._model.to(self.device)
        # -------------------------------------------------------------------

    def segment(self, image: Any, concept: str) -> dict[str, Any]:
        """Return detections ({score, box:[x0,y0,x1,y1]}) for one text concept."""
        width, height = image.size
        if self.fake:
            return _fake_detections(concept, width, height)

        self._ensure_loaded()
        # --- ADAPT HERE to the released SAM 3.1 API -------------------------
        # Promptable concept segmentation: text concept -> instance masks +
        # scores. Convert each returned mask to its bounding box + score.
        import numpy as np
        import torch

        inputs = self._processor(images=image, text=concept, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            outputs = self._model(**inputs)
        masks = self._processor.post_process_masks(
            outputs.pred_masks, [(height, width)]
        )[0]
        scores = outputs.iou_scores.squeeze().tolist()
        scores = scores if isinstance(scores, list) else [scores]
        detections: list[dict[str, Any]] = []
        for mask, score in zip(np.asarray(masks), scores, strict=False):
            ys, xs = np.where(mask.squeeze() > 0.5)
            if xs.size == 0:
                continue
            detections.append(
                {
                    "score": round(float(score), 3),
                    "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                }
            )
        # -------------------------------------------------------------------
        return {"detections": detections, "image_size": [width, height]}


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _fake_detections(concept: str, width: int, height: int) -> dict[str, Any]:
    """Deterministic detections mirroring ``sam_probe._fake_transport`` polarity."""
    seed = zlib.crc32(concept.encode()) % 100 / 100.0
    looks_absent = any(w in concept.lower() for w in ("green", "purple", "banana", "wrench"))
    score = 0.15 + 0.2 * seed if looks_absent else 0.72 + 0.2 * seed
    detections: list[dict[str, Any]] = []
    if score >= 0.3:
        cx, cy = int(width * 0.45), int(height * 0.5)
        box = [cx - width // 8, cy - height // 8, cx + width // 8, cy + height // 8]
        detections.append({"score": round(score, 3), "box": box})
    return {"detections": detections, "image_size": [width, height]}


def _decode_image(data_uri: str) -> Any:
    from PIL import Image

    raw = data_uri.split(",", 1)[1] if data_uri.startswith("data:") else data_uri
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def make_handler(backend: SamBackend) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/info":
                self._send(
                    200,
                    {"model": backend.model_id, "device": backend.device, "fake": backend.fake},
                )
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/segment":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                image = _decode_image(body["image"])
                result = backend.segment(image, str(body["concept"]))
                self._send(200, result)
            except Exception as exc:
                self._send(500, {"error": str(exc), "detections": []})

        def log_message(self, *_args: Any) -> None:  # quiet
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--model", default="facebook/sam3.1")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument(
        "--fake",
        action="store_true",
        help="serve deterministic detections with no weights (or set SAM_SERVER_FAKE=1)",
    )
    args = parser.parse_args()

    fake = args.fake or os.environ.get("SAM_SERVER_FAKE") == "1"
    backend = SamBackend(args.model, args.score_threshold, fake)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(backend))
    mode = "FAKE (no weights)" if fake else backend.model_id
    print(f"SAM 3.1 object-verification server on {args.host}:{args.port} — {mode}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
