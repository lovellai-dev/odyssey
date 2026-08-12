"""Probe a served SAM 3.1 as a SPECIALIST candidate (object verification).

Custom-eval script (``evaluation_type: custom`` contract: ``--checkpoint`` /
``--out-json`` + passthrough flags). SAM 3.1 arm of the Specialist Model Map
v0.5 **object-verification** bake-off, the challenger to the RoboBrain 2.5 arm
(``robobrain_probe.py``). Both arms answer the same paper — "which of these
named objects does the model see in this frame?" — over the SAME rollout frames
and the SAME object set, so any verdict delta is the model.

Interface asymmetry (see README.md): SAM is **promptable concept segmentation**,
not a YES/NO chat, so this arm does NOT reuse ``OpenAICompatCompletionJudge``.
It POSTs each (frame, concept) pair to a served SAM 3.1 endpoint and turns the
returned mask/score into the same present/absent verdict the RoboBrain arm
emits. What is held constant across arms is the object set + frames, not a
verbatim prompt.

Served endpoint contract (a minimal reference server ships as ``sam_server.py``
in this dir; SYMMETRIC with the RoboBrain arm's vLLM endpoint — its own port,
its own venv):

    POST {base_url}/segment   {"image": <data-uri>, "concept": "<object name>"}
      -> {"detections": [{"score": <float 0..1>, "box": [x0,y0,x1,y1]}...],
          "image_size": [W, H]}          # box coords are pixels in the frame
    GET  {base_url}/info      -> {"model": "<id>"}    (optional; logged only)

A concept is judged PRESENT when its best detection score clears
``--score_threshold``. Per-object present/absent verdicts, the best score, the
detection count, and the recall / FPR summary land in the metrics JSON — the
same shape the RoboBrain arm writes, so a single downstream read compares them.

``--fake`` answers deterministically with no server, for wiring/offline checks.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.request
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Reuse the RoboBrain arm's frame helpers so both arms crop/sample IDENTICALLY.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from robobrain_probe import (  # noqa: E402
    FRAME_POSITIONS,
    load_probe_frames,
)

Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


def _png_data_uri(array: Any) -> str:
    from PIL import Image

    img = Image.fromarray(array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _http_transport(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fake_transport(url: str, body: dict[str, Any]) -> dict[str, Any]:
    """Deterministic offline detections (wiring test only, no server).

    Present-looking concepts (short names, no "green"/"purple" distractor hint)
    score high; the crc is only to make it vary frame-to-frame without a server.
    """
    concept = str(body.get("concept", ""))
    seed = zlib.crc32(concept.encode()) % 100 / 100.0
    looks_absent = any(w in concept.lower() for w in ("green", "purple", "banana", "wrench"))
    score = 0.15 + 0.2 * seed if looks_absent else 0.72 + 0.2 * seed
    detections = [] if score < 0.3 else [{"score": round(score, 3), "box": [40, 40, 120, 120]}]
    return {"detections": detections, "image_size": [224, 224]}


def build_concepts(
    objects: list[str], distractors: list[str]
) -> list[tuple[str, str]]:
    """(question_name, concept) rows — mirrors the RoboBrain arm's questions.

    Control uses a concept every manipulation frame contains (``robot arm``);
    present objects are the recall signal, distractors the false-positive one.
    """
    rows: list[tuple[str, str]] = [("control", "robot arm")]
    rows += [(f"present:{obj}", obj) for obj in objects]
    rows += [(f"absent:{obj}", obj) for obj in distractors]
    return rows


def segment(
    transport: Transport, base_url: str, image_uri: str, concept: str
) -> dict[str, Any]:
    try:
        return transport(f"{base_url}/segment", {"image": image_uri, "concept": concept})
    except Exception as exc:  # transport failure -> recorded, judged absent
        return {"detections": [], "error": str(exc)}


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="recorded in the summary only")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:8003")
    parser.add_argument("--model", default="facebook/sam3.1")
    parser.add_argument("--videos_dir", required=True, help="dir of rollout MP4s to probe")
    parser.add_argument("--instruction", required=True, help="the task the rollouts attempted")
    parser.add_argument("--objects", required=True, help="comma-separated present objects")
    parser.add_argument("--distractors", default="", help="comma-separated absent objects")
    parser.add_argument("--max_videos", type=int, default=4)
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="best-detection score at/above which a concept counts as PRESENT",
    )
    parser.add_argument("--timeout_seconds", type=float, default=180.0)
    parser.add_argument("--view", choices=["full", "wrist", "side"], default="full")
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--fake", action="store_true", help="no server: deterministic detections")
    args = parser.parse_args()

    objects = _split_csv(args.objects)
    distractors = _split_csv(args.distractors)
    if not objects:
        raise SystemExit("--objects must name at least one present object")
    concepts = build_concepts(objects, distractors)

    frames = load_probe_frames(
        Path(args.videos_dir).expanduser(), args.max_videos, args.view, args.upscale
    )

    if args.fake:
        transport: Transport = _fake_transport
    else:
        def transport(url: str, body: dict[str, Any]) -> dict[str, Any]:
            return _http_transport(url, body, args.timeout_seconds)

    verdicts: list[dict[str, Any]] = []
    latencies: list[float] = []

    for frame in frames:
        image_uri = _png_data_uri(frame["image"])
        for name, concept in concepts:
            start = time.monotonic()
            result = segment(transport, args.base_url, image_uri, concept)
            elapsed = time.monotonic() - start
            latencies.append(elapsed)
            detections = result.get("detections") or []
            best = max((d.get("score", 0.0) for d in detections), default=0.0)
            present = best >= args.score_threshold
            verdicts.append(
                {
                    "question": name,
                    "concept": concept,
                    "video": frame["video"],
                    "position": frame["position"],
                    "answer": "YES" if present else "NO",
                    "best_score": round(float(best), 3),
                    "num_detections": len(detections),
                    "latency_s": round(elapsed, 2),
                    "call_failed": "error" in result,
                    "error": result.get("error"),
                }
            )
            print(
                f"[{name:>22}] {frame['video']} ({frame['position']}): "
                f"{'YES' if present else 'NO'} score={best:.2f} ({elapsed:.1f}s)",
                flush=True,
            )

    def yes_rate(question: str) -> float | None:
        rows = [v for v in verdicts if v["question"] == question]
        return round(sum(v["answer"] == "YES" for v in rows) / len(rows), 3) if rows else None

    present_rates = [r for obj in objects if (r := yes_rate(f"present:{obj}")) is not None]
    distractor_rates = [r for obj in distractors if (r := yes_rate(f"absent:{obj}")) is not None]

    def _mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    calls_ok = sum(not v["call_failed"] for v in verdicts)
    payload = {
        "num_episodes": len(frames),
        "metrics": {
            "endpoint": args.base_url,
            "model": args.model,
            "frames_probed": len(frames),
            "objects": objects,
            "distractors": distractors,
            "score_threshold": args.score_threshold,
            "view": args.view,
            "upscale": args.upscale,
            "segment_calls": len(verdicts),
            "call_success_rate": round(calls_ok / len(verdicts), 3),
            "latency_s_mean": round(sum(latencies) / len(latencies), 2),
            "latency_s_max": round(max(latencies), 2),
            "control_yes_rate": yes_rate("control"),
            # Same headline metrics as the RoboBrain arm, so one read compares them.
            "present_recall": _mean(present_rates),
            "distractor_fpr": _mean(distractor_rates),
            "per_object_yes_rate": {obj: yes_rate(f"present:{obj}") for obj in objects},
            "per_distractor_yes_rate": {obj: yes_rate(f"absent:{obj}") for obj in distractors},
            "verdicts": verdicts,
        },
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"wrote metrics -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()


# Re-exported for the viewer so both share the exact frame sampling positions.
__all__ = ["FRAME_POSITIONS", "_fake_transport", "_png_data_uri", "build_concepts", "segment"]
