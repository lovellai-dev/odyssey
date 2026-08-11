"""Iteration 2 — retry as a PROGRESS CURVE, not a per-frame YES/NO.

The ep000_fail lesson: an eventless failure (robot moves nominally, never
progresses) is invisible to single-frame retry judging — every frame honestly
looks like "a nominal in-progress attempt". The failure lives in the
*sequence*. RoboBrain 2.5's model card claims exactly the matching native
capability (Temporal Value Estimation / Dense Progress Prediction: "estimates
success, failure, error occurrence"), so this probe asks it for a task
progress percentage per sampled frame and reads the *curve*:

* successful rollout  -> the curve rises toward 100;
* eventless failure   -> the curve stays flat (or regresses).

A stalled curve IS the retry signal — a threshold on a continuous, GT-
verifiable quantity instead of a binary opinion. (The released
UnifiedInference code has no dedicated value-estimation template, so this
uses the model's general-VQA mode with a strict integer-only prompt.)

Custom-eval script (``evaluation_type: custom`` contract: ``--checkpoint`` /
``--out-json`` + passthrough flags), so a mission can run it; standalone use:

    python examples/robobrain-retry-probe/robobrain_value_probe.py \\
        --videos_dir ~/cosmos3_probe_videos_success \\
        --instruction "pick up the red capsule and place it in the blue tray" \\
        --view wrist --upscale 3 --stride 5 --out-json /tmp/value_probe.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from robobrain_probe import prepare_frame

VALUE_TEMPLATE = (
    "You are monitoring a robot manipulation task. "
    "The robot is attempting this instruction: {instruction!r}. "
    "Look at the image and estimate how far the task has progressed, "
    "as an integer percentage from 0 (not started) to 100 (fully completed). "
    "Answer with ONLY the integer."
)

SPARK = "▁▂▃▄▅▆▇█"

# A useful retry threshold: by this fraction of the episode, a healthy attempt
# should have risen at least MIN_RISE points over its opening value.
STALL_CHECKPOINT = 0.6
MIN_RISE = 15


def ask_value(
    frame: Any, instruction: str, args: argparse.Namespace
) -> tuple[int | None, str, float]:
    """One general-VQA call -> (parsed 0-100 value or None, raw reply, latency)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG")
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(buf.getvalue()).decode("ascii")
                        },
                    },
                    {"type": "text", "text": VALUE_TEMPLATE.format(instruction=instruction)},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    try:
        reply = json.load(urllib.request.urlopen(request, timeout=args.timeout_seconds))
        text = str(reply["choices"][0]["message"]["content"])
    except Exception as error:  # endpoint/transport problems -> visible, not fatal
        return None, f"<call failed: {error}>", time.monotonic() - start
    match = re.search(r"\d+", text)
    value = min(int(match.group()), 100) if match else None
    return value, text.strip(), time.monotonic() - start


def sparkline(values: list[int | None]) -> str:
    return "".join("·" if v is None else SPARK[min(v, 99) * len(SPARK) // 100] for v in values)


def probe_video(video: Path, args: argparse.Namespace) -> dict[str, Any]:
    import imageio.v3 as iio

    stack = iio.imread(video)
    try:
        fps = float(iio.immeta(video).get("fps") or 20.0)
    except Exception:
        fps = 20.0

    points: list[dict[str, Any]] = []
    for index in range(0, len(stack), args.stride):
        frame = prepare_frame(stack[index], args.view, args.upscale)
        value, raw, latency = ask_value(frame, args.instruction, args)
        points.append(
            {
                "frame": index,
                "time_s": round(index / fps, 2),
                "value": value,
                "latency_s": round(latency, 2),
                "raw_reply": raw[:120],
            }
        )
        print(f"  frame {index:4d}  value={'?' if value is None else value:>3}  ({latency:.1f}s)")

    values = [p["value"] for p in points]
    known = [v for v in values if v is not None]
    opening = known[0] if known else None
    # Progress reached by the stall checkpoint (fraction of the episode).
    checkpoint = [
        v for v in values[: max(1, int(len(values) * STALL_CHECKPOINT))] if v is not None
    ]
    rise_by_checkpoint = (max(checkpoint) - opening) if checkpoint and opening is not None else None
    stalled = rise_by_checkpoint is not None and rise_by_checkpoint < MIN_RISE
    return {
        "video": video.name,
        "curve": sparkline(values),
        "points": points,
        "opening": opening,
        "final": known[-1] if known else None,
        "peak": max(known) if known else None,
        "rise_by_checkpoint": rise_by_checkpoint,
        "stalled": stalled,
        "parse_rate": round(len(known) / len(values), 3) if values else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="recorded in the summary only")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="BAAI/RoboBrain2.5-8B-NV")
    parser.add_argument("--videos_dir", required=True, help="dir of rollout MP4s to probe")
    parser.add_argument("--instruction", required=True, help="the task the rollouts attempted")
    parser.add_argument("--max_videos", type=int, default=8)
    parser.add_argument("--stride", type=int, default=5, help="query every Nth frame")
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--timeout_seconds", type=float, default=120.0)
    parser.add_argument("--view", choices=["full", "wrist", "side"], default="full")
    parser.add_argument("--upscale", type=int, default=1)
    args = parser.parse_args()

    videos = sorted(Path(args.videos_dir).expanduser().glob("*.mp4"))[: args.max_videos]
    if not videos:
        raise SystemExit(f"no .mp4 rollouts found under {args.videos_dir}")

    results = []
    for video in videos:
        print(f"\n{video.name}")
        results.append(probe_video(video, args))

    print("\n=== progress curves ===")
    for result in results:
        flag = "STALLED -> RETRY" if result["stalled"] else "progressing"
        print(
            f"{result['video']:>36}  {result['curve']}  "
            f"open={result['opening']} peak={result['peak']} final={result['final']}  [{flag}]"
        )

    payload = {
        "num_episodes": len(results),
        "metrics": {
            "endpoint": args.base_url,
            "model": args.model,
            "prompt": VALUE_TEMPLATE,
            "view": args.view,
            "upscale": args.upscale,
            "stride": args.stride,
            "stall_checkpoint": STALL_CHECKPOINT,
            "min_rise": MIN_RISE,
            "videos": results,
        },
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote metrics -> {args.out_json}")


if __name__ == "__main__":
    main()
