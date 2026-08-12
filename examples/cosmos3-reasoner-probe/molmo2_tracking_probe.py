"""Direction B — grasp verification in Molmo 2's NATIVE modality: the sequence.

Direction A (mission-molmo2.yaml) measures Molmo 2 as a single-frame VQA
judge with the Cosmos arm's verbatim prompts — apples-to-apples, but not what
Molmo 2 is for. Its specialty is video: pointing, tracking, timestamps. This
probe asks the question the way the map phrases the role — "does the vial
travel WITH the gripper during the lift?" — over K chronological frames sent
in ONE multi-image chat request. Deliberately NOT comparable with direction
A; the two directions are pre-registered as separate measurements.

Two questions per rollout (not per frame):

* ``carry``       — the headline: was the object grasped and does it travel
  with the gripper through the lift?  YES/NO over the sequence. Expect YES on
  successful rollouts, NO on failures.
* ``grasp_frame`` — "in which frame (0-indexed) does the grasp first occur?
  Answer with only the integer."  Measurable against the known early pick
  window (~first 15% of these episodes); a sequence-blind model cannot answer
  it consistently.

Custom-eval script (``evaluation_type: custom`` contract: ``--checkpoint`` /
``--out-json`` + passthrough flags). The server must allow multi-image
requests (``--limit-mm-per-prompt '{"image": 16}'`` — see the serving recipe
in mission-molmo2.yaml).
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

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reasoner_probe import prepare_frame  # noqa: E402

CARRY_TEMPLATE = (
    "These images are chronological frames from one robot manipulation "
    "episode, in temporal order. The robot is attempting this instruction: "
    "{instruction!r}. Reason over the SEQUENCE: did the robot grasp the "
    "object, and does the object travel WITH the gripper during the lift "
    "(moving together across consecutive frames, not just touching in one "
    "frame)? Answer with exactly one word: YES or NO."
)

GRASP_FRAME_TEMPLATE = (
    "These images are chronological frames (indexed 0 to {last_index}) from "
    "one robot manipulation episode, in temporal order. The robot is "
    "attempting this instruction: {instruction!r}. In which frame index does "
    "the robot FIRST have the object grasped in its gripper? If it never "
    "grasps the object, answer -1. Answer with ONLY the integer."
)


def frames_data_uris(
    video: Path, num_frames: int, view: str, upscale: int, window: str = "0.0,1.0"
) -> tuple[list[str], list[int], int]:
    """K evenly-spaced frames within an episode-fraction window, as data URIs.

    ``window`` (iteration 2): "0.0,0.4" densifies sampling where the action
    actually is — the pick lands in the first ~15% of these episodes, so
    whole-episode spacing wastes most frames on the post-place idle tail.
    """
    import imageio.v3 as iio
    from PIL import Image

    stack = iio.imread(video)
    start_frac, end_frac = (float(x) for x in window.split(","))
    lo = int(start_frac * (len(stack) - 1))
    hi = max(int(end_frac * (len(stack) - 1)), lo + 1)
    indexes = [
        min(lo + int(i * (hi - lo) / max(num_frames - 1, 1)), len(stack) - 1)
        for i in range(num_frames)
    ]
    total_frames = len(stack)
    uris = []
    for index in indexes:
        array = prepare_frame(stack[index], view, upscale)
        buf = io.BytesIO()
        Image.fromarray(array).save(buf, format="PNG")
        uris.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
    return uris, indexes, total_frames


def ask_sequence(
    uris: list[str], text: str, args: argparse.Namespace
) -> tuple[str | None, float]:
    """One multi-image chat request -> (raw reply or None, latency)."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": uri}} for uri in uris
    ]
    content.append({"type": "text", "text": text})
    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": content}],
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    try:
        reply = json.load(urllib.request.urlopen(request, timeout=args.timeout_seconds))
        return str(reply["choices"][0]["message"]["content"]), time.monotonic() - start
    except Exception as error:
        print(f"  call failed: {error}", flush=True)
        return None, time.monotonic() - start


def parse_yes_no(reply: str | None) -> str:
    if reply is None:
        return "FAIL"
    match = re.search(r"\b(YES|NO)\b", reply.upper())
    return match.group(1) if match else "UNPARSED"


def parse_frame_index(reply: str | None) -> int | None:
    if reply is None:
        return None
    match = re.search(r"-?\d+", reply)
    return int(match.group()) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="recorded in the summary only")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--model", default="allenai/Molmo2-8B")
    parser.add_argument("--videos_dir", required=True, help="dir of rollout MP4s to probe")
    parser.add_argument("--instruction", required=True, help="the task the rollouts attempted")
    parser.add_argument("--max_videos", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=8, help="frames per request (<= server limit)")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--timeout_seconds", type=float, default=300.0)
    parser.add_argument("--view", choices=["full", "wrist", "side"], default="full")
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument(
        "--window",
        default="0.0,1.0",
        help='episode-fraction sampling window, e.g. "0.0,0.4" (iteration 2: densify the pick)',
    )
    args = parser.parse_args()

    videos = sorted(Path(args.videos_dir).expanduser().glob("*.mp4"))[: args.max_videos]
    if not videos:
        raise SystemExit(f"no .mp4 rollouts found under {args.videos_dir}")

    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    for video in videos:
        uris, indexes, total_frames = frames_data_uris(
            video, args.num_frames, args.view, args.upscale, args.window
        )
        carry_raw, carry_latency = ask_sequence(
            uris, CARRY_TEMPLATE.format(instruction=args.instruction), args
        )
        frame_raw, frame_latency = ask_sequence(
            uris,
            GRASP_FRAME_TEMPLATE.format(
                last_index=len(uris) - 1, instruction=args.instruction
            ),
            args,
        )
        latencies += [carry_latency, frame_latency]
        grasp_frame = parse_frame_index(frame_raw)
        # Which real video frame the claimed grasp index maps to, as a fraction
        # of the WHOLE episode (not the window) — comparable against the known
        # early pick window regardless of sampling.
        grasp_fraction = (
            round(indexes[grasp_frame] / max(total_frames - 1, 1), 3)
            if grasp_frame is not None and 0 <= grasp_frame < len(indexes)
            else None
        )
        results.append(
            {
                "video": video.name,
                "carry": parse_yes_no(carry_raw),
                "carry_raw": (carry_raw or "")[:120] or None,
                "grasp_frame": grasp_frame,
                "grasp_fraction": grasp_fraction,
                "grasp_frame_raw": (frame_raw or "")[:120] or None,
                "sampled_indexes": indexes,
                "latency_s": round(carry_latency + frame_latency, 2),
            }
        )
        print(
            f"{video.name}: carry={results[-1]['carry']} "
            f"grasp_frame={grasp_frame} (fraction={grasp_fraction}) "
            f"({results[-1]['latency_s']}s)",
            flush=True,
        )

    calls = [r for r in results for _ in (0, 1)]
    ok = sum(r["carry"] not in ("FAIL",) for r in results) + sum(
        r["grasp_frame_raw"] is not None for r in results
    )
    payload = {
        "num_episodes": len(results),
        "metrics": {
            "endpoint": args.base_url,
            "model": args.model,
            "num_frames_per_request": args.num_frames,
            "window": args.window,
            "view": args.view,
            "upscale": args.upscale,
            "call_success_rate": round(ok / len(calls), 3) if calls else 0.0,
            "latency_s_mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "carry_yes_rate": round(
                sum(r["carry"] == "YES" for r in results) / len(results), 3
            ),
            "videos": results,
        },
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"wrote metrics -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
