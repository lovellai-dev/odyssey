"""Probe a served Cosmos 3 Reasoner as a SPECIALIST candidate (grasp verification).

Custom-eval script (``evaluation_type: custom`` contract: ``--checkpoint`` /
``--out-json`` + passthrough flags). It exercises ``OpenAICompatCompletionJudge``
— the exact ``CompletionDetector`` surface the multi-agent runtimes gate on —
against an OpenAI-compatible endpoint serving a Cosmos 3 Reasoner (vLLM-Omni),
in the DELEGATION spirit of the closed planner-vs-delegation experiment
(PR #68): the SPECIALIST authors no plan; it is asked perception questions on
demand. Four YES/NO questions per sampled frame:

* ``control``    — "is a robot arm clearly visible?"  Detects the always-NO
  degenerate mode issue #78 found in the Gemma int4 ``check_done`` judge.
* ``grasp``      — **the headline check**: "is the object currently secured in
  the gripper?"  Expect NO on pre-grasp / post-place frames and YES only while
  the object is actually held (mid-episode of successful rollouts).
* ``completion`` — the judge's stock done-check. Expect YES only at the end of
  successful rollouts.
* ``retry``      — "is the robot failed/stuck such that it should abort and
  retry?"  Expect YES only late in failed rollouts.

Frames are sampled at fractional positions through each rollout MP4 so the
grasp window of successful episodes is covered. Metric-only: rates + latency +
per-frame verdicts, no fabricated success_rate (position ground truth is
qualitative).

The Cosmos 3 Reasoner emits chain-of-thought; the judge's 8-token default would
truncate it before any YES/NO token (parsing as a spurious NO), so ``--max_tokens``
defaults high here and the raw replies are captured for diagnosis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Runs under the eval env (env_pilot_cosmos3 etc.) where odyssey may not be
# installed — resolve the repo's src/ relative to this file, like the other
# example bridges.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

from odyssey.runners.agents.openai_judge import (  # noqa: E402
    DEFAULT_PROMPT_TEMPLATE,
    OpenAICompatCompletionJudge,
)

CONTROL_TEMPLATE = (
    "You are a strict visual judge for a robot manipulation scene. "
    "Look at the image and answer whether a robot arm and a tabletop scene "
    "are clearly visible. Answer with exactly one word: YES or NO."
    "{instruction:.0s}"  # template must consume the instruction slot
)

# Iteration 2 (dense-sweep + visualizer finding): the original long,
# task-anchored phrasing ("the robot is attempting {instruction} ... secured
# ... NOT a grasp if ...") biased to NO even inside the true grasp window,
# while short pure-perception phrasing tracked it. Grasp verification is
# asked as perception, not as task judgement.
GRASP_TEMPLATE = (
    "You are a strict visual judge for a robot manipulation scene. "
    "Look at the image and answer whether the robot gripper is CURRENTLY "
    "holding an object between its fingers. "
    "Answer with exactly one word: YES or NO."
    "{instruction:.0s}"
)

RETRY_TEMPLATE = (
    "You are a strict visual judge monitoring a robot manipulation task. "
    "The robot is attempting this instruction: {instruction!r}. "
    "Look at the image and answer whether the robot is in a FAILED or STUCK "
    "state such that it should abort and RETRY the task from the start "
    "(e.g. object dropped or knocked over, arm wedged or flailing away from "
    "the target, target unreachable from the current pose). "
    "A nominal in-progress attempt is NOT a retry state. "
    "Answer with exactly one word: YES or NO."
)

QUESTIONS = (
    ("control", CONTROL_TEMPLATE),
    ("grasp", GRASP_TEMPLATE),
    ("completion", DEFAULT_PROMPT_TEMPLATE),
    ("retry", RETRY_TEMPLATE),
)

FRAME_POSITIONS = (0.0, 0.5, 0.75, 1.0)


def prepare_frame(frame: Any, view: str, upscale: int) -> Any:
    """Crop to one half of a concat_view frame and/or upscale.

    The issue #78 lesson (``check_crop``/``check_upscale``): small concat sim
    frames starve the judge of pixels on the region that matters. ``wrist``
    keeps the right half and ``side`` the left half of a 2:1-wide concat frame
    (non-concat frames pass through untouched); ``upscale`` LANCZOS-resizes by
    an integer factor.
    """
    import numpy as np
    from PIL import Image

    array = np.asarray(frame)
    height, width = array.shape[:2]
    if view in ("wrist", "side") and width == 2 * height:
        array = array[:, width // 2 :] if view == "wrist" else array[:, : width // 2]
    if upscale > 1:
        img = Image.fromarray(array)
        img = img.resize((img.width * upscale, img.height * upscale), Image.LANCZOS)
        array = np.asarray(img)
    return array


def load_probe_frames(
    videos_dir: Path, max_videos: int, view: str = "full", upscale: int = 1
) -> list[dict[str, Any]]:
    """Frames at ``FRAME_POSITIONS`` of each rollout MP4 (sorted for determinism)."""
    import imageio.v3 as iio

    videos = sorted(videos_dir.glob("*.mp4"))[:max_videos]
    if not videos:
        raise SystemExit(f"no .mp4 rollouts found under {videos_dir}")
    frames: list[dict[str, Any]] = []
    for video in videos:
        stack = iio.imread(video)
        for fraction in FRAME_POSITIONS:
            index = min(int(fraction * (len(stack) - 1)), len(stack) - 1)
            frames.append(
                {
                    "video": video.name,
                    "position": f"p{int(fraction * 100)}",
                    "image": prepare_frame(stack[index], view, upscale),
                }
            )
    return frames


def make_judge(args: argparse.Namespace, template: str) -> OpenAICompatCompletionJudge:
    return OpenAICompatCompletionJudge(
        base_url=args.base_url,
        model=args.model,
        prompt_template=template,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        # vLLM-Omni routes image+text chat to IMAGE GENERATION (50 diffusion
        # steps, image reply, no YES/NO text) unless the request selects the
        # text output modality explicitly.
        extra_body={"modalities": ["text"]},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="recorded in the summary only")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="nvidia/Cosmos3-Nano")
    parser.add_argument("--videos_dir", required=True, help="dir of rollout MP4s to probe")
    parser.add_argument("--instruction", required=True, help="the task the rollouts attempted")
    parser.add_argument("--max_videos", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--timeout_seconds", type=float, default=180.0)
    parser.add_argument(
        "--view",
        choices=["full", "wrist", "side"],
        default="full",
        help="crop half of a 2:1 concat_view frame (issue #78 zoom lesson)",
    )
    parser.add_argument("--upscale", type=int, default=1, help="integer LANCZOS upscale factor")
    args = parser.parse_args()

    frames = load_probe_frames(
        Path(args.videos_dir).expanduser(), args.max_videos, args.view, args.upscale
    )

    verdicts: list[dict[str, Any]] = []
    latencies: list[float] = []
    raw_replies: list[str | None] = []

    for name, template in QUESTIONS:
        judge = make_judge(args, template)
        original_post = judge._post

        def capturing_post(
            payload: dict[str, Any], _post: Any = original_post
        ) -> dict[str, Any]:
            response = _post(payload)
            try:
                raw_replies.append(str(response["choices"][0]["message"]["content"]))
            except Exception:
                raw_replies.append(None)
            return response

        judge._post = capturing_post  # type: ignore[method-assign]

        for frame in frames:
            raw_before = len(raw_replies)
            start = time.monotonic()
            answer = judge.is_complete(frame["image"], args.instruction)
            elapsed = time.monotonic() - start
            latencies.append(elapsed)
            reply = raw_replies[raw_before] if len(raw_replies) > raw_before else None
            verdicts.append(
                {
                    "question": name,
                    "video": frame["video"],
                    "position": frame["position"],
                    "answer": "YES" if answer else "NO",
                    "latency_s": round(elapsed, 2),
                    "reply_excerpt": (reply or "")[-200:] or None,
                    "call_failed": reply is None,
                }
            )
            print(
                f"[{name:>10}] {frame['video']} ({frame['position']}): "
                f"{'YES' if answer else 'NO'} ({elapsed:.1f}s)",
                flush=True,
            )

    def yes_rate(question: str) -> float | None:
        rows = [v for v in verdicts if v["question"] == question]
        return round(sum(v["answer"] == "YES" for v in rows) / len(rows), 3) if rows else None

    calls_ok = sum(not v["call_failed"] for v in verdicts)
    payload = {
        "num_episodes": len(frames),
        "metrics": {
            "endpoint": args.base_url,
            "model": args.model,
            "frames_probed": len(frames),
            "view": args.view,
            "upscale": args.upscale,
            "judge_calls": len(verdicts),
            "call_success_rate": round(calls_ok / len(verdicts), 3),
            "latency_s_mean": round(sum(latencies) / len(latencies), 2),
            "latency_s_max": round(max(latencies), 2),
            # Aggregate YES rates per question; the discriminative signal lives
            # in the per-frame verdicts below (which video x position said YES).
            "control_yes_rate": yes_rate("control"),
            "grasp_yes_rate": yes_rate("grasp"),
            "completion_yes_rate": yes_rate("completion"),
            "retry_yes_rate": yes_rate("retry"),
            "verdicts": verdicts,
        },
    }
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"wrote metrics -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
