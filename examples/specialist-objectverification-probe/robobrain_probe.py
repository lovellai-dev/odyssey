"""Probe a served RoboBrain 2.5 as a SPECIALIST candidate (object verification).

Custom-eval script (``evaluation_type: custom`` contract: ``--checkpoint`` /
``--out-json`` + passthrough flags). RoboBrain 2.5 arm of the Specialist Model
Map v0.5 **object-verification** bake-off, challenged against SAM 3.1 (probed by
``sam_probe.py`` in this same directory). Both arms answer the same paper —
"which of these named objects does the model see in this frame?" — over the same
rollout frames, so any verdict delta is the model, not the pixels.

The two arms do NOT share a prompt surface (SAM is promptable *concept
segmentation*, not a YES/NO chat), so what is held constant across arms is the
**object set** and the **frames**, not a verbatim prompt. See README.md.

This arm exercises ``OpenAICompatCompletionJudge`` — the exact
``CompletionDetector`` surface the multi-agent runtimes gate on — against a
*vanilla* vLLM endpoint (RoboBrain 2.5 is Qwen3-VL-8B-based; no vLLM-Omni, no
``modalities`` routing knob needed). Per sampled frame it asks one YES/NO
question per named object:

* ``control``       — "is a robot arm and tabletop clearly visible?"  Detects
  the always-NO degenerate mode issue #78 found in the Gemma int4 judge.
* ``present:<obj>`` — for each object the scene DOES contain (e.g. the drug-sort
  ``red capsule`` and ``blue tray``): a good detector answers YES → this is the
  recall / recognition signal.
* ``absent:<obj>``  — for each DISTRACTOR object the scene does NOT contain
  (e.g. ``green bottle``): a good detector answers NO → this is the
  false-positive / specificity signal (an always-YES model is as useless as an
  always-NO one).

Frames are sampled at fractional positions through each rollout MP4. Metric-
only: per-object YES rates + a present-recall / distractor-FPR summary +
latency + per-frame verdicts. No fabricated success_rate.

RoboBrain 2.5 reasons before answering; the judge's 8-token default would
truncate that before any YES/NO token (parsing as a spurious NO), so
``--max_tokens`` defaults high here and the raw replies are captured for
diagnosis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Runs under the eval env where odyssey may not be installed — resolve the
# repo's src/ relative to this file, like the other example bridges.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

from odyssey.runners.agents.openai_judge import (  # noqa: E402
    OpenAICompatCompletionJudge,
)

# The control question is a verbatim copy from the retry/grasp probes — the
# always-NO degeneracy detector must stay identical across every specialist role
# so its reading is comparable. ``{instruction:.0s}`` consumes the query slot.
CONTROL_TEMPLATE = (
    "You are a strict visual judge for a robot manipulation scene. "
    "Look at the image and answer whether a robot arm and a tabletop scene "
    "are clearly visible. Answer with exactly one word: YES or NO."
    "{instruction:.0s}"
)

# The presence question. The object name is fed through the judge's ``instruction``
# slot (``is_complete(frame, object_name)``), so ONE template serves every object,
# present or distractor — the polarity is decided by whether the scene contains it,
# not by the wording. Open-vocabulary, matching SAM's promptable-concept surface.
PRESENCE_TEMPLATE = (
    "You are a strict open-vocabulary object detector for a robot manipulation "
    "scene. Look at the image and answer whether a {instruction} is visible "
    "somewhere in the scene. Only answer YES if you can actually see it. "
    "Answer with exactly one word: YES or NO."
)

FRAME_POSITIONS = (0.0, 0.5, 0.75, 1.0)


def build_questions(
    objects: list[str], distractors: list[str]
) -> list[tuple[str, str, str]]:
    """(question_name, judge_template, query) rows.

    ``query`` is what gets passed to ``is_complete`` as the ``instruction`` slot:
    ignored for control, the object name for presence/absence rows.
    """
    questions: list[tuple[str, str, str]] = [("control", CONTROL_TEMPLATE, "")]
    for obj in objects:
        questions.append((f"present:{obj}", PRESENCE_TEMPLATE, obj))
    for obj in distractors:
        questions.append((f"absent:{obj}", PRESENCE_TEMPLATE, obj))
    return questions


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
        # Vanilla vLLM: no extra_body. (The Cosmos arm needs
        # {"modalities": ["text"]} to dodge vLLM-Omni's diffusion routing;
        # RoboBrain has no such surface.)
    )


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="recorded in the summary only")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="BAAI/RoboBrain2.5-8B-NV")
    parser.add_argument("--videos_dir", required=True, help="dir of rollout MP4s to probe")
    parser.add_argument("--instruction", required=True, help="the task the rollouts attempted")
    parser.add_argument(
        "--objects",
        required=True,
        help="comma-separated objects the scene CONTAINS (recall signal)",
    )
    parser.add_argument(
        "--distractors",
        default="",
        help="comma-separated objects the scene does NOT contain (FPR signal)",
    )
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

    objects = _split_csv(args.objects)
    distractors = _split_csv(args.distractors)
    if not objects:
        raise SystemExit("--objects must name at least one present object")
    questions = build_questions(objects, distractors)

    frames = load_probe_frames(
        Path(args.videos_dir).expanduser(), args.max_videos, args.view, args.upscale
    )

    verdicts: list[dict[str, Any]] = []
    latencies: list[float] = []
    raw_replies: list[str | None] = []

    for name, template, query in questions:
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
            answer = judge.is_complete(frame["image"], query)
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
                f"[{name:>22}] {frame['video']} ({frame['position']}): "
                f"{'YES' if answer else 'NO'} ({elapsed:.1f}s)",
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
            "view": args.view,
            "upscale": args.upscale,
            "judge_calls": len(verdicts),
            "call_success_rate": round(calls_ok / len(verdicts), 3),
            "latency_s_mean": round(sum(latencies) / len(latencies), 2),
            "latency_s_max": round(max(latencies), 2),
            "control_yes_rate": yes_rate("control"),
            # Headline object-verification metrics: recall on present objects
            # (want high) vs false-positive rate on distractors (want low).
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
