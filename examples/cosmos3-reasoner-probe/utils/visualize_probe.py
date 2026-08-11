"""Watch the probe interrogate the SPECIALIST, frame by frame.

Makes the experiment transparent: the model never sees the *video* — it sees
isolated frames, and per frame each question is one independent image+text
HTTP call. This tool replays that loop over a single rollout MP4 and shows it:

* terminal — one table row per (frame, question) as each answer arrives;
* HTML report — the playable video, the judged (cropped/upscaled) thumbnail
  per sampled frame, and the growing Q/A table; clicking a row seeks the
  video to that frame. The file is rewritten after every frame and
  auto-refreshes while the run is live, so you can watch it fill in.

Usage (server recipe in ../README.md; tunnel with `ssh -L 8002:127.0.0.1:8002`
if the model is served on the H100):

    python examples/cosmos3-reasoner-probe/utils/visualize_probe.py \\
        --video ~/videos/rollout_ep001_success.mp4 \\
        --instruction "pick up the red capsule and place it in the blue tray" \\
        --view wrist --upscale 3 --stride 10 --out /tmp/probe_report.html

    open /tmp/probe_report.html      # while the run is live, or after

``--fake`` answers deterministically without any server — for checking the
viewer itself.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # reasoner_probe (the templates under test)
_REPO_SRC = _HERE.parents[3] / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))         # odyssey (the judge)

from reasoner_probe import (  # noqa: E402
    CONTROL_TEMPLATE,
    GRASP_TEMPLATE,
    RETRY_TEMPLATE,
    prepare_frame,
)

from odyssey.runners.agents.openai_judge import (  # noqa: E402
    DEFAULT_PROMPT_TEMPLATE,
    OpenAICompatCompletionJudge,
)

# The paired opposite of "grasp" — the cheap degeneracy detector from the
# experiment's dense diagnostic sweep (an honest judge answers grasp/empty
# consistently; an always-NO judge answers NO to both).
EMPTY_TEMPLATE = (
    "You are a strict visual judge for a robot manipulation scene. "
    "Look at the image and answer whether the robot gripper is EMPTY — "
    "nothing held between its fingers. "
    "Answer with exactly one word: YES or NO."
    "{instruction:.0s}"
)

QUESTIONS = (
    ("control", CONTROL_TEMPLATE),
    ("grasp", GRASP_TEMPLATE),
    ("empty", EMPTY_TEMPLATE),
    ("completion", DEFAULT_PROMPT_TEMPLATE),
    ("retry", RETRY_TEMPLATE),
)

ANSWER_COLOR = {"YES": "#1a7f37", "NO": "#b3261e", "FAIL": "#8a8a8a"}


def _fake_transport(payload: dict[str, Any]) -> dict[str, Any]:
    """Deterministic offline answers (viewer testing only, no server)."""
    text = payload["messages"][0]["content"][1]["text"]
    verdict = "YES" if zlib.crc32(text.encode()) % 3 == 0 else "NO"
    return {"choices": [{"message": {"content": verdict}}]}


def _png_data_uri(array: Any, max_side: int = 240) -> str:
    from PIL import Image

    img = Image.fromarray(array)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_html(
    out: Path,
    *,
    video_path: Path,
    records: list[dict[str, Any]],
    meta: dict[str, Any],
    done: bool,
) -> None:
    video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    refresh = "" if done else '<meta http-equiv="refresh" content="2">'
    status = "finished" if done else "RUNNING — page auto-refreshes"
    names = [name for name, _ in QUESTIONS]

    badges = "".join(
        f'<span class="badge" id="badge-{n}"><small>{n}</small><b id="bv-{n}">?</b></span>'
        for n in names
    )
    tl_rows = "".join(
        f'<div class="tlrow"><span class="tlabel">{n}</span>'
        f'<div class="tband" data-q="{n}" id="band-{n}"></div></div>'
        for n in names
    )
    rows = []
    for r in records:
        color = ANSWER_COLOR[r["answer"]]
        thumb = (
            f'<img src="{r["thumb"]}" height="72">' if r["question"] == QUESTIONS[0][0] else ""
        )
        rows.append(
            f'<tr id="r{r["frame"]}-{r["question"]}" onclick="seek({r["time_s"]:.2f})">'
            f'<td>{r["frame"]}</td><td>{r["time_s"]:.1f}s</td><td>{thumb}</td>'
            f'<td>{r["question"]}</td>'
            f'<td style="color:{color};font-weight:bold">{r["answer"]}</td>'
            f'<td>{r["latency_s"]:.2f}s</td>'
            f'<td class="q">{html.escape(r["prompt"])}</td></tr>'
        )
    lean = [
        {k: r[k] for k in ("frame", "time_s", "question", "answer")} for r in records
    ]
    data = json.dumps({"records": lean, "questions": names, "colors": ANSWER_COLOR})
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8">{refresh}
<title>Reasoner probe — {html.escape(video_path.name)}</title><style>
body {{ font-family: -apple-system, sans-serif; margin: 1.5rem; }}
table {{ border-collapse: collapse; font-size: 13px; margin-top: 1rem; }}
td, th {{ border: 1px solid #ddd; padding: 4px 8px; vertical-align: middle; }}
tr:hover {{ background: #f6f8fa; cursor: pointer; }}
tr.now {{ background: #fff3c4; }}
td.q {{ max-width: 440px; color: #555; font-size: 11px; }}
.meta {{ color: #555; margin-bottom: 1rem; }}
.badge {{ display: inline-flex; flex-direction: column; align-items: center;
  border: 1px solid #ddd; border-radius: 8px; padding: 4px 10px; margin: 2px;
  min-width: 76px; }}
.badge b {{ font-size: 18px; }}
.speed button {{ margin-right: 4px; }}
#timeline {{ margin-top: 10px; width: 512px; position: relative; }}
.tlrow {{ display: flex; align-items: center; height: 20px; margin: 2px 0; }}
.tlabel {{ width: 84px; font-size: 11px; color: #555; text-align: right;
  padding-right: 6px; }}
.tband {{ position: relative; flex: 1; height: 16px; background: #eee;
  cursor: pointer; border-radius: 3px; overflow: hidden; }}
.seg {{ position: absolute; top: 0; bottom: 0; }}
#playhead {{ position: absolute; top: 0; bottom: 0; width: 2px;
  background: #000; pointer-events: none; left: 84px; }}
</style></head><body>
<h2>Grasp-verification probe — {html.escape(video_path.name)} <small>({status})</small></h2>
<div class="meta">model <b>{html.escape(meta["model"])}</b> · instruction
“{html.escape(meta["instruction"])}” · view {meta["view"]} x{meta["upscale"]} ·
stride {meta["stride"]} · {len(records)} judgements · slow-motion 0.25x by
default — the badges and timeline follow the playhead; click a band or a row
to seek</div>
<video id="v" controls width="512" src="data:video/mp4;base64,{video_b64}"></video>
<div class="speed">speed:
<button onclick="rate(0.1)">0.1x</button><button onclick="rate(0.25)">0.25x</button>
<button onclick="rate(0.5)">0.5x</button><button onclick="rate(1)">1x</button></div>
<div style="margin-top:8px">{badges}</div>
<div id="timeline">{tl_rows}<div id="playhead"></div></div>
<table><tr><th>frame</th><th>t</th><th>judged image</th><th>question</th>
<th>answer</th><th>latency</th><th>full prompt sent</th></tr>
{"".join(rows)}
</table><script>
const DATA = {data};
const v = document.getElementById("v");
v.addEventListener("loadedmetadata", () => {{ v.playbackRate = 0.25; buildTimeline(); }});
function rate(x) {{ v.playbackRate = x; }}
function seek(t) {{ v.currentTime = t + 0.001; }}
const byQ = {{}};
for (const q of DATA.questions) byQ[q] = DATA.records
    .filter(r => r.question === q).sort((a, b) => a.time_s - b.time_s);
function buildTimeline() {{
  const D = v.duration || 1;
  for (const q of DATA.questions) {{
    const band = document.getElementById("band-" + q);
    band.innerHTML = "";
    const pts = byQ[q];
    for (let i = 0; i < pts.length; i++) {{
      const start = pts[i].time_s, end = (i + 1 < pts.length) ? pts[i + 1].time_s : D;
      const seg = document.createElement("div");
      seg.className = "seg";
      seg.style.left = (start / D * 100) + "%";
      seg.style.width = (Math.max(end - start, 0.02) / D * 100) + "%";
      seg.style.background = DATA.colors[pts[i].answer];
      band.appendChild(seg);
    }}
    band.onclick = (e) => {{
      const rect = band.getBoundingClientRect();
      seek((e.clientX - rect.left) / rect.width * D);
    }};
  }}
}}
let lastFrame = null;
function refreshUI() {{
  const t = v.currentTime, D = v.duration || 1;
  const band = document.querySelector(".tband");
  document.getElementById("playhead").style.left =
      (band.offsetLeft + t / D * band.offsetWidth) + "px";
  let current = null;
  for (const r of byQ[DATA.questions[0]]) if (r.time_s <= t + 1e-6) current = r.frame;
  for (const q of DATA.questions) {{
    let pt = null;
    for (const r of byQ[q]) if (r.time_s <= t + 1e-6) pt = r;
    const ans = pt ? pt.answer : "?";
    const el = document.getElementById("bv-" + q);
    el.textContent = ans;
    el.style.color = DATA.colors[ans] || "#333";
  }}
  if (current !== lastFrame) {{
    document.querySelectorAll("tr.now").forEach(el => el.classList.remove("now"));
    for (const q of DATA.questions) {{
      const row = document.getElementById("r" + current + "-" + q);
      if (row) row.classList.add("now");
    }}
    lastFrame = current;
  }}
}}
v.addEventListener("timeupdate", refreshUI);
setInterval(refreshUI, 150);
</script></body></html>""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="nvidia/Cosmos3-Nano")
    parser.add_argument("--view", choices=["full", "wrist", "side"], default="full")
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--stride", type=int, default=10, help="judge every Nth frame")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--timeout_seconds", type=float, default=120.0)
    parser.add_argument("--out", default="/tmp/probe_report.html")
    parser.add_argument("--fake", action="store_true", help="no server: deterministic answers")
    args = parser.parse_args()

    import imageio.v3 as iio

    video_path = Path(args.video).expanduser()
    stack = iio.imread(video_path)
    try:
        fps = float(iio.immeta(video_path).get("fps") or 20.0)
    except Exception:
        fps = 20.0

    replies: list[str | None] = []

    def _capturing(judge: OpenAICompatCompletionJudge) -> OpenAICompatCompletionJudge:
        """Record each raw reply so the viewer can tell FAIL apart from NO
        (the judge itself maps transport failures to a silent NO)."""
        original_post = judge._post

        def post(payload: dict[str, Any], _post: Any = original_post) -> dict[str, Any]:
            response = _post(payload)
            try:
                replies.append(str(response["choices"][0]["message"]["content"]))
            except Exception:
                replies.append(None)
            return response

        judge._post = post  # type: ignore[method-assign]
        return judge

    judges = {
        name: _capturing(
            OpenAICompatCompletionJudge(
                base_url=args.base_url,
                model=args.model,
                prompt_template=template,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds,
                extra_body={"modalities": ["text"]},
                transport=_fake_transport if args.fake else None,
            )
        )
        for name, template in QUESTIONS
    }
    templates = dict(QUESTIONS)
    meta = {
        "model": args.model, "instruction": args.instruction,
        "view": args.view, "upscale": args.upscale, "stride": args.stride,
    }
    out = Path(args.out).expanduser()
    records: list[dict[str, Any]] = []

    indexes = list(range(0, len(stack), args.stride))
    print(f"{len(stack)} frames, judging {len(indexes)} of them x {len(QUESTIONS)} questions")
    print(f"report: {out}  (open it now — it fills in live)\n")
    print(f"{'frame':>6} {'question':>10}  answer  latency")
    for index in indexes:
        frame = prepare_frame(stack[index], args.view, args.upscale)
        thumb = _png_data_uri(frame)
        for name, _template in QUESTIONS:
            replies_before = len(replies)
            start = time.monotonic()
            answer = "YES" if judges[name].is_complete(frame, args.instruction) else "NO"
            elapsed = time.monotonic() - start
            if len(replies) == replies_before:  # transport failed -> judge's silent NO
                answer = "FAIL"
            records.append(
                {
                    "frame": index,
                    "time_s": index / fps,
                    "question": name,
                    "answer": answer,
                    "latency_s": elapsed,
                    "prompt": templates[name].format(instruction=args.instruction),
                    "thumb": thumb,
                }
            )
            print(f"{index:6d} {name:>10}  {answer:<6} {elapsed:.2f}s")
        render_html(out, video_path=video_path, records=records, meta=meta, done=False)
    render_html(out, video_path=video_path, records=records, meta=meta, done=True)

    summary_path = out.with_suffix(".json")
    lean = [{k: v for k, v in r.items() if k != "thumb"} for r in records]
    summary_path.write_text(json.dumps({"meta": meta, "records": lean}, indent=2))
    print(f"\ndone -> {out}  (+ raw records in {summary_path})")


if __name__ == "__main__":
    main()
