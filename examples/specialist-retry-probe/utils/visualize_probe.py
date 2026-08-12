"""Watch the probe interrogate the SPECIALIST, frame by frame.

RoboBrain arm of the retry bake-off — a copy of the cosmos3-reasoner-probe
viewer with the serving glue adapted (vanilla vLLM, no Omni ``modalities``
knob, RoboBrain default model id). Makes the experiment transparent: the model
never sees the *video* — it sees isolated frames, and per frame each question
is one independent image+text HTTP call. This tool replays that loop over a
single rollout MP4 and shows it:

* terminal — one table row per (frame, question) as each answer arrives;
* HTML report — the playable video (slow-motion by default), live answer
  badges, per-question timeline bands with a synced playhead, and the Q/A
  table. The page loads ONCE and polls a sidecar ``*_data.js`` file, so new
  rows stream in with no page reload — playback is never interrupted.
  Clicking a timeline band or a table row seeks the video.

Usage (server recipe in ../README.md; tunnel with `ssh -L 8002:127.0.0.1:8002`
if the model is served on the H100):

    python examples/specialist-retry-probe/utils/visualize_probe.py \\
        --video ~/videos/rollout_ep001_success.mp4 \\
        --instruction "pick up the red capsule and place it in the blue tray" \\
        --view wrist --upscale 3 --stride 5 --out /tmp/probe_report.html

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
sys.path.insert(0, str(_HERE.parent))          # robobrain_probe (the templates under test)
_REPO_SRC = _HERE.parents[2] / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))         # odyssey (the judge)

from robobrain_probe import (  # noqa: E402
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

# Serene Ocean semantic colors (lai-trainer command-center theme).
ANSWER_COLOR = {"YES": "#34d399", "NO": "#c94a4a", "FAIL": "#5a7a8f"}


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


def data_path_for(out: Path) -> Path:
    return out.with_name(out.stem + "_data.js")


def write_data(out: Path, records: list[dict[str, Any]], done: bool) -> None:
    """Sidecar the page polls — atomic-ish single write per judged frame."""
    payload = json.dumps({"records": records, "done": done})
    data_path_for(out).write_text(f"window.PROBE_DATA = {payload};")


def render_shell(out: Path, *, video_path: Path, meta: dict[str, Any]) -> None:
    """Static page, written ONCE: video + empty containers. All rows/segments
    are rendered client-side from the polled sidecar, so nothing here ever
    reloads and playback is never interrupted."""
    video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    names = [name for name, _ in QUESTIONS]
    badges = "".join(
        f'<span class="badge" id="badge-{n}"><small>{n}</small><b id="bv-{n}">?</b></span>'
        for n in names
    )
    tl_rows = "".join(
        f'<div class="tlrow"><span class="tlabel">{n}</span>'
        f'<div class="tband" id="band-{n}"></div></div>'
        for n in names
    )
    config = json.dumps(
        {"questions": names, "colors": ANSWER_COLOR, "data_src": data_path_for(out).name}
    )
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>RoboBrain probe — {html.escape(video_path.name)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<style>
/* Serene Ocean Oasis — lai-trainer command-center theme tokens */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
:root {{
  --bg-primary:#070c10; --bg-secondary:#0c1318; --bg-tertiary:#121c22;
  --bg-elevated:#18252d; --bg-glass:rgba(145,174,193,.07);
  --bg-glass-hover:rgba(145,174,193,.12);
  --text-primary:#eaf2f7; --text-secondary:#9ab5c7; --text-muted:#5a7a8f;
  --border-primary:rgba(145,174,193,.12); --border-secondary:rgba(145,174,193,.22);
  --primary:#508ca4; --pale-sky:#bfd7ea; --sea-green:#0a8754;
  --emerald:#34d399; --error:#c94a4a; --warning:#d9a441;
  --gradient-card:linear-gradient(160deg,rgba(12,19,24,.94) 0%,rgba(7,12,16,.98) 100%);
  --shadow-md:0 3px 8px rgba(0,0,0,.3),0 2px 4px rgba(0,0,0,.2);
  --font-primary:'DM Sans',-apple-system,sans-serif;
  --font-mono:'Space Mono','Fira Code',monospace;
}}
body {{ font-family:var(--font-primary); margin:0; padding:1.5rem 2rem;
  background:var(--bg-primary); color:var(--text-primary);
  background-image:radial-gradient(ellipse at 30% 20%,rgba(80,140,164,.08) 0%,transparent 40%),
    radial-gradient(ellipse at 70% 80%,rgba(10,135,84,.06) 0%,transparent 40%); }}
h2 {{ font-weight:600; letter-spacing:-.015em; margin:0 0 4px; font-size:1.375rem; }}
.card {{ background:var(--gradient-card); border:1px solid var(--border-primary);
  border-radius:14px; box-shadow:var(--shadow-md); padding:14px 16px;
  margin-bottom:14px; backdrop-filter:blur(16px); }}
.layout {{ max-width: 1120px; }}
.status-badge {{ display:inline-flex; align-items:center; gap:6px;
  font-family:var(--font-mono); font-size:.6875rem; letter-spacing:.08em;
  text-transform:uppercase; padding:3px 10px; border-radius:999px;
  vertical-align:middle; margin-left:10px; }}
.status-badge.running {{ background:rgba(10,135,84,.12); color:var(--emerald);
  border:1px solid rgba(10,135,84,.25); }}
.status-badge.running::before {{ content:''; width:6px; height:6px;
  border-radius:50%; background:var(--emerald); animation:pulse 1.4s infinite; }}
.status-badge.done {{ background:rgba(80,140,164,.12); color:var(--pale-sky);
  border:1px solid rgba(80,140,164,.3); }}
@keyframes pulse {{ 50% {{ opacity:.3; }} }}
.meta {{ color:var(--text-secondary); font-size:.8125rem; margin:0 0 16px;
  line-height:1.55; }}
.meta b {{ color:var(--text-primary); font-weight:500; }}
video {{ border-radius:10px; border:1px solid var(--border-primary);
  display:block; background:#000; }}
.speed {{ margin-top:8px; }}
.speed span {{ font-family:var(--font-mono); font-size:.6875rem;
  letter-spacing:.08em; text-transform:uppercase; color:var(--text-muted);
  margin-right:6px; }}
.speed button {{ font-family:var(--font-mono); font-size:.6875rem;
  background:var(--bg-glass); color:var(--text-secondary);
  border:1px solid var(--border-primary); border-radius:7px;
  padding:4px 10px; margin-right:4px; cursor:pointer; }}
.speed button:hover {{ background:var(--bg-glass-hover); color:var(--text-primary);
  border-color:var(--border-secondary); }}
.badges {{ display:flex; gap:8px; margin:12px 0 2px; flex-wrap:wrap; }}
.badge {{ display:inline-flex; flex-direction:column; align-items:center;
  background:var(--bg-glass); border:1px solid var(--border-primary);
  border-radius:10px; padding:6px 14px; min-width:82px; }}
.badge small {{ font-family:var(--font-mono); font-size:.625rem;
  letter-spacing:.08em; text-transform:uppercase; color:var(--text-muted); }}
.badge b {{ font-size:1.125rem; font-family:var(--font-mono); }}
#timeline {{ margin-top:4px; width:560px; position:relative; }}
.tlrow {{ display:flex; align-items:center; height:22px; margin:3px 0; }}
.tlabel {{ width:92px; font-family:var(--font-mono); font-size:.625rem;
  letter-spacing:.08em; text-transform:uppercase; color:var(--text-muted);
  text-align:right; padding-right:8px; }}
.tband {{ position:relative; flex:1; height:14px; background:var(--bg-tertiary);
  cursor:pointer; border-radius:4px; overflow:hidden;
  border:1px solid var(--border-primary); }}
.seg {{ position:absolute; top:0; bottom:0; opacity:.85; }}
#playhead {{ position:absolute; top:0; bottom:0; width:2px;
  background:var(--pale-sky); box-shadow:0 0 8px rgba(191,215,234,.6);
  pointer-events:none; left:92px; }}
table {{ border-collapse:collapse; font-size:.8125rem; width:100%; }}
th {{ font-family:var(--font-mono); font-size:.625rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--text-muted); text-align:left; }}
td, th {{ border-bottom:1px solid var(--border-primary); padding:6px 10px;
  vertical-align:middle; }}
tr:hover td {{ background:var(--bg-glass-hover); cursor:pointer; }}
tr.now td {{ background:rgba(217,164,65,.12);
  box-shadow:inset 2px 0 0 var(--warning); }}
td.q {{ max-width:440px; color:var(--text-muted); font-size:.6875rem;
  line-height:1.4; }}
td img {{ border-radius:6px; border:1px solid var(--border-primary); }}
.ans {{ font-family:var(--font-mono); font-weight:700; }}
</style></head><body><div class="layout">
<h2>Retry-strategy probe — {html.escape(video_path.name)}
<span class="status-badge running" id="status">waiting</span></h2>
<div class="meta">model <b>{html.escape(meta["model"])}</b> · instruction
“{html.escape(meta["instruction"])}” · view <b>{meta["view"]}</b>
x{meta["upscale"]} · stride {meta["stride"]} · slow-motion 0.25x by default —
badges and timeline follow the playhead; click a band or a row to seek; new
judgements stream in without reloading</div>
<div class="card">
<video id="v" controls width="560" src="data:video/mp4;base64,{video_b64}"></video>
<div class="speed"><span>speed</span>
<button onclick="rate(0.1)">0.1x</button><button onclick="rate(0.25)">0.25x</button>
<button onclick="rate(0.5)">0.5x</button><button onclick="rate(1)">1x</button></div>
<div class="badges">{badges}</div>
<div id="timeline">{tl_rows}<div id="playhead"></div></div>
</div>
<div class="card">
<table id="tbl"><tr><th>frame</th><th>t</th><th>judged image</th><th>question</th>
<th>answer</th><th>latency</th><th>full prompt sent</th></tr></table>
</div>
</div><script>
const CFG = {config};
const v = document.getElementById("v");
v.addEventListener("loadedmetadata", () => {{ v.playbackRate = 0.25; }});
function rate(x) {{ v.playbackRate = x; }}
function seek(t) {{ v.currentTime = t + 0.001; }}
let records = [], done = false, rendered = 0, thumbs = {{}};

function onData(d) {{
  if (!d || d.records.length === records.length && done === d.done) return;
  records = d.records; done = d.done;
  const st = document.getElementById("status");
  st.textContent = (done ? "finished · " : "running · ") + records.length + " judgements";
  st.className = "status-badge " + (done ? "done" : "running");
  const tbl = document.getElementById("tbl");
  for (; rendered < records.length; rendered++) {{
    const r = records[rendered];
    if (r.thumb) thumbs[r.frame] = r.thumb;
    const tr = document.createElement("tr");
    tr.id = "r" + r.frame + "-" + r.question;
    tr.onclick = () => seek(r.time_s);
    const thumb = r.thumb ? '<img src="' + r.thumb + '" height="72">' : "";
    tr.innerHTML = "<td>" + r.frame + "</td><td>" + r.time_s.toFixed(1) +
        "s</td><td>" + thumb + "</td><td>" + r.question +
        '</td><td class="ans" style="color:' + CFG.colors[r.answer] + '">' +
        r.answer + "</td><td>" + r.latency_s.toFixed(2) + "s</td>" +
        '<td class="q"></td>';
    tr.lastChild.textContent = r.prompt;
    tbl.appendChild(tr);
  }}
  buildTimeline();
}}

function byQuestion(q) {{
  return records.filter(r => r.question === q).sort((a, b) => a.time_s - b.time_s);
}}

function buildTimeline() {{
  const D = v.duration;
  if (!D) {{ v.addEventListener("loadedmetadata", buildTimeline, {{once: true}}); return; }}
  for (const q of CFG.questions) {{
    const band = document.getElementById("band-" + q);
    band.innerHTML = "";
    const pts = byQuestion(q);
    for (let i = 0; i < pts.length; i++) {{
      const start = pts[i].time_s;
      const end = (i + 1 < pts.length) ? pts[i + 1].time_s : (done ? D : start + 0.2);
      const seg = document.createElement("div");
      seg.className = "seg";
      seg.style.left = (start / D * 100) + "%";
      seg.style.width = (Math.max(end - start, 0.02) / D * 100) + "%";
      seg.style.background = CFG.colors[pts[i].answer];
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
  for (const r of byQuestion(CFG.questions[0])) if (r.time_s <= t + 1e-6) current = r.frame;
  for (const q of CFG.questions) {{
    let pt = null;
    for (const r of byQuestion(q)) if (r.time_s <= t + 1e-6) pt = r;
    const ans = pt ? pt.answer : "?";
    const el = document.getElementById("bv-" + q);
    el.textContent = ans;
    el.style.color = CFG.colors[ans] || "#333";
  }}
  if (current !== lastFrame) {{
    document.querySelectorAll("tr.now").forEach(el => el.classList.remove("now"));
    for (const q of CFG.questions) {{
      const row = document.getElementById("r" + current + "-" + q);
      if (row) row.classList.add("now");
    }}
    lastFrame = current;
  }}
}}
v.addEventListener("timeupdate", refreshUI);
setInterval(refreshUI, 150);

function poll() {{
  if (done) return;
  const s = document.createElement("script");
  s.src = CFG.data_src + "?t=" + Date.now();
  s.onload = () => {{ s.remove(); onData(window.PROBE_DATA); }};
  s.onerror = () => s.remove();
  document.body.appendChild(s);
}}
poll();
setInterval(poll, 1500);
</script></body></html>""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--base_url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="BAAI/RoboBrain2.5-8B-NV")
    parser.add_argument("--view", choices=["full", "wrist", "side"], default="full")
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--stride", type=int, default=5, help="judge every Nth frame")
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
                # Vanilla vLLM: no Omni `modalities` extra_body knob.
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
    render_shell(out, video_path=video_path, meta=meta)
    write_data(out, records, done=False)

    indexes = list(range(0, len(stack), args.stride))
    print(f"{len(stack)} frames, judging {len(indexes)} of them x {len(QUESTIONS)} questions")
    print(f"report: {out}  (open it now — rows stream in without reloading)\n")
    print(f"{'frame':>6} {'question':>10}  answer  latency")
    for index in indexes:
        frame = prepare_frame(stack[index], args.view, args.upscale)
        thumb = _png_data_uri(frame)
        for position, (name, _template) in enumerate(QUESTIONS):
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
                    "thumb": thumb if position == 0 else None,
                }
            )
            print(f"{index:6d} {name:>10}  {answer:<6} {elapsed:.2f}s")
        write_data(out, records, done=False)
    write_data(out, records, done=True)

    summary_path = out.with_suffix(".json")
    lean = [{k: v for k, v in r.items() if k != "thumb"} for r in records]
    summary_path.write_text(json.dumps({"meta": meta, "records": lean}, indent=2))
    print(f"\ndone -> {out}  (+ raw records in {summary_path})")


if __name__ == "__main__":
    main()
