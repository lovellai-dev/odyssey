"""Watch the probe interrogate one or several SPECIALISTs, frame by frame.

Makes the experiment transparent: the model never sees the *video* — it sees
isolated frames, and per frame each question is one independent image+text
HTTP call. This tool replays that loop over a single rollout MP4 and shows it:

* terminal — one table row per (frame, arm, question) as each answer arrives;
* HTML report — the playable video (slow-motion by default), live answer
  badges, per-question timeline bands with a synced playhead, and the Q/A
  table. The page loads ONCE and polls a sidecar ``*_data.js`` file, so new
  rows stream in with no page reload — playback is never interrupted.
  Clicking a timeline band or a table row seeks the video.

COMPARATIVE MODE: pass ``--arm`` (repeatable, JSON) to judge the same frames
with several models side by side — one timeline band per question x arm, an
arm column in the table, badges grouped per arm. Same frames, same verbatim
prompts; the only variable is the model:

    python examples/cosmos3-reasoner-probe/utils/visualize_probe.py \\
        --video ~/videos/rollout_ep001_success.mp4 \\
        --instruction "pick up the red capsule and place it in the blue tray" \\
        --view wrist --upscale 3 --stride 5 --out /tmp/probe_report.html \\
        --arm '{"label": "cosmos3", "model": "nvidia/Cosmos3-Nano",
                "base_url": "http://127.0.0.1:8002/v1",
                "extra_body": {"modalities": ["text"]}}' \\
        --arm '{"label": "molmo2", "model": "allenai/Molmo2-8B",
                "base_url": "http://127.0.0.1:8003/v1"}'

With no ``--arm``, the single-model flags (``--model``/``--base_url``) form
the only arm — the original behaviour. ``--fake`` answers deterministically
without any server, for checking the viewer itself.

(Server recipes in ../README.md and the mission headers; tunnel with
`ssh -L 8002:127.0.0.1:8002 -L 8003:127.0.0.1:8003` if serving on the H100.)
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

# Serene Ocean semantic colors (lai-trainer command-center theme).
ANSWER_COLOR = {"YES": "#34d399", "NO": "#c94a4a", "FAIL": "#5a7a8f"}


def _fake_transport(payload: dict[str, Any]) -> dict[str, Any]:
    """Deterministic offline answers (viewer testing only, no server).
    Salted with the model id so fake arms disagree — exercises the
    comparative rendering."""
    text = payload["messages"][0]["content"][1]["text"] + payload.get("model", "")
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


def resolve_arms(args: argparse.Namespace) -> list[dict[str, Any]]:
    """--arm JSON entries, or the single-model flags as the only arm."""
    if not args.arm:
        return [
            {
                "label": args.model.split("/")[-1],
                "model": args.model,
                "base_url": args.base_url,
                # vLLM-Omni routes image+text chat to IMAGE GENERATION unless
                # the request selects the text modality — historic default.
                "extra_body": {"modalities": ["text"]},
            }
        ]
    arms = []
    for raw in args.arm:
        arm = json.loads(raw)
        if "label" not in arm or "model" not in arm or "base_url" not in arm:
            raise SystemExit(f"--arm needs label/model/base_url: {raw}")
        arm.setdefault("extra_body", {})
        arms.append(arm)
    return arms


def render_shell(
    out: Path, *, video_path: Path, meta: dict[str, Any], arms: list[dict[str, Any]]
) -> None:
    """Static page, written ONCE: video + empty containers. All rows/segments
    are rendered client-side from the polled sidecar, so nothing here ever
    reloads and playback is never interrupted."""
    video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    names = [name for name, _ in QUESTIONS]
    labels = [arm["label"] for arm in arms]
    badge_rows = "".join(
        f'<div class="badges"><span class="armtag">{html.escape(label)}</span>'
        + "".join(
            f'<span class="badge"><small>{n}</small>'
            f'<b id="bv-{n}-{html.escape(label)}">?</b></span>'
            for n in names
        )
        + "</div>"
        for label in labels
    )
    tl_rows = "".join(
        f'<div class="tlrow"><span class="tlabel">{n}'
        + (f" · {html.escape(label)}" if len(labels) > 1 else "")
        + f'</span><div class="tband" id="band-{n}-{html.escape(label)}"></div></div>'
        for n in names
        for label in labels
    )
    arms_meta = " vs ".join(f"{a['label']} ({a['model']})" for a in arms)
    config = json.dumps(
        {
            "questions": names,
            "arms": labels,
            "colors": ANSWER_COLOR,
            "data_src": data_path_for(out).name,
            "budget_ms": meta["latency_budget_ms"],
        }
    )
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Reasoner probe — {html.escape(video_path.name)}</title>
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
.badges {{ display:flex; gap:8px; margin:12px 0 2px; flex-wrap:wrap;
  align-items:center; }}
.armtag {{ font-family:var(--font-mono); font-size:.6875rem;
  letter-spacing:.08em; text-transform:uppercase; color:var(--pale-sky);
  min-width:84px; text-align:right; padding-right:4px; }}
.badge {{ display:inline-flex; flex-direction:column; align-items:center;
  background:var(--bg-glass); border:1px solid var(--border-primary);
  border-radius:10px; padding:6px 14px; min-width:82px; }}
.badge small {{ font-family:var(--font-mono); font-size:.625rem;
  letter-spacing:.08em; text-transform:uppercase; color:var(--text-muted); }}
.badge b {{ font-size:1.125rem; font-family:var(--font-mono); }}
#timeline {{ margin-top:4px; width:640px; position:relative; }}
.tlrow {{ display:flex; align-items:center; height:20px; margin:3px 0; }}
.tlabel {{ width:170px; font-family:var(--font-mono); font-size:.625rem;
  letter-spacing:.06em; text-transform:uppercase; color:var(--text-muted);
  text-align:right; padding-right:8px; white-space:nowrap; overflow:hidden; }}
.tband {{ position:relative; flex:1; height:13px; background:var(--bg-tertiary);
  cursor:pointer; border-radius:4px; overflow:hidden;
  border:1px solid var(--border-primary); }}
.seg {{ position:absolute; top:0; bottom:0; opacity:.85; }}
#playhead {{ position:absolute; top:0; bottom:0; width:2px;
  background:var(--pale-sky); box-shadow:0 0 8px rgba(191,215,234,.6);
  pointer-events:none; left:170px; }}
table {{ border-collapse:collapse; font-size:.8125rem; width:100%; }}
th {{ font-family:var(--font-mono); font-size:.625rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--text-muted); text-align:left; }}
td, th {{ border-bottom:1px solid var(--border-primary); padding:6px 10px;
  vertical-align:middle; }}
tr:hover td {{ background:var(--bg-glass-hover); cursor:pointer; }}
tr.now td {{ background:rgba(217,164,65,.12);
  box-shadow:inset 2px 0 0 var(--warning); }}
td.q {{ max-width:380px; color:var(--text-muted); font-size:.6875rem;
  line-height:1.4; }}
td.arm {{ font-family:var(--font-mono); font-size:.6875rem;
  color:var(--pale-sky); }}
td img {{ border-radius:6px; border:1px solid var(--border-primary); }}
.ans {{ font-family:var(--font-mono); font-weight:700; }}
h3 {{ font-weight:600; font-size:.9375rem; margin:0 0 10px; }}
h3 small {{ color:var(--text-muted); font-weight:400; font-size:.75rem; }}
.latrow {{ display:flex; align-items:center; gap:10px; margin:6px 0; }}
.latstats {{ font-family:var(--font-mono); font-size:.6875rem;
  color:var(--text-secondary); min-width:340px; }}
.latstats b {{ color:var(--text-primary); }}
.latbar {{ position:relative; flex:1; height:14px; background:var(--bg-tertiary);
  border-radius:4px; border:1px solid var(--border-primary); overflow:visible; }}
.latfill {{ position:absolute; top:0; bottom:0; left:0; border-radius:3px;
  opacity:.85; }}
.latbudget {{ position:absolute; top:-3px; bottom:-3px; width:2px;
  background:var(--warning); box-shadow:0 0 6px rgba(217,164,65,.6); }}
.latverdict {{ font-family:var(--font-mono); font-size:.6875rem; font-weight:700;
  min-width:110px; text-align:right; }}
</style></head><body><div class="layout">
<h2>Grasp-verification probe — {html.escape(video_path.name)}
<span class="status-badge running" id="status">waiting</span></h2>
<div class="meta">arms <b>{html.escape(arms_meta)}</b> · instruction
“{html.escape(meta["instruction"])}” · view <b>{meta["view"]}</b>
x{meta["upscale"]} · stride {meta["stride"]} · slow-motion 0.25x by default —
same frames, same verbatim prompts for every arm; badges and timeline follow
the playhead; click a band or a row to seek; new judgements stream in without
reloading</div>
<div class="card">
<video id="v" controls width="640" src="data:video/mp4;base64,{video_b64}"></video>
<div class="speed"><span>speed</span>
<button onclick="rate(0.1)">0.1x</button><button onclick="rate(0.25)">0.25x</button>
<button onclick="rate(0.5)">0.5x</button><button onclick="rate(1)">1x</button></div>
{badge_rows}
<div id="timeline">{tl_rows}<div id="playhead"></div></div>
</div>
<div class="card">
<h3>latency per decision <small>budget {meta["latency_budget_ms"]} ms
(Specialist Map v0.5) · bar = p95 on a log scale · updates live</small></h3>
<div id="latbox"></div>
</div>
<div class="card">
<table id="tbl"><tr><th>frame</th><th>t</th><th>judged image</th><th>arm</th>
<th>question</th><th>answer</th><th>latency</th><th>full prompt sent</th></tr></table>
</div>
</div><script>
const CFG = {config};
const v = document.getElementById("v");
v.addEventListener("loadedmetadata", () => {{ v.playbackRate = 0.25; }});
function rate(x) {{ v.playbackRate = x; }}
function seek(t) {{ v.currentTime = t + 0.001; }}
let records = [], done = false, rendered = 0;

function onData(d) {{
  if (!d || d.records.length === records.length && done === d.done) return;
  records = d.records; done = d.done;
  const st = document.getElementById("status");
  st.textContent = (done ? "finished · " : "running · ") + records.length + " judgements";
  st.className = "status-badge " + (done ? "done" : "running");
  const tbl = document.getElementById("tbl");
  for (; rendered < records.length; rendered++) {{
    const r = records[rendered];
    const tr = document.createElement("tr");
    tr.id = "r" + r.frame + "-" + r.question + "-" + r.arm;
    tr.onclick = () => seek(r.time_s);
    const thumb = r.thumb ? '<img src="' + r.thumb + '" height="72">' : "";
    tr.innerHTML = "<td>" + r.frame + "</td><td>" + r.time_s.toFixed(1) +
        "s</td><td>" + thumb + '</td><td class="arm">' + r.arm +
        "</td><td>" + r.question +
        '</td><td class="ans" style="color:' + CFG.colors[r.answer] + '">' +
        r.answer + "</td><td>" + r.latency_s.toFixed(2) + "s</td>" +
        '<td class="q"></td>';
    tr.lastChild.textContent = r.prompt;
    tbl.appendChild(tr);
  }}
  buildTimeline();
  buildLatency();
}}

function quantile(sorted, q) {{
  if (!sorted.length) return null;
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos);
  return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (pos - lo);
}}

function buildLatency() {{
  // Live per-arm latency stats from the streamed records. The bar is p95 on
  // a log scale (10ms..10s) against the pre-registered per-decision budget.
  const box = document.getElementById("latbox");
  const budget = CFG.budget_ms / 1000;
  const logPos = s => Math.max(0, Math.min(1,
      (Math.log10(s) - Math.log10(0.01)) / (Math.log10(10) - Math.log10(0.01))));
  let html = "";
  for (const arm of CFG.arms) {{
    const lats = records.filter(r => r.arm === arm && r.answer !== "FAIL")
        .map(r => r.latency_s).sort((a, b) => a - b);
    if (!lats.length) {{ html += ""; continue; }}
    const mean = lats.reduce((a, b) => a + b, 0) / lats.length;
    const p50 = quantile(lats, 0.5), p95 = quantile(lats, 0.95);
    const max = lats[lats.length - 1];
    const under = p95 <= budget;
    const color = under ? CFG.colors.YES : CFG.colors.NO;
    html += '<div class="latrow"><span class="armtag">' + arm + "</span>" +
        '<span class="latstats">n=' + lats.length +
        " · mean <b>" + (mean * 1000).toFixed(0) + "ms</b>" +
        " · p50 <b>" + (p50 * 1000).toFixed(0) + "ms</b>" +
        " · p95 <b>" + (p95 * 1000).toFixed(0) + "ms</b>" +
        " · max <b>" + (max * 1000).toFixed(0) + "ms</b></span>" +
        '<span class="latbar"><span class="latfill" style="width:' +
        (logPos(p95) * 100) + "%;background:" + color + '"></span>' +
        '<span class="latbudget" style="left:' + (logPos(budget) * 100) +
        '%"></span></span>' +
        '<span class="latverdict" style="color:' + color + '">' +
        (under ? "UNDER BUDGET" : "OVER BUDGET") + "</span></div>";
  }}
  box.innerHTML = html;
}}

function byQA(q, arm) {{
  return records.filter(r => r.question === q && r.arm === arm)
      .sort((a, b) => a.time_s - b.time_s);
}}

function buildTimeline() {{
  const D = v.duration;
  if (!D) {{ v.addEventListener("loadedmetadata", buildTimeline, {{once: true}}); return; }}
  for (const q of CFG.questions) for (const arm of CFG.arms) {{
    const band = document.getElementById("band-" + q + "-" + arm);
    if (!band) continue;
    band.innerHTML = "";
    const pts = byQA(q, arm);
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
  if (band) document.getElementById("playhead").style.left =
      (band.offsetLeft + t / D * band.offsetWidth) + "px";
  let current = null;
  for (const r of byQA(CFG.questions[0], CFG.arms[0]))
    if (r.time_s <= t + 1e-6) current = r.frame;
  for (const q of CFG.questions) for (const arm of CFG.arms) {{
    let pt = null;
    for (const r of byQA(q, arm)) if (r.time_s <= t + 1e-6) pt = r;
    const ans = pt ? pt.answer : "?";
    const el = document.getElementById("bv-" + q + "-" + arm);
    if (!el) continue;
    el.textContent = ans;
    el.style.color = CFG.colors[ans] || "#333";
  }}
  if (current !== lastFrame) {{
    document.querySelectorAll("tr.now").forEach(el => el.classList.remove("now"));
    for (const q of CFG.questions) for (const arm of CFG.arms) {{
      const row = document.getElementById("r" + current + "-" + q + "-" + arm);
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
    parser.add_argument("--model", default="nvidia/Cosmos3-Nano")
    parser.add_argument(
        "--arm",
        action="append",
        help='repeatable JSON arm: {"label", "model", "base_url", "extra_body"?}; '
        "overrides --model/--base_url",
    )
    parser.add_argument("--view", choices=["full", "wrist", "side"], default="full")
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--stride", type=int, default=5, help="judge every Nth frame")
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--timeout_seconds", type=float, default=120.0)
    parser.add_argument("--out", default="/tmp/probe_report.html")
    parser.add_argument(
        "--latency_budget_ms",
        type=int,
        default=300,
        help="per-decision latency budget (Specialist Map v0.5) drawn in the latency panel",
    )
    parser.add_argument("--fake", action="store_true", help="no server: deterministic answers")
    args = parser.parse_args()

    import imageio.v3 as iio

    video_path = Path(args.video).expanduser()
    stack = iio.imread(video_path)
    try:
        fps = float(iio.immeta(video_path).get("fps") or 20.0)
    except Exception:
        fps = 20.0

    arms = resolve_arms(args)
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
        (arm["label"], name): _capturing(
            OpenAICompatCompletionJudge(
                base_url=arm["base_url"],
                model=arm["model"],
                prompt_template=template,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds,
                extra_body=arm["extra_body"],
                transport=_fake_transport if args.fake else None,
            )
        )
        for arm in arms
        for name, template in QUESTIONS
    }
    templates = dict(QUESTIONS)
    meta = {
        "instruction": args.instruction,
        "view": args.view, "upscale": args.upscale, "stride": args.stride,
        "latency_budget_ms": args.latency_budget_ms,
    }
    out = Path(args.out).expanduser()
    records: list[dict[str, Any]] = []
    render_shell(out, video_path=video_path, meta=meta, arms=arms)
    write_data(out, records, done=False)

    indexes = list(range(0, len(stack), args.stride))
    total = len(indexes) * len(QUESTIONS) * len(arms)
    print(f"{len(stack)} frames; judging {len(indexes)} x {len(QUESTIONS)} questions"
          f" x {len(arms)} arms = {total} calls")
    print(f"report: {out}  (open it now — rows stream in without reloading)\n")
    print(f"{'frame':>6} {'arm':>10} {'question':>10}  answer  latency")
    for index in indexes:
        frame = prepare_frame(stack[index], args.view, args.upscale)
        thumb = _png_data_uri(frame)
        first_record_of_frame = True
        for arm in arms:
            for name, _template in QUESTIONS:
                replies_before = len(replies)
                start = time.monotonic()
                answer = (
                    "YES"
                    if judges[(arm["label"], name)].is_complete(frame, args.instruction)
                    else "NO"
                )
                elapsed = time.monotonic() - start
                if len(replies) == replies_before:  # transport failed -> silent NO
                    answer = "FAIL"
                records.append(
                    {
                        "frame": index,
                        "time_s": index / fps,
                        "question": name,
                        "arm": arm["label"],
                        "answer": answer,
                        "latency_s": elapsed,
                        "prompt": templates[name].format(instruction=args.instruction),
                        "thumb": thumb if first_record_of_frame else None,
                    }
                )
                first_record_of_frame = False
                print(f"{index:6d} {arm['label']:>10} {name:>10}  {answer:<6} {elapsed:.2f}s")
        write_data(out, records, done=False)
    write_data(out, records, done=True)

    summary_path = out.with_suffix(".json")
    lean = [{k: v for k, v in r.items() if k != "thumb"} for r in records]
    summary_path.write_text(
        json.dumps({"meta": meta, "arms": arms, "records": lean}, indent=2)
    )
    print(f"\ndone -> {out}  (+ raw records in {summary_path})")


if __name__ == "__main__":
    main()
