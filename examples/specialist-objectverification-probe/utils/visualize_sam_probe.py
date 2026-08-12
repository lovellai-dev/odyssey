"""Watch the SAM 3.1 arm segment named concepts, LIVE over the playing video.

The SAM counterpart of ``visualize_probe.py`` — the **variation** the object-
verification bake-off needs because SAM does not answer YES/NO: it returns
masks/boxes. The headline view here is a **bounding box drawn on the playing
video**, synced to the playhead: as the rollout plays you see, per named
concept, the box SAM put around the object it recognised and its score. Per
sampled frame each concept is one POST to the served SAM endpoint ("segment
<concept>"); a concept counts PRESENT when its best score clears
``--score_threshold``.

Crop→full-frame mapping: the model judges a *crop* of the 2:1 concat frame
(``wrist`` = right half, ``side`` = left half), but the video shows the whole
concat frame, so the returned boxes are remapped to full-frame coordinates —
they land on the actual object on screen, not the wrong half.

Also keeps the streaming shell (Serene Ocean theme, sidecar ``*_data.js`` polled
with no reload, per-concept timeline bands, synced playhead) and a per-row
canvas showing the exact judged crop with its boxes.

Usage (server recipe in ../README.md; tunnel with `ssh -L 8006:127.0.0.1:8006`
if SAM is served on the H100):

    python utils/visualize_sam_probe.py \\
        --video ~/videos/rollout_ep001_success.mp4 \\
        --instruction "pick up the red capsule and place it in the blue tray" \\
        --objects "red capsule, blue tray" --distractors "green bottle" \\
        --view wrist --upscale 3 --stride 5 --out /tmp/sam_report.html

    open /tmp/sam_report.html      # while the run is live, or after

``--fake`` draws deterministic boxes without any server — for checking the
viewer itself (the live overlay works the same, with fake boxes).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # sam_probe (transport + concept builder)

from robobrain_probe import prepare_frame  # noqa: E402
from sam_probe import (  # noqa: E402
    _fake_transport,
    _http_transport,
    build_concepts,
    segment,
)

# PRESENT / ABSENT / FAIL — Serene Ocean semantic colors (for the % / table).
ANSWER_COLOR = {"PRESENT": "#34d399", "ABSENT": "#c94a4a", "FAIL": "#5a7a8f"}
# Distinct per-concept colors for the boxes drawn on the video.
CONCEPT_PALETTE = ["#34d399", "#508ca4", "#d9a441", "#c084fc", "#f472b6", "#38bdf8"]


def _png_data_uri(array: Any, max_side: int = 320) -> str:
    from PIL import Image

    img = Image.fromarray(array)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def data_path_for(out: Path) -> Path:
    return out.with_name(out.stem + "_data.js")


def write_data(out: Path, records: list[dict[str, Any]], done: bool) -> None:
    payload = json.dumps({"records": records, "done": done})
    data_path_for(out).write_text(f"window.PROBE_DATA = {payload};")


def _normalize_boxes(result: dict[str, Any], frame: Any) -> list[list[float]]:
    """SAM returns pixel boxes + image_size; normalize to 0..1 over the CROP."""
    import numpy as np

    detections = result.get("detections") or []
    size = result.get("image_size")
    if size and size[0] and size[1]:
        width, height = float(size[0]), float(size[1])
    else:
        arr = np.asarray(frame)
        height, width = float(arr.shape[0]), float(arr.shape[1])
    boxes: list[list[float]] = []
    for det in detections:
        box = det.get("box")
        if not box:
            continue
        boxes.append([box[0] / width, box[1] / height, box[2] / width, box[3] / height])
    return boxes


def _to_full_frame(boxes: list[list[float]], view: str) -> list[list[float]]:
    """Map crop-normalized xyxy boxes onto the full 2:1 concat frame.

    ``wrist`` judged the right half -> x' = 0.5 + x/2; ``side`` the left half ->
    x' = x/2; ``full`` passes through. Height is untouched (the crop keeps all
    rows).
    """
    if view == "full":
        return boxes
    shift = 0.5 if view == "wrist" else 0.0
    return [[shift + b[0] / 2, b[1], shift + b[2] / 2, b[3]] for b in boxes]


def render_shell(
    out: Path, *, video_path: Path, meta: dict[str, Any], concepts: list[tuple[str, str]]
) -> None:
    names = [n for n, _ in concepts]
    concept_colors = {
        q: CONCEPT_PALETTE[i % len(CONCEPT_PALETTE)] for i, (q, _) in enumerate(concepts)
    }
    video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    badges = "".join(
        f'<span class="badge" id="badge-{html.escape(n)}"><small>{html.escape(n)}</small>'
        f'<b id="bv-{html.escape(n)}">?</b></span>'
        for n in names
    )
    tl_rows = "".join(
        f'<div class="tlrow"><span class="tlabel">{html.escape(n)}</span>'
        f'<div class="tband" id="band-{html.escape(n)}"></div></div>'
        for n in names
    )
    # legend: which colour is which concept on the video
    legend = "".join(
        f'<span class="lg"><i style="background:{concept_colors[n]}"></i>{html.escape(n)}</span>'
        for n in names
        if n != "control"
    )
    config = json.dumps(
        {
            "questions": names,
            "colors": ANSWER_COLOR,
            "conceptColors": concept_colors,
            "data_src": data_path_for(out).name,
            "threshold": meta["score_threshold"],
        }
    )
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>SAM object-verification — {html.escape(video_path.name)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
:root {{
  --bg-primary:#070c10; --bg-secondary:#0c1318; --bg-tertiary:#121c22;
  --bg-glass:rgba(145,174,193,.07); --bg-glass-hover:rgba(145,174,193,.12);
  --text-primary:#eaf2f7; --text-secondary:#9ab5c7; --text-muted:#5a7a8f;
  --border-primary:rgba(145,174,193,.12); --border-secondary:rgba(145,174,193,.22);
  --primary:#508ca4; --pale-sky:#bfd7ea; --emerald:#34d399; --error:#c94a4a;
  --warning:#d9a441;
  --gradient-card:linear-gradient(160deg,rgba(12,19,24,.94) 0%,rgba(7,12,16,.98) 100%);
  --shadow-md:0 3px 8px rgba(0,0,0,.3),0 2px 4px rgba(0,0,0,.2);
  --font-primary:'DM Sans',-apple-system,sans-serif; --font-mono:'Space Mono',monospace;
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
.meta {{ color:var(--text-secondary); font-size:.8125rem; margin:0 0 16px; line-height:1.55; }}
.meta b {{ color:var(--text-primary); font-weight:500; }}
.viewrow {{ display:flex; gap:18px; align-items:flex-start; }}
#det {{ border-radius:12px; border:1px solid var(--border-secondary); background:#000;
  display:block; box-shadow:var(--shadow-md); }}
.side {{ width:260px; }}
video {{ border-radius:8px; border:1px solid var(--border-primary);
  display:block; background:#000; width:260px; }}
.legend {{ display:flex; gap:14px; margin:8px 0 2px; flex-wrap:wrap;
  font-family:var(--font-mono); font-size:.6875rem; color:var(--text-secondary); }}
.legend .lg i {{ display:inline-block; width:11px; height:11px; border-radius:3px;
  margin-right:5px; vertical-align:-1px; }}
.speed {{ margin-top:8px; }}
.speed span {{ font-family:var(--font-mono); font-size:.6875rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--text-muted); margin-right:6px; }}
.speed button {{ font-family:var(--font-mono); font-size:.6875rem; background:var(--bg-glass);
  color:var(--text-secondary); border:1px solid var(--border-primary); border-radius:7px;
  padding:4px 10px; margin-right:4px; cursor:pointer; }}
.speed button:hover {{ background:var(--bg-glass-hover); color:var(--text-primary); }}
.badges {{ display:flex; gap:8px; margin:12px 0 2px; flex-wrap:wrap; }}
.badge {{ display:inline-flex; flex-direction:column; align-items:center;
  background:var(--bg-glass); border:1px solid var(--border-primary);
  border-radius:10px; padding:6px 14px; min-width:96px; }}
.badge small {{ font-family:var(--font-mono); font-size:.625rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--text-muted); }}
.badge b {{ font-size:1.125rem; font-family:var(--font-mono); }}
#timeline {{ margin-top:4px; width:560px; position:relative; }}
.tlrow {{ display:flex; align-items:center; height:22px; margin:3px 0; }}
.tlabel {{ width:132px; font-family:var(--font-mono); font-size:.625rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--text-muted); text-align:right; padding-right:8px; }}
.tband {{ position:relative; flex:1; height:14px; background:var(--bg-tertiary); cursor:pointer;
  border-radius:4px; overflow:hidden; border:1px solid var(--border-primary); }}
.seg {{ position:absolute; top:0; bottom:0; opacity:.85; }}
#playhead {{ position:absolute; top:0; bottom:0; width:2px; background:var(--pale-sky);
  box-shadow:0 0 8px rgba(191,215,234,.6); pointer-events:none; left:132px; }}
table {{ border-collapse:collapse; font-size:.8125rem; width:100%; }}
th {{ font-family:var(--font-mono); font-size:.625rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--text-muted); text-align:left; }}
td, th {{ border-bottom:1px solid var(--border-primary); padding:6px 10px; vertical-align:middle; }}
tr:hover td {{ background:var(--bg-glass-hover); cursor:pointer; }}
tr.now td {{ background:rgba(217,164,65,.12); box-shadow:inset 2px 0 0 var(--warning); }}
canvas.overlay {{ border-radius:6px; border:1px solid var(--border-primary); display:block; }}
.score {{ font-family:var(--font-mono); font-weight:700; }}
</style></head><body><div class="layout">
<h2>SAM object-verification — {html.escape(video_path.name)}
<span class="status-badge running" id="status">waiting</span></h2>
<div class="meta">model <b>{html.escape(meta["model"])}</b> · instruction
“{html.escape(meta["instruction"])}” · view <b>{meta["view"]}</b>
x{meta["upscale"]} · stride {meta["stride"]} · PRESENT when best score ≥
<b>{meta["score_threshold"]}</b> · big panel = the exact crop SAM judges, zoomed,
with its detection boxes; click a band or row to seek</div>
<div class="card">
<div class="viewrow">
<canvas id="det" width="480" height="480"></canvas>
<div class="side">
<video id="v" controls src="data:video/mp4;base64,{video_b64}"></video>
<div class="speed"><span>speed</span>
<button onclick="rate(0.1)">0.1x</button><button onclick="rate(0.25)">0.25x</button>
<button onclick="rate(0.5)">0.5x</button><button onclick="rate(1)">1x</button></div>
<div class="legend">{legend}</div>
<div class="badges">{badges}</div>
</div>
</div>
<div id="timeline">{tl_rows}<div id="playhead"></div></div>
</div>
<div class="card">
<table id="tbl"><tr><th>frame</th><th>t</th><th>concept</th><th>crop + boxes</th>
<th>present</th><th>best score</th><th>#det</th><th>latency</th></tr></table>
</div>
</div><script>
const CFG = {config};
const v = document.getElementById("v");
const det = document.getElementById("det"), dctx = det.getContext("2d");
v.addEventListener("loadedmetadata", () => {{ v.playbackRate = 0.25; sizeDet(); }});
window.addEventListener("resize", sizeDet);
function rate(x) {{ v.playbackRate = x; }}
function seek(t) {{ v.currentTime = t + 0.001; }}
function idFor(s) {{ return s.replace(/[^a-zA-Z0-9_-]/g, "_"); }}
let records = [], done = false, rendered = 0, thumbs = {{}};

// source-pixel crop rect matching the view the model judged (2:1 concat frame)
function cropRect() {{
  const VW = v.videoWidth || 2, VH = v.videoHeight || 1;
  const concat = Math.abs(VW - 2*VH) <= 2;
  if (CFG.view === "wrist" && concat) return {{sx:VW/2, sy:0, sw:VW/2, sh:VH}};
  if (CFG.view === "side"  && concat) return {{sx:0,    sy:0, sw:VW/2, sh:VH}};
  return {{sx:0, sy:0, sw:VW, sh:VH}};
}}
function sizeDet() {{
  const c = cropRect(), MAX = 480;
  det.width = MAX; det.height = Math.max(1, Math.round(MAX * c.sh / c.sw));
}}

function frameGroups() {{
  const by = {{}};
  for (const r of records) {{ (by[r.frame] = by[r.frame] || {{time_s:r.time_s, items:[]}}).items.push(r); }}
  return Object.keys(by).map(k => ({{frame:+k, ...by[k]}})).sort((a,b)=>a.time_s-b.time_s);
}}

// Live detection panel: draw the current crop zoomed to fill, then the boxes
// for the frame nearest the playhead (boxes are crop-normalized -> land exactly).
function drawDet() {{
  requestAnimationFrame(drawDet);
  if (!v.videoWidth) return;
  if (!det.height || det.height < 2) sizeDet();
  const c = cropRect(), W = det.width, H = det.height;
  try {{ dctx.drawImage(v, c.sx, c.sy, c.sw, c.sh, 0, 0, W, H); }}
  catch (e) {{ dctx.fillStyle = "#000"; dctx.fillRect(0, 0, W, H); }}
  const t = v.currentTime, groups = frameGroups();
  let g = null; for (const x of groups) if (x.time_s <= t + 1e-6) g = x;
  if (!g) return;
  for (const r of g.items) {{
    if (r.question === "control") continue;
    const boxes = r.boxes || [];
    if (!boxes.length) continue;
    const present = r.answer === "PRESENT";
    const col = CFG.conceptColors[r.question] || "#fff";
    dctx.lineWidth = present ? 3 : 1.5;
    dctx.globalAlpha = present ? 1 : 0.4;
    dctx.strokeStyle = col; dctx.shadowColor = present ? col : "transparent";
    dctx.shadowBlur = present ? 8 : 0;
    for (const b of boxes) {{
      const x0=b[0]*W, y0=b[1]*H, bw=(b[2]-b[0])*W, bh=(b[3]-b[1])*H;
      dctx.strokeRect(x0, y0, bw, bh);
      if (present) {{
        dctx.shadowBlur = 0;
        const label = r.question + "  " + r.best_score.toFixed(2);
        dctx.font = "700 13px 'Space Mono', monospace";
        const tw = dctx.measureText(label).width + 10, ly = y0 > 20 ? y0-18 : y0+bh+2;
        dctx.globalAlpha = 1; dctx.fillStyle = col;
        dctx.fillRect(x0 - 1.5, ly, tw, 17);
        dctx.fillStyle = "#07120f"; dctx.fillText(label, x0 + 4, ly + 13);
        dctx.strokeStyle = col; dctx.shadowColor = col; dctx.shadowBlur = 8;
      }}
    }}
  }}
  dctx.globalAlpha = 1; dctx.shadowBlur = 0;
}}
requestAnimationFrame(drawDet);

function drawOverlay(td, r) {{  // per-row crop thumbnail with its boxes
  const img = new Image();
  img.onload = () => {{
    const cv = document.createElement("canvas");
    const H = 80, scale = H / img.height, W = img.width * scale;
    cv.width = W; cv.height = H; cv.className = "overlay";
    const ctx = cv.getContext("2d");
    ctx.drawImage(img, 0, 0, W, H);
    ctx.lineWidth = 2; ctx.strokeStyle = CFG.colors[r.answer] || "#fff";
    for (const b of (r.boxes || [])) ctx.strokeRect(b[0]*W, b[1]*H, (b[2]-b[0])*W, (b[3]-b[1])*H);
    td.innerHTML = ""; td.appendChild(cv);
  }};
  img.src = thumbs[r.frame] || "";
}}

function onData(d) {{
  if (!d || d.records.length === records.length && done === d.done) return;
  records = d.records; done = d.done;
  const st = document.getElementById("status");
  st.textContent = (done ? "finished · " : "running · ") + records.length + " detections";
  st.className = "status-badge " + (done ? "done" : "running");
  const tbl = document.getElementById("tbl");
  for (; rendered < records.length; rendered++) {{
    const r = records[rendered];
    if (r.thumb) thumbs[r.frame] = r.thumb;
    const tr = document.createElement("tr");
    tr.id = "r" + r.frame + "-" + idFor(r.question);
    tr.onclick = () => seek(r.time_s);
    tr.innerHTML = "<td>" + r.frame + "</td><td>" + r.time_s.toFixed(1) +
        "s</td><td>" + r.question + '</td><td class="ov"></td>' +
        '<td class="score" style="color:' + CFG.colors[r.answer] + '">' +
        r.answer + "</td><td>" + r.best_score.toFixed(2) + "</td><td>" +
        r.num_detections + "</td><td>" + r.latency_s.toFixed(2) + "s</td>";
    tbl.appendChild(tr);
    drawOverlay(tr.querySelector(".ov"), r);
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
    const band = document.getElementById("band-" + idFor(q));
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
    const el = document.getElementById("bv-" + idFor(q));
    el.textContent = pt ? pt.best_score.toFixed(2) : "?";
    el.style.color = pt ? (CFG.colors[pt.answer] || "#333") : "#333";
  }}
  if (current !== lastFrame) {{
    document.querySelectorAll("tr.now").forEach(el => el.classList.remove("now"));
    for (const q of CFG.questions) {{
      const row = document.getElementById("r" + current + "-" + idFor(q));
      if (row) row.classList.add("now");
    }}
    lastFrame = current;
  }}
}}
v.addEventListener("timeupdate", refreshUI);
setInterval(refreshUI, 120);

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


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--objects", required=True, help="comma-separated present objects")
    parser.add_argument("--distractors", default="", help="comma-separated absent objects")
    parser.add_argument("--base_url", default="http://127.0.0.1:8006")
    parser.add_argument("--model", default="facebook/sam3.1")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--view", choices=["full", "wrist", "side"], default="full")
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--stride", type=int, default=5, help="segment every Nth frame")
    parser.add_argument("--timeout_seconds", type=float, default=120.0)
    parser.add_argument("--out", default="/tmp/sam_report.html")
    parser.add_argument("--fake", action="store_true", help="no server: deterministic boxes")
    args = parser.parse_args()

    import imageio.v3 as iio

    objects = _split_csv(args.objects)
    distractors = _split_csv(args.distractors)
    concepts = build_concepts(objects, distractors)

    video_path = Path(args.video).expanduser()
    stack = iio.imread(video_path)
    try:
        fps = float(iio.immeta(video_path).get("fps") or 20.0)
    except Exception:
        fps = 20.0

    if args.fake:
        def transport(url: str, body: dict[str, Any]) -> dict[str, Any]:
            return _fake_transport(url, body)
    else:
        def transport(url: str, body: dict[str, Any]) -> dict[str, Any]:
            return _http_transport(url, body, args.timeout_seconds)

    meta = {
        "model": args.model, "instruction": args.instruction,
        "view": args.view, "upscale": args.upscale, "stride": args.stride,
        "score_threshold": args.score_threshold,
    }
    out = Path(args.out).expanduser()
    records: list[dict[str, Any]] = []
    render_shell(out, video_path=video_path, meta=meta, concepts=concepts)
    write_data(out, records, done=False)

    indexes = list(range(0, len(stack), args.stride))
    print(f"{len(stack)} frames, segmenting {len(indexes)} of them x {len(concepts)} concepts")
    print(f"report: {out}  (open it now — rows stream in without reloading)\n")
    print(f"{'frame':>6} {'concept':>22}  present  score  latency")
    from sam_probe import _png_data_uri as _full_uri  # full-size crop for POSTing

    for index in indexes:
        frame = prepare_frame(stack[index], args.view, args.upscale)
        thumb = _png_data_uri(frame)
        image_uri = _full_uri(frame)
        for position, (name, concept) in enumerate(concepts):
            start = time.monotonic()
            result = segment(transport, args.base_url, image_uri, concept)
            elapsed = time.monotonic() - start
            detections = result.get("detections") or []
            best = max((d.get("score", 0.0) for d in detections), default=0.0)
            if "error" in result:
                answer = "FAIL"
            else:
                answer = "PRESENT" if best >= args.score_threshold else "ABSENT"
            boxes_crop = _normalize_boxes(result, frame)
            records.append(
                {
                    "frame": index,
                    "time_s": index / fps,
                    "question": name,
                    "answer": answer,
                    "best_score": round(float(best), 3),
                    "num_detections": len(detections),
                    "boxes": boxes_crop,
                    "boxes_full": _to_full_frame(boxes_crop, args.view),
                    "latency_s": elapsed,
                    "thumb": thumb if position == 0 else None,
                }
            )
            print(f"{index:6d} {name:>22}  {answer:<7} {best:.2f}  {elapsed:.2f}s")
        write_data(out, records, done=False)
    write_data(out, records, done=True)

    summary_path = out.with_suffix(".json")
    lean = [{k: val for k, val in r.items() if k != "thumb"} for r in records]
    summary_path.write_text(json.dumps({"meta": meta, "records": lean}, indent=2))
    print(f"\ndone -> {out}  (+ raw records in {summary_path})")


if __name__ == "__main__":
    main()
