"""Watch the value probe estimate task progress, frame by frame.

The Iteration-2 companion of ``visualize_probe.py`` (same streaming pattern):
the model never sees the *video* — per sampled frame it is asked one
independent image+text question, "how far has the task progressed, 0-100?".
This tool replays that loop over a single rollout MP4 and shows it:

* terminal — one row per frame as each estimate arrives;
* HTML report — the playable video (slow-motion by default), a live progress
  curve that grows as estimates stream in, the live value + stall verdict,
  and the full per-frame table. The page loads ONCE and polls a sidecar
  ``*_data.js`` file, so new points stream in with no page reload — playback
  is never interrupted. Clicking the curve or a table row seeks the video.

The verdict mirrors ``robobrain_value_probe.py``: if progress has not risen
at least MIN_RISE points over its opening value by the STALL_CHECKPOINT
fraction of the episode, the rollout is flagged STALLED -> RETRY.

Usage (server recipe in ../README.md; tunnel with `ssh -L 8002:127.0.0.1:8002`
if the model is served on the H100):

    python examples/specialist-retry-probe/utils/visualize_value_probe.py \\
        --video ~/videos/rollout_ep000_fail.mp4 \\
        --instruction "pick up the red capsule and place it in the blue tray" \\
        --view side --upscale 2 --stride 5 --out /tmp/value_report.html

    open /tmp/value_report.html      # while the run is live, or after

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
import zlib
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))          # the probe modules under test
_REPO_SRC = _HERE.parents[2] / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

from robobrain_probe import prepare_frame  # noqa: E402
from robobrain_value_probe import (  # noqa: E402
    MIN_RISE,
    STALL_CHECKPOINT,
    VALUE_TEMPLATE,
    ask_value,
    stall_verdict,
)

# Serene Ocean semantic colors (lai-trainer command-center theme).
CURVE_COLOR = "#508ca4"
OK_COLOR = "#34d399"
FAIL_COLOR = "#c94a4a"


def _fake_value(index: int, total: int) -> int:
    """Deterministic rising-with-noise curve (viewer testing only)."""
    base = int(90 * index / max(total - 1, 1))
    return min(100, base + zlib.crc32(str(index).encode()) % 15)


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
    """Sidecar the page polls — atomic-ish single write per judged frame.

    The verdict ships COMPUTED (via the probe's own ``stall_verdict``, on the
    median-smoothed curve), so the page displays exactly what the batch probe
    would decide — no duplicated JS rule to drift."""
    verdict = stall_verdict([r["value"] for r in records])
    payload = json.dumps({"records": records, "done": done, "verdict": verdict})
    data_path_for(out).write_text(f"window.PROBE_DATA = {payload};")


def render_shell(out: Path, *, video_path: Path, meta: dict[str, Any]) -> None:
    """Static page, written ONCE: video + empty containers. Curve and rows are
    rendered client-side from the polled sidecar, so nothing here ever reloads
    and playback is never interrupted."""
    video_b64 = base64.b64encode(video_path.read_bytes()).decode("ascii")
    config = json.dumps(
        {
            "data_src": data_path_for(out).name,
            "checkpoint": STALL_CHECKPOINT,
            "min_rise": MIN_RISE,
            "colors": {"curve": CURVE_COLOR, "ok": OK_COLOR, "fail": FAIL_COLOR},
        }
    )
    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Value probe — {html.escape(video_path.name)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<style>
/* Serene Ocean Oasis — lai-trainer command-center theme tokens */
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
.badges {{ display:flex; gap:8px; margin:12px 0 2px; }}
.badge {{ display:inline-flex; flex-direction:column; align-items:center;
  background:var(--bg-glass); border:1px solid var(--border-primary);
  border-radius:10px; padding:6px 14px; min-width:96px; }}
.badge small {{ font-family:var(--font-mono); font-size:.625rem;
  letter-spacing:.08em; text-transform:uppercase; color:var(--text-muted); }}
.badge b {{ font-size:1.125rem; font-family:var(--font-mono); }}
#curvebox {{ margin-top:10px; position:relative; }}
table {{ border-collapse:collapse; font-size:.8125rem; width:100%; }}
th {{ font-family:var(--font-mono); font-size:.625rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--text-muted); text-align:left; }}
td, th {{ border-bottom:1px solid var(--border-primary); padding:6px 10px;
  vertical-align:middle; }}
tr:hover td {{ background:var(--bg-glass-hover); cursor:pointer; }}
tr.now td {{ background:rgba(217,164,65,.12);
  box-shadow:inset 2px 0 0 var(--warning); }}
td img {{ border-radius:6px; border:1px solid var(--border-primary); }}
td.raw {{ color:var(--text-muted); font-size:.6875rem; max-width:320px; }}
.val {{ font-family:var(--font-mono); font-weight:700; }}
</style></head><body><div class="layout">
<h2>Retry value probe — {html.escape(video_path.name)}
<span class="status-badge running" id="status">waiting</span></h2>
<div class="meta">model <b>{html.escape(meta["model"])}</b> · instruction
“{html.escape(meta["instruction"])}” · view <b>{meta["view"]}</b>
x{meta["upscale"]} · stride {meta["stride"]} · progress estimate 0-100 per frame;
verdict = STALLED → RETRY if the median-smoothed rise over opening &lt;
{MIN_RISE} points by the {int(STALL_CHECKPOINT * 100)}% checkpoint (needs
stride &le; 5) — the curve and rows stream in without reloading; click the
curve or a row to seek</div>
<div class="card">
<video id="v" controls width="560" src="data:video/mp4;base64,{video_b64}"></video>
<div class="speed"><span>speed</span>
<button onclick="rate(0.1)">0.1x</button><button onclick="rate(0.25)">0.25x</button>
<button onclick="rate(0.5)">0.5x</button><button onclick="rate(1)">1x</button></div>
<div class="badges">
<span class="badge"><small>progress</small><b id="bv-now">?</b></span>
<span class="badge"><small>opening</small><b id="bv-open">?</b></span>
<span class="badge"><small>peak</small><b id="bv-peak">?</b></span>
<span class="badge"><small>verdict</small><b id="bv-verdict">?</b></span>
</div>
<div id="curvebox">
<svg id="curve" width="560" height="170" style="background:var(--bg-secondary);border-radius:10px"></svg>
</div>
</div>
<div class="card">
<table id="tbl"><tr><th>frame</th><th>t</th><th>judged image</th><th>value</th>
<th>latency</th><th>raw reply</th></tr></table>
</div>
</div><script>
const CFG = {config};
const v = document.getElementById("v");
v.addEventListener("loadedmetadata", () => {{ v.playbackRate = 0.25; }});
function rate(x) {{ v.playbackRate = x; }}
function seek(t) {{ v.currentTime = t + 0.001; }}
const W = 560, H = 170, PAD = 26;
let records = [], done = false, rendered = 0, verdictData = null;

function xOf(t) {{ return PAD + (t / (v.duration || 1)) * (W - 2 * PAD); }}
function yOf(val) {{ return H - PAD - (val / 100) * (H - 2 * PAD); }}

function verdict() {{
  // Streamed pre-computed from Python (the probe's own smoothed stall rule);
  // provisional while the run is still filling the curve.
  if (!verdictData || verdictData.rise_by_checkpoint === null)
      return ["?", CFG.colors.curve];
  if (!done) return ["rise " + verdictData.rise_by_checkpoint, CFG.colors.curve];
  return verdictData.stalled
      ? ["STALLED → RETRY", CFG.colors.fail] : ["progressing", CFG.colors.ok];
}}

function drawCurve() {{
  const svg = document.getElementById("curve");
  const D = v.duration;
  if (!D) {{ v.addEventListener("loadedmetadata", drawCurve, {{once: true}}); return; }}
  let parts = "";
  for (const g of [0, 50, 100]) parts +=
      `<line x1="${{PAD}}" y1="${{yOf(g)}}" x2="${{W - PAD}}" y2="${{yOf(g)}}"
       stroke="rgba(145,174,193,.15)"/>` +
      `<text x="4" y="${{yOf(g) + 4}}" fill="#5a7a8f" font-size="10">${{g}}</text>`;
  const cx = xOf(CFG.checkpoint * D);
  parts += `<line x1="${{cx}}" y1="${{PAD}}" x2="${{cx}}" y2="${{H - PAD}}"
      stroke="rgba(217,164,65,.4)" stroke-dasharray="4 4"/>` +
      `<text x="${{cx + 4}}" y="${{PAD + 10}}" fill="#d9a441"
       font-size="10">${{Math.round(CFG.checkpoint * 100)}}%</text>`;
  const known = records.filter(r => r.value !== null);
  const pts = known.map(r => `${{xOf(r.time_s)}},${{yOf(r.value)}}`).join(" ");
  parts += `<polyline points="${{pts}}" fill="none" stroke="${{CFG.colors.curve}}"
      stroke-width="2.5"/>`;
  for (const r of known) parts +=
      `<circle cx="${{xOf(r.time_s)}}" cy="${{yOf(r.value)}}" r="3"
       fill="${{CFG.colors.curve}}"/>`;
  parts += `<line id="ph" x1="0" y1="${{PAD}}" x2="0" y2="${{H - PAD}}"
      stroke="#bfd7ea" stroke-width="2" style="filter:drop-shadow(0 0 4px rgba(191,215,234,.6))"/>`;
  svg.innerHTML = parts;
  svg.onclick = (e) => {{
    const rect = svg.getBoundingClientRect();
    seek((e.clientX - rect.left - PAD) / (W - 2 * PAD) * D);
  }};
}}

function onData(d) {{
  if (!d || d.records.length === records.length && done === d.done) return;
  records = d.records; done = d.done; verdictData = d.verdict;
  const st = document.getElementById("status");
  st.textContent = (done ? "finished · " : "running · ") + records.length + " estimates";
  st.className = "status-badge " + (done ? "done" : "running");
  const tbl = document.getElementById("tbl");
  for (; rendered < records.length; rendered++) {{
    const r = records[rendered];
    const tr = document.createElement("tr");
    tr.id = "r" + r.frame;
    tr.onclick = () => seek(r.time_s);
    const val = r.value === null ? "FAIL" : r.value;
    const color = r.value === null ? CFG.colors.fail : CFG.colors.curve;
    tr.innerHTML = "<td>" + r.frame + "</td><td>" + r.time_s.toFixed(1) +
        "s</td><td>" + (r.thumb ? '<img src="' + r.thumb + '" height="72">' : "") +
        '</td><td class="val" style="color:' + color + '">' + val +
        "</td><td>" + r.latency_s.toFixed(2) + 's</td><td class="raw"></td>';
    tr.lastChild.textContent = r.raw_reply;
    tbl.appendChild(tr);
  }}
  drawCurve();
}}

let lastFrame = null;
function refreshUI() {{
  const t = v.currentTime, D = v.duration || 1;
  const ph = document.getElementById("ph");
  if (ph) {{ const x = xOf(t); ph.setAttribute("x1", x); ph.setAttribute("x2", x); }}
  let current = null;
  for (const r of records) if (r.time_s <= t + 1e-6) current = r;
  document.getElementById("bv-now").textContent =
      current && current.value !== null ? current.value : "?";
  document.getElementById("bv-open").textContent =
      verdictData && verdictData.opening !== null ? verdictData.opening : "?";
  document.getElementById("bv-peak").textContent =
      verdictData && verdictData.peak !== null ? verdictData.peak : "?";
  const [verdictText, verdictColor] = verdict();
  const bv = document.getElementById("bv-verdict");
  bv.textContent = verdictText; bv.style.color = verdictColor;
  const frame = current ? current.frame : null;
  if (frame !== lastFrame) {{
    document.querySelectorAll("tr.now").forEach(el => el.classList.remove("now"));
    const row = document.getElementById("r" + frame);
    if (row) row.classList.add("now");
    lastFrame = frame;
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
    parser.add_argument("--stride", type=int, default=5, help="query every Nth frame")
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument("--timeout_seconds", type=float, default=120.0)
    parser.add_argument("--out", default="/tmp/value_report.html")
    parser.add_argument("--fake", action="store_true", help="no server: deterministic values")
    args = parser.parse_args()

    import imageio.v3 as iio

    video_path = Path(args.video).expanduser()
    stack = iio.imread(video_path)
    try:
        fps = float(iio.immeta(video_path).get("fps") or 20.0)
    except Exception:
        fps = 20.0

    meta = {
        "model": args.model, "instruction": args.instruction,
        "view": args.view, "upscale": args.upscale, "stride": args.stride,
    }
    out = Path(args.out).expanduser()
    records: list[dict[str, Any]] = []
    render_shell(out, video_path=video_path, meta=meta)
    write_data(out, records, done=False)

    indexes = list(range(0, len(stack), args.stride))
    print(f"{len(stack)} frames, estimating progress on {len(indexes)} of them")
    print(f"prompt: {VALUE_TEMPLATE.format(instruction=args.instruction)[:88]}...")
    print(f"report: {out}  (open it now — the curve streams in without reloading)\n")
    print(f"{'frame':>6}  value  latency")
    for index in indexes:
        frame = prepare_frame(stack[index], args.view, args.upscale)
        if args.fake:
            value, raw, latency = _fake_value(index, len(stack)), "fake", 0.0
        else:
            value, raw, latency = ask_value(frame, args.instruction, args)
        records.append(
            {
                "frame": index,
                "time_s": round(index / fps, 2),
                "value": value,
                "latency_s": round(latency, 2),
                "raw_reply": raw[:120],
                "thumb": _png_data_uri(frame),
            }
        )
        print(f"{index:6d}  {'?' if value is None else value:>5}  {latency:.2f}s")
        write_data(out, records, done=False)
    write_data(out, records, done=True)

    summary_path = out.with_suffix(".json")
    lean = [{k: v for k, v in r.items() if k != "thumb"} for r in records]
    summary_path.write_text(json.dumps({"meta": meta, "records": lean}, indent=2))
    print(f"\ndone -> {out}  (+ raw records in {summary_path})")


if __name__ == "__main__":
    main()
