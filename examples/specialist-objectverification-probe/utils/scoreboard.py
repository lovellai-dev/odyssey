"""Object-verification SCOREBOARD — read the bake-off verdict at a glance.

The streaming per-model viewers (``visualize_probe.py`` / ``visualize_sam_probe.py``)
are for *inspecting* one rollout frame by frame; this tool is for *reading the
benchmark result*: which named objects each model actually recognises, over the
whole run, without pressing play. It is a static HTML report built from the
probe result JSONs (the ``--out-json`` each arm writes).

Layout — rows are objects, columns are models, so RoboBrain and SAM sit side by
side per object:

    OBJECT VERIFICATION
                       RoboBrain·side   SAM 3.1
      present objects (want high recall)
      red capsule      ██████░░ 50%     (pending)
      blue tray        ████████ 100%    (pending)
      distractors (want 0% — false positives)
      green bottle     ░░░░░░░░  0%     (pending)
      control (sanity) ████████ 100%    (pending)

Each cell shows the recall (present) / false-positive rate (distractor) /
sanity rate (control) as a colored bar PLUS a per-frame heat-strip. Colour is
**correctness**, so the whole board reads green = good: a present object should
be YES, a distractor NO, control YES — green means the model gave the expected
verdict, red means it did not, grey a failed call. Models with no data yet are
rendered as a ``pending`` column (``--pending "SAM 3.1"``).

    python utils/scoreboard.py \\
        --result "RoboBrain·side=/tmp/rb_side.json" \\
        --result "RoboBrain·wrist=/tmp/rb_objverif.json" \\
        --pending "SAM 3.1" \\
        --out /tmp/objverif_scoreboard.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

# Serene Ocean semantic colors.
GOOD = "#34d399"
BAD = "#c94a4a"
FAILC = "#5a7a8f"


def _goodness_color(goodness: float) -> str:
    """Red (0) -> amber -> green (1) via HSL hue sweep."""
    hue = int(max(0.0, min(1.0, goodness)) * 130)  # 0=red, 130=green
    return f"hsl({hue}, 62%, 52%)"


def load_result(path: Path) -> dict[str, Any]:
    metrics = json.loads(path.read_text())["metrics"]
    return metrics


def verdicts_for(metrics: dict[str, Any], question: str) -> list[dict[str, Any]]:
    return [v for v in metrics.get("verdicts", []) if v["question"] == question]


def cell_data(metrics: dict[str, Any], kind: str, name: str) -> dict[str, Any]:
    """One (model, row) cell: headline pct, goodness 0..1, and heat cells.

    ``kind`` in {"present", "absent", "control"}. Heat correctness: present
    wants YES, absent wants NO, control wants YES.
    """
    if kind == "present":
        question = f"present:{name}"
        pct = (metrics.get("per_object_yes_rate") or {}).get(name)
        want_yes = True
        label = "recall"
    elif kind == "absent":
        question = f"absent:{name}"
        pct = (metrics.get("per_distractor_yes_rate") or {}).get(name)
        want_yes = False
        label = "false-pos"
    else:
        question = "control"
        pct = metrics.get("control_yes_rate")
        want_yes = True
        label = "sanity"

    rows = verdicts_for(metrics, question)
    heat = []
    for v in rows:
        if v.get("call_failed"):
            heat.append(("fail", v))
        else:
            yes = v["answer"] == "YES"
            correct = yes == want_yes
            heat.append(("good" if correct else "bad", v))

    goodness = None if pct is None else (pct if want_yes else 1.0 - pct)
    return {"pct": pct, "goodness": goodness, "label": label, "heat": heat}


def render_cell(cell: dict[str, Any]) -> str:
    if cell["pct"] is None:
        return '<td class="mcell"><span class="nodata">—</span></td>'
    pct = cell["pct"]
    goodness = cell["goodness"]
    color = _goodness_color(goodness)
    strip = "".join(
        f'<i class="hc {klass}" title="{html.escape(v["video"])} @ {v["position"]}: '
        f'{v["answer"]}"></i>'
        for klass, v in cell["heat"]
    )
    return (
        '<td class="mcell">'
        f'<div class="pctline"><b style="color:{color}">{round(pct * 100)}%</b>'
        f'<span class="lbl">{cell["label"]}</span></div>'
        f'<div class="bar"><span style="width:{round(pct * 100)}%;background:{color}"></span></div>'
        f'<div class="heat">{strip}</div>'
        "</td>"
    )


def render(
    out: Path,
    *,
    title: str,
    results: list[tuple[str, dict[str, Any]]],
    pending: list[str],
) -> None:
    ref = results[0][1] if results else {}
    objects = ref.get("objects", [])
    distractors = ref.get("distractors", [])
    n_frames = ref.get("frames_probed", "?")
    model_labels = [label for label, _ in results] + pending

    head_cols = "".join(
        f'<th class="mhead">{html.escape(lbl)}</th>' for lbl in model_labels
    )

    def summary(metrics: dict[str, Any]) -> str:
        pr = metrics.get("present_recall")
        fpr = metrics.get("distractor_fpr")
        lat = metrics.get("latency_s_mean")
        pr_s = f"{round(pr * 100)}%" if pr is not None else "—"
        fpr_s = f"{round(fpr * 100)}%" if fpr is not None else "—"
        lat_s = f"{lat:.2f}s/call" if lat is not None else "—"
        return (
            f'<div class="msum">recall <b>{pr_s}</b> · FP <b>{fpr_s}</b> · '
            f'lat <b>{lat_s}</b><br>model {html.escape(str(metrics.get("model", "?")))}</div>'
        )

    sum_cols = "".join(f'<th class="mhead">{summary(m)}</th>' for _, m in results)
    sum_cols += "".join('<th class="mhead"><div class="msum pend">no data yet</div></th>' for _ in pending)

    def row(kind: str, name: str, display: str, css: str = "") -> str:
        cells = "".join(render_cell(cell_data(m, kind, name)) for _, m in results)
        cells += "".join('<td class="mcell pend">pending</td>' for _ in pending)
        return f'<tr class="{css}"><td class="oname">{html.escape(display)}</td>{cells}</tr>'

    body = []
    body.append(
        f'<tr class="section"><td colspan="{len(model_labels) + 1}">'
        f"present objects &nbsp;<span>want high recall</span></td></tr>"
    )
    for obj in objects:
        body.append(row("present", obj, obj))
    body.append(
        f'<tr class="section"><td colspan="{len(model_labels) + 1}">'
        f"distractors &nbsp;<span>want 0% — false positives</span></td></tr>"
    )
    for obj in distractors:
        body.append(row("absent", obj, obj, "distractor"))
    body.append(
        f'<tr class="section"><td colspan="{len(model_labels) + 1}">'
        f"sanity</td></tr>"
    )
    body.append(row("control", "control", "control (arm visible)"))

    out.write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
:root {{
  --bg-primary:#070c10; --bg-secondary:#0c1318; --bg-tertiary:#121c22;
  --bg-glass:rgba(145,174,193,.07); --text-primary:#eaf2f7;
  --text-secondary:#9ab5c7; --text-muted:#5a7a8f;
  --border-primary:rgba(145,174,193,.14); --border-secondary:rgba(145,174,193,.24);
  --pale-sky:#bfd7ea; --warning:#d9a441;
  --gradient-card:linear-gradient(160deg,rgba(12,19,24,.94) 0%,rgba(7,12,16,.98) 100%);
  --font-primary:'DM Sans',-apple-system,sans-serif; --font-mono:'Space Mono',monospace;
}}
body {{ font-family:var(--font-primary); margin:0; padding:2rem 2.5rem;
  background:var(--bg-primary); color:var(--text-primary);
  background-image:radial-gradient(ellipse at 30% 10%,rgba(80,140,164,.08) 0%,transparent 45%),
    radial-gradient(ellipse at 80% 90%,rgba(10,135,84,.06) 0%,transparent 45%); }}
h1 {{ font-weight:600; letter-spacing:-.02em; font-size:1.5rem; margin:0 0 2px; }}
.sub {{ color:var(--text-secondary); font-size:.8125rem; margin:0 0 20px; }}
.legend {{ display:flex; gap:16px; margin:0 0 18px; font-size:.75rem;
  color:var(--text-secondary); align-items:center; flex-wrap:wrap; }}
.legend i {{ display:inline-block; width:11px; height:11px; border-radius:3px;
  margin-right:5px; vertical-align:-1px; }}
table {{ border-collapse:collapse; width:100%; max-width:1000px;
  background:var(--gradient-card); border:1px solid var(--border-primary);
  border-radius:14px; overflow:hidden; }}
th, td {{ padding:11px 16px; text-align:left; vertical-align:middle; }}
thead th {{ background:var(--bg-tertiary); border-bottom:1px solid var(--border-secondary); }}
.mhead {{ font-family:var(--font-mono); font-size:.8125rem; color:var(--pale-sky);
  font-weight:700; min-width:172px; }}
.msum {{ font-family:var(--font-mono); font-size:.625rem; font-weight:400;
  color:var(--text-muted); margin-top:3px; letter-spacing:.02em; }}
.msum b {{ color:var(--text-secondary); }}
.msum.pend {{ color:var(--text-muted); font-style:italic; }}
tr.section td {{ font-family:var(--font-mono); font-size:.625rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--text-muted); padding:14px 16px 6px;
  border-top:1px solid var(--border-primary); }}
tr.section span {{ color:var(--warning); text-transform:none; letter-spacing:.02em;
  font-size:.6875rem; }}
.oname {{ font-weight:500; font-size:.9375rem; }}
tr.distractor .oname {{ color:var(--text-secondary); }}
.mcell {{ border-left:1px solid var(--border-primary); }}
.pctline b {{ font-family:var(--font-mono); font-size:1.0625rem; }}
.pctline .lbl {{ font-family:var(--font-mono); font-size:.5625rem; color:var(--text-muted);
  text-transform:uppercase; letter-spacing:.08em; margin-left:6px; }}
.bar {{ height:6px; background:var(--bg-tertiary); border-radius:3px; margin:5px 0 6px;
  overflow:hidden; max-width:150px; }}
.bar span {{ display:block; height:100%; border-radius:3px; }}
.heat {{ display:flex; gap:2px; flex-wrap:wrap; max-width:150px; }}
.hc {{ width:9px; height:9px; border-radius:2px; display:inline-block; }}
.hc.good {{ background:{GOOD}; }} .hc.bad {{ background:{BAD}; }}
.hc.fail {{ background:{FAILC}; opacity:.5; }}
.mcell.pend {{ color:var(--text-muted); font-style:italic; font-family:var(--font-mono);
  font-size:.75rem; }}
.nodata {{ color:var(--text-muted); }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(str(len(objects)))} objects · {html.escape(str(len(distractors)))}
distractors · {html.escape(str(n_frames))} frames judged per model · each square = one judged frame</p>
<div class="legend">
  <span><i style="background:{GOOD}"></i>model gave the expected verdict</span>
  <span><i style="background:{BAD}"></i>wrong verdict</span>
  <span><i style="background:{FAILC};opacity:.5"></i>failed call</span>
  <span>bar/% = recall (present) · false-positive rate (distractor) · sanity (control)</span>
</div>
<table><thead>
<tr><th>object</th>{head_cols}</tr>
<tr><th></th>{sum_cols}</tr>
</thead><tbody>
{''.join(body)}
</tbody></table>
</body></html>""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="a probe out-json to include as a model column (repeatable)",
    )
    parser.add_argument(
        "--pending",
        action="append",
        default=[],
        metavar="LABEL",
        help="a model with no data yet — rendered as a pending column (repeatable)",
    )
    parser.add_argument("--title", default="Object verification — bake-off scoreboard")
    parser.add_argument("--out", default="/tmp/objverif_scoreboard.html")
    args = parser.parse_args()

    results: list[tuple[str, dict[str, Any]]] = []
    for spec in args.result:
        if "=" not in spec:
            raise SystemExit(f"--result must be LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        results.append((label, load_result(Path(path).expanduser())))
    if not results:
        raise SystemExit("need at least one --result LABEL=PATH")

    out = Path(args.out).expanduser()
    render(out, title=args.title, results=results, pending=args.pending)
    print(f"wrote scoreboard -> {out}")


if __name__ == "__main__":
    main()
