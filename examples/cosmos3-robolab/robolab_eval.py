#!/usr/bin/env python3
"""RoboLab bridge for `evaluation_type: custom` — score a Cosmos 3 DROID policy.

NVIDIA evaluates the published Cosmos3-{Nano,Edge}-Policy-DROID checkpoints on
`RoboLab <https://github.com/NVlabs/RoboLab>`_, a client/server simulation
benchmark: a cosmos-framework **policy server** streams action chunks to the
RoboLab **client** driving the sim. Odyssey has no first-class RoboLab runner
(yet — promote one only if this becomes recurring); instead this bridge script
satisfies the ``CustomEvalRunner`` contract (see
``src/odyssey/runners/evals/custom.py``)::

    <eval_python> robolab_eval.py --checkpoint <id> --out-json <path> \
        --robolab_root <checkout> [--task ... --num_envs ... ...]

It launches RoboLab's ``policies/cosmos3/run.py`` for each task, parses the
rollout results, and writes ``{success_rate, num_episodes, metrics}`` to
``--out-json``. The policy server is EXTERNALLY-SERVED — start it first in the
cosmos-framework env (any family member; Edge needs the JSON prompt flag)::

    python -m cosmos_framework.scripts.action_policy_server_robolab \
        --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8000

Result parsing is deliberately tolerant: it prefers a JSON results file when
RoboLab writes one (``--results_glob``), falls back to scanning stdout for a
``success rate``-style summary, and otherwise reports metric-only (no fabricated
success_rate) with the stdout tail attached for debugging. ⚠ Pin the exact
RoboLab output format on the first GPU smoke and tighten this parser.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

_SUCCESS_RE = re.compile(
    r"success[\s_-]*rate[^0-9]*([0-9]*\.?[0-9]+)\s*(%?)", re.IGNORECASE
)
_EPISODES_RE = re.compile(r"([0-9]+)\s*/\s*([0-9]+)\s+episodes", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="RoboLab eval bridge for Cosmos3 policies.")
    # --- CustomEvalRunner contract ---
    ap.add_argument("--checkpoint", required=True,
                    help="Recorded in metrics; the policy SERVER holds the weights.")
    ap.add_argument("--out-json", required=True, dest="out_json")
    # --- RoboLab launch (config passthrough) ---
    ap.add_argument("--robolab_root", required=True,
                    help="Path to the NVlabs/RoboLab checkout (client side).")
    ap.add_argument("--robolab_python", default=sys.executable,
                    help="Interpreter for RoboLab's run.py (its docker/venv).")
    ap.add_argument("--task", default="BananaInBowlTask",
                    help="Comma-separated RoboLab task name(s).")
    ap.add_argument("--num_envs", type=int, default=10)
    ap.add_argument("--headless", default="true")
    ap.add_argument("--results_glob", default="",
                    help="Optional glob (relative to robolab_root) of JSON result "
                         "files run.py writes; preferred over stdout scraping.")
    ap.add_argument("--extra_args", default="",
                    help="Extra whitespace-separated args forwarded to run.py "
                         "(e.g. server host/port flags).")
    return ap


def parse_stdout_metrics(stdout: str) -> dict:
    """Best-effort scrape of a success-rate summary from run.py stdout."""
    payload: dict = {}
    m = _SUCCESS_RE.search(stdout)
    if m:
        rate = float(m.group(1))
        if m.group(2) == "%" or rate > 1.0:
            rate /= 100.0
        payload["success_rate"] = rate
    m = _EPISODES_RE.search(stdout)
    if m:
        payload.setdefault("metrics", {})["successes"] = int(m.group(1))
        payload["num_episodes"] = int(m.group(2))
    return payload


def collect_results_files(root: Path, pattern: str) -> dict:
    """Aggregate ``{success_rate, num_episodes}`` across RoboLab JSON results."""
    successes = episodes = 0
    for path in sorted(glob.glob(str(root / pattern))):
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            episodes += int(data.get("num_episodes", data.get("episodes", 0)))
            successes += int(data.get("successes", data.get("num_successes", 0)))
    if episodes <= 0:
        return {}
    return {
        "success_rate": successes / episodes,
        "num_episodes": episodes,
        "metrics": {"successes": successes},
    }


def run_task(args: argparse.Namespace, task: str) -> tuple[int, str]:
    cmd = [
        args.robolab_python, "policies/cosmos3/run.py",
        "--task", task,
        "--num-envs", str(args.num_envs),
    ]
    if str(args.headless).strip().lower() in ("1", "true", "yes", "on"):
        cmd.append("--headless")
    if args.extra_args:
        cmd += args.extra_args.split()
    print(f"[robolab_eval] launching: {' '.join(cmd)} (cwd={args.robolab_root})", flush=True)
    proc = subprocess.run(
        cmd, cwd=args.robolab_root, capture_output=True, text=True
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode, proc.stdout


def main() -> None:
    args = build_parser().parse_args()
    tasks = [t.strip() for t in str(args.task).split(",") if t.strip()]
    per_task: dict[str, dict] = {}
    stdout_all: list[str] = []
    for task in tasks:
        rc, stdout = run_task(args, task)
        stdout_all.append(stdout)
        if rc != 0:
            raise SystemExit(f"RoboLab run.py failed for task {task!r} (rc={rc})")
        per_task[task] = parse_stdout_metrics(stdout)

    payload: dict = {}
    if args.results_glob:
        payload = collect_results_files(Path(args.robolab_root), args.results_glob)
    if not payload:
        rates = [p["success_rate"] for p in per_task.values() if "success_rate" in p]
        if rates:
            payload["success_rate"] = sum(rates) / len(rates)
            episodes = sum(p.get("num_episodes", 0) for p in per_task.values())
            if episodes:
                payload["num_episodes"] = episodes

    metrics = payload.setdefault("metrics", {})
    metrics["checkpoint"] = args.checkpoint
    metrics["tasks"] = tasks
    metrics["per_task"] = per_task
    if "success_rate" not in payload:
        # Metric-only rather than a fabricated 0.0 — pin the parser on GPU smoke.
        metrics["stdout_tail"] = "\n".join(stdout_all)[-2000:]

    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"[robolab_eval] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
