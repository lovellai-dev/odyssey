"""Tests for LiberoRunner's recovery plumbing (``evals/libero.py``).

The runner resolves ``--recovery_dir`` exactly like ``--video_dir`` (the
recipe writes in place, nothing is copied) and registers the written
artifacts on the result summary after ``summarize``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from odyssey.runners.evals.libero import (
    _attach_recovery_artifacts,
    _resolve_recovery_dir,
    build_gr00t_libero_argv,
    build_pi05_libero_argv,
)
from odyssey.spec import EvaluationTask, EvaluationType


def _eval_task(**overrides: Any) -> EvaluationTask:
    fields: dict[str, Any] = {
        "name": "recovery-eval",
        "evaluation_type": EvaluationType.LIBERO,
        "benchmark_name": "libero_object",
        "num_episodes": 2,
    }
    fields.update(overrides)
    return EvaluationTask(**fields)


# ---------------------------------------------------------------------------
# _resolve_recovery_dir — the --video_dir pattern.
# ---------------------------------------------------------------------------


def test_resolve_recovery_dir_off_by_default() -> None:
    assert _resolve_recovery_dir({}, Path("/out")) is None
    assert _resolve_recovery_dir({"recovery": False}, Path("/out")) is None


def test_resolve_recovery_dir_for_each_recovery_flag() -> None:
    for key in ("recovery", "shadow_mode", "log_actions"):
        resolved = _resolve_recovery_dir({key: True}, Path("/out"))
        assert resolved == Path("/out/recovery"), key
        # Forwarded-string truthiness too (YAML values may arrive as strings).
        assert _resolve_recovery_dir({key: "true"}, Path("/out")) == Path("/out/recovery")


def test_resolve_recovery_dir_requires_output_dir() -> None:
    assert _resolve_recovery_dir({"recovery": True}, None) is None


def test_resolve_recovery_dir_explicit_config_wins() -> None:
    cfg = {"recovery": True, "recovery_dir": "/elsewhere/rec"}
    assert _resolve_recovery_dir(cfg, Path("/out")) == Path("/elsewhere/rec")
    # Explicit dir works even without any recovery flag (standalone corpus runs).
    assert _resolve_recovery_dir({"recovery_dir": "/e"}, None) == Path("/e")


# ---------------------------------------------------------------------------
# argv builders — --recovery_dir appended, raw recovery_dir key not forwarded.
# ---------------------------------------------------------------------------


def test_builders_append_recovery_dir() -> None:
    task = _eval_task(config={"pilot": "gr00t", "recovery": True})
    for build in (build_gr00t_libero_argv, build_pi05_libero_argv):
        argv = build(
            spec=task, checkpoint=Path("/ckpt"), video_dir=None,
            recovery_dir=Path("/out/recovery"),
        )
        assert argv[argv.index("--recovery_dir") + 1] == "/out/recovery"
        assert argv[argv.index("--recovery") + 1] == "True"  # flag still forwarded


def test_builders_omit_recovery_dir_when_none() -> None:
    task = _eval_task(config={"pilot": "pi05"})
    for build in (build_gr00t_libero_argv, build_pi05_libero_argv):
        argv = build(spec=task, checkpoint=Path("/ckpt"), video_dir=None)
        assert "--recovery_dir" not in argv


def test_builders_drop_raw_recovery_dir_config_key() -> None:
    # The runner consumes recovery_dir itself; the resolved value is what the
    # recipe sees — never the raw config string a second time.
    task = _eval_task(config={"recovery_dir": "/from/config"})
    argv = build_gr00t_libero_argv(
        spec=task, checkpoint=Path("/ckpt"), video_dir=None,
        recovery_dir=Path("/from/config"),
    )
    assert argv.count("--recovery_dir") == 1


# ---------------------------------------------------------------------------
# _attach_recovery_artifacts.
# ---------------------------------------------------------------------------


def test_attach_recovery_artifacts_registers_events_and_rollouts(tmp_path: Path) -> None:
    (tmp_path / "recovery_events.jsonl").write_text('{"kind": "flush"}\n')
    (tmp_path / "rollouts").mkdir()
    (tmp_path / "rollouts" / "episode_01_FAIL.npz").write_bytes(b"npz")

    summary: dict[str, Any] = {"success_rate": 0.0}
    _attach_recovery_artifacts(summary, tmp_path)

    assert summary["artifacts"]["recovery_events"].endswith("recovery_events.jsonl")
    assert len(summary["artifacts"]["rollout_logs"]) == 1


def test_attach_recovery_artifacts_noops_when_nothing_written(tmp_path: Path) -> None:
    summary: dict[str, Any] = {"success_rate": 0.0}
    _attach_recovery_artifacts(summary, tmp_path / "never-created")
    assert "artifacts" not in summary
