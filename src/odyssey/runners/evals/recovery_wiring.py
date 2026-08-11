"""Shared recovery wiring for the LIBERO eval recipes (GR00T + π0.5).

The two recipes are deliberate siblings; this module keeps their recovery
surface identical instead of letting two copies drift: one argparse block
(:func:`add_recovery_args`), one assembly path from parsed args to a
:class:`~odyssey.runners.agents.recovery.RecoveryPolicy`
(:func:`make_recovery`), and the small env/obs helpers the loop hooks need.

Everything imports lazily and the module itself is stdlib-only at import time
(the recipes import it inside ``build_parser``/``run_eval``, preserving their
bare-stdlib import contract). Strict-mypy clean — the recipes are the exempt
layer, this glue is not.

Default-off by construction: with none of the flags set,
:func:`recovery_enabled` is ``False``, :func:`make_recovery` returns ``None``
and every loop hook is skipped — the recipes behave byte-identically to the
pre-recovery code.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_SPECIALIST_MODEL = "nvidia/Cosmos3-Nano"
_sim_state_warned = False


def add_recovery_args(
    ap: argparse.ArgumentParser, *, bool_type: Callable[[str], bool]
) -> None:
    """Add the recovery flag block (shared verbatim by both recipes).

    ``bool_type`` is the recipe's ``--flag <value>``-style boolean parser (the
    runner forwards config keys as ``--key value``, so ``store_true`` would
    choke on the trailing value).
    """
    g = ap.add_argument_group("recovery", "closed-loop stuck detection + rollback")
    g.add_argument("--recovery", type=bool_type, default=False,
                   help="enable interventions (flush / rollback) on stuck triggers.")
    g.add_argument("--shadow_mode", type=bool_type, default=False,
                   help="detect + log would-be interventions, never intervene.")
    g.add_argument("--recovery_mode", choices=["command", "teleport"], default="command",
                   help="rollback style: drive the arm back (command) or restore "
                        "sim state (teleport; degrades to an event if unsupported).")
    g.add_argument("--max_recoveries", type=int, default=3,
                   help="intervention budget per episode; past it, triggers only log.")
    g.add_argument("--recovery_steps", type=int, default=24,
                   help="max command-mode drive steps back to the snapshot.")
    g.add_argument("--recovery_settle_steps", type=int, default=5,
                   help="steps the monitors stay muted after an intervention.")
    g.add_argument("--stuck_eps_m", type=float, default=0.005,
                   help="tier-1: EE displacement below this over the window = stuck.")
    g.add_argument("--stuck_window_steps", type=int, default=12,
                   help="tier-1 kinematic window length in steps.")
    g.add_argument("--stuck_min_commanded", type=float, default=0.05,
                   help="tier-1: minimum commanded translation over the window.")
    g.add_argument("--continuity_eps", type=float, default=0.0,
                   help="tier-2 inter-chunk discrepancy threshold (0 = disabled).")
    g.add_argument("--poll_every_chunks", type=int, default=2,
                   help="specialist poll throttle (every Nth chunk boundary).")
    g.add_argument("--specialist_base_url", default="",
                   help="OpenAI-compatible endpoint serving the stuck judge "
                        "(empty = no specialist tier).")
    g.add_argument("--specialist_model", default="",
                   help=f"served judge model id (default {_DEFAULT_SPECIALIST_MODEL}).")
    g.add_argument("--specialist_max_tokens", type=int, default=256,
                   help="reasoner CoT needs headroom before its YES/NO token.")
    g.add_argument("--specialist_timeout_s", type=float, default=30.0)
    g.add_argument("--specialist_text_modality", type=bool_type, default=True,
                   help="send {'modalities': ['text']} (vLLM-Omni image-gen footgun).")
    g.add_argument("--specialist_api_key_env", default="",
                   help="env var holding the judge endpoint's bearer token, if any.")
    g.add_argument("--log_actions", type=bool_type, default=False,
                   help="log per-step (frame, action) npz — the LVM training corpus.")
    g.add_argument("--recovery_dir", default="",
                   help="where recovery_events.jsonl + rollout npz land "
                        "(the runner resolves this like --video_dir).")


def recovery_enabled(args: argparse.Namespace) -> bool:
    return bool(args.recovery or args.shadow_mode)


def make_recovery(args: argparse.Namespace) -> Any | None:
    """Assemble the RecoveryPolicy from parsed args; ``None`` when disabled."""
    if not recovery_enabled(args):
        return None
    from odyssey.runners.agents.recovery import (
        ChunkLedger,
        RecoveryController,
        RecoveryPolicy,
        StuckMonitor,
    )

    specialist = None
    if args.specialist_base_url:
        from odyssey.runners.agents.openai_judge import OpenAICompatCompletionJudge
        from odyssey.runners.agents.specialist_gate import (
            STUCK_PROMPT_TEMPLATE,
            SpecialistGate,
        )

        judge = OpenAICompatCompletionJudge(
            base_url=args.specialist_base_url,
            model=args.specialist_model or _DEFAULT_SPECIALIST_MODEL,
            prompt_template=STUCK_PROMPT_TEMPLATE,
            max_tokens=args.specialist_max_tokens,
            timeout_seconds=args.specialist_timeout_s,
            api_key_env=args.specialist_api_key_env or None,
            extra_body=(
                {"modalities": ["text"]} if args.specialist_text_modality else None
            ),
        )
        specialist = SpecialistGate(
            detector=judge, poll_every_chunks=args.poll_every_chunks
        )

    return RecoveryPolicy(
        ledger=ChunkLedger(),
        monitor=StuckMonitor(
            window_steps=args.stuck_window_steps,
            eps_m=args.stuck_eps_m,
            min_commanded=args.stuck_min_commanded,
            continuity_eps=args.continuity_eps,
        ),
        controller=RecoveryController(max_steps=args.recovery_steps),
        specialist=specialist,
        mode=args.recovery_mode,
        shadow=args.shadow_mode,
        max_recoveries=args.max_recoveries,
        settle_steps=args.recovery_settle_steps,
    )


def make_rollout_log(args: argparse.Namespace) -> Any | None:
    """The (frame, action) npz corpus logger; ``None`` unless flagged + dir set."""
    if not (args.log_actions and args.recovery_dir):
        if args.log_actions:
            logger.warning("log_actions set but no recovery_dir; skipping npz corpus")
        return None
    from odyssey.runners.agents.recovery import RolloutLog

    return RolloutLog(Path(args.recovery_dir) / "rollouts")


def maybe_sim_state(env: Any) -> Any | None:
    """Best-effort MuJoCo sim-state snapshot for teleport mode.

    ``get_sim_state`` is not pinned across LIBERO versions, hence the hasattr
    chain with a one-time warning; ``None`` degrades teleport rollbacks to a
    ``teleport_unavailable`` event (command mode is unaffected).
    """
    global _sim_state_warned
    getter = getattr(env, "get_sim_state", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            logger.debug("get_sim_state() failed", exc_info=True)
    sim = getattr(env, "sim", None)
    if sim is not None:
        try:
            return sim.get_state().flatten()
        except Exception:
            logger.debug("sim.get_state() failed", exc_info=True)
    if not _sim_state_warned:
        logger.warning(
            "teleport recovery: env exposes no sim state; rollbacks will degrade"
        )
        _sim_state_warned = True
    return None


def proprio(obs: Any) -> tuple[list[float], list[float], list[float]]:
    """(eef_pos, eef_quat_xyzw, gripper_qpos) as plain float lists."""

    def _vec(key: str, n: int) -> list[float]:
        raw = obs[key]
        flat = raw.reshape(-1) if hasattr(raw, "reshape") else raw
        return [float(v) for v in list(flat)[:n]]

    return (
        _vec("robot0_eef_pos", 3),
        _vec("robot0_eef_quat", 4),
        _vec("robot0_gripper_qpos", 2),
    )


def finalize(policy: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Write events (when a dir is set), return the metrics block, close."""
    metrics: dict[str, Any] = dict(policy.metrics())
    if args.recovery_dir:
        policy.write_events(Path(args.recovery_dir) / "recovery_events.jsonl")
    policy.close()
    return metrics
