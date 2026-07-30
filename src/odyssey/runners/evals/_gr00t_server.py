"""Shared GR00T policy-server lifecycle for eval recipes (Isaac Lab + LIBERO).

Boot the upstream ``gr00t.eval.run_gr00t_server`` on a checkpoint, wait until it
accepts TCP connections, yield, then tear it down; plus a thin ``PolicyClient``
connector. Extracted verbatim from ``gr00t_isaac_eval.py`` so the LIBERO recipe
reuses the exact same, already-hardened lifecycle (deadlock-safe stdout redirect
+ ``PR_SET_PDEATHSIG`` reap-on-crash) rather than a second copy that drifts.

Stdlib-only at import time (``subprocess`` / ``socket`` / ``ctypes`` are imported
lazily inside the functions); the heavy ``gr00t`` import is deferred into
``connect_policy_client``. So this module stays unit-testable on a CPU box
without gr00t / torch installed.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
from collections.abc import Iterator
from typing import Any

log = logging.getLogger("gr00t_server")

# The upstream GR00T policy-server entry point (closed-loop auto-serve).
_SERVER_ENTRY = "gr00t.eval.run_gr00t_server"


def _default_server_python() -> str:
    """Interpreter for the out-of-process GR00T server when none is configured.

    GR00T's model stack lives in a SEPARATE venv (its torch/CUDA ABI clashes with the
    eval env), so the server can't run under this process's python. Prefer the
    ``setup.sh --pilot gr00t`` venv — ``$GR00T_VENV_PYTHON``, then
    ``$ISAAC_GR00T_DIR/.venv/bin/python`` (default ``~/Isaac-GR00T/.venv/bin/python``).
    Falls back to this process's python only if neither exists (gr00t must then be
    importable here — usually it isn't, and the server will fail with a clear error).
    """
    cand = os.environ.get("GR00T_VENV_PYTHON")
    if cand and os.path.exists(cand):
        return cand
    gdir = os.environ.get("ISAAC_GR00T_DIR", os.path.expanduser("~/Isaac-GR00T"))
    cand = os.path.join(gdir, ".venv", "bin", "python")
    return cand if os.path.exists(cand) else sys.executable


def build_server_command(
    *,
    checkpoint: str,
    embodiment_tag: str,
    port: int,
    server_python: str | None = None,
    device: str = "cuda:0",
    modality_config_path: str | None = None,
    denoising_steps: int = 0,
    sim_policy_wrapper: bool = False,
) -> list[str]:
    """argv to serve a (finetuned) GR00T checkpoint as a policy server.

    Points ``--model-path`` at the checkpoint to serve.

    ``sim_policy_wrapper`` toggles ``--use-sim-policy-wrapper``. The Isaac/DROID
    recipe leaves it OFF (default): it builds raw NESTED obs (``build_gr00t_obs``),
    which the un-wrapped server expects. The LIBERO recipe turns it ON — the
    GR00T-N1.7-LIBERO checkpoint is served through the sim policy wrapper (matching
    NVIDIA's own ``gr00t/eval/sim/LIBERO`` eval), fed the flat ``video.*``/``state.*``
    obs from ``build_gr00t_libero_obs``.
    """
    py = server_python or _default_server_python()
    argv = [
        py, "-m", _SERVER_ENTRY,
        "--model-path", str(checkpoint),
        "--embodiment-tag", str(embodiment_tag),
        "--port", str(int(port)),
        "--device", str(device),
    ]
    if sim_policy_wrapper:
        argv += ["--use-sim-policy-wrapper"]
    if modality_config_path:
        argv += ["--modality-config-path", str(modality_config_path)]
    if denoising_steps and int(denoising_steps) > 0:
        argv += ["--denoising-steps", str(int(denoising_steps))]
    return argv


def wait_for_server(host: str, port: int, timeout_s: int, proc: Any = None) -> bool:
    """Block until ``(host, port)`` accepts a TCP connection (server ready).

    Returns ``True`` once reachable, ``False`` on timeout. Fails fast if the
    server ``proc`` exits during startup (so we surface its error instead of
    waiting out the whole timeout).
    """
    import socket
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False  # server died during startup
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return True
        except OSError:
            time.sleep(2.0)
    return False


@contextlib.contextmanager
def serve_checkpoint(
    *,
    checkpoint: str,
    embodiment_tag: str,
    host: str = "127.0.0.1",
    port: int = 5555,
    server_python: str | None = None,
    device: str = "cuda:0",
    modality_config_path: str | None = None,
    denoising_steps: int = 0,
    ready_timeout: int = 900,
    sim_policy_wrapper: bool = False,
) -> Iterator[None]:
    """Start a GR00T policy server on ``checkpoint``, wait until ready, yield,
    then tear it down. The body runs the eval against ``host:port``.
    """
    import ctypes
    import signal
    import subprocess
    import tempfile

    argv = build_server_command(
        checkpoint=checkpoint,
        embodiment_tag=embodiment_tag,
        port=port,
        server_python=server_python or None,
        device=device,
        modality_config_path=modality_config_path or None,
        denoising_steps=denoising_steps,
        sim_policy_wrapper=sim_policy_wrapper,
    )
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")       # gated Cosmos backbone -> offline cache
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    log.info("auto-serve: launching GR00T server: %s", " ".join(argv))

    # DEADLOCK FIX (carried from the Isaac recipe): the server MUST NOT inherit
    # this recipe's stdout/stderr — those are the pipe the Odyssey runner reads.
    # The recipe may hard-exit (sim close / SystemExit), so the teardown `finally`
    # below might not run; an orphan holding the pipe's write end would leave the
    # runner's read blocked with no EOF until a timeout. (1) redirect the server's
    # output to a log file; (2) PR_SET_PDEATHSIG so the kernel SIGKILLs the server
    # if we hard-exit (no GPU leak).
    server_log_path = os.path.join(tempfile.gettempdir(), f"gr00t_server_{port}.log")
    server_log = open(server_log_path, "wb")  # noqa: SIM115  handle outlives this scope (Popen + finally)

    def _die_with_parent() -> None:  # Linux: reap this server when the recipe process dies
        with contextlib.suppress(Exception):
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0)

    proc = subprocess.Popen(
        argv, env=env,
        stdout=server_log, stderr=subprocess.STDOUT,
        preexec_fn=_die_with_parent,
    )
    log.info("auto-serve: GR00T server pid=%s (log: %s)", proc.pid, server_log_path)
    try:
        if not wait_for_server(host, port, ready_timeout, proc):
            raise RuntimeError(
                f"GR00T server failed to become ready on {host}:{port} "
                f"within {ready_timeout}s (server rc={proc.poll()}; see {server_log_path})"
            )
        log.info("auto-serve: GR00T server ready on %s:%d", host, port)
        yield
    finally:
        log.info("auto-serve: tearing down GR00T server (pid=%s)", proc.pid)
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
        with contextlib.suppress(Exception):
            server_log.close()


def connect_policy_client(
    *, host: str, port: int, timeout_ms: int, isaac_gr00t_dir: str | None = None
) -> Any:
    """Return a connected GR00T ``PolicyClient`` (light deps: msgpack/pyzmq).

    Adds the Isaac-GR00T checkout to ``sys.path`` (``$ISAAC_GR00T_DIR``, default
    ``~/Isaac-GR00T``) before importing, mirroring the Isaac recipe. Import is
    deferred here so the module stays importable without gr00t.
    """
    g = isaac_gr00t_dir or os.environ.get(
        "ISAAC_GR00T_DIR", os.path.expanduser("~/Isaac-GR00T")
    )
    if g not in sys.path:
        sys.path.insert(0, g)
    from gr00t.policy.server_client import PolicyClient

    return PolicyClient(host=host, port=port, timeout_ms=timeout_ms)
