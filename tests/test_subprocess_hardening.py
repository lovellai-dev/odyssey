"""Tests for F17/F18/F61 — further training/eval subprocess hardening.

F17 reap-on-exit: kill the child's group on *any* exit from
     run_training_subprocess (cancelled task / unexpected error), not just the
     deadline path — PDEATHSIG only covers a whole-runner crash.
F18 PYTHONPATH isolation: don't leak the orchestrator's PYTHONPATH into the
     child, where it can shadow the training env's own torch / prismatic.
F61 resource classification: a non-zero exit whose output shows CUDA OOM or a
     full disk raises SubprocessResourceError with a clear reason, not "rc=1".
"""
import asyncio
import os

from odyssey.runners.subprocess import (
    SubprocessResourceError,
    TrainingProcessSpec,
    _build_child_env,
    _classify_failure,
    _reap_if_alive,
)


async def _spawn(*argv):
    return await asyncio.create_subprocess_exec(
        *argv,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


# ---- F17 ----------------------------------------------------------------

async def test_reap_if_alive_kills_running_child():
    proc = await _spawn("sleep", "30")
    await _reap_if_alive(proc, grace=0.5)
    assert proc.returncode is not None  # reaped, not orphaned


async def test_reap_if_alive_noop_when_already_exited():
    proc = await _spawn("true")
    await proc.wait()
    await _reap_if_alive(proc, grace=0.5)  # no-op, no error
    assert proc.returncode == 0


# ---- F18 ----------------------------------------------------------------

def test_build_child_env_strips_leaked_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/opt/ros/jazzy:/home/x/odyssey/src")
    env = _build_child_env(TrainingProcessSpec(entry_module="x"))
    assert "PYTHONPATH" not in env


def test_build_child_env_keeps_explicit_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/opt/ros/jazzy")
    spec = TrainingProcessSpec(entry_module="x", env={"PYTHONPATH": "/wanted"})
    assert _build_child_env(spec)["PYTHONPATH"] == "/wanted"


def test_build_child_env_inherits_other_vars(monkeypatch):
    monkeypatch.setenv("SOME_VAR", "keepme")
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert _build_child_env(TrainingProcessSpec(entry_module="x"))["SOME_VAR"] == "keepme"


# ---- F61 ----------------------------------------------------------------

def test_classify_failure_detects_cuda_oom():
    reason = _classify_failure(
        ["epoch 3 step 40",
         "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"]
    )
    assert reason is not None and "OOM" in reason


def test_classify_failure_detects_disk_full():
    reason = _classify_failure(
        ["saving checkpoint", "OSError: [Errno 28] No space left on device"]
    )
    assert reason is not None and "disk" in reason.lower()


def test_classify_failure_none_for_benign_output():
    assert _classify_failure(["step 100 loss 0.5", "training complete"]) is None


def test_resource_error_is_runtime_error():
    # Engine error handling catches RuntimeError; the classified error must fit.
    assert issubclass(SubprocessResourceError, RuntimeError)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
