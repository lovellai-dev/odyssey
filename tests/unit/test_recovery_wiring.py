"""Tests for the shared recipe recovery glue (``evals/recovery_wiring.py``).

The end-to-end loop behavior is pinned through the recipes' own tests
(``test_gr00t_libero_eval.py`` / ``test_pi05_libero_eval.py``); these cover
the assembly surface directly: enablement, policy construction (with and
without a specialist endpoint), the proprio/state helpers.
"""

from __future__ import annotations

import argparse
from typing import Any

from odyssey.runners.agents.recovery import RecoveryPolicy
from odyssey.runners.evals import recovery_wiring as rw


def _args(**overrides: Any) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    rw.add_recovery_args(
        ap, bool_type=lambda v: str(v).strip().lower() in ("1", "true", "yes", "on")
    )
    ns = ap.parse_args([])
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def test_recovery_disabled_by_default_and_make_returns_none() -> None:
    args = _args()
    assert rw.recovery_enabled(args) is False
    assert rw.make_recovery(args) is None
    assert rw.make_rollout_log(args) is None


def test_make_recovery_without_specialist_builds_policy() -> None:
    policy = rw.make_recovery(_args(recovery=True))
    assert isinstance(policy, RecoveryPolicy)
    assert not policy.recovering
    policy.close()  # no specialist: close is a no-op, must not raise


def test_make_recovery_shadow_only_counts_as_enabled() -> None:
    policy = rw.make_recovery(_args(shadow_mode=True))
    assert isinstance(policy, RecoveryPolicy)
    policy.close()


def test_make_recovery_with_specialist_wires_gate_and_judge() -> None:
    policy = rw.make_recovery(
        _args(recovery=True, specialist_base_url="http://judge:8002/v1")
    )
    assert isinstance(policy, RecoveryPolicy)
    # The gate exists and its lifecycle flows through the policy.
    assert policy._specialist is not None
    policy.close()


def test_proprio_extracts_plain_float_lists() -> None:
    import numpy as np

    obs = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }
    pos, quat, grip = rw.proprio(obs)
    assert pos == [0.1, 0.2, 0.3]
    assert quat == [0.0, 0.0, 0.0, 1.0]
    assert grip == [0.04, -0.04]
    assert all(isinstance(v, float) for v in pos + quat + grip)


def test_maybe_sim_state_prefers_get_sim_state_then_degrades() -> None:
    class _WithGetter:
        def get_sim_state(self) -> str:
            return "the-state"

    class _WithSim:
        class sim:
            @staticmethod
            def get_state() -> Any:
                class _S:
                    @staticmethod
                    def flatten() -> str:
                        return "flat-state"

                return _S()

    class _Bare:
        pass

    assert rw.maybe_sim_state(_WithGetter()) == "the-state"
    assert rw.maybe_sim_state(_WithSim()) == "flat-state"
    assert rw.maybe_sim_state(_Bare()) is None  # one-time warning path
