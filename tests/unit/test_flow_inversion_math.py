"""TDD for the pure flow-inversion math (probe_flow_inversion_groot).

The inverter ``invert_perstep_fp`` is a GR00T-agnostic function of a velocity
callable ``v(x, step_idx)``. We exercise it against SYNTHETIC analytic velocity
fields whose forward map + inverse are known in closed form, so these tests need
no GR00T / checkpoint / GPU — only torch (guarded with importorskip; the local
.venv-ur5e has no torch so the suite also runs on the VM's GR00T venv).

Two field families with a provable answer:

* constant field  ``v(x,i) = c``  => forward map ``x_N = w + c`` (since N*dt=1).
  The fixed point converges in ONE iteration (v is independent of x), so the
  inverse must recover ``w`` to machine precision.

* contractive affine field ``v(x,i) = A x + b`` with ``dt*||A|| < 1``. The
  per-step fixed-point map ``x -> x_{i+1} - dt*(A x + b)`` is then a contraction
  (Lipschitz constant ``dt*||A|| < 1``), so Banach guarantees convergence and
  the residuals must shrink monotonically.

We also assert the derived GR00T bucket schedule {0,250,500,750}.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from probe_flow_inversion_groot import (  # noqa: E402
    bucket_schedule,
    forward_euler_sample,
    invert_perstep_fp,
)

N_STEPS = 4
DT = 1.0 / N_STEPS


def _mse(a, b) -> float:
    return float(((a - b) ** 2).mean())


# --------------------------------------------------------------------------- #
# bucket schedule (the GR00T convention we derived from the source)
# --------------------------------------------------------------------------- #
def test_bucket_schedule_matches_groot_inference_loop():
    assert bucket_schedule(4, 1000) == [0, 250, 500, 750]
    assert bucket_schedule(10, 1000) == [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]


# --------------------------------------------------------------------------- #
# constant field: exact recovery
# --------------------------------------------------------------------------- #
def test_constant_field_inverts_exactly():
    torch.manual_seed(0)
    c = torch.randn(1, 16, 7)
    w = torch.randn(1, 16, 7)

    def v(x, i):
        return c

    x_n = forward_euler_sample(w, v, N_STEPS, DT)
    # forward map of a constant field is exactly w + c
    assert _mse(x_n, w + c) < 1e-12

    w_rec, residuals = invert_perstep_fp(x_n, v, N_STEPS, DT, fp_iters=8)
    assert _mse(w_rec, w) < 1e-10
    # constant field => the seed already solves the step => residual is ~0
    assert max(residuals) < 1e-6


# --------------------------------------------------------------------------- #
# contractive affine field: convergence + recovery
# --------------------------------------------------------------------------- #
def _affine_field(A, b):
    def v(x, i):
        # x: (B, T, D); A: (D, D)
        return x @ A.T + b

    return v


def test_contractive_affine_inverts_and_residuals_shrink():
    torch.manual_seed(1)
    D = 7
    # small-norm A so dt*||A|| << 1 (guaranteed contraction of the FP map)
    A = 0.3 * torch.randn(D, D)
    A = A / (torch.linalg.matrix_norm(A, ord=2) + 1e-9) * 0.5  # spectral norm 0.5
    b = 0.1 * torch.randn(D)
    v = _affine_field(A, b)

    w = torch.randn(3, 16, D)
    x_n = forward_euler_sample(w, v, N_STEPS, DT)
    w_rec, residuals = invert_perstep_fp(x_n, v, N_STEPS, DT, fp_iters=30)

    # round-trip: decoding the recovered noise reproduces the target
    x_rt = forward_euler_sample(w_rec, v, N_STEPS, DT)
    assert _mse(x_rt, x_n) < 1e-8
    # and we recover the actual latent noise
    assert _mse(w_rec, w) < 1e-6
    # every per-step fixed point converged
    assert max(residuals) < 1e-5


def test_fp_residual_decreases_with_more_iters():
    torch.manual_seed(2)
    D = 7
    A = torch.eye(D) * 0.6  # dt*0.6 = 0.15 contraction
    b = torch.zeros(D)
    v = _affine_field(A, b)
    w = torch.randn(1, 16, D)
    x_n = forward_euler_sample(w, v, N_STEPS, DT)

    _, res_few = invert_perstep_fp(x_n, v, N_STEPS, DT, fp_iters=2)
    _, res_many = invert_perstep_fp(x_n, v, N_STEPS, DT, fp_iters=20)
    # more fixed-point iterations => smaller final residual at every step
    assert max(res_many) <= max(res_few)
    assert max(res_many) < 1e-6


def test_step_dependent_field_still_inverts():
    # velocity that depends on the step index (like GR00T's t-bucket conditioning)
    torch.manual_seed(3)
    D = 7
    scales = [0.2, 0.4, 0.1, 0.3]

    def v(x, i):
        return -scales[i] * x  # contraction, i-dependent

    w = torch.randn(2, 16, D)
    x_n = forward_euler_sample(w, v, N_STEPS, DT)
    w_rec, residuals = invert_perstep_fp(x_n, v, N_STEPS, DT, fp_iters=30)
    x_rt = forward_euler_sample(w_rec, v, N_STEPS, DT)
    assert _mse(x_rt, x_n) < 1e-8
    assert _mse(w_rec, w) < 1e-6
