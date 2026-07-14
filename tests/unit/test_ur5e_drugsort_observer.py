"""Tests for the grasp-target Observer (odyssey ur5e-drugsort perception stack).

The Observer maps the two deploy cameras to the 3D grasp target that fixes the
"closes ~7 cm short" descent failure. These tests pin the bits that make it
usable WITHOUT downloading a DINO backbone or touching a GPU:

  * the pure metric / coordinate helpers (cm error, world->base) are correct;
  * construction is lazy (no backbone/head built until ``load``);
  * with an **injected** tiny backbone the full frames->xyz wiring runs on CPU,
    the head is the only trainable part, and save/load round-trips.

Heavy deps (torch / transformers) are deferred into ``load``/``forward``; the
torch-dependent tests ``importorskip`` torch so the pure-helper tests still run
under a bare interpreter.
"""

from __future__ import annotations

import numpy as np
import pytest

from odyssey.embodiments.ur5e_drugsort import observer as O

# --- pure helpers -----------------------------------------------------------

def test_keypoint_error_cm_known_values() -> None:
    pred = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]])
    gt = np.array([[0.0, 0.0, 0.03], [0.10, 0.0, 0.0]])
    err = O.keypoint_error_cm(pred, gt)
    assert err.shape == (2,)
    assert err[0] == pytest.approx(3.0)  # 0.03 m -> 3 cm
    assert err[1] == pytest.approx(0.0)


def test_keypoint_error_cm_accepts_single_vector() -> None:
    err = O.keypoint_error_cm([0.0, 0.04, 0.0], [0.0, 0.0, 0.0])
    assert err.shape == (1,)
    assert err[0] == pytest.approx(4.0)


def test_world_to_base_identity_is_translation() -> None:
    p = O.world_to_base([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], np.eye(3))
    assert np.allclose(p, [0.5, 1.5, 2.5])


def test_world_to_base_applies_inverse_rotation() -> None:
    # base rotated +90deg about z (columns = base axes in world)
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    p = O.world_to_base([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], rz)
    assert np.allclose(p, [0.0, -1.0, 0.0], atol=1e-9)


def test_constants_are_sane() -> None:
    assert O.TARGET_DIM == 3
    assert O.DEFAULT_BACKBONES[0].startswith("facebook/dinov3")
    assert "dinov2" in O.DEFAULT_BACKBONES[1]


# --- laziness ---------------------------------------------------------------

def test_construction_is_lazy() -> None:
    obs = O.GraspTargetObserver()
    assert obs._head is None
    assert obs.backbone_id is None  # backbone not resolved until load()


# --- wiring with an injected backbone (no DINO download) --------------------

def _tiny_backbone(feat_dim: int):  # type: ignore[no-untyped-def]
    from torch import nn

    lin = nn.Linear(3 * 16 * 16, feat_dim)

    def call(view_t):  # type: ignore[no-untyped-def]
        import torch.nn.functional as fn

        x = fn.interpolate(view_t, size=(16, 16), mode="bilinear", align_corners=False)
        return lin(x.reshape(x.shape[0], -1))

    return call, lin


def test_forward_and_predict_with_injected_backbone() -> None:
    torch = pytest.importorskip("torch")
    call, _ = _tiny_backbone(8)
    obs = O.GraspTargetObserver(backbone=call, feat_dim=8, num_views=2, img_size=32)
    obs.load()
    ext = np.random.default_rng(0).integers(0, 256, (4, 40, 40, 3), dtype=np.uint8)
    wrist = np.random.default_rng(1).integers(0, 256, (4, 40, 40, 3), dtype=np.uint8)
    out = obs.forward([ext, wrist])
    assert tuple(out.shape) == (4, 3)
    # deploy path: single pair of frames -> (3,)
    xyz = obs.predict(ext[0], wrist[0])
    assert xyz.shape == (3,)
    assert np.isfinite(xyz).all()
    # only the head is trainable (backbone injected/frozen)
    n_head = sum(p.numel() for p in obs.head_parameters())
    assert n_head > 0
    assert isinstance(out, torch.Tensor)


def test_head_can_overfit_a_constant_target() -> None:
    torch = pytest.importorskip("torch")
    call, _ = _tiny_backbone(8)
    obs = O.GraspTargetObserver(backbone=call, feat_dim=8, num_views=1, img_size=32)
    obs.load()
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (8, 32, 32, 3), dtype=np.uint8)
    target = torch.tensor([[0.4, 0.1, 0.3]] * 8, dtype=torch.float32)
    opt = torch.optim.Adam(obs.head_parameters(), lr=1e-2)
    for _ in range(150):
        opt.zero_grad()
        loss = ((obs.forward([frames]) - target) ** 2).mean()
        loss.backward()
        opt.step()
    assert loss.item() < 1e-3  # head learns to hit the constant target


def test_save_and_load_head_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    call, _ = _tiny_backbone(8)
    obs = O.GraspTargetObserver(backbone=call, feat_dim=8, num_views=2, img_size=32)
    obs.load()
    ext = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    wrist = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    before = obs.predict(ext[0], wrist[0])
    path = str(tmp_path / "head.pt")
    obs.save(path)
    obs2 = O.GraspTargetObserver(backbone=call, feat_dim=8, num_views=2, img_size=32)
    obs2.load_head(path)
    after = obs2.predict(ext[0], wrist[0])
    assert np.allclose(before, after, atol=1e-5)
