"""Tests for the distilled success / phase classifier (ur5e-drugsort).

The classifier distils the free sim-GT phase signal into a fast CNN that gives
the RL loop its terminal reward (``seated`` == success) and a deploy-time
success signal. These tests pin the parts that make it correct + portable
WITHOUT torch:

  * the GT label mapping (``label_from_state``) hits every phase boundary;
  * the Cosmos-Reason text parser (``parse_phase_from_text``) resolves phase
    words most-advanced-first and returns ``None`` when absent;
  * the ``CosmosReasonTeacher`` adapter uses an injected sidecar (no 2B VLM) and
    never raises;

and — with ``importorskip('torch')`` — the CNN wiring: single- and multi-view
forward/predict shapes, that ``predict`` is eval-mode safe at batch size 1
(BatchNorm), and weight save/load round-trips.
"""

from __future__ import annotations

import numpy as np
import pytest

from odyssey.embodiments.ur5e_drugsort import success_classifier as S

# --- GT label mapping -------------------------------------------------------

def _label(**kw) -> int:  # type: ignore[no-untyped-def]
    base = dict(grip_closure=0.0, vial_to_pinch_m=0.30, lift_height_m=0.0,
                place_dist_xy_m=0.30, vial_z_m=0.22, phase_name="approach")
    base.update(kw)
    return S.label_from_state(**base)  # type: ignore[arg-type]


def test_label_reaching_when_gripper_open() -> None:
    assert _label(grip_closure=0.0) == 0


def test_label_grasped_when_closed_on_vial_not_lifted() -> None:
    assert _label(grip_closure=0.9, vial_to_pinch_m=0.02, lift_height_m=0.0,
                  phase_name="close") == 1


def test_label_lifted_when_closed_and_raised() -> None:
    assert _label(grip_closure=0.9, vial_to_pinch_m=0.02, lift_height_m=0.10,
                  phase_name="lift") == 2


def test_label_seated_takes_priority() -> None:
    # in the pocket, at nest height, after release -> success regardless of grip
    assert _label(grip_closure=0.0, place_dist_xy_m=0.01, vial_z_m=0.25,
                  phase_name="retract") == 3


def test_seated_requires_release_phase() -> None:
    # geometrically in the pocket but still mid-transport phase -> not seated
    assert _label(grip_closure=0.9, place_dist_xy_m=0.01, vial_z_m=0.25,
                  lift_height_m=0.10, phase_name="lower") == 2


def test_is_success_only_for_seated() -> None:
    assert S.is_success(3) is True
    assert all(not S.is_success(c) for c in (0, 1, 2))


# --- Cosmos-Reason text parsing --------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("the arm is reaching toward the vial", 0),
    ("the gripper has grasped the vial", 1),
    ("the vial is lifted off the tray", 2),
    ("the vial is now seated in the rack", 3),
    ("placed it in the pocket", 3),
])
def test_parse_phase_from_text(text: str, expected: int) -> None:
    assert S.parse_phase_from_text(text) == expected


def test_parse_phase_prefers_more_advanced_phase() -> None:
    # mentions both grasped and lifted -> the further-along phase wins
    assert S.parse_phase_from_text("it grasped then lifted the vial") == 2


def test_parse_phase_returns_none_without_phase_word() -> None:
    assert S.parse_phase_from_text("the scene is well lit") is None
    assert S.parse_phase_from_text("") is None


# --- Cosmos-Reason teacher adapter (injected sidecar) -----------------------

class _FakeSidecar:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def reason(self, frame, instruction):  # type: ignore[no-untyped-def]
        self.calls = self.calls + 1
        return self._text


class _BrokenSidecar:
    def reason(self, frame, instruction):  # type: ignore[no-untyped-def]
        raise RuntimeError("VLM unavailable")


def test_teacher_labels_via_injected_sidecar() -> None:
    t = S.CosmosReasonTeacher(sidecar=_FakeSidecar("the vial is lifted"))
    assert t.label(np.zeros((4, 4, 3), dtype=np.uint8)) == 2


def test_teacher_never_raises_on_broken_sidecar() -> None:
    t = S.CosmosReasonTeacher(sidecar=_BrokenSidecar())
    assert t.label(np.zeros((4, 4, 3), dtype=np.uint8)) is None


def test_teacher_returns_none_on_unparseable_reply() -> None:
    t = S.CosmosReasonTeacher(sidecar=_FakeSidecar("hmm, hard to tell"))
    assert t.label(np.zeros((4, 4, 3), dtype=np.uint8)) is None


# --- laziness ---------------------------------------------------------------

def test_construction_is_lazy() -> None:
    clf = S.SuccessClassifier()
    assert clf._model is None
    assert clf.in_ch == 3  # num_views defaults to 1


def test_multiview_sets_input_channels() -> None:
    clf = S.SuccessClassifier(num_views=2)
    assert clf.in_ch == 6


# --- CNN wiring (real torch) ------------------------------------------------

def test_single_view_forward_and_predict() -> None:
    pytest.importorskip("torch")
    clf = S.SuccessClassifier(num_views=1, img_size=32)
    frames = np.random.default_rng(0).integers(0, 256, (5, 48, 48, 3), dtype=np.uint8)
    logits = clf.forward(frames)
    assert tuple(logits.shape) == (5, S.NUM_CLASSES)
    label, probs = clf.predict(frames[0])
    assert 0 <= label < S.NUM_CLASSES
    assert probs.shape == (S.NUM_CLASSES,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_multiview_forward_and_predict_success() -> None:
    pytest.importorskip("torch")
    clf = S.SuccessClassifier(num_views=2, img_size=32)
    ext = np.zeros((3, 40, 40, 3), dtype=np.uint8)
    wrist = np.zeros((3, 40, 40, 3), dtype=np.uint8)
    logits = clf.forward([ext, wrist])
    assert tuple(logits.shape) == (3, S.NUM_CLASSES)
    assert isinstance(clf.predict_success([ext[0], wrist[0]]), bool)


def test_predict_is_batchnorm_safe_at_batch_one() -> None:
    # a single frame must give a confident, finite distribution in eval mode
    pytest.importorskip("torch")
    clf = S.SuccessClassifier(num_views=1, img_size=32)
    frame = np.random.default_rng(3).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    _, probs = clf.predict(frame)
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_save_and_load_weights_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    clf = S.SuccessClassifier(num_views=2, img_size=32)
    ext = np.full((1, 40, 40, 3), 120, dtype=np.uint8)
    wrist = np.full((1, 40, 40, 3), 30, dtype=np.uint8)
    before_label, before_probs = clf.predict([ext[0], wrist[0]])
    path = str(tmp_path / "cnn.pt")
    clf.save(path)
    clf2 = S.SuccessClassifier()
    clf2.load_weights(path)
    assert clf2.num_views == 2
    after_label, after_probs = clf2.predict([ext[0], wrist[0]])
    assert before_label == after_label
    assert np.allclose(before_probs, after_probs, atol=1e-5)
