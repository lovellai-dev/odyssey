"""Distilled success / phase classifier for the UR5e drug-sorting cell.

A fast, small CNN that reads a rollout frame and reports the grasp **phase** —
``reaching`` / ``grasped`` / ``lifted`` / ``seated`` — where ``seated`` is task
**success** (vial resting in the nest pocket). It provides two things the RL
loop needs on the hot path:

* the **terminal reward** — a per-episode (or per-frame) success signal, and
* **deploy-time success detection** — the same signal at inference, from RGB
  only.

Why distil: the ``ReasoningSidecar`` (Cosmos-Reason-2B, ``runners/evals/
cosmos_reason.py``) can *judge* a frame in natural language, but at ~1-2 s per
call it is far too slow for a reward on every episode/step. In sim we also get
the phase **for free** from ground truth (gripper closure, vial-to-pinch
distance, lift height, place distance — see :func:`label_from_state`). So the
teacher signal is: **sim GT** (exact, free) — optionally cross-checked by the
Cosmos-Reason VLM via :class:`CosmosReasonTeacher` — and the **student** is this
CNN, which runs in well under a millisecond.

Following the repo convention, the label/parse helpers are pure (stdlib/numpy,
unit-testable without torch) and ``torch`` is imported lazily. The classifier is
composed from ``nn.Sequential`` rather than subclassing ``nn.Module`` so the
typed surface stays ``mypy --strict`` clean.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Phase classes, ordered by task progress. Index 3 (``seated``) == success.
PHASE_CLASSES: tuple[str, ...] = ("reaching", "grasped", "lifted", "seated")
SUCCESS_CLASS = 3
NUM_CLASSES = len(PHASE_CLASSES)

# GT thresholds (kept in sync with scripts/gen_ur5e_perception_data.py).
GRIP_CLOSED_THRESH = 0.30
NEAR_PINCH_THRESH_M = 0.055
LIFT_THRESH_M = 0.03
SEATED_XY_THRESH_M = 0.05
SEATED_Z_MIN_M = 0.20
_SEATED_PHASES = frozenset({"release", "retract", "home2", "finished"})

DEFAULT_IMG_SIZE = 96


# ---------------------------------------------------------------------------
# Pure label / parse helpers (stdlib/numpy -> unit-testable without torch)
# ---------------------------------------------------------------------------

def label_from_state(
    *,
    grip_closure: float,
    vial_to_pinch_m: float,
    lift_height_m: float,
    place_dist_xy_m: float,
    vial_z_m: float,
    phase_name: str,
) -> int:
    """Map sim ground truth to a phase class in ``[0, 3]``.

    Priority ``seated > lifted > grasped > reaching``: the vial seated in the
    pocket (near it in xy, at nest height, after release) is success; otherwise a
    closed gripper that has raised the vial is ``lifted``; a closed gripper on
    the vial is ``grasped``; everything else is ``reaching``.
    """
    grip_closed = grip_closure > GRIP_CLOSED_THRESH
    seated = (
        place_dist_xy_m < SEATED_XY_THRESH_M
        and vial_z_m > SEATED_Z_MIN_M
        and phase_name in _SEATED_PHASES
    )
    if seated:
        return 3
    if grip_closed and lift_height_m > LIFT_THRESH_M:
        return 2
    if grip_closed and vial_to_pinch_m < NEAR_PINCH_THRESH_M:
        return 1
    return 0


def is_success(label: int) -> bool:
    """``True`` iff the phase label is the terminal success class (``seated``)."""
    return int(label) == SUCCESS_CLASS


def parse_phase_from_text(text: str) -> int | None:
    """Parse a Cosmos-Reason (VLM) judgement string into a phase class.

    The teacher is prompted to answer with one phase word; this maps its free
    text onto ``PHASE_CLASSES``, scanning most-advanced-first so a sentence like
    "the vial is grasped and lifted" resolves to the further-along phase.
    Returns ``None`` when no phase word is present (caller falls back to GT).
    """
    low = (text or "").lower()
    synonyms: tuple[tuple[int, tuple[str, ...]], ...] = (
        (3, ("seated", "placed", "in the rack", "in the pocket", "in the nest", "released")),
        (2, ("lifted", "raised", "picked up", "holding it up", "in the air")),
        (1, ("grasped", "gripped", "grabbed", "grasping", "closed on")),
        (0, ("reaching", "approaching", "moving toward", "locating", "searching")),
    )
    for cls, words in synonyms:
        if any(w in low for w in words):
            return cls
    return None


def build_cnn(num_classes: int = NUM_CLASSES, in_ch: int = 3, img_size: int = DEFAULT_IMG_SIZE) -> Any:
    """A small, fast phase-classifier CNN as an ``nn.Sequential``.

    ~5 conv stages (stride-2 downsampling) + global-average-pool + linear head:
    tiny enough to run in well under a millisecond, so it is viable as a hot-path
    reward. Returns a bare ``nn.Sequential`` (no ``nn.Module`` subclassing).
    """
    from torch import nn

    def block(cin: int, cout: int) -> Any:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    return nn.Sequential(
        block(in_ch, 32),   # img/2
        block(32, 64),      # img/4
        block(64, 128),     # img/8
        block(128, 128),    # img/16
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.1),
        nn.Linear(128, num_classes),
    )


# ---------------------------------------------------------------------------
# Fast CNN student (heavy model deferred to load())
# ---------------------------------------------------------------------------

class SuccessClassifier:
    """Small CNN mapping camera view(s) -> phase logits (``seated`` == success).

    Multi-view: with ``num_views=2`` it channel-stacks the ``exterior`` +
    ``wrist`` frames the Playground already publishes. The single overhead view
    can't tell the near-identical home (start) vs home2 (end) arm poses apart —
    the wrist eye-in-hand view resolves it, sharply cutting ``reaching``->``seated``
    false positives.

    Lazy: constructing is cheap; the CNN builds on :meth:`load` / first use. Pass
    ``model`` to inject a callable ``(N,C,H,W) -> (N, num_classes)`` (tests avoid
    torch model construction that way).
    """

    def __init__(
        self,
        *,
        num_classes: int = NUM_CLASSES,
        num_views: int = 1,
        img_size: int = DEFAULT_IMG_SIZE,
        device: str = "cpu",
        model: Any = None,
    ) -> None:
        self.num_classes = num_classes
        self.num_views = num_views
        self.in_ch = 3 * num_views
        self.img_size = img_size
        self.device = device
        self._model: Any = model

    def load(self) -> None:
        if self._model is None:
            self._model = build_cnn(self.num_classes, self.in_ch, self.img_size).to(self.device)

    def _prep_one(self, frames: Any) -> Any:
        """``(N,H,W,3) uint8`` (or ``(H,W,3)``) -> normalised ``(N,3,img,img)``."""
        import torch
        import torch.nn.functional as fn

        arr = np.asarray(frames)
        if arr.ndim == 3:
            arr = arr[None]
        t = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        t = t.permute(0, 3, 1, 2) / 255.0
        if t.shape[-1] != self.img_size or t.shape[-2] != self.img_size:
            t = fn.interpolate(
                t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )
        return t

    def _stack(self, views: Any) -> Any:
        """Prep + channel-stack one array or a list of per-view arrays."""
        import torch

        seq = views if isinstance(views, list | tuple) else [views]
        return torch.cat([self._prep_one(v) for v in seq], dim=1)

    def forward(self, views: Any) -> Any:
        """View(s) (array or list of ``(N,H,W,3)`` uint8) -> ``(N, num_classes)``."""
        self.load()  # idempotent
        return self._model(self._stack(views))

    def predict(self, views: Any) -> tuple[int, Any]:
        """One frame ``(H,W,3)`` or a list of per-view frames -> ``(label, probs)``."""
        import torch

        self.load()
        # Deploy inference must run in eval mode: the CNN has BatchNorm, which
        # would otherwise use (undefined) batch stats for a single frame.
        self._model.eval()
        seq = views if isinstance(views, list | tuple) else [views]
        batched = [v[None] for v in seq]
        with torch.no_grad():
            logits = self.forward(batched)
            probs = torch.softmax(logits, dim=1)[0]
        return int(torch.argmax(probs).item()), probs.detach().cpu().numpy()

    def predict_success(self, views: Any) -> bool:
        """Deploy-time terminal success signal for one frame or per-view frames."""
        label, _ = self.predict(views)
        return is_success(label)

    def parameters(self) -> Any:
        self.load()
        return self._model.parameters()

    def train_mode(self, flag: bool = True) -> None:
        if self._model is not None:
            self._model.train(flag)

    def save(self, path: str) -> None:
        import torch

        if self._model is None:
            raise RuntimeError("nothing to save: model not built")
        torch.save(
            {"model": self._model.state_dict(), "num_classes": self.num_classes,
             "num_views": self.num_views, "img_size": self.img_size},
            path,
        )

    def load_weights(self, path: str) -> None:
        import torch

        ckpt = torch.load(path, map_location=self.device)
        self.num_classes = int(ckpt["num_classes"])
        self.num_views = int(ckpt["num_views"])
        self.in_ch = 3 * self.num_views
        self.img_size = int(ckpt["img_size"])
        model = build_cnn(self.num_classes, self.in_ch, self.img_size).to(self.device)
        model.load_state_dict(ckpt["model"])
        model.eval()  # loaded weights default to deploy/eval mode
        self._model = model


# ---------------------------------------------------------------------------
# Cosmos-Reason VLM teacher (optional cross-check of the free GT labels)
# ---------------------------------------------------------------------------

class CosmosReasonTeacher:
    """Wrap the ``ReasoningSidecar`` (Cosmos-Reason-2B) as a phase labeller.

    Used to *cross-validate* the free sim-GT labels the CNN distils from (and, on
    a real robot where GT is absent, to bootstrap labels). The sidecar's own
    ``reason()`` is a plan-narration surface (it always asks the VLM for a 3-step
    plan), so this teacher instead issues a **direct one-word phase prompt**
    through the sidecar's loaded model/processor and parses the reply with
    :func:`parse_phase_from_text`. Slow (~2 s/frame) — a labelling-time teacher,
    never the hot path (that is exactly why we distil it into the CNN). Never
    raises: returns ``None`` when the VLM is unavailable or gives no phase word.

    A sidecar exposing only ``reason(frame, instruction)`` (e.g. a test double)
    is driven through that method instead of the direct model path.
    """

    JUDGE_INSTRUCTION = (
        "This is a robot arm sorting a vial into a rack. Reply with exactly ONE "
        "word naming the current stage: 'reaching' (arm approaching, not holding "
        "the vial), 'grasped' (gripper closed on the vial, still low), 'lifted' "
        "(holding the vial up in the air), or 'seated' (vial released, resting in "
        "the rack). One word only."
    )

    def __init__(self, sidecar: Any = None, max_new_tokens: int = 12) -> None:
        self._sidecar = sidecar
        self._max_new_tokens = max_new_tokens

    def _ensure(self) -> Any:
        if self._sidecar is None:
            import importlib.util
            import os

            here = os.path.dirname(__file__)
            path = os.path.normpath(
                os.path.join(here, "..", "..", "runners", "evals", "cosmos_reason.py")
            )
            spec = importlib.util.spec_from_file_location("cosmos_reason", path)
            if spec is None or spec.loader is None:  # pragma: no cover - import guard
                raise RuntimeError(f"could not load cosmos_reason from {path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._sidecar = mod.ReasoningSidecar(max_new_tokens=self._max_new_tokens)
        return self._sidecar

    def _judge_text(self, sidecar: Any, frame: Any) -> str:
        """Ask the VLM the one-word phase question and return its raw reply."""
        # Test doubles expose only reason(); real sidecars expose the model.
        if not hasattr(sidecar, "load"):
            return str(sidecar.reason(frame, self.JUDGE_INSTRUCTION))
        import numpy as _np
        import torch
        from PIL import Image

        sidecar.load()
        arr = _np.ascontiguousarray(frame, dtype=_np.uint8)
        img = frame if isinstance(frame, Image.Image) else Image.fromarray(arr)
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": self.JUDGE_INSTRUCTION}]}]
        proc, model = sidecar._proc, sidecar._model
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(sidecar.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=self._max_new_tokens, do_sample=False)
        return str(proc.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])

    def label(self, frame: Any) -> int | None:
        """VLM phase judgement for one frame, or ``None`` on failure/no-phase."""
        try:
            text = self._judge_text(self._ensure(), frame)
        except Exception:  # pragma: no cover - teacher must never break labelling
            return None
        return parse_phase_from_text(text)
