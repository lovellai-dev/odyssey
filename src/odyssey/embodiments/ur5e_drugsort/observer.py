"""Grasp-target **Observer** for the UR5e / Robotiq drug-sorting cell.

The iteration-2 policy fired the gripper but closed ~7 cm short on the final
descent (closed-loop eval 0/20, every episode ``lifted=False``): a single
overhead camera cannot resolve descent *depth*. This Observer attacks that
failure directly. It maps the two deploy-time camera frames (``exterior`` +
``wrist`` eye-in-hand) to the **3D grasp target** — the vial cap/neck point in
the robot **base** frame (metres). That absolute 3D target is:

* a **reward signal** for residual RL (penalise the gripper-target gap on the
  descent, exactly where the policy stalls), and
* a **deployable conditioning input**: fed alongside the images it tells the
  policy where to descend, with no ground truth needed at inference (it is
  regressed from RGB, so it runs from real cameras).

Design (mirrors the repo's lazy-heavy-deps convention, e.g. ``cosmos_reason``):
a **frozen** vision backbone + a small trained regression head. DINOv3 is the
first-choice backbone; it is gated on HuggingFace, so the loader falls back to
the open, compact **DINOv2-small** (22 M params, frozen) — a ViT whose dense
features localise the cap/neck well with only a tiny head to train. Only the
head is optimised, so training is cheap (CPU or a brief GPU touch) and never
competes with the big ``launch_finetune`` runs.

The pure helpers (coordinate/metric math, backbone selection) use only
numpy/stdlib and are unit-testable without torch; ``torch``/``transformers`` are
imported lazily inside :meth:`GraspTargetObserver.load`. The class deliberately
does **not** subclass ``torch.nn.Module`` (it composes ``nn.Sequential`` +
a backbone as attributes) so the typed surface stays ``mypy --strict`` clean.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Backbone preference order: DINOv3 (gated) -> DINOv2-small (open, compact).
DEFAULT_BACKBONES: tuple[str, ...] = (
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov2-small",
)
# ImageNet normalisation (both DINOv2 and DINOv3 expect it).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
DEFAULT_IMG_SIZE = 224
TARGET_DIM = 3  # (x, y, z) grasp target in the robot base frame (metres)
VIEW_KEYS: tuple[str, ...] = ("exterior", "wrist")


# ---------------------------------------------------------------------------
# Pure helpers (numpy/stdlib only -> unit-testable without torch)
# ---------------------------------------------------------------------------

def keypoint_error_cm(pred: Any, gt: Any) -> Any:
    """Per-sample Euclidean grasp-target error in **centimetres**.

    ``pred`` / ``gt`` are ``(N, 3)`` (or ``(3,)``) arrays of metres. Returns the
    per-sample error in cm — the headline Observer accuracy metric.
    """
    p = np.asarray(pred, dtype=np.float64).reshape(-1, TARGET_DIM)
    g = np.asarray(gt, dtype=np.float64).reshape(-1, TARGET_DIM)
    return np.linalg.norm(p - g, axis=1) * 100.0


def world_to_base(p_world: Any, base_pos: Any, base_mat: Any) -> Any:
    """Express a world point in the robot base frame: ``R_base^T (p - base_pos)``.

    ``base_mat`` is the 3x3 base rotation (columns = base axes in world), as
    given by MuJoCo ``data.xmat``. Kept here so the deploy client and the
    label generator share one convention.
    """
    r = np.asarray(base_mat, dtype=np.float64).reshape(3, 3)
    return r.T @ (np.asarray(p_world, dtype=np.float64) - np.asarray(base_pos, dtype=np.float64))


def select_backbone(candidates: tuple[str, ...]) -> tuple[str, Any]:
    """Return the first loadable ``(model_id, AutoModel)`` from ``candidates``.

    Tries each id in order (DINOv3 first, DINOv2 fallback); skips gated /
    unreachable weights. Raises ``RuntimeError`` if none load.
    """
    from transformers import AutoModel  # lazy heavy dep

    last_err: Exception | None = None
    for model_id in candidates:
        try:
            model = AutoModel.from_pretrained(model_id)
            return model_id, model
        except Exception as exc:  # gated repo / offline / missing — try the next
            last_err = exc
    raise RuntimeError(
        f"no backbone loadable from {candidates!r}: {last_err}"
    )


# ---------------------------------------------------------------------------
# Observer (heavy model deferred to load())
# ---------------------------------------------------------------------------

class GraspTargetObserver:
    """Frozen ViT backbone + small head: two RGB views -> 3D grasp target.

    Lazy: constructing is cheap (no torch import); the backbone + head load on
    :meth:`load` (or first :meth:`predict`). Pass ``backbone`` to inject a
    callable ``(N,3,H,W) float tensor -> (N, feat_dim) tensor`` (used by tests
    to avoid downloading DINO); otherwise it is resolved via
    :func:`select_backbone`.
    """

    def __init__(
        self,
        *,
        backbones: tuple[str, ...] = DEFAULT_BACKBONES,
        num_views: int = 2,
        head_hidden: int = 256,
        img_size: int = DEFAULT_IMG_SIZE,
        device: str = "cpu",
        feat_dim: int | None = None,
        backbone: Any = None,
    ) -> None:
        self.backbones = backbones
        self.num_views = num_views
        self.head_hidden = head_hidden
        self.img_size = img_size
        self.device = device
        self.backbone_id: str | None = "injected" if backbone is not None else None
        self._feat_dim = feat_dim
        self._backbone: Any = backbone
        self._head: Any = None

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> None:
        """Resolve the frozen backbone (if not injected) and build the head."""
        import torch
        from torch import nn

        if self._backbone is None:
            self.backbone_id, self._backbone = select_backbone(self.backbones)
            self._feat_dim = int(self._backbone.config.hidden_size)
            self._backbone = self._backbone.to(self.device).eval()
            for p in self._backbone.parameters():
                p.requires_grad_(False)
        if self._feat_dim is None:
            raise ValueError("feat_dim must be given when injecting a backbone")
        if self._head is None:
            self._head = nn.Sequential(
                nn.Linear(self._feat_dim * self.num_views, self.head_hidden),
                nn.GELU(),
                nn.Linear(self.head_hidden, self.head_hidden),
                nn.GELU(),
                nn.Linear(self.head_hidden, TARGET_DIM),
            ).to(self.device)
        # torch is only needed for the isinstance guard below in predict()
        _ = torch

    # -- tensor path (training / forward) -----------------------------------
    def _prep(self, frames: Any) -> Any:
        """``(N,H,W,3) uint8`` -> normalised ``(N,3,img,img)`` float tensor."""
        import torch
        import torch.nn.functional as fn

        arr = np.asarray(frames)
        if arr.ndim == 3:
            arr = arr[None]
        t = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        t = t.permute(0, 3, 1, 2) / 255.0
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        t = (t - mean) / std
        if t.shape[-1] != self.img_size or t.shape[-2] != self.img_size:
            t = fn.interpolate(
                t, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )
        return t

    def _embed_view(self, view_t: Any) -> Any:
        """Frozen backbone features for one preprocessed view -> ``(N, feat_dim)``."""
        import torch

        with torch.no_grad():
            if callable(self._backbone) and not hasattr(self._backbone, "config"):
                return self._backbone(view_t)  # injected test backbone
            out = self._backbone(pixel_values=view_t)
            # DINOv2/DINOv3: CLS token is row 0 of the last hidden state.
            return out.last_hidden_state[:, 0]

    def forward(self, views: list[Any]) -> Any:
        """Preprocessed/raw views (list of ``(N,H,W,3)`` uint8) -> ``(N,3)`` target.

        The backbone runs frozen (no grad); the head is trainable.
        """
        import torch

        self.load()  # idempotent; ensures backbone + head are built
        feats = [self._embed_view(self._prep(v)) for v in views]
        cat = torch.cat(feats, dim=1)
        return self._head(cat)

    # -- deploy inference ---------------------------------------------------
    def predict(self, exterior: Any, wrist: Any) -> Any:
        """Real-camera inference: two ``(H,W,3)`` uint8 frames -> ``(3,)`` xyz (m).

        No ground truth needed — this is the deployable grasp-target signal.
        """
        import torch

        with torch.no_grad():  # forward() loads lazily
            out = self.forward([exterior[None], wrist[None]])
        return out.detach().cpu().numpy().reshape(TARGET_DIM)

    # -- optimisation surface ----------------------------------------------
    def head_parameters(self) -> Any:
        """The trainable head parameters (the backbone stays frozen)."""
        self.load()
        return self._head.parameters()

    def train_mode(self, flag: bool = True) -> None:
        if self._head is not None:
            self._head.train(flag)

    def save(self, path: str) -> None:
        import torch

        if self._head is None:
            raise RuntimeError("nothing to save: head not built")
        torch.save(
            {"head": self._head.state_dict(), "backbone_id": self.backbone_id,
             "feat_dim": self._feat_dim, "img_size": self.img_size,
             "num_views": self.num_views, "head_hidden": self.head_hidden},
            path,
        )

    def load_head(self, path: str) -> None:
        import torch

        ckpt = torch.load(path, map_location=self.device)
        self._feat_dim = int(ckpt["feat_dim"])
        self.img_size = int(ckpt["img_size"])
        self.num_views = int(ckpt["num_views"])
        self.head_hidden = int(ckpt["head_hidden"])
        self.load()
        self._head.load_state_dict(ckpt["head"])
