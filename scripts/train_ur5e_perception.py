#!/usr/bin/env python3
"""Train + evaluate the UR5e drug-sorting perception stack.

Consumes the auto-labelled dataset from ``gen_ur5e_perception_data.py`` and
trains the two reusable modules:

* :class:`~odyssey.embodiments.ur5e_drugsort.observer.GraspTargetObserver` — a
  frozen DINOv2/DINOv3 backbone + small head regressing the 3D grasp target
  (base frame). Reports held-out keypoint error in **cm** (incl. visual-DR).
* :class:`~odyssey.embodiments.ur5e_drugsort.success_classifier.SuccessClassifier`
  — a small CNN distilled from the free GT phase labels. Reports held-out
  4-class accuracy and binary ``seated`` (terminal-success) accuracy.

Splits by **episode** so the validation poses/appearances are unseen. Backbone
features are precomputed once (a brief GPU touch) so head/CNN training is cheap
and never competes with the remote ``launch_finetune`` runs. Writes weights +
``metrics.json`` to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from odyssey.embodiments.ur5e_drugsort.observer import (
    GraspTargetObserver,
    keypoint_error_cm,
)
from odyssey.embodiments.ur5e_drugsort.success_classifier import (
    NUM_CLASSES,
    PHASE_CLASSES,
    SUCCESS_CLASS,
    SuccessClassifier,
)


def load_dataset(d: Path):  # type: ignore[no-untyped-def]
    return {
        "ext": np.load(d / "ext.npy"),
        "wrist": np.load(d / "wrist.npy"),
        "target": np.load(d / "grasp_target_base.npy"),
        "labels": np.load(d / "labels.npy"),
        "episode": np.load(d / "episode.npy"),
        "success": np.load(d / "success.npy"),
    }


def precompute_features(obs, ext, wrist, device, batch=64):  # type: ignore[no-untyped-def]
    """Frozen-backbone features for both views -> (N, 2*feat_dim) numpy."""
    import torch

    feats = []
    n = ext.shape[0]
    for i in range(0, n, batch):
        e = obs._embed_view(obs._prep(ext[i:i + batch]))
        w = obs._embed_view(obs._prep(wrist[i:i + batch]))
        feats.append(torch.cat([e, w], dim=1).detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def train_observer(data, tr, va, device, epochs, out):  # type: ignore[no-untyped-def]
    import torch
    from torch import nn

    obs = GraspTargetObserver(device=device)
    obs.load()
    print(f"[obs] backbone={obs.backbone_id} feat_dim={obs._feat_dim} "
          f"(head params={sum(p.numel() for p in obs.head_parameters())})")
    cache = out / "feat_cache.npy"
    if cache.is_file():
        feats = np.load(cache)
        print(f"[obs] loaded cached features {feats.shape}")
    else:
        feats = precompute_features(obs, data["ext"], data["wrist"], device)
        np.save(cache, feats)
    ft = torch.tensor(feats, dtype=torch.float32, device=device)
    tgt = torch.tensor(data["target"], dtype=torch.float32, device=device)

    head = obs._head
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.SmoothL1Loss(beta=0.02)
    rng = np.random.default_rng(0)
    head.train()
    for ep in range(epochs):
        perm = tr[rng.permutation(len(tr))]
        for i in range(0, len(perm), 128):
            idx = torch.tensor(perm[i:i + 128], device=device)
            opt.zero_grad()
            pred = head(ft[idx])
            loss = lossf(pred, tgt[idx])
            loss.backward()
            opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            head.eval()
            with torch.no_grad():
                va_t = torch.tensor(va, device=device)
                err = keypoint_error_cm(
                    head(ft[va_t]).cpu().numpy(), data["target"][va])
            print(f"[obs] epoch {ep:3d} val median={np.median(err):.2f}cm "
                  f"mean={err.mean():.2f}cm p90={np.percentile(err,90):.2f}cm")
            head.train()

    head.eval()
    with torch.no_grad():
        va_t = torch.tensor(va, device=device)
        err = keypoint_error_cm(head(ft[va_t]).cpu().numpy(), data["target"][va])
    obs.save(str(out / "observer_head.pt"))
    return {
        "backbone": obs.backbone_id,
        "n_train": int(len(tr)), "n_val": int(len(va)),
        "median_cm": float(np.median(err)), "mean_cm": float(err.mean()),
        "p90_cm": float(np.percentile(err, 90)), "max_cm": float(err.max()),
    }


def train_classifier(data, tr, va, device, epochs, out):  # type: ignore[no-untyped-def]
    import torch
    from torch import nn

    clf = SuccessClassifier(device=device, num_views=2)
    clf.load()
    # Pre-resize both views to the CNN input size ONCE (uint8) so per-step _prep
    # only normalises — the 224->96 interpolate every step was the bottleneck.
    views = [_resize_uint8(data["ext"], clf.img_size, device),
             _resize_uint8(data["wrist"], clf.img_size, device)]
    labels = data["labels"]
    print(f"[clf] CNN params={sum(p.numel() for p in clf.parameters())} "
          f"num_views={clf.num_views}")
    counts = np.bincount(labels[tr], minlength=NUM_CLASSES).astype(np.float64)
    weights = torch.tensor((counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))),
                           dtype=torch.float32, device=device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(weight=weights)
    lab_t = torch.tensor(labels, dtype=torch.long, device=device)
    rng = np.random.default_rng(0)
    best = (-1.0, None)
    clf.train_mode(True)
    for ep in range(epochs):
        perm = tr[rng.permutation(len(tr))]
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            logits = clf.forward([v[idx] for v in views])
            loss = lossf(logits, lab_t[torch.tensor(idx, device=device)])
            loss.backward()
            opt.step()
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            acc, sacc, _ = _eval_clf(clf, views, labels, va, device)
            score = acc + sacc
            if score > best[0]:
                best = (score, {k: v.detach().cpu().clone()
                                for k, v in clf._model.state_dict().items()})
            print(f"[clf] epoch {ep:3d} val 4class={acc*100:.1f}% seated={sacc*100:.1f}%")
    if best[1] is not None:
        clf._model.load_state_dict(best[1])  # keep the best checkpoint
    clf.train_mode(False)
    acc, sacc, cm = _eval_clf(clf, views, labels, va, device)
    clf.save(str(out / "success_cnn.pt"))
    return {
        "views": ["exterior", "wrist"], "n_train": int(len(tr)), "n_val": int(len(va)),
        "acc_4class": float(acc), "acc_seated_success": float(sacc),
        "confusion": cm.tolist(), "class_names": list(PHASE_CLASSES),
    }


def _resize_uint8(frames, size, device, batch=512):  # type: ignore[no-untyped-def]
    """Batched 224->``size`` resize of a ``(N,H,W,3)`` uint8 stack (one-time)."""
    import torch
    import torch.nn.functional as fn

    outs = []
    for i in range(0, frames.shape[0], batch):
        t = torch.as_tensor(frames[i:i + batch], dtype=torch.float32, device=device)
        t = t.permute(0, 3, 1, 2)
        t = fn.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        outs.append(t.permute(0, 2, 3, 1).round().clamp(0, 255).byte().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _eval_clf(clf, views, labels, va, device, batch=128):  # type: ignore[no-untyped-def]
    import torch

    preds = []
    clf.train_mode(False)
    with torch.no_grad():
        for i in range(0, len(va), batch):
            idx = va[i:i + batch]
            logits = clf.forward([v[idx] for v in views])
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    pred = np.concatenate(preds)
    true = labels[va]
    acc = float((pred == true).mean())
    sacc = float(((pred == SUCCESS_CLASS) == (true == SUCCESS_CLASS)).mean())
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(true, pred, strict=True):
        cm[t, p] += 1
    return acc, sacc, cm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-episodes", default="4,9,14,19,23",
                    help="comma-separated episode ids held out for validation")
    ap.add_argument("--obs-epochs", type=int, default=60)
    ap.add_argument("--clf-epochs", type=int, default=25)
    ap.add_argument("--skip-observer", action="store_true")
    ap.add_argument("--skip-classifier", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(Path(args.data))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    val_eps = {int(x) for x in args.val_episodes.split(",") if x != ""}
    ep = data["episode"]
    va = np.where(np.isin(ep, list(val_eps)))[0]
    tr = np.where(~np.isin(ep, list(val_eps)))[0]
    print(f"[data] frames={len(ep)} train={len(tr)} val={len(va)} "
          f"val_episodes={sorted(val_eps)} device={device}")
    print(f"[data] val class counts (r/g/l/s)="
          f"{np.bincount(data['labels'][va], minlength=NUM_CLASSES).tolist()}")

    prev = {}
    mpath = out / "metrics.json"
    if mpath.is_file():
        prev = json.loads(mpath.read_text())

    obs_metrics = (prev.get("observer") if args.skip_observer
                   else train_observer(data, tr, va, device, args.obs_epochs, out))
    clf_metrics = (prev.get("classifier") if args.skip_classifier
                   else train_classifier(data, tr, va, device, args.clf_epochs, out))

    metrics = {"observer": obs_metrics, "classifier": clf_metrics,
               "val_episodes": sorted(val_eps), "device": device}
    mpath.write_text(json.dumps(metrics, indent=2) + "\n")
    print("\n==== SUMMARY ====")
    if obs_metrics:
        print(f"Observer ({obs_metrics['backbone']}): grasp-target error "
              f"median={obs_metrics['median_cm']:.2f}cm mean={obs_metrics['mean_cm']:.2f}cm "
              f"p90={obs_metrics['p90_cm']:.2f}cm  (val n={obs_metrics['n_val']})")
    if clf_metrics:
        print(f"Classifier: 4-class acc={clf_metrics['acc_4class']*100:.1f}%  "
              f"seated/success acc={clf_metrics['acc_seated_success']*100:.1f}%  "
              f"(val n={clf_metrics['n_val']})")
    print(f"[out] wrote weights + metrics.json to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
