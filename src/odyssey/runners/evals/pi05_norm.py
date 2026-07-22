"""π0.5 (openpi) action-space normalization — the analog of OpenVLA's ``unnorm_key``.

See ``docs/pi05-scoping.md`` → "Mapeo de espacio de acción". The one-line summary:

  **π0.5 has no ``unnorm_key``.** Where OpenVLA passes a string at inference time to
  pick which de-normalization statistics to apply to the 7-D action, π0.5/openpi
  bakes a ``norm_stats.json`` **inside the checkpoint** (under ``assets/<asset_id>/``)
  and selects it by the ``asset_id``/``repo_id`` it was trained with — NOT by a key
  the mission YAML carries. For ``pi05_libero`` that ``asset_id`` is
  ``physical-intelligence/libero``.

So the "knob" analogous to ``unnorm_key`` is the **asset_id** (which norm_stats to
use); the *math* it drives is a plain affine (mean/std) or quantile map between the
model's normalized space and the simulator's physical units. openpi runs that math
server-side inside ``Normalize``/``Unnormalize``; this module mirrors it in numpy so
the odyssey side can (a) reason about the mapping, (b) resolve the asset_id, and
(c) reject a stray ``unnorm_key`` in a π0.5 mission config — all without a GPU, a
served checkpoint, or downloaded weights.

Pipeline this module models (both directions), matching openpi's LIBERO recipe:

  observation  8-D Franka Panda state ──pad_to_dim(32)──► normalize(norm_stats) ──► model
  action       model (H,32) normalized ──unnormalize(norm_stats)──► slice[:7] ──► env.step

numpy-only, so it imports and unit-tests without openpi / torch / jax.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# PI0/π0.5 family model width; the 8-D state is padded up to this and the action
# chunk comes out at this width before it is sliced back to LIBERO's 7-DoF.
PI0_ACTION_DIM = 32
# LIBERO's native OSC_POSE action applied to env.step: [dx,dy,dz,droll,dpitch,dyaw,gripper].
LIBERO_ACTION_DIM = 7
# openpi's normalization epsilon (src/openpi/transforms.py) — keep in lockstep so the
# numpy mirror round-trips against the served de-normalization.
NORM_EPS = 1e-6

# The ``unnorm_key`` analog: which baked ``norm_stats`` asset a π0.5 checkpoint uses.
# Keyed by a checkpoint-name substring; the value is openpi's ``asset_id`` (the string
# under ``assets/`` in the checkpoint). This is selected at TRAIN time and travels with
# the weights — it is NOT a mission-config knob (unlike OpenVLA's ``unnorm_key``).
PI05_NORM_ASSETS = {
    "pi05_libero": "physical-intelligence/libero",
}


@dataclass(frozen=True)
class NormStats:
    """Per-dimension normalization statistics, mirroring openpi's ``NormStats``.

    ``mean``/``std`` drive the default affine map; ``q01``/``q99`` drive the quantile
    map (``use_quantiles=True``). Only the pair for the chosen mode is required.
    """

    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None

    @staticmethod
    def from_mean_std(mean, std) -> "NormStats":
        return NormStats(mean=_as_1d(mean), std=_as_1d(std))

    @staticmethod
    def from_quantiles(q01, q99) -> "NormStats":
        return NormStats(q01=_as_1d(q01), q99=_as_1d(q99))


def _as_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def normalize(x, stats: NormStats, *, use_quantiles: bool = False) -> np.ndarray:
    """Map physical-unit values into the model's normalized space (openpi ``Normalize``).

    ``mean_std`` (default): ``(x - mean) / (std + eps)``.
    ``quantile``: ``(x - q01) / (q99 - q01 + eps) * 2 - 1`` (target range ``[-1, 1]``).
    """
    x = np.asarray(x, dtype=np.float64)
    if use_quantiles:
        _require(stats.q01, stats.q99, mode="quantile")
        return (x - stats.q01) / (stats.q99 - stats.q01 + NORM_EPS) * 2.0 - 1.0
    _require(stats.mean, stats.std, mode="mean_std")
    return (x - stats.mean) / (stats.std + NORM_EPS)


def unnormalize(x, stats: NormStats, *, use_quantiles: bool = False) -> np.ndarray:
    """Inverse of :func:`normalize` — model-normalized values back to physical units.

    This is the exact step openpi's ``Unnormalize`` runs on the action chunk with the
    checkpoint's baked ``norm_stats``; it is the direct analog of OpenVLA applying its
    ``unnorm_key`` statistics inside ``predict_action``.
    """
    x = np.asarray(x, dtype=np.float64)
    if use_quantiles:
        _require(stats.q01, stats.q99, mode="quantile")
        return (x + 1.0) / 2.0 * (stats.q99 - stats.q01 + NORM_EPS) + stats.q01
    _require(stats.mean, stats.std, mode="mean_std")
    return x * (stats.std + NORM_EPS) + stats.mean


def _require(*fields, mode: str) -> None:
    if any(f is None for f in fields):
        raise ValueError(
            f"NormStats is missing the fields required for '{mode}' normalization"
        )


def pad_to_dim(x, target_dim: int = PI0_ACTION_DIM) -> np.ndarray:
    """Zero-pad the last axis up to ``target_dim`` (openpi ``pad_to_dim``).

    The 8-D Franka Panda state is padded to the model's 32-D width before it is
    normalized; a no-op when ``x`` already meets/exceeds ``target_dim``.
    """
    x = np.asarray(x, dtype=np.float64)
    current = x.shape[-1]
    if current >= target_dim:
        return x
    pad = [(0, 0)] * x.ndim
    pad[-1] = (0, target_dim - current)
    return np.pad(x, pad)


def slice_to_libero(x, dim: int = LIBERO_ACTION_DIM) -> np.ndarray:
    """Slice the last axis to LIBERO's 7-DoF (openpi ``LiberoOutputs`` ``actions[..., :7]``).

    The tail of the 32-D model action is padding; only the first 7 dims are the
    ``[dx,dy,dz,droll,dpitch,dyaw,gripper]`` OSC_POSE command env.step consumes.
    """
    return np.asarray(x, dtype=np.float64)[..., :dim]


def pi05_state_to_model(state, stats: NormStats, *,
                        use_quantiles: bool = False,
                        target_dim: int = PI0_ACTION_DIM) -> np.ndarray:
    """8-D proprio state → padded-to-32 → normalized: what openpi feeds the model."""
    return normalize(pad_to_dim(state, target_dim), stats, use_quantiles=use_quantiles)


def pi05_action_to_sim(action, stats: NormStats, *,
                       use_quantiles: bool = False,
                       dim: int = LIBERO_ACTION_DIM) -> np.ndarray:
    """Model action (normalized, ≥7-D) → un-normalized → sliced to LIBERO's 7-DoF.

    The gripper channel is passed through with NO fix-up — π0.5's checkpoint is baked
    to LIBERO's gripper convention (contrast GR00T, which normalizes+inverts). See
    ``docs/pi05-scoping.md``: ⚠ confirm gripper polarity on the first GPU rollout.
    """
    return slice_to_libero(
        unnormalize(action, stats, use_quantiles=use_quantiles), dim
    )


def resolve_pi05_asset_id(checkpoint: str, *, override: str | None = None) -> str:
    """Resolve the baked ``asset_id`` (the ``unnorm_key`` analog) for a π0.5 checkpoint.

    Matches a known checkpoint-name substring (e.g. ``lerobot/pi05_libero`` or
    ``gs://openpi-assets/checkpoints/pi05_libero`` → ``physical-intelligence/libero``).
    ``override`` short-circuits the lookup for a custom fine-tune whose asset differs
    from its checkpoint name. Raises ``KeyError`` for an unknown checkpoint so a
    mission fails loudly rather than silently de-normalizing with the wrong stats.
    """
    if override:
        return override
    name = str(checkpoint).rstrip("/").rsplit("/", 1)[-1].lower()
    for key, asset in PI05_NORM_ASSETS.items():
        if key in name:
            return asset
    raise KeyError(
        f"No baked norm-stats asset_id known for π0.5 checkpoint {checkpoint!r}; "
        f"known checkpoints: {sorted(PI05_NORM_ASSETS)}. Pass override=... for a "
        f"custom fine-tune."
    )


def assert_no_unnorm_key(config) -> None:
    """Guard: a π0.5 mission config must NOT carry an ``unnorm_key``.

    Unlike OpenVLA, π0.5's de-normalization is self-contained in the checkpoint
    (``norm_stats``/``asset_id``), so an ``unnorm_key`` in the YAML would be inert or
    misleading (``docs/pi05-scoping.md`` → "No añadir ``unnorm_key`` al YAML de π0.5").
    Reject it early to keep configs honest.
    """
    if isinstance(config, dict) and "unnorm_key" in config:
        raise ValueError(
            "π0.5 configs must NOT set 'unnorm_key': de-normalization is baked into "
            "the checkpoint (norm_stats/asset_id), so the key is inert/misleading. "
            "Remove it; the asset_id is resolved from the checkpoint instead."
        )
