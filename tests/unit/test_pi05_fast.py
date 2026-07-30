"""Tests for π0.5 FAST action de/tokenization (issue #74).

FAST (Frequency-space Action Sequence Tokenization) is what turns π0.5's discrete
action tokens back into the ``(action_horizon, action_dim)`` chunk the LIBERO eval
consumes. The real codec is the HF ``physical-intelligence/fast`` ``AutoProcessor``
(pulled via ``transformers`` + ``scipy``), but ``runners/evals/pi05_fast.py`` is
pure integration glue over it, so the whole encode/decode/normalize path is
exercisable WITHOUT a GPU, ``transformers``, or any weight download:

  * a dependency-free ``_FakeFastProcessor`` stands in for the HF AutoProcessor,
    mimicking its batched ``__call__``/``decode(..., time_horizon=, action_dim=)``
    contract with a lossless round-trip so reconstruction is assertable;
  * ``FastActionCodec`` (the batch-dim juggling + reshape) and the ``[-1, 1]``
    normalization helpers are numpy-only;
  * ``load_fast_tokenizer`` defers the ``transformers`` import, so the injection
    path and the missing-dependency contract are both pinned without it installed.

All tests are named ``test_pi05_fast_*`` (the ``-k pi05_fast`` gate — which does
NOT match the existing ``fails_fast`` tests). No GPU, no served checkpoint, no weights.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from odyssey.runners.evals.pi05_fast import (
    DEFAULT_FAST_TOKENIZER,
    FastActionCodec,
    load_fast_tokenizer,
    normalize_to_unit,
    unnormalize_from_unit,
)

# ---------------------------------------------------------------------------
# Fake FAST processor — a dependency-free stand-in for the HF AutoProcessor.
#
# Faithful to the API the real codec drives (openpi's FASTTokenizer path):
#   processor(action_data)                    -> list[batch] of list[int] tokens
#   processor.decode(tokens, time_horizon=, action_dim=) -> (batch, T, D) array
# The transform here is a lossless fixed-point quantizer (not a real DCT) so the
# round-trip reconstructs within 1/SCALE — enough to test OUR wiring, not FAST.
# ---------------------------------------------------------------------------

class _FakeFastProcessor:
    SCALE = 100_000       # quantization resolution -> ~1e-5 round-trip error
    OFFSET = 1_000_000    # keep token ids non-negative, like real vocab ids

    def __init__(self) -> None:
        self.encode_calls: list[np.ndarray] = []
        self.decode_calls: list[dict[str, Any]] = []

    def __call__(self, action_data) -> list[list[int]]:
        arr = np.asarray(action_data, dtype=np.float64)
        assert arr.ndim == 3, f"FAST expects (batch, T, D), got {arr.shape}"
        self.encode_calls.append(arr)
        out: list[list[int]] = []
        for item in arr:  # (T, D)
            out.append([round(v * self.SCALE) + self.OFFSET
                        for v in item.reshape(-1)])
        return out

    def decode(self, tokens, *, time_horizon: int, action_dim: int) -> np.ndarray:
        self.decode_calls.append(
            {"tokens": tokens, "time_horizon": time_horizon, "action_dim": action_dim}
        )
        out = []
        for toks in tokens:
            flat = np.array([(t - self.OFFSET) / self.SCALE for t in toks],
                            dtype=np.float64)
            out.append(flat.reshape(time_horizon, action_dim))
        return np.stack(out)


def _codec(time_horizon: int = 10, action_dim: int = 32) -> FastActionCodec:
    return FastActionCodec(
        _FakeFastProcessor(), time_horizon=time_horizon, action_dim=action_dim
    )


# ---------------------------------------------------------------------------
# Round-trip: tokenize -> detokenize reconstructs the chunk.
# ---------------------------------------------------------------------------

def test_pi05_fast_roundtrip_reconstructs_chunk() -> None:
    codec = _codec(time_horizon=10, action_dim=32)
    rng = np.random.RandomState(0)
    chunk = rng.uniform(-1.0, 1.0, size=(10, 32)).astype(np.float32)

    tokens = codec.encode(chunk)
    recovered = codec.decode(tokens)

    assert recovered.shape == (10, 32)
    np.testing.assert_allclose(recovered, chunk, atol=1e-4)


def test_pi05_fast_roundtrip_preserves_dtype_and_horizon_one() -> None:
    # A single flat action is treated as a horizon-1 chunk (matches openpi's
    # actions[None] tokenize convention).
    codec = _codec(time_horizon=1, action_dim=7)
    action = np.linspace(-1.0, 1.0, 7, dtype=np.float32)

    recovered = codec.decode(codec.encode(action))

    assert recovered.shape == (1, 7)
    assert recovered.dtype == np.float32
    np.testing.assert_allclose(recovered[0], action, atol=1e-4)


# ---------------------------------------------------------------------------
# Decode of a synthetic token stream (no encode step) reconstructs known values.
# ---------------------------------------------------------------------------

def test_pi05_fast_decode_of_synthetic_tokens() -> None:
    codec = _codec(time_horizon=2, action_dim=3)
    # Hand-build tokens for a known (2, 3) chunk under the fake's fixed-point scheme.
    values = np.array([[0.0, 0.5, -0.5], [0.25, -0.25, 1.0]], dtype=np.float64)
    P = _FakeFastProcessor
    tokens = [round(v * P.SCALE) + P.OFFSET for v in values.reshape(-1)]

    chunk = codec.decode(tokens, time_horizon=2, action_dim=3)

    assert chunk.shape == (2, 3)
    np.testing.assert_allclose(chunk, values, atol=1e-6)


def test_pi05_fast_decode_accepts_ndarray_tokens() -> None:
    # π0.5 hands token ids out as an ndarray; decode must accept .tolist()-ables.
    codec = _codec(time_horizon=1, action_dim=4)
    chunk_in = np.array([[0.1, -0.2, 0.3, -0.4]], dtype=np.float32)
    tokens = np.asarray(codec.encode(chunk_in))  # ndarray, not list

    chunk = codec.decode(tokens)
    np.testing.assert_allclose(chunk, chunk_in, atol=1e-4)


# ---------------------------------------------------------------------------
# Batch-axis juggling + decode reshape are owned by the codec (single-chunk API).
# ---------------------------------------------------------------------------

def test_pi05_fast_encode_batches_and_decode_strips_batch_axis() -> None:
    proc = _FakeFastProcessor()
    codec = FastActionCodec(proc, time_horizon=5, action_dim=8)
    chunk = np.zeros((5, 8), dtype=np.float32)

    tokens = codec.encode(chunk)
    # Caller passed a 2-D chunk; the codec fed the processor a batched (1, T, D).
    assert proc.encode_calls[0].shape == (1, 5, 8)
    # ...and encode returned that single batch item's tokens, not a length-1 list.
    assert isinstance(tokens, list) and isinstance(tokens[0], int)

    out = codec.decode(tokens)
    # decode wrapped the tokens as a length-1 batch and stripped it back to 2-D.
    assert proc.decode_calls[0]["tokens"] == [tokens]
    assert out.shape == (5, 8)


def test_pi05_fast_decode_uses_codec_horizon_and_dim_by_default() -> None:
    proc = _FakeFastProcessor()
    codec = FastActionCodec(proc, time_horizon=10, action_dim=32)
    tokens = codec.encode(np.zeros((10, 32), dtype=np.float32))

    codec.decode(tokens)
    call = proc.decode_calls[0]
    assert (call["time_horizon"], call["action_dim"]) == (10, 32)


def test_pi05_fast_decode_overrides_horizon_and_dim_per_call() -> None:
    proc = _FakeFastProcessor()
    codec = FastActionCodec(proc, time_horizon=10, action_dim=32)
    tokens = codec.encode(np.zeros((3, 7), dtype=np.float32))

    out = codec.decode(tokens, time_horizon=3, action_dim=7)
    assert out.shape == (3, 7)
    call = proc.decode_calls[0]
    assert (call["time_horizon"], call["action_dim"]) == (3, 7)


def test_pi05_fast_codec_rejects_non_positive_shape() -> None:
    with pytest.raises(ValueError, match="positive"):
        FastActionCodec(_FakeFastProcessor(), time_horizon=0, action_dim=32)


def test_pi05_fast_encode_rejects_3d_input() -> None:
    codec = _codec()
    with pytest.raises(ValueError, match="time_horizon, action_dim"):
        codec.encode(np.zeros((2, 10, 32), dtype=np.float32))


# ---------------------------------------------------------------------------
# [-1, 1] normalization helpers (the only numeric integration work FAST needs).
# ---------------------------------------------------------------------------

def test_pi05_fast_normalize_roundtrip() -> None:
    lo = np.array([-2.0, 0.0, -1.0])
    hi = np.array([2.0, 4.0, 1.0])
    physical = np.array([[-2.0, 4.0, 0.0], [1.0, 1.0, -0.5]])

    unit = normalize_to_unit(physical, lo, hi)
    back = unnormalize_from_unit(unit, lo, hi)

    np.testing.assert_allclose(back, physical, atol=1e-9)


def test_pi05_fast_normalize_maps_bounds_to_plus_minus_one() -> None:
    lo, hi = -3.0, 5.0
    np.testing.assert_allclose(normalize_to_unit(lo, lo, hi), -1.0)
    np.testing.assert_allclose(normalize_to_unit(hi, lo, hi), 1.0)
    np.testing.assert_allclose(normalize_to_unit((lo + hi) / 2, lo, hi), 0.0)


def test_pi05_fast_normalize_clips_out_of_range() -> None:
    # Values beyond [lo, hi] are clamped so FAST never sees input outside [-1, 1].
    out = normalize_to_unit([-10.0, 10.0], lo=-1.0, hi=1.0)
    np.testing.assert_allclose(out, [-1.0, 1.0])


def test_pi05_fast_normalize_handles_flat_dimension() -> None:
    # A degenerate dim (hi == lo) must not divide by zero; it maps to 0 / lo.
    unit = normalize_to_unit([5.0], lo=[5.0], hi=[5.0])
    np.testing.assert_allclose(unit, [0.0])
    back = unnormalize_from_unit(unit, lo=[5.0], hi=[5.0])
    np.testing.assert_allclose(back, [5.0])


# ---------------------------------------------------------------------------
# Integration: a FAST-decoded chunk feeds pi05_action_to_libero unchanged.
# ---------------------------------------------------------------------------

def test_pi05_fast_decoded_chunk_feeds_pi05_action_to_libero() -> None:
    from odyssey.runners.evals.pi05_transforms import pi05_action_to_libero

    codec = _codec(time_horizon=4, action_dim=32)
    rng = np.random.RandomState(1)
    chunk = rng.uniform(-1.0, 1.0, size=(4, 32)).astype(np.float32)
    decoded = codec.decode(codec.encode(chunk))

    # The detokenized chunk slices to LIBERO's 7-DoF action with no gripper fix-up.
    action = pi05_action_to_libero({"actions": decoded}, 2)
    assert action.shape == (7,)
    np.testing.assert_allclose(action, chunk[2, :7], atol=1e-4)


# ---------------------------------------------------------------------------
# load_fast_tokenizer — injection bypasses transformers; absence is contracted.
# ---------------------------------------------------------------------------

def test_pi05_fast_load_returns_injected_tokenizer() -> None:
    sentinel = _FakeFastProcessor()
    assert load_fast_tokenizer(tokenizer=sentinel) is sentinel


def test_pi05_fast_default_checkpoint_is_universal_fast() -> None:
    assert DEFAULT_FAST_TOKENIZER == "physical-intelligence/fast"


def test_pi05_fast_load_raises_without_transformers() -> None:
    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers installed; the missing-dep path is not exercised")
    with pytest.raises(NotImplementedError, match="transformers"):
        load_fast_tokenizer()  # no injected tokenizer -> tries the deferred import


# ---------------------------------------------------------------------------
# The codec module defers heavy deps (transformers/torch/jax) — bare stdlib+numpy.
# ---------------------------------------------------------------------------

def test_pi05_fast_module_imports_without_heavy_deps() -> None:
    heavy = ("transformers", "torch", "jax", "scipy")
    # Fresh interpreter pointed at THIS worktree's src (it doesn't inherit pytest's
    # `pythonpath`, and an editable-install .pth may redirect elsewhere).
    src_dir = Path(__file__).resolve().parents[2] / "src"
    script = (
        "import importlib, json, sys\n"
        f"sys.path.insert(0, {str(src_dir)!r})\n"
        "importlib.import_module('odyssey.runners.evals.pi05_fast')\n"
        f"print(json.dumps([m for m in {heavy!r} if m in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f"pi05_fast module failed to import cleanly:\n{result.stderr}"
    )
    leaked = json.loads(result.stdout.strip().splitlines()[-1])
    assert leaked == [], f"pi05_fast imported heavy deps at load: {leaked}"
