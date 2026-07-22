"""FAST (Frequency-space Action Sequence Tokenization) codec for π0.5 chunks.

π0.5's autoregressive path emits **discrete action tokens**, not a continuous
chunk: FAST compresses each action sequence with a per-dimension DCT + JPEG-style
quantization (see ``docs/pi05-scoping.md`` → "FAST tokenizer"). To turn those
tokens back into the ``(action_horizon, action_dim)`` chunk the LIBERO recipe
consumes, they have to be **detokenized** with the exact same FAST codec.

FAST is a **reusable library, not something to reimplement** — Physical
Intelligence ships it as the HF ``AutoProcessor`` checkpoint
``physical-intelligence/fast`` (Apache 2.0), which openpi's ``FASTTokenizer``
wraps underneath. So this module is pure integration glue:

  * ``load_fast_tokenizer`` — lazily builds the HF ``AutoProcessor`` (or takes an
    injected one), matching the missing-optional-backend contract used elsewhere
    (openvla / gr00t): a ``NotImplementedError`` with an actionable message when
    ``transformers`` is absent, so nothing downloads weights at import time;
  * ``FastActionCodec`` — the thin encode/decode wrapper around that tokenizer,
    single-chunk in / single-chunk out (it owns the batch-dim juggling that the
    FAST API requires, exactly like openpi's ``tokenize`` / ``extract_actions``);
  * ``normalize_to_unit`` / ``unnormalize_from_unit`` — the ``[-1, 1]`` mapping
    FAST expects on its input, the *only* numeric work the scoping doc calls out
    ("normalización a ``[-1, 1]``, wiring del chunk"), kept as pure functions.

The decoded chunk feeds ``pi05_action_to_libero`` unchanged (slice to 7-D, no
gripper fix-up). numpy-only at import — ``transformers`` is deferred — so the
whole codec unit-tests on a CPU box with an injected fake tokenizer, no GPU and
no weight download.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# The universal FAST+ tokenizer on the Hub (trained on 1M real action sequences).
# openpi's FASTTokenizer defaults to this exact checkpoint underneath.
DEFAULT_FAST_TOKENIZER = "physical-intelligence/fast"


def load_fast_tokenizer(
    path: str = DEFAULT_FAST_TOKENIZER,
    *,
    tokenizer: Any = None,
) -> Any:
    """Return a FAST tokenizer (HF ``AutoProcessor``) or the injected ``tokenizer``.

    ``trust_remote_code=True`` is required: the FAST algorithm (DCT + quantize)
    travels as code in the Hub repo. Deferred import of ``transformers`` keeps
    this module numpy-only at load; a missing dep raises ``NotImplementedError``
    with an actionable message (same contract as ``make_pi05_pilot`` for the
    missing openpi client) rather than an opaque ``ImportError``.
    """
    if tokenizer is not None:
        return tokenizer
    try:
        from transformers import AutoProcessor
    except ImportError as e:  # pragma: no cover - exercised via load test when absent
        raise NotImplementedError(
            "FAST action detokenization requires 'transformers' (+ 'scipy'). "
            "Install them and let the processor pull the "
            f"'{DEFAULT_FAST_TOKENIZER}' checkpoint, or inject a tokenizer."
        ) from e
    return AutoProcessor.from_pretrained(path, trust_remote_code=True)


def normalize_to_unit(actions, lo, hi) -> np.ndarray:
    """Map physical actions into FAST's ``[-1, 1]`` input range via ``[lo, hi]``.

    ``lo``/``hi`` are per-dimension bounds (e.g. the checkpoint's q01/q99 action
    quantiles); scalars broadcast. Degenerate dims (``hi == lo``) map to 0 rather
    than dividing by zero. Output is clipped to ``[-1, 1]`` so out-of-range
    samples can't push FAST outside its trained domain.
    """
    arr = np.asarray(actions, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    span = hi - lo
    safe = np.where(span == 0.0, 1.0, span)  # avoid /0 on flat dims
    unit = 2.0 * (arr - lo) / safe - 1.0
    unit = np.where(span == 0.0, 0.0, unit)
    return np.clip(unit, -1.0, 1.0)


def unnormalize_from_unit(unit, lo, hi) -> np.ndarray:
    """Inverse of :func:`normalize_to_unit`: ``[-1, 1]`` → physical ``[lo, hi]``.

    The exact inverse for in-range values (clipping in the forward map is the
    only lossy step). Flat dims (``hi == lo``) return ``lo``.
    """
    u = np.asarray(unit, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return lo + (u + 1.0) / 2.0 * (hi - lo)


class FastActionCodec:
    """Encode/detokenize a **single** π0.5 action chunk with a FAST tokenizer.

    Wraps the FAST processor's batched API (``processor(actions)`` → tokens,
    ``processor.decode(tokens, time_horizon=, action_dim=)`` → actions) so callers
    work one chunk at a time — the codec adds/strips the leading batch axis itself,
    mirroring openpi's ``tokenize`` / ``extract_actions``. ``time_horizon`` and
    ``action_dim`` default the decode reshape (10×32 for ``pi05_libero``); either
    can be overridden per call.
    """

    def __init__(self, tokenizer: Any, *, time_horizon: int, action_dim: int) -> None:
        if time_horizon <= 0 or action_dim <= 0:
            raise ValueError("time_horizon and action_dim must be positive")
        self._tok = tokenizer
        self.time_horizon = int(time_horizon)
        self.action_dim = int(action_dim)

    def encode(self, actions) -> Any:
        """Tokenize one ``(time_horizon, action_dim)`` chunk (in ``[-1, 1]``).

        Returns the token sequence for that chunk (FAST yields one token list per
        batch item; the leading batch axis added here is stripped off the result).
        """
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 1:  # a single flat action -> horizon-1 chunk
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(
                f"expected a (time_horizon, action_dim) chunk, got shape {arr.shape}"
            )
        tokens = self._tok(arr[None])  # add batch axis -> list of length 1
        return tokens[0]

    def decode(self, tokens, *, time_horizon: int | None = None,
               action_dim: int | None = None) -> np.ndarray:
        """Detokenize one token sequence back to a ``(time_horizon, action_dim)`` chunk.

        Passes ``time_horizon``/``action_dim`` to the FAST decode (they're needed
        to un-flatten the variable-length token stream) and strips the batch axis.
        """
        th = int(time_horizon) if time_horizon is not None else self.time_horizon
        ad = int(action_dim) if action_dim is not None else self.action_dim
        toks = tokens.tolist() if hasattr(tokens, "tolist") else list(tokens)
        actions = self._tok.decode([toks], time_horizon=th, action_dim=ad)
        return np.asarray(actions[0], dtype=np.float32).reshape(th, ad)
