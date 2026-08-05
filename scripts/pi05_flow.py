"""pi05_flow.py — π0.5 (openpi) flow wrapper for FlowDAgger noise steering.

Analog of ``GrootFlow`` (``probe_flow_inversion_groot.py``) for the π0.5 pilot. Wraps a
frozen openpi π0.5 ``Policy`` + the vendored ``FlowMatchingInverter``
(``vendor/flowdagger_pi05/``, from microsoft/FlowDAgger, MIT) so the SAME numpy steering
pipeline (``train_steering_ur5e`` / ``flowdagger_offline_gate_ur5e`` / ``serve_pi05_bestofn``)
works on π0.5 exactly as it does on GR00T.

Key differences vs GR00T that this wrapper hides:
  * Flow-matching convention is openpi/pi0's (``x_{t+dt}=x_t+dt*v``, dt=-1/N, t: 1→0 = noise
    at t=1, clean at t=0) — the INVERSE time direction of GR00T's sampler. So invert/decode
    ALWAYS go through this wrapper; never mix with GrootFlow's ``forward_euler_sample``.
  * The action tensor is (action_horizon, action_dim) with ``action_dim=32`` zero-padded; the
    REAL UR10e dims are the first ``REAL_DIMS=7`` (6 arm joints + 1 gripper). There is no extra
    horizon padding (unlike GR00T's 40×132), so ``pad_horizon == action_horizon``.
  * Observations and the normalized+padded action target are produced by openpi's OWN input
    transforms (repack → LiberoInputs → DeltaActions → Normalize → model transforms), loaded
    from the checkpoint via ``create_trained_policy`` — so target actions live in "pi0 internal
    space", which is exactly what ``FlowMatchingInverter.invert`` expects.

Runs inside openpi's venv (jax). Shares device arrays with the policy → inversion adds no
extra GPU memory. Heavy imports are done lazily in ``__init__`` so the module stays importable
(for constant introspection) in a jax-less shell.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Put the vendored microsoft inverter on the path (sibling ``vendor/`` dir).
_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE / "vendor" / "flowdagger_pi05"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

# UR10e real action dims: 6 arm joints + 1 gripper. The remaining action_dim-REAL_DIMS
# channels are openpi zero-pad (steering fills them with fixed-seed noise at deploy).
REAL_DIMS = 7
DEFAULT_INSTRUCTION = "pick up the vial and place it in the rack"


class Pi05Flow:
    """Frozen π0.5 policy + flow inverter, exposing the ops the steering pipeline needs.

    Public API mirrors what ``flow_inverter_pi05`` / the gate / the server call:
      * ``process_frame(...)``  → (observation, target_internal|None)
      * ``invert(obs, target)`` → (w_star (H,D) numpy, recon_mse float)
      * ``decode_internal(obs, noise)`` → internal actions (H,D) numpy (for recon checks)
      * ``decode_physical(obs, noise)`` → physical action chunk (H, REAL_DIMS) numpy
      * ``pool_backbone(obs)`` → pooled prefix features (D_embed,) numpy  [v4 conditioning]
    """

    def __init__(
        self,
        config_name: str,
        checkpoint_dir: str | Path,
        *,
        num_denoise_steps: int = 10,
        method: str = "perstep_fp",
        fp_per_step: int = 5,
        default_prompt: str | None = DEFAULT_INSTRUCTION,
        seed: int = 0,
    ) -> None:
        import jax
        import jax.numpy as jnp
        from flax import nnx

        import openpi.models.model as _model
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        self._jax = jax
        self._jnp = jnp
        self._model_mod = _model

        train_config = _config.get_config(config_name)
        # The repack transform (LeRobot column names → openpi ``observation/*`` keys) lives on
        # the data config, NOT in ``create_trained_policy``'s default input pipeline — pass it
        # explicitly so raw ``observation.images.*`` frames are accepted exactly as in training.
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        # Loads the raw pi0 model + the exact train-time transforms + checkpoint norm stats.
        self.policy = _policy_config.create_trained_policy(
            train_config, str(checkpoint_dir), default_prompt=default_prompt,
            repack_transforms=data_config.repack_transforms,
        )
        self.model = self.policy._model
        self.action_horizon = int(self.model.action_horizon)
        self.action_dim = int(self.model.action_dim)
        self.real_dims = REAL_DIMS

        from flow_matching_inverter import FlowMatchingInverter  # vendored

        self.inverter = FlowMatchingInverter(
            self.model,
            method=method,
            num_denoise_steps=num_denoise_steps,
            fp_per_step=fp_per_step,
            seed=seed,
        )

        # A small jitted prefix-pooling fn for v4 image-conditioned features. Uses its own
        # split of the SAME model (shares device arrays, no extra memory).
        graphdef, state = nnx.split(self.model)
        self._pool_state = state

        def _pool_fn(st: Any, observation: Any) -> Any:
            model = nnx.merge(graphdef, st)
            observation = _model.preprocess_observation(None, observation, train=False)
            prefix_tokens, prefix_mask, _ = model.embed_prefix(observation)
            mask = prefix_mask[..., None].astype(prefix_tokens.dtype)
            summed = jnp.sum(prefix_tokens * mask, axis=1)
            count = jnp.clip(jnp.sum(mask, axis=1), a_min=1.0, a_max=None)
            return summed / count  # (B, D_embed)

        self._pool_jit = jax.jit(_pool_fn)

    # ------------------------------------------------------------------ obs / target
    def process_frame(
        self,
        *,
        exterior: np.ndarray,
        wrist: np.ndarray,
        state: np.ndarray,
        prompt: str | None = None,
        action_chunk: np.ndarray | None = None,
    ) -> tuple[Any, Any]:
        """Raw frame → (batched Observation, target_internal (1,H,D)|None).

        ``exterior``/``wrist``: uint8 HxWx3. ``state``: raw proprio (unnormalized).
        ``action_chunk``: (H, REAL_DIMS) ABSOLUTE expert actions, or None. When provided it is
        pushed through the SAME openpi input transforms as the observation, so the returned
        ``target_internal`` is normalized + padded to (1, action_horizon, action_dim) — exactly
        what ``FlowMatchingInverter.invert`` consumes.
        """
        raw: dict[str, Any] = {
            "observation.images.exterior": np.asarray(exterior),
            "observation.images.wrist": np.asarray(wrist),
            "observation.state": np.asarray(state, dtype=np.float32),
        }
        if prompt is not None:
            raw["prompt"] = prompt
        have_action = action_chunk is not None
        # The repack transform maps ``actions <- action``, so the ``action`` key MUST exist
        # even on the obs-only path (features / serving). Inject a zero dummy when there is no
        # expert chunk — the observation (prefix: images+state+prompt) is independent of the
        # action value, so pooled features are unaffected; we just don't read a target back.
        raw["action"] = (np.asarray(action_chunk, dtype=np.float32) if have_action
                         else np.zeros((self.action_horizon, self.real_dims), np.float32))

        inputs = self.policy._input_transform(raw)
        obs = self._observation_from_inputs(inputs)
        target_internal = None
        if have_action:
            target_internal = self._jnp.asarray(inputs["actions"])[np.newaxis, ...]
        return obs, target_internal

    def _observation_from_inputs(self, inputs: dict[str, Any]) -> Any:
        jnp = self._jnp
        batched = self._jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return self._model_mod.Observation.from_dict(batched)

    # ------------------------------------------------------------------ invert / decode
    def invert(self, observation: Any, target_internal: Any) -> tuple[np.ndarray, float]:
        """Recover noise w* s.t. denoise(obs, w*) ≈ target_internal. Returns (w* (H,D), mse)."""
        noise, err = self.inverter.invert(observation, target_internal)
        return np.asarray(noise[0]), float(np.asarray(err).reshape(-1)[0])

    def decode_internal(self, observation: Any, noise: np.ndarray) -> np.ndarray:
        """Forward denoise a noise tensor → internal (normalized, padded) action chunk (H,D)."""
        jnp = self._jnp
        w = jnp.asarray(noise)
        if w.ndim == 2:
            w = w[jnp.newaxis, ...]
        out = self.inverter._denoise(observation, w)
        return np.asarray(out[0])

    def decode_physical(self, observation: Any, noise: np.ndarray) -> np.ndarray:
        """Decode a noise tensor all the way to a PHYSICAL action chunk (H, REAL_DIMS).

        Runs the internal decode then openpi's output transforms (Unnormalize + AbsoluteActions
        + un-pad), matching what the served policy would emit. The state fed to the output
        transform is the SAME normalized state carried on the observation (as in Policy.infer,
        where ``outputs["state"] = inputs["state"]``).
        """
        internal = self.decode_internal(observation, noise)  # (H, action_dim)
        state = np.asarray(observation.state[0], dtype=np.float32)  # normalized, padded
        outputs = {"state": state, "actions": internal}
        physical = self.policy._output_transform(outputs)["actions"]
        return np.asarray(physical, dtype=np.float32)

    # ------------------------------------------------------------------ v4 features
    def pool_backbone(self, observation: Any) -> np.ndarray:
        """Masked-mean of the π0.5 prefix (image+language) embeddings → (D_embed,) numpy.

        INVARIANT: the SAME pooling must be used at invert / gate / serve — any mismatch
        recreates the target-noise floor the steering head exists to break.
        """
        pooled = self._pool_jit(self._pool_state, observation)
        return np.asarray(pooled[0], dtype=np.float32)

    # ------------------------------------------------------------------ real-dim helpers
    def real_slice(self, chunk_hd: np.ndarray) -> np.ndarray:
        """(H, action_dim) → (H, REAL_DIMS): the meaningful UR10e action channels."""
        return np.asarray(chunk_hd)[: self.action_horizon, : self.real_dims]
