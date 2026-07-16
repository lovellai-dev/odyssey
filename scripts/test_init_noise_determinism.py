#!/usr/bin/env python3
"""A1 determinism test for the FlowDAgger ``init_noise`` patch (over the wire).

Proves the ``options["init_noise"]`` latent-steering hook survives the FULL client
path — msgpack/ZMQ request -> ``PolicyServer`` handler -> ``BasePolicy.get_action``
-> ``Gr00tPolicy._get_action`` -> ``model.get_action(..., options)`` ->
``get_action_with_features`` — and controls the flow sampler deterministically.

Design (zero new deps: numpy / msgpack / msgpack_numpy / pyzmq only — nothing is
pip-installed into the shared GR00T venv). We drive a REAL running GR00T server
(``python -m gr00t.eval.run_gr00t_server``) with a raw msgpack REQ client, exactly
like ``scripts/eval_ur5e_drugsort_groot.py``'s ``GrootPolicyClient``.

Assertions:
  (a) same ``init_noise`` sent twice  -> BYTE-IDENTICAL action chunks
      (random sampling could never do this; therefore the dict reached the sampler)
  (b) two DIFFERENT ``init_noise``    -> different action chunks
  (c) ``options=None``                -> still returns a valid chunk (back-compat)
  (+) steered (init_noise) chunk differs from a stock ``options=None`` draw
      (the injected noise actually changed the trajectory)

Exit 0 + prints ``INIT_NOISE_DETERMINISM PASS`` / a JSON blob on success.
"""
from __future__ import annotations

import argparse
import json
import sys

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

# Padded action tensor geometry for this checkpoint (established facts).
ACTION_HORIZON = 40
ACTION_DIM = 132
INSTRUCTION = "pick up the vial and place it in the rack"


class RawClient:
    """Minimal msgpack/ZMQ REQ client (mirrors eval GrootPolicyClient)."""

    def __init__(self, host: str, port: int, timeout_ms: int = 180000) -> None:
        self._ctx = zmq.Context.instance()
        self._host, self._port, self._timeout_ms = host, port, timeout_ms
        self._connect()

    def _connect(self) -> None:
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._sock.connect(f"tcp://{self._host}:{self._port}")

    def _call(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        req: dict = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data
        self._sock.send(msgpack.packb(req, default=mnp.encode))
        msg = self._sock.recv()
        resp = msgpack.unpackb(msg, object_hook=mnp.decode, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"GR00T server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self._call("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._connect()
            return False

    def get_action(self, observation: dict, options: dict | None) -> dict:
        action, _info = self._call(
            "get_action", {"observation": observation, "options": options}
        )
        return action


def load_state10(dataset: str, episode: int, frame: int) -> np.ndarray:
    """Pull one real 10-D observation.state row (pyarrow only, no video decode)."""
    import pyarrow.parquet as pq
    from pathlib import Path

    pf = Path(dataset) / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    tbl = pq.read_table(pf, columns=["observation.state"])
    return np.asarray(tbl.column("observation.state")[frame].as_py(), dtype=np.float32)


def build_obs(state10: np.ndarray) -> dict:
    """Nested B=1,T=1 observation for the obscond checkpoint (10-D state).

    Images are zeros — a determinism test does not depend on pixel content, only
    on strict-validation shapes/dtypes (avoids an imageio/ffmpeg dependency).
    """
    s = np.asarray(state10, dtype=np.float32).reshape(-1)
    zeros_img = np.zeros((1, 1, 256, 256, 3), dtype=np.uint8)
    return {
        "video": {"exterior": zeros_img, "wrist": zeros_img.copy()},
        "state": {
            "single_arm": s[:6].reshape(1, 1, 6).astype(np.float32),
            "gripper": s[6:7].reshape(1, 1, 1).astype(np.float32),
            "grasp_target": s[7:10].reshape(1, 1, 3).astype(np.float32),
        },
        "language": {"annotation.human.task_description": [[INSTRUCTION]]},
    }


def _chunk_bytes(action: dict) -> bytes:
    arm = np.asarray(action["single_arm"], dtype=np.float32)
    grip = np.asarray(action["gripper"], dtype=np.float32)
    return arm.tobytes() + grip.tobytes()


def _valid_chunk(action: dict) -> bool:
    try:
        arm = np.asarray(action["single_arm"], dtype=np.float32)
        grip = np.asarray(action["gripper"], dtype=np.float32)
    except Exception:
        return False
    return arm.ndim >= 2 and arm.shape[-1] == 6 and grip.shape[-1] == 1 and np.isfinite(arm).all()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5561)
    ap.add_argument("--dataset", default="/home/ubuntu/ur5e_drugsort_obscond")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    client = RawClient(args.host, args.port)
    assert client.ping(), "GR00T server not answering ping"

    obs = build_obs(load_state10(args.dataset, args.episode, args.frame))

    rng = np.random.default_rng(0)
    noise_a = rng.standard_normal((1, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
    noise_b = rng.standard_normal((1, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)

    act_a1 = client.get_action(obs, {"init_noise": noise_a})
    act_a2 = client.get_action(obs, {"init_noise": noise_a})
    act_b = client.get_action(obs, {"init_noise": noise_b})
    act_none = client.get_action(obs, None)          # (c) back-compat: no options
    act_none2 = client.get_action(obs, None)          # stock draws differ (random)

    ba1, ba2, bb = _chunk_bytes(act_a1), _chunk_bytes(act_a2), _chunk_bytes(act_b)
    b_none, b_none2 = _chunk_bytes(act_none), _chunk_bytes(act_none2)

    same_noise_identical = (ba1 == ba2)          # (a)
    diff_noise_differ = (ba1 != bb)               # (b)
    none_works = _valid_chunk(act_none)            # (c)
    steered_vs_stock_differ = (ba1 != b_none)     # injected noise changed the output
    stock_is_random = (b_none != b_none2)          # sanity: stock path is not frozen

    # max |Δ| diagnostics
    def _maxabs(x, y):
        ax = np.frombuffer(x, dtype=np.float32)
        ay = np.frombuffer(y, dtype=np.float32)
        n = min(ax.size, ay.size)
        return float(np.abs(ax[:n] - ay[:n]).max()) if n else float("nan")

    determinism_pass = bool(same_noise_identical and diff_noise_differ and none_works)
    wire_options_pass = bool(same_noise_identical and steered_vs_stock_differ)

    result = {
        "same_init_noise_byte_identical": bool(same_noise_identical),
        "different_init_noise_differ": bool(diff_noise_differ),
        "no_options_works": bool(none_works),
        "steered_differs_from_stock": bool(steered_vs_stock_differ),
        "stock_path_is_random": bool(stock_is_random),
        "maxabs_same_noise": _maxabs(ba1, ba2),
        "maxabs_diff_noise": _maxabs(ba1, bb),
        "maxabs_steered_vs_stock": _maxabs(ba1, b_none),
        "determinism_pass": determinism_pass,
        "wire_options_pass": wire_options_pass,
    }
    text = json.dumps(result, indent=2)
    print(text, flush=True)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(text)
    if determinism_pass and wire_options_pass:
        print("INIT_NOISE_DETERMINISM PASS", flush=True)
        return 0
    print("INIT_NOISE_DETERMINISM FAIL", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
