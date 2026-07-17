#!/usr/bin/env python3
"""Best-of-N GR00T policy service (R2 rung) — drop-in for the HTTP bridge.

Replaces the ZMQ-server + HTTP-bridge pair on the GPU host with ONE in-process
service that, per control query:

1. encodes the observation ONCE (backbone cached; the expensive part),
2. samples K initial noises = steering mean (``init_noise`` from the sidecar,
   if provided) + ``sigma`` * N(0,1) jitter — candidate 0 is ALWAYS the pure
   steering mean, so K=1/sigma=0 reproduces the Stage-B behavior exactly,
3. decodes all K in ONE batched flow pass (broadcast cache),
4. asks the FK sidecar service for pinch trajectories,
5. selects via bestofn_select: CBF filters (vial-protection, clearance, motion
   caps), CLF ranks (predicate-gated Lyapunov progress), HOLD fallback,
6. returns the chosen chunk in the exact bridge response contract, plus a
   ``bestofn`` provenance report (the per-decision safety certificate).

GR00T's frozen flow head remains the sole source of actuator values — every
candidate is its own decode; selection among its own proposals preserves the
authenticity invariant.

Request extras over the bridge contract (all optional; absent -> stock K=1):
  init_noise_b64/init_noise_shape  — steering mean (sidecar attaches, Stage B)
  bestofn: {k, sigma, phase, grasped, vial, pocket}   — sidecar attaches

Run (GR00T venv, next to the FK service):
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ~/Isaac-GR00T/.venv/bin/python \
    serve_groot_bestofn.py --model-path <ckpt> --http-port 5596 --fk-url http://127.0.0.1:5560
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_INSTRUCTION = "pick up the vial and place it in the rack"
REAL_STEPS, REAL_DIMS = 16, 7


class BestOfNService:
    def __init__(self, model_path: str, embodiment_tag: str, device: str,
                 fk_url: str, steering_net: str | None = None) -> None:
        import torch  # noqa: F401  (ensures torch import errors surface early)
        from probe_flow_inversion_groot import GrootFlow, forward_euler_sample
        from flow_inverter_groot import pool_backbone
        import bestofn_select as bon
        import cbf_constraints_ur5e as cbf

        self.flow = GrootFlow(model_path, embodiment_tag, device=device)
        self._sample = forward_euler_sample
        self._pool = pool_backbone
        self.bon = bon
        self.cbf = cbf
        # v4 image-conditioned steering head: mean = head(pooled feats + low15).
        # Lives HERE because the pooled features exist per-query in-process —
        # the exact pool_backbone used at training (any mismatch re-creates the
        # target-noise floor).
        self._steer_forward = None
        self._steer_meta = {}
        if steering_net:
            from flowdagger_offline_gate_ur5e import load_steering_net
            self._steer_forward, self._steer_meta = load_steering_net(steering_net)
            print(f"[bestofn] v4 steering head loaded: in={self._steer_meta.get('in_dim')} "
                  f"out={self._steer_meta.get('out_dim')}", flush=True)
        self.fk_url = fk_url.rstrip("/")
        self.device = device
        self._lock = threading.Lock()   # one query at a time (GPU + FK)
        self.n_queries = 0
        self.n_fallback_hold = 0
        self.rejection_hist: dict[str, int] = {}   # aggregated CBF rejections
        print(f"[bestofn] ready | ckpt={model_path} fk={self.fk_url}", flush=True)

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _decode_frame(b64: str) -> np.ndarray:
        from PIL import Image
        raw = str(b64).split(",")[-1]
        im = Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
        if im.size != (256, 256):
            im = im.resize((256, 256))
        return np.asarray(im, dtype=np.uint8)

    def _fk_batch(self, qs: np.ndarray) -> np.ndarray:
        body = json.dumps({"q": np.asarray(qs, float).reshape(-1, 6).tolist()}).encode()
        req = urllib.request.Request(f"{self.fk_url}/fk", data=body,
                                     headers={"content-type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return np.asarray(json.loads(r.read().decode())["pinch"], float)

    # -- one query -------------------------------------------------------------
    def get_action(self, req: dict) -> dict:
        import torch

        with self._lock:
            ext = self._decode_frame(req["image_b64"])
            wrist = self._decode_frame(req.get("image_b64_wrist") or req["image_b64"])
            state = np.asarray(req["state"], dtype=np.float32).reshape(-1)
            state10 = np.zeros(10, dtype=np.float32)
            state10[:min(10, len(state))] = state[:10]
            instr = req.get("instruction") or DEFAULT_INSTRUCTION
            cfg = req.get("bestofn") or {}
            K = int(cfg.get("k", 1))
            sigma = float(cfg.get("sigma", 0.0))

            cache, _x, _m = self.flow.encode(ext, wrist, state10, instruction=instr)

            mean = None
            steer_head_used = False
            if self._steer_forward is not None and req.get("steer_low15"):
                low = np.asarray(req["steer_low15"], dtype=np.float32).reshape(-1)
                pooled = self._pool(cache)[0].astype(np.float32)
                x = np.concatenate([low, pooled])[None]
                if x.shape[1] == int(self._steer_meta.get("in_dim", -1)):
                    mean = self._steer_forward(x).reshape(
                        self.flow.action_horizon, self.flow.action_dim
                    ).astype(np.float32)
                    steer_head_used = True
                else:
                    print(f"[bestofn] steer_low15 dim mismatch "
                          f"{x.shape[1]} vs {self._steer_meta.get('in_dim')}", flush=True)
            if mean is None and req.get("init_noise_b64"):
                shape = tuple(req.get("init_noise_shape") or ())
                mean = np.frombuffer(base64.b64decode(req["init_noise_b64"]),
                                     dtype=np.float32).reshape(shape)
            H, D = self.flow.action_horizon, self.flow.action_dim
            if mean is None:
                mean = np.random.default_rng().standard_normal((H, D)).astype(np.float32)
            seeds = np.repeat(mean[None], K, axis=0).astype(np.float32)
            if K > 1 and sigma > 0:
                rng = np.random.default_rng()
                seeds[1:] += (sigma * rng.standard_normal((K - 1, H, D))).astype(np.float32)

            vfn = self.flow.velocity_fn(self.flow.broadcast_cache(cache, K))
            w = torch.as_tensor(seeds, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                x = self._sample(w, vfn, self.flow.n_steps, self.flow.dt)
            phys = self.flow.decode_to_physical(x)
            arm = np.asarray(phys["single_arm"], dtype=np.float32)[:, :REAL_STEPS, :]   # (K,16,6)
            grip = np.asarray(phys["gripper"], dtype=np.float32)[:, :REAL_STEPS, 0]     # (K,16)

            # Per-request shaping config (powered runs pin these explicitly)
            import clf_reward_ur5e as clfmod
            clfmod.CENTERING_MODE = str(cfg.get("clf_centering", "graded"))
            lim = self.cbf.Limits()
            if bool(cfg.get("cbf_flat_vial_cap", False)):
                lim.vial_step_floor = lim.vial_near_step_max   # flat (run-1) cap

            def _select(arm_c, grip_c):
                Kc = len(arm_c)
                pinch_flat = self._fk_batch(arm_c.reshape(-1, 6)).reshape(Kc, REAL_STEPS, 3)
                fk_lut = {}
                for kk in range(Kc):
                    for t in range(REAL_STEPS):
                        fk_lut[arm_c[kk, t].tobytes()] = pinch_flat[kk, t]
                return self.bon.select(
                    arm_c, grip_c,
                    lambda q6: fk_lut[np.asarray(q6, np.float32).tobytes()],
                    vial=cfg.get("vial") or state10[7:10],
                    grasp_target=cfg.get("grasp_target") or state10[7:10],
                    pocket=cfg.get("pocket"),
                    phase=int(cfg.get("phase", 0)),
                    grasped=bool(cfg.get("grasped", False)),
                    exec_horizon=int(cfg.get("exec_horizon", 8)),
                    lim=lim,
                )

            chosen, report = 0, {"k": K, "selection": "single"}
            if K > 1:
                idx, report = _select(arm, grip)
                # CEM refinement at GRASP phase: resample around round-1's
                # winner at sigma/2, decode, re-select; keep the better round.
                cem_rounds = int(cfg.get("cem_rounds", 1))
                if (idx is not None and cem_rounds > 1
                        and int(cfg.get("phase", 0)) == 1):
                    r1_reward = (report.get("chosen_clf") or {}).get("total_reward", 0.0)
                    seeds2 = self.bon.make_cem_seeds(seeds[idx], K, sigma * 0.5)
                    w2 = torch.as_tensor(seeds2, dtype=torch.float32, device=self.device)
                    with torch.no_grad():
                        x2 = self._sample(w2, vfn, self.flow.n_steps, self.flow.dt)
                    phys2 = self.flow.decode_to_physical(x2)
                    arm2 = np.asarray(phys2["single_arm"], dtype=np.float32)[:, :REAL_STEPS, :]
                    grip2 = np.asarray(phys2["gripper"], dtype=np.float32)[:, :REAL_STEPS, 0]
                    idx2, rep2 = _select(arm2, grip2)
                    r2_reward = (rep2.get("chosen_clf") or {}).get("total_reward", -1e9) \
                        if idx2 is not None else -1e9
                    if idx2 is not None and r2_reward > r1_reward:
                        arm, grip, idx, report = arm2, grip2, idx2, rep2
                        report["cem"] = {"rounds": 2, "used_round": 2,
                                         "r1_reward": r1_reward, "r2_reward": r2_reward}
                    else:
                        report["cem"] = {"rounds": 2, "used_round": 1,
                                         "r1_reward": r1_reward, "r2_reward": r2_reward}
                for name, cnt in (report.get("cbf_rejections") or {}).items():
                    self.rejection_hist[name] = self.rejection_hist.get(name, 0) + int(cnt)
                if self.n_queries % 25 == 0:
                    print(f"[bestofn] q={self.n_queries} holds={self.n_fallback_hold} "
                          f"rejections={self.rejection_hist}", flush=True)
                if idx is None:
                    self.n_fallback_hold += 1
                    q_now = state10[:6]
                    arm_h, grip_h = self.bon.hold_chunk(q_now, float(state10[6]),
                                                        horizon=REAL_STEPS)
                    arm = arm_h[None]
                    grip = grip_h[None]
                    chosen = 0
                else:
                    chosen = int(idx)

            self.n_queries += 1
            return {
                "q": [float(v) for v in arm[chosen, 0]],
                "grip": float(np.clip(grip[chosen, 0], 0.0, 1.0)),
                "chunk_q": [[float(v) for v in row] for row in arm[chosen]],
                "chunk_grip": [float(np.clip(g, 0.0, 1.0)) for g in grip[chosen]],
                "bestofn": report,
                "steer_head": ("v4" if steer_head_used else None),
            }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--http-host", default="127.0.0.1")
    ap.add_argument("--http-port", type=int, default=5596)
    ap.add_argument("--fk-url", default="http://127.0.0.1:5560")
    ap.add_argument("--steering-net", default=None,
                    help="v4 image-conditioned steering head .npz (server-side mean)")
    args = ap.parse_args()

    svc = BestOfNService(args.model_path, args.embodiment_tag, args.device,
                         args.fk_url, steering_net=args.steering_net)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/health"):
                self._send(200, {"ok": True, "bestofn": True,
                                 "queries": svc.n_queries,
                                 "fallback_holds": svc.n_fallback_hold})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/get_action"):
                self._send(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("content-length", 0) or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                self._send(200, svc.get_action(req))
            except Exception as exc:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                self._send(500, {"error": repr(exc)})

    print(f"[bestofn] http on {args.http_host}:{args.http_port}", flush=True)
    ThreadingHTTPServer((args.http_host, args.http_port), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
