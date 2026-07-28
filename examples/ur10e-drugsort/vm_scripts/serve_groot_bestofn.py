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
import os
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
        # Render-gap DIAGNOSTIC (opt-in, STEER_DIAG=1): for the first N queries,
        # dump the received BROWSER frames + state + the pure head-mean chunk's
        # decoded pinch-vs-target. Lets an offline pass replay the SAME states on
        # DATASET frames and confirm whether the head-mean reaches offline but
        # parks online (the v4-head online-park hypothesis). Zero effect when off.
        self.diag = os.environ.get("STEER_DIAG", "0") == "1"
        self.diag_path = os.environ.get("STEER_DIAG_PATH", "/home/ubuntu/steer_diag.jsonl")
        self.diag_frames = os.environ.get("STEER_DIAG_FRAMES", "/home/ubuntu/steer_diag_frames")
        self.diag_max = int(os.environ.get("STEER_DIAG_MAX", "40"))
        self.diag_n = 0
        if self.diag:
            import os as _os
            _os.makedirs(self.diag_frames, exist_ok=True)
            print(f"[bestofn] STEER_DIAG on -> {self.diag_path} (max {self.diag_max})", flush=True)
        self.n_queries = 0
        self.n_fallback_hold = 0
        self.rejection_hist: dict[str, int] = {}   # aggregated CBF rejections
        # L5 lever — bounded final-centimeter visual servo (STEER_SERVO=1, opt-in).
        # OFF => the V4 selection path is byte-identical (no "servo" key emitted).
        self.servo_enabled = os.environ.get("STEER_SERVO", "0") == "1"
        # Observer->IK HANDOFF (STEER_HANDOFF=1): full-authority final-cm capture.
        # Stronger than the bounded servo — hands the sub-cm step to the IK oracle.
        self.handoff_enabled = os.environ.get("STEER_HANDOFF", "0") == "1"
        self.servo = None
        self.servo_kwargs: dict = {}
        self.n_servo_applied = 0
        self.handoff_kwargs: dict = {}
        self.n_handoff_applied = 0
        self.n_handoff_retries = 0
        self.handoff_smooth = 1
        self.handoff_covmax = float("inf")
        self.handoff_nearwin = False
        self.n_cov_rejected = 0
        # Per-episode handoff retry state, keyed by the harness's per-episode sid.
        self._ho_states: dict[str, dict] = {}
        self._brain: dict = {}
        self._brain_hist: list = []
        if self.servo_enabled or self.handoff_enabled:
            import servo_ur5e as servo
            self.servo = servo
        if self.servo_enabled:
            self.servo_kwargs = dict(
                radius=float(os.environ.get("STEER_SERVO_RADIUS", "0.08")),
                gain=float(os.environ.get("STEER_SERVO_GAIN", "0.8")),
                damping=float(os.environ.get("STEER_SERVO_DAMPING", "0.08")),
                bound=float(os.environ.get("STEER_SERVO_BOUND", "0.08")),
                max_dominance=float(os.environ.get("STEER_SERVO_DOMINANCE", "0.5")),
            )
            print(f"[bestofn] L5 servo ENABLED | {self.servo_kwargs}", flush=True)
        if self.handoff_enabled:
            self.handoff_kwargs = dict(
                capture_zone=float(os.environ.get("STEER_HANDOFF_ZONE", "0.10")),
                descend_offset=float(os.environ.get("STEER_HANDOFF_DESCEND", "0.05")),
                grasp_tol=float(os.environ.get("STEER_HANDOFF_GRASPTOL", "0.015")),
                move_cap=float(os.environ.get("STEER_HANDOFF_MOVECAP", "0.4")),
                xy_from_groot=os.environ.get("STEER_HANDOFF_XY", "groot") == "groot",
            )
            _gz = os.environ.get("STEER_HANDOFF_GRASPZ")   # optional fixed FK-frame grasp height
            if _gz:
                self.handoff_kwargs["grasp_z"] = float(_gz)
            # Window (in queries) of the running-median Observer filter; 1 = off.
            self.handoff_smooth = int(os.environ.get("STEER_HANDOFF_SMOOTH", "15"))
            self.handoff_covmax = float(os.environ.get("STEER_HANDOFF_COVMAX", "2.0"))
            self.handoff_nearwin = os.environ.get("STEER_HANDOFF_NEARWIN", "1") == "1"
            self.handoff_kwargs.update(
                verify_queries=int(os.environ.get("STEER_HANDOFF_VERIFY", "3")),
                max_attempts=int(os.environ.get("STEER_HANDOFF_ATTEMPTS", "4")),
                retract=float(os.environ.get("STEER_HANDOFF_RETRACT", "0.06")),
                retry_jitter=float(os.environ.get("STEER_HANDOFF_JITTER", "0.012")),
                place_z_hi=float(os.environ.get("STEER_PLACE_Z_HI", "0.34")),
                place_z_lo=float(os.environ.get("STEER_PLACE_Z_LO", "0.28")),
                place_tol=float(os.environ.get("STEER_PLACE_TOL", "0.02")),
                descend_step=(float(os.environ["STEER_HANDOFF_DESCEND_STEP"])
                              if os.environ.get("STEER_HANDOFF_DESCEND_STEP") else None),
                close_ramp=os.environ.get("STEER_HANDOFF_CLOSE_RAMP", "0") == "1",
                approach_budget=(int(os.environ["STEER_HANDOFF_APPROACH_BUDGET"])
                                 if os.environ.get("STEER_HANDOFF_APPROACH_BUDGET") else None),
                seat_queries=int(os.environ.get("STEER_HANDOFF_SEAT", "0")),
                rise_step=(float(os.environ["STEER_HANDOFF_RISE_STEP"])
                           if os.environ.get("STEER_HANDOFF_RISE_STEP") else None),
                transit_step=(float(os.environ["STEER_HANDOFF_TRANSIT_STEP"])
                              if os.environ.get("STEER_HANDOFF_TRANSIT_STEP") else None),
                descend_latch=os.environ.get("STEER_HANDOFF_DESCEND_LATCH", "0") == "1",
                drop_retry=os.environ.get("STEER_HANDOFF_DROP_RETRY", "0") == "1",
                carry_mode=os.environ.get("STEER_HANDOFF_CARRY", "owned"),
                close_commit_through=os.environ.get("STEER_HANDOFF_COMMIT_THROUGH", "0") == "1",
            )
            print(f"[bestofn] Observer->IK HANDOFF (grasp-aware) ENABLED | {self.handoff_kwargs}", flush=True)
        print(f"[bestofn] ready | ckpt={model_path} fk={self.fk_url} "
              f"servo={'on' if self.servo_enabled else 'off'} "
              f"handoff={'on' if self.handoff_enabled else 'off'}", flush=True)

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

    _PHASES = ("reach", "grasp", "transport", "place")
    _IK_STAGES = {"align", "settle", "descend", "close_hold", "lift_test",
                  "retract", "grasp_failed_retry", "grasp_failed_exhausted"}

    def _update_brain(self, req, cfg, tgt, ho_state, tgt_settled,
                      handoff_report, pre_arm0, state10) -> None:
        """One JSON snapshot per query for the live demo dashboard."""
        hr = handoff_report or {}
        stage = hr.get("stage")
        if hr.get("applied") and stage in self._IK_STAGES:
            authority = "ik"
        elif stage == "carry_hold":
            authority = "pilot"        # VLA carries; arbiter only pins the grip
        else:
            authority = "pilot"
        try:
            pp = self._fk_batch(pre_arm0[None])[0]
            pinch_proposal = [round(float(v), 4) for v in np.asarray(pp).reshape(-1)[:3]]
        except Exception:
            pinch_proposal = None
        pm = hr.get("pinch")
        phase = int(cfg.get("phase", 0))
        dxy = hr.get("pinch_err_before_cm")
        snap = {
            "q": int(self.n_queries),
            "sid": req.get("sid"),
            "phase": phase,
            "phase_name": self._PHASES[phase] if 0 <= phase < 4 else "?",
            "grasped": bool(cfg.get("grasped", False)),
            "observer": {
                "target": ([round(float(v), 4) for v in np.asarray(tgt).reshape(-1)[:3]]
                           if tgt is not None else None),
                "cov_cm": cfg.get("frame_cov_cm"),
                "reads": len((ho_state or {}).get("tgt_buf", [])),
                "settled": bool(tgt_settled),
            },
            "pilot": {
                "pinch_proposal": pinch_proposal,
                "pinch_measured": pm,
                "in_zone": (dxy is not None and
                            dxy <= self.handoff_kwargs.get("capture_zone", 0.10) * 100.0),
                "k": int(cfg.get("k", 1)),
            },
            "arbiter": {
                "stage": stage,
                "authority": authority,
                "attempt": int(hr.get("attempt") or 0),
                "dxy_cm": dxy,
                "d_gp_cm": hr.get("d_gp_cm"),
                "cbf_scale": hr.get("cbf_scale"),
                "retries_total": int(self.n_handoff_retries),
                "handoff_applied_total": int(self.n_handoff_applied),
                "commit_latched": bool(ho_state and "commit_gp" in ho_state),
            },
        }
        self._brain_hist.append({"q": snap["q"], "stage": stage,
                                 "authority": authority, "dxy": dxy})
        if len(self._brain_hist) > 60:
            del self._brain_hist[0]
        snap["history"] = list(self._brain_hist)
        self._brain = snap

    def _handoff_state(self, sid: Any) -> dict:
        """Retry state for one episode. The harness sends a fresh sid per episode,
        so a new key == a new episode; keep the map bounded for long sessions."""
        key = str(sid or "default")
        st = self._ho_states.get(key)
        if st is None:
            if len(self._ho_states) > 256:
                self._ho_states.clear()
            st = {}
            self._ho_states[key] = st
        return st

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

            # Render-gap diagnostic: capture the PURE head-mean chunk (arm[0] =
            # decoded undithered mean) pinch-vs-target + the browser frames, so an
            # offline pass can replay the same states on dataset frames.
            if self.diag and self.diag_n < self.diag_max and steer_head_used:
                # Diagnostics must NEVER break serving — swallow any error.
                try:
                    import base64 as _b64
                    i = self.diag_n
                    mean_pinch = np.asarray(self._fk_batch(arm[0]), float).reshape(-1, 3)  # (16,3)
                    tgt_arr = np.asarray(cfg.get("grasp_target") or cfg.get("vial")
                                         or state10[7:10], float).reshape(3)
                    dd = np.linalg.norm(mean_pinch - tgt_arr[None], axis=1)
                    for tag, b64 in (("ext", req["image_b64"]),
                                     ("wrist", req.get("image_b64_wrist") or req["image_b64"])):
                        # Browser frames arrive as data URLs ("data:image/png;base64,..");
                        # strip the header before decoding the raw base64 payload.
                        raw = b64.split(",", 1)[-1] if isinstance(b64, str) else b64
                        with open(f"{self.diag_frames}/{i:03d}_{tag}.png", "wb") as fh:
                            fh.write(_b64.b64decode(raw))
                    with open(self.diag_path, "a") as fh:
                        fh.write(json.dumps({
                            "i": i, "phase": int(cfg.get("phase", 0)),
                            "state10": [float(v) for v in state10],
                            "target": [float(v) for v in tgt_arr],
                            "pooled_norm": float(np.linalg.norm(self._pool(cache)[0])),
                            "mean_pinch_first": [float(v) for v in mean_pinch[0]],
                            "mean_to_tgt_cm_first": float(dd[0] * 100.0),
                            "mean_to_tgt_cm_min": float(dd.min() * 100.0),
                        }) + "\n")
                    self.diag_n += 1
                except Exception as _e:
                    print(f"[bestofn] STEER_DIAG capture skipped: {_e}", flush=True)

            # Per-request shaping config (powered runs pin these explicitly)
            import clf_reward_ur5e as clfmod
            clfmod.CENTERING_MODE = str(cfg.get("clf_centering", "graded"))
            lim = self.cbf.Limits()   # v4-calibrated tapered vial barrier (default)

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

            # L5 — bounded final-centimeter visual servo. Composed onto GR00T's
            # ALREADY-SELECTED chunk (arm dims only; grip untouched), gated to
            # REACH/GRASP within a small radius, dominance-capped so GR00T owns
            # the majority of motion, and re-checked through the SAME CBF filter.
            # When STEER_SERVO is off, `chosen_arm` is exactly `arm[chosen]` and
            # no "servo" key is added => the response is byte-identical to V4.
            chosen_arm = arm[chosen]
            chosen_grip = np.asarray(grip[chosen], dtype=np.float32).copy()  # (16,)
            servo_report = None
            handoff_report = None
            if self.servo_enabled:
                tgt = cfg.get("grasp_target") or cfg.get("vial")
                if tgt is not None:
                    corrected, servo_report = self.servo.servo_correction(
                        chosen_arm, self._fk_batch, tgt,
                        int(cfg.get("phase", 0)), bool(cfg.get("grasped", False)),
                        lim, exec_horizon=int(cfg.get("exec_horizon", 8)),
                        **self.servo_kwargs,
                    )
                    chosen_arm = corrected
                    if servo_report.get("applied"):
                        self.n_servo_applied += 1
            tgt = None; ho_state = None; tgt_settled = True; handoff_report = {}
            pre_arm0 = np.asarray(chosen_arm[0], dtype=np.float64).copy()
            if self.handoff_enabled:
                # Full-authority Observer->IK handoff: takes over the arm AND grip
                # for the final centimetre once GR00T's reach is in the zone.
                # Fall back to the conditioner's state[7:10] target, which is
                # always present, so the handoff still runs when no best-of-N cfg
                # was attached (steering off).
                tgt = cfg.get("grasp_target") or cfg.get("vial")
                if tgt is None and float(np.abs(state10[7:10]).sum()) > 0.0:
                    tgt = [float(v) for v in state10[7:10]]
                if tgt is not None:
                    ho_state = self._handoff_state(req.get("sid"))
                    # The Observer is a per-frame regressor and its read jitters by
                    # a few cm query-to-query; the handoff was chasing that noise
                    # instead of converging. The vial does not move until we touch
                    # it, so take a running MEDIAN (robust to the occasional wild
                    # read) over a short window of recent estimates.
                    tgt_settled = True
                    if self.handoff_smooth > 1 and not bool(cfg.get("grasped", False)):
                        buf = ho_state.setdefault("tgt_buf", [])
                        # E2 filter-input guard: the conditioner's uncertainty head
                        # flags reads whose predicted residual is large (the
                        # rejected ~6% carry ~33cm median error) — keep them OUT
                        # of the median so one wild read cannot poison the window.
                        cov = cfg.get("frame_cov_cm")
                        if cov is not None and float(cov) > self.handoff_covmax:
                            self.n_cov_rejected += 1
                        else:
                            buf.append([float(v) for v in tgt])
                        if len(buf) > self.handoff_smooth:
                            del buf[0]
                        if buf:
                            tgt = np.median(np.asarray(buf, dtype=np.float64), axis=0)
                        # "settled" = enough reads in the CURRENT window to trust
                        # the median. The window is reset when descend begins (see
                        # below), so right after reset this blocks the close until
                        # the near-vial reads dominate.
                        tgt_settled = len(buf) >= 5
                    # Version-tolerant call: filter kwargs to the deployed
                    # servo build's signature so ANY servo_ur5e.py generation
                    # (incl. archaeological replicates) can be dropped in.
                    if not hasattr(self, "_handoff_sig"):
                        import inspect
                        self._handoff_sig = set(
                            inspect.signature(self.servo.ik_handoff).parameters)
                    _kw = dict(exec_horizon=int(cfg.get("exec_horizon", 8)),
                               state=ho_state, q_now=state10[:6],
                               tgt_settled=tgt_settled, pocket=cfg.get("pocket"))
                    _kw.update(self.handoff_kwargs)
                    _kw = {k: v for k, v in _kw.items() if k in self._handoff_sig}
                    a_ho, g_ho, handoff_report = self.servo.ik_handoff(
                        chosen_arm, chosen_grip, self._fk_batch, tgt,
                        int(cfg.get("phase", 0)), bool(cfg.get("grasped", False)),
                        lim, **_kw,
                    )
                    chosen_arm = a_ho
                    chosen_grip = np.asarray(g_ho, dtype=np.float32)
                    if handoff_report.get("applied"):
                        self.n_handoff_applied += 1
                    if os.environ.get("STEER_HANDOFF_LOG", "0") == "1":
                        print(f"[handoff] q={self.n_queries} {handoff_report.get('reason')}/"
                              f"{handoff_report.get('stage')} dxy={handoff_report.get('pinch_err_before_cm')}cm "
                              f"d_gp={handoff_report.get('d_gp_cm')}cm cbf={handoff_report.get('cbf_scale')} "
                              f"att={handoff_report.get('attempt')} it={handoff_report.get('iters')} "
                              f"dq={handoff_report.get('dq_norm')} ikres={handoff_report.get('ik_resid_cm')}cm "
                              f"pinch={handoff_report.get('pinch')} "
                              f"gp={handoff_report.get('grasp_point')}", flush=True)
                    if (self.handoff_nearwin
                            and handoff_report.get("stage") in ("descend", "close_hold")
                            and not ho_state.get("near_window")):
                        ho_state["near_window"] = True
                        nb = ho_state.get("tgt_buf")
                        if nb:
                            del nb[:]   # rebuild the median from close-range reads
                        print(f"[handoff] sid={req.get('sid')} descend reached -> "
                              f"filter window reset (near-vial reads take over)", flush=True)
                    if handoff_report.get("stage") == "grasp_failed_retry":
                        self.n_handoff_retries += 1
                        print(f"[handoff] sid={req.get('sid')} grasp attempt "
                              f"{handoff_report.get('attempt')} failed to weld -> "
                              f"retract + retry", flush=True)

            # --- live brain telemetry (demo dashboard; GET /brain/state) --------
            if self.handoff_enabled:
                try:
                    self._update_brain(req, cfg, tgt, ho_state, tgt_settled,
                                       handoff_report, pre_arm0, state10)
                except Exception as _be:  # telemetry must never break serving
                    print(f"[bestofn] brain telemetry skipped: {_be}", flush=True)

            out = {
                "q": [float(v) for v in chosen_arm[0]],
                "grip": float(np.clip(chosen_grip[0], 0.0, 1.0)),
                "chunk_q": [[float(v) for v in row] for row in chosen_arm],
                "chunk_grip": [float(np.clip(g, 0.0, 1.0)) for g in chosen_grip],
                "bestofn": report,
                "steer_head": ("v4" if steer_head_used else None),
            }
            if self.servo_enabled:
                out["servo"] = servo_report
            if self.handoff_enabled:
                out["handoff"] = handoff_report
            return out


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
            if self.path.startswith("/brain/state"):
                self._send(200, svc._brain or {"q": 0, "waiting": True})
                return
            if self.path.startswith("/health"):
                self._send(200, {"ok": True, "bestofn": True,
                                 "queries": svc.n_queries,
                                 "fallback_holds": svc.n_fallback_hold,
                                 "handoff_applied": svc.n_handoff_applied,
                                 "handoff_retries": svc.n_handoff_retries,
                                 "cov_rejected": svc.n_cov_rejected})
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
