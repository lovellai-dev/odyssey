#!/usr/bin/env python3
"""Deploy-time **Observer state-conditioning sidecar** for the served GR00T pilot.

Roadmap Step 2 (Observer-conditioned GR00T). The render-gap close proved the
Observer localises the vial cap to sub-cm on the deployment (Three.js) frames, yet
the pixel-only browser GR00T pilot still scored 0/15 — it could not infer *descent
depth* from pixels. Step 2 retrains GR00T with the grasp target as an EXPLICIT
state input (state 7 -> 10). This sidecar is the deploy half of that contract: it
is a **drop-in for the GR00T HTTP<->ZMQ bridge** (same ``POST /get_action`` +
``GET /health``) that, per control tick:

1. runs the Observer's DINOv2 grasp-target head on the exterior+wrist frames the
   browser already sends -> the 3D vial cap in the robot base frame (``cap_base``);
2. appends that target to the pilot's 7-D proprio state -> the 10-D state the
   Observer-conditioned checkpoint was trained on;
3. forwards the (now 10-D) observation to the real GR00T bridge and returns its
   action unchanged (GR00T keeps full actuator authority).

Point the agent-service ``GROOT_BRIDGE_URL`` (and/or ``GROOT_STATE_CONDITIONER_URL``)
at this sidecar; it forwards to the true bridge (``--bridge`` / ``GROOT_BRIDGE_URL``
of THIS process). It is a **dev/local-only** seam — the GR00T serving path is
GPU-local, never Cloud Run.

Robust by construction: if the Observer (or a frame) errors, it falls back to the
last good target for that session, then to a fixed nominal target — it ALWAYS emits
a 10-D state, so the 10-D model never sees a missing key and the pilot never freezes.

Run (torch+transformers interpreter; odyssey src on the path)::

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      OBSERVER_WEIGHTS=/path/to/percep_weights_browser \
      CONDITION_BRIDGE_URL=http://127.0.0.1:5596 \
      PYTHONPATH=/home/daniel/LovellAI/odyssey-ur5e/src \
      /home/daniel/Isaac-GR00T/.venv/bin/python scripts/serve_observer_conditioning.py --port 5604

FlowDAgger Stage B — latent-noise STEERING (optional, env-toggled)
------------------------------------------------------------------
When ``STEERING_WEIGHTS`` points at the A4 steering-net ``.npz``, the sidecar
additionally computes, per tick, the flow sampler's initial noise from the
low-dim obs ``[proprio7, observer target xyz3, inferred phase_onehot4]`` and
attaches it to the forwarded JSON as ``init_noise_b64`` (base64 of the float32
``(40, 132)`` tensor) + ``init_noise_shape``. The patched GR00T bridge decodes
it and passes ``options={"init_noise": ...}`` into the ZMQ ``get_action`` (see
``scripts/patches/isaac_groot_init_noise.patch``). ``STEERING_WEIGHTS`` unset
-> byte-identical to the pre-Stage-B conditioner (the stock arm of the A/B).

The phase input was GT-sidecar-derived at training (A3 ``PHASE_TO_BUCKET``);
at deploy it is INFERRED by :class:`PhaseInference` — a per-session state
machine over the grip command + the FK'd ``gr_pinch`` xy relative to the xy
locked at the open->close transition, matching the A3 bucket semantics
(0 REACH = home/approach/descend; 1 GRASP = close/lift, gripper closed at the
grasp column; 2 TRANSPORT = transport/lower, gripper closed and departed the
grasp column; 3 PLACE = release/retract/home2, gripper reopened after a close).
Known tail mismatch: the GT labels the post-place return home as bucket 3
while the episode-boundary reset here re-labels near-home+open as bucket 0.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import traceback
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts (steering helpers)

import numpy as np
from PIL import Image

# Nominal grasp target (robot base frame, metres) used only if the Observer errors
# before any good reading exists — the dataset grasp_target mean (a safe centre of
# the vial workspace). Env-overridable.
NOMINAL_TARGET = np.array(
    [float(x) for x in os.environ.get("GRASP_TARGET_FALLBACK", "-0.425,-0.015,0.272").split(",")],
    dtype=np.float32,
)


def _decode_frame(data_url: str) -> np.ndarray:
    b64 = data_url.split(",", 1)[1] if "," in str(data_url) else data_url
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# FlowDAgger Stage B — deploy-time steering (pure helpers, unit-testable)
# ---------------------------------------------------------------------------
ACTION_HORIZON = 40
ACTION_DIM = 132
N_PHASES = 4
PHASE_NAMES = ("reach", "grasp", "transport", "place")
DEFAULT_HOME_Q = (-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
DEFAULT_FK_XML = (
    "/home/daniel/LovellAI/lai-agent-multiagent/src/embodiments/urdf/"
    "aseptipack_description/aseptipack.xml"
)


def encode_init_noise(noise: np.ndarray) -> tuple[str, list[int]]:
    """(H, D) float32 -> (base64 payload, shape list) for the forwarded JSON."""
    arr = np.ascontiguousarray(np.asarray(noise, dtype=np.float32))
    return base64.b64encode(arr.tobytes()).decode("ascii"), list(arr.shape)


class PhaseInference:
    """Per-session 4-bucket phase state machine matching the A3 label semantics.

    Inputs per tick: the request's 7-D proprio (arm6 + grip command 0..1) and a
    ``pinch_fn(q6) -> xyz`` FK callable (injected, so the logic is testable
    without mujoco). Transitions:

    * grip < close_thresh, never closed          -> 0 REACH
    * near home_q with gripper open              -> episode-boundary RESET -> 0
    * grip >= close_thresh: lock the pinch xy at the open->close transition;
      while closed, xy-dist(pinch, locked) <= xy_lock_radius -> 1 GRASP
      (close + straight-up lift), else -> 2 TRANSPORT (carry/lower departed
      the grasp column; the scripted vial->pocket carry is ~33 cm)
    * grip < close_thresh after a close          -> 3 PLACE (release/retract)
      (a later re-close re-locks and returns to 1/2)
    """

    REACH, GRASP, TRANSPORT, PLACE = 0, 1, 2, 3

    def __init__(
        self,
        pinch_fn: Callable[[np.ndarray], np.ndarray],
        home_q: Any = DEFAULT_HOME_Q,
        *,
        close_thresh: float = 0.5,
        xy_lock_radius: float = 0.10,
        home_tol: float = 0.15,
        open_thresh: float = 0.30,
        promote_radius: float = 0.035,
        promote_dz: float = 0.05,
        dwell_n: int = 2,
    ) -> None:
        self._pinch = pinch_fn
        self._home_q = np.asarray(home_q, dtype=np.float64).reshape(6)
        self._close = float(close_thresh)
        self._rad = float(xy_lock_radius)
        self._home_tol = float(home_tol)
        self._open = float(open_thresh)
        self._promote_rad = float(promote_radius)
        self._promote_dz = float(promote_dz)
        self._dwell_n = int(dwell_n)
        self._sess: dict[str, dict[str, Any]] = {}

    def reset(self, sid: str) -> None:
        self._sess.pop(sid, None)

    def infer(self, sid: str, state7: Any, tgt3: Any = None) -> int:
        s7 = np.asarray(state7, dtype=np.float64).reshape(-1)[:7]
        q6, grip = s7[:6], float(s7[6])
        st = self._sess.setdefault(
            sid, {"has_closed": False, "was_closed": False, "locked_xy": None,
                  "promoted": False, "dwell": 0})
        closed = grip >= self._close
        # Episode-boundary reset: back at (or still at) home with the gripper
        # open. Clears has_closed so a new episode starts in REACH. (Known GT
        # mismatch: the post-place home2 frames are bucket 3 in training.)
        if (not closed and grip < self._open
                and float(np.max(np.abs(q6 - self._home_q))) < self._home_tol):
            st.update(has_closed=False, was_closed=False, locked_xy=None,
                      promoted=False, dwell=0)
            return self.REACH
        if closed:
            pinch = np.asarray(self._pinch(q6), dtype=np.float64).reshape(3)
            if not st["was_closed"]:      # open -> close: lock the grasp column xy
                st["locked_xy"] = pinch[:2].copy()
                st["has_closed"] = True
                st["was_closed"] = True
                st["promoted"] = False
                st["dwell"] = 0
            xy = float(np.linalg.norm(pinch[:2] - st["locked_xy"]))
            return self.GRASP if xy <= self._rad else self.TRANSPORT
        st["was_closed"] = False
        if st["has_closed"]:
            return self.PLACE
        # v0.1/v0.2 DWELL PROMOTION — breaks the grasp deadlock found in the
        # Stage-B A/B (steered arm reached 0.2-0.7 cm but never closed): the
        # phase used to advance only on the policy's OWN grip command, while
        # REACH-phase noise never commands closure. v0.2 promotes on FULL 3D
        # proximity (the v0.1 xy-only check fired while the pinch was still
        # 5-10 cm ABOVE the vial -> 81% of ticks spent hovering in a
        # (GRASP, early-t, high-z) input combo that never occurs in training).
        # Hysteresis at 1.5x the radius demotes if the arm drifts while open.
        if tgt3 is not None:
            tgt = np.asarray(tgt3, dtype=np.float64).reshape(3)
            pinch = np.asarray(self._pinch(q6), dtype=np.float64).reshape(3)
            xy = float(np.linalg.norm(pinch[:2] - tgt[:2]))
            dz = abs(float(pinch[2] - tgt[2]))
            near = xy <= self._promote_rad and dz <= self._promote_dz
            if st["promoted"]:
                if xy <= 1.5 * self._promote_rad and dz <= 1.5 * self._promote_dz:
                    return self.GRASP
                st.update(promoted=False, dwell=0)
                return self.REACH
            st["dwell"] = st["dwell"] + 1 if near else 0
            if st["dwell"] >= self._dwell_n:
                st["promoted"] = True
                return self.GRASP
        return self.REACH


class MjPinchFK:
    """``gr_pinch`` site FK (robot base frame) from the 6 arm joint angles.

    Loads the aseptipack MJCF once on a private MjData; mujoco is imported
    lazily so the module stays importable in a mujoco-less venv.
    """

    _ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")

    def __init__(self, xml_path: str = DEFAULT_FK_XML) -> None:
        import mujoco as mj

        self._mj = mj
        self._model = mj.MjModel.from_xml_path(xml_path)
        self._data = mj.MjData(self._model)
        self._qadr = []
        for name in self._ARM_JOINTS:
            j = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise RuntimeError(f"arm joint {name!r} not found in {xml_path}")
            self._qadr.append(int(self._model.jnt_qposadr[j]))
        self._site = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_SITE, "gr_pinch")
        if self._site < 0:
            raise RuntimeError(f"gr_pinch site not found in {xml_path}")
        hk = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_KEY, "home")
        if hk >= 0:
            mj.mj_resetDataKeyframe(self._model, self._data, hk)
        mj.mj_forward(self._model, self._data)
        base = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_BODY, "base")
        self._bpos = np.array(self._data.xpos[base], dtype=np.float64)
        self._bmat = np.array(self._data.xmat[base], dtype=np.float64).reshape(3, 3)

    def __call__(self, q6: Any) -> np.ndarray:
        q = np.asarray(q6, dtype=np.float64).reshape(6)
        for a, v in zip(self._qadr, q):
            self._data.qpos[a] = float(v)
        self._mj.mj_kinematics(self._model, self._data)
        p = np.array(self._data.site_xpos[self._site], dtype=np.float64)
        return self._bmat.T @ (p - self._bpos)


class SteeringEngine:
    """Loads the A4 steering-net npz; produces one (40,132) init_noise per tick.

    The MLP forward is the SAME numpy implementation that passed the A5 offline
    gate (``flowdagger_offline_gate_ur5e.load_steering_net``) — deterministic,
    CPU, ~1 ms/tick for the 14->5280 net (no GPU contention with the Observer).
    """

    def __init__(self, weights: str, pinch_fn: Callable[[np.ndarray], np.ndarray],
                 home_q: Any = DEFAULT_HOME_Q) -> None:
        from flowdagger_offline_gate_ur5e import load_steering_net
        import train_steering_ur5e as ts

        self._forward, meta = load_steering_net(weights)
        self.design = str(meta.get("output_design", "full5280"))
        self.in_dim = int(meta.get("in_dim", 14))   # 14 = v1; 15 = v2 (+ t_norm)
        self._ts = ts
        self._fixed_pads = None if self.design == "full5280" else ts.deploy_pads(1)[0]
        self.phase_inf = PhaseInference(pinch_fn, home_q)
        self.n_attached = 0
        self.phase_ticks = [0, 0, 0, 0]
        # v0.1 t_norm feature: executed ticks per SESSION / nominal episode ticks.
        # Correct t_norm relies on the client sending a FRESH sid per episode
        # (eval_browser_groot.js does since v0.1). Deliberately NOT reset by the
        # phase machine's home detection — a home-frozen arm pinned at t_norm=0
        # would recreate the exact absorbing fixed point this feature removes.
        self._ticks_per_query = int(os.environ.get("STEER_TICKS_PER_QUERY", "8"))
        self._ep_ticks = float(os.environ.get("STEER_EP_TICKS", "900"))
        self._ticks: dict[str, int] = {}
        # v3 recent-motion feature (in_dim >= 21): dq = q_now - q_prev_query,
        # matching the trainer's per-8-frame window. Zero on a session's first
        # query — consistent with the dataset's episode-start convention.
        self._prev_q: dict[str, np.ndarray] = {}
        print(f"[steer] ready | weights={weights} design={self.design} "
              f"in_dim={self.in_dim} out_dim={meta.get('out_dim')} "
              f"t_norm={'ON' if self.in_dim >= 15 else 'off'}", flush=True)

    def init_noise(self, sid: str, state7: Any, tgt3: Any) -> tuple[np.ndarray, int]:
        phase = self.phase_inf.infer(sid, state7, tgt3)
        oh = np.zeros(N_PHASES, dtype=np.float32)
        oh[phase] = 1.0
        cols = [
            np.asarray(state7, dtype=np.float32).reshape(-1)[:7],
            np.asarray(tgt3, dtype=np.float32).reshape(3), oh,
        ]
        if self.in_dim >= 15:
            if len(self._ticks) > 64:                  # bound per-session state
                self._ticks.pop(next(iter(self._ticks)))
            t = min(1.0, self._ticks.get(sid, 0) / self._ep_ticks)
            self._ticks[sid] = self._ticks.get(sid, 0) + self._ticks_per_query
            cols.append(np.array([t], dtype=np.float32))
        if self.in_dim >= 21:
            q_now = np.asarray(state7, dtype=np.float32).reshape(-1)[:6]
            if len(self._prev_q) > 64:
                self._prev_q.pop(next(iter(self._prev_q)))
            dq = q_now - self._prev_q.get(sid, q_now)   # first query -> zeros
            self._prev_q[sid] = q_now.copy()
            cols.append(dq.astype(np.float32))
        x = np.concatenate(cols).astype(np.float32)[None]             # (1, 14|15|21)
        pred = self._forward(x)
        if self.design == "full5280":
            noise = pred.reshape(ACTION_HORIZON, ACTION_DIM).astype(np.float32)
        else:  # real112 + fixed-seed pads (deploy convention)
            noise = self._ts.assemble_init_noise(
                pred.reshape(1, self._ts.REAL_STEPS, self._ts.REAL_DIMS),
                pads=self._fixed_pads[None])[0].astype(np.float32)
        self.n_attached += 1
        self.phase_ticks[phase] += 1
        return noise, phase


class ConditioningService:
    """Loads the Observer grasp-target head once; conditions one request per call."""

    def __init__(self, *, weights: str, bridge_url: str, device: str,
                 steering_weights: str = "") -> None:
        from odyssey.embodiments.ur5e_drugsort.observer import GraspTargetObserver

        self.bridge_url = bridge_url.rstrip("/")
        self._lock = threading.Lock()
        self._last: dict[str, np.ndarray] = {}     # per-session last good target (hold-last fallback)
        self._n = 0
        self._n_fallback = 0
        # Ablation switch: when set, ignore the Observer and inject the fixed nominal
        # target every tick. Isolates the *live-target signal* vs a constant — if the
        # Observer-conditioned success collapses to the nominal-target success, the
        # per-frame Observer localisation is what carries the policy.
        self._freeze_nominal = os.environ.get("CONDITIONER_FREEZE_NOMINAL", "").strip() in ("1", "true", "True")
        # FlowDAgger Stage B: latent-noise steering (None => stock behaviour).
        self._steering: SteeringEngine | None = None
        if steering_weights:
            fk = MjPinchFK(os.environ.get("STEERING_FK_XML", DEFAULT_FK_XML))
            home_q = np.array(
                [float(x) for x in os.environ.get(
                    "STEERING_HOME_Q",
                    ",".join(str(v) for v in DEFAULT_HOME_Q)).split(",")],
                dtype=np.float64,
            )
            self._steering = SteeringEngine(steering_weights, fk, home_q)
        self._observer = GraspTargetObserver(device=device)
        self._observer.load_head(str(Path(weights) / "observer_head.pt"))
        warm = np.zeros((256, 256, 3), np.uint8)
        self._observer.predict(warm, warm)         # warm the CUDA graphs
        print(
            f"[conditioner] ready | backbone={self._observer.backbone_id} device={device} "
            f"bridge={self.bridge_url} nominal_target={NOMINAL_TARGET.tolist()} "
            f"steering={'ON' if self._steering else 'OFF'}",
            flush=True,
        )

    def _target(self, payload: dict, sid: str) -> tuple[np.ndarray, str]:
        """Observer grasp target (base frame) for this tick, with layered fallback."""
        if self._freeze_nominal:
            return NOMINAL_TARGET.copy(), "frozen-nominal"
        try:
            ext = _decode_frame(payload["image_b64"])
            wrist = _decode_frame(payload.get("image_b64_wrist") or payload["image_b64"])
            with self._lock:
                tgt = np.asarray(self._observer.predict(ext, wrist), dtype=np.float32).reshape(3)
            if not np.all(np.isfinite(tgt)):
                raise ValueError("non-finite observer target")
            self._last[sid] = tgt
            if len(self._last) > 32:
                self._last.pop(next(iter(self._last)))
            return tgt, "observer"
        except Exception as exc:  # noqa: BLE001 — never freeze; hold-last then nominal
            self._n_fallback += 1
            if sid in self._last:
                return self._last[sid], f"hold-last({exc!r})"
            return NOMINAL_TARGET.copy(), f"nominal({exc!r})"

    def _query_bridge(self, body: bytes) -> tuple[int, dict[str, Any]]:
        req = urllib.request.Request(
            f"{self.bridge_url}/get_action", data=body,
            headers={"content-type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=180.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))

    def get_action(self, body: bytes) -> tuple[int, dict[str, Any]]:
        payload = json.loads(body.decode("utf-8"))
        sid = str(payload.get("sid") or "default")
        state = list(np.asarray(payload.get("state", []), dtype=np.float32).reshape(-1)[:7])
        while len(state) < 7:
            state.append(0.0)
        tgt, mode = self._target(payload, sid)
        payload["state"] = [float(x) for x in state] + [float(x) for x in tgt]   # 7 -> 10
        self._n += 1
        phase = None
        if self._steering is not None:
            noise, phase = self._steering.init_noise(sid, state, tgt)
            payload["init_noise_b64"], payload["init_noise_shape"] = encode_init_noise(noise)
            # R2 best-of-N passthrough: when STEER_BESTOFN_K > 1 the downstream
            # service samples K jittered candidates around this steering mean and
            # selects via CBF-filter + CLF-rank. grasped = the phase machine's
            # closure latch for this session (drives the CBF vial-protection
            # exemption and the CLF transport/place gates).
            k_bon = int(os.environ.get("STEER_BESTOFN_K", "1"))
            if k_bon > 1:
                sess = self._steering.phase_inf._sess.get(sid, {})
                # Adaptive K: finer selection granularity exactly where capture
                # precision matters (GRASP phase), default 2x the base K.
                if int(phase) == PhaseInference.GRASP:
                    k_bon = int(os.environ.get("STEER_BESTOFN_K_GRASP", str(2 * k_bon)))
                payload["bestofn"] = {
                    "k": k_bon,
                    "sigma": float(os.environ.get("STEER_BESTOFN_SIGMA", "0.35")),
                    "phase": int(phase),
                    "grasped": bool(sess.get("was_closed", False)),
                    "vial": [float(x) for x in tgt],
                    "grasp_target": [float(x) for x in tgt],
                    # Cell geometry (base frame) so the TRANSPORT/PLACE CLF aims
                    # the held vial at the rack, not at its own pickup point.
                    "pocket": [float(x) for x in os.environ.get(
                        "STEER_POCKET", "-0.39,0.174,0.228").split(",")],
                    # Powered-run config pins (defaults = current shaping)
                    "clf_centering": os.environ.get("STEER_CLF_CENTERING", "graded"),
                    "cbf_flat_vial_cap": os.environ.get("STEER_CBF_FLAT", "0") == "1",
                    "cem_rounds": int(os.environ.get("STEER_CEM_ROUNDS", "1")),
                }
            print(f"[steer] n={self._steering.n_attached} sid={sid} "
                  f"phase={phase}({PHASE_NAMES[phase]}) grip={state[6]:.3f}", flush=True)
        if self._n <= 3 or self._n % 200 == 0:
            print(f"[conditioner] tick={self._n} sid={sid} mode={mode} "
                  f"target={np.round(tgt, 4).tolist()} fallbacks={self._n_fallback}", flush=True)
        code, out = self._query_bridge(json.dumps(payload).encode("utf-8"))
        if isinstance(out, dict):
            out["grasp_target"] = [float(x) for x in tgt]
            out["conditioner"] = {"mode": mode}
            if phase is not None:
                out["steering_phase"] = int(phase)
        return code, out

    def health(self) -> dict[str, Any]:
        ok = False
        try:
            with urllib.request.urlopen(f"{self.bridge_url}/health", timeout=5.0) as resp:
                ok = bool(json.loads(resp.read().decode("utf-8")).get("ok"))
        except Exception:  # noqa: BLE001
            ok = False
        # `ok` mirrors the downstream bridge so the agent-service /api/groot/health
        # wraps it as {"bridge_health":{"ok":true}} exactly like the raw bridge.
        out: dict[str, Any] = {
            "ok": ok, "observer_ready": True, "bridge": self.bridge_url,
            "conditioned": True, "ticks": self._n, "fallbacks": self._n_fallback,
            "steering_enabled": self._steering is not None,
        }
        if self._steering is not None:
            out["steering"] = {
                "design": self._steering.design,
                "attached": self._steering.n_attached,
                "phase_ticks": {PHASE_NAMES[i]: int(n)
                                for i, n in enumerate(self._steering.phase_ticks)},
            }
        return out


def _make_handler(service: ConditioningService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:  # quiet
            return

        def _send(self, code: int, obj: dict[str, Any]) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/health":
                self._send(200, service.health())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            # Accept BOTH the plain bridge path (?agents=groot -> /get_action) and the
            # observer path (?agents=groot-observer -> /get_action_corrected) so the
            # conditioner is a drop-in for either agent-service route.
            if self.path.rstrip("/") not in ("/get_action", "/get_action_corrected"):
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("content-length", 0) or 0)
            body = self.rfile.read(length)
            try:
                code, out = service.get_action(body)
                self._send(code, out)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._send(502, {"error": f"conditioner failure: {exc!r}"})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=int(os.environ.get("CONDITIONER_PORT", "5604")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--weights", default=os.environ.get("OBSERVER_WEIGHTS", ""))
    ap.add_argument("--bridge", default=os.environ.get("CONDITION_BRIDGE_URL",
                                                        os.environ.get("GROOT_BRIDGE_URL", "http://127.0.0.1:5596")))
    ap.add_argument("--device", default=os.environ.get("OBSERVER_DEVICE", "cuda"))
    ap.add_argument("--steering-weights", default=os.environ.get("STEERING_WEIGHTS", ""),
                    help="A4 steering-net .npz; set => attach init_noise per tick (Stage B)")
    args = ap.parse_args()
    if not args.weights:
        raise SystemExit("--weights (or OBSERVER_WEIGHTS) is required: dir with observer_head.pt")

    service = ConditioningService(weights=args.weights, bridge_url=args.bridge,
                                  device=args.device, steering_weights=args.steering_weights)
    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(service))
    print(f"[conditioner] serving on http://{args.host}:{args.port} "
          f"(POST /get_action|/get_action_corrected, GET /health) -> {args.bridge}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
