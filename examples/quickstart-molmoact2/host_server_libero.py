"""MolmoAct2-LIBERO inference server (FastAPI ``/act``, json_numpy wire).

Adapted from Ai2's ``examples/droid/host_server_droid.py`` in
https://github.com/allenai/molmoact2 (Apache-2.0). Ai2 ships /act servers for
DROID and YAM only; this is the same server with the LIBERO schema, so odyssey's
``pilot: molmoact2`` can drive ``allenai/MolmoAct2-LIBERO`` out-of-process.

It runs in a SEPARATE GPU env (torch + transformers + fastapi + json-numpy),
never in the odyssey/LIBERO env. Easiest: the uv env of an ``allenai/molmoact2``
checkout (its ``pyproject.toml`` pins all of these)::

    cd molmoact2 && uv sync
    uv run python /path/to/odyssey/examples/quickstart-molmoact2/host_server_libero.py \
        --repo-id allenai/MolmoAct2-LIBERO --port 8000

Wire protocol (matches ``odyssey.runners.models.molmoact2.MolmoAct2HttpClient``)::

    GET  /act -> {"status": "ok", "repo_id", "norm_tag", "action_horizon", ...}
    POST /act   request (json_numpy):
        {"image":       ndarray(H, W, 3) uint8 RGB   # agentview, 180°-flipped
         "wrist_image": ndarray(H, W, 3) uint8 RGB   # eye-in-hand, 180°-flipped
         "state":       ndarray(8,) float32          # eef_pos(3) + axis-angle(3) + gripper_qpos(2)
         "instruction": str,
         "num_steps":   int (optional, flow solver steps, default 10)}
              response (json_numpy):
        {"actions": ndarray(N, 7) float32, "dt_ms": float}   |   {"error": str}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from typing import Any

import json_numpy
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

json_numpy.patch()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("molmoact2.libero_server")

REPO_ID = "allenai/MolmoAct2-LIBERO"
NORM_TAG = "libero"
STATE_DIM = 8
DEFAULT_NUM_STEPS = 10


def _patch_modeling_for_bf16(local_dir: str) -> None:
    """Same idempotent bf16 patches as Ai2's DROID server.

    The upstream flow trajectory is hardcoded to float32 and ``_to_array`` calls
    ``.numpy()`` on bf16 tensors; both break bf16 inference. Patches the snapshot
    copy and the ``trust_remote_code`` module cache copy.
    """
    patches = [
        (
            "device=device,\n            dtype=torch.float32,\n            generator=generator,",
            "device=device,\n"
            "            dtype=source_tensor.dtype,  # patched_bf16_dtype\n"
            "            generator=generator,",
            "patched_bf16_dtype",
        ),
        (
            "return value.detach().cpu().numpy().astype(np.float32, copy=False)",
            "return value.detach().cpu().float().numpy().astype(np.float32, copy=False)"
            "  # patched_bf16_to_array",
            "patched_bf16_to_array",
        ),
    ]
    candidates = [os.path.join(local_dir, "modeling_molmoact2.py")]
    modules_root = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    if os.path.isdir(modules_root):
        for sub in os.listdir(modules_root):
            p = os.path.join(modules_root, sub, "modeling_molmoact2.py")
            if os.path.isfile(p):
                candidates.append(p)
    for path in candidates:
        try:
            with open(path, encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        new_src = src
        for needle, replacement, marker in patches:
            if marker in new_src:
                continue
            if needle not in new_src:
                log.warning("patch %s: needle not found in %s", marker, path)
                continue
            new_src = new_src.replace(needle, replacement, 1)
        if new_src != src:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_src)
            log.info("Applied bf16 patches in %s", path)


def _action_horizon(local_dir: str) -> int:
    """``action_horizon`` for NORM_TAG from the checkpoint's norm_stats.json."""
    try:
        with open(os.path.join(local_dir, "norm_stats.json"), encoding="utf-8") as f:
            meta = json.load(f)["metadata_by_tag"][NORM_TAG]
        return int(meta.get("n_action_steps") or meta.get("action_horizon") or 10)
    except (OSError, KeyError, ValueError):
        return 10


class Policy:
    """Loaded model + processor; serializes inference calls."""

    def __init__(self, repo_id: str, device: str, dtype: torch.dtype,
                 enable_cuda_graph: bool = False) -> None:
        self.repo_id = repo_id
        self.default_cuda_graph = enable_cuda_graph
        # The model reads norm_stats.json from config._name_or_path, so load from
        # the resolved local snapshot dir, not the repo id.
        local_dir = snapshot_download(repo_id=repo_id)
        log.info("Resolved snapshot dir: %s", local_dir)
        self.action_horizon = _action_horizon(local_dir)
        if dtype != torch.float32:
            _patch_modeling_for_bf16(local_dir)

        # tokenizer_config ships extra_special_tokens as a list; newer
        # transformers want a dict (same workaround as Ai2's server).
        self.processor = AutoProcessor.from_pretrained(
            local_dir, trust_remote_code=True, extra_special_tokens={}
        )
        log.info("Loading model (dtype=%s, device=%s)", dtype, device)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                local_dir, trust_remote_code=True, torch_dtype=dtype
            )
            .to(device)
            .eval()
        )
        self.device = device

        # Cast float inputs to the model dtype (processor emits fp32 pixel_values).
        target_dtype = next(self.model.parameters()).dtype

        def _move_and_cast(inputs: Any, dev: Any, _target: torch.dtype = target_dtype
                           ) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    value = value.to(dev)
                    if value.is_floating_point() and value.dtype != _target:
                        value = value.to(_target)
                out[key] = value
            return out

        self.model._move_inputs_to_device = _move_and_cast
        self._lock = threading.Lock()

    @torch.inference_mode()
    def predict(self, image: np.ndarray, wrist_image: np.ndarray, instruction: str,
                state: np.ndarray, num_steps: int = DEFAULT_NUM_STEPS,
                enable_cuda_graph: bool = False) -> np.ndarray:
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (STATE_DIM,):
            raise ValueError(f"state must be shape ({STATE_DIM},), got {state_f32.shape}")
        with self._lock:
            out = self.model.predict_action(
                processor=self.processor,
                images=[_to_pil(image), _to_pil(wrist_image)],  # agentview, then wrist
                task=instruction,
                state=state_f32,
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=num_steps,
                normalize_language=True,
                enable_cuda_graph=enable_cuda_graph,
            )
        raw = out.actions
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return actions


def _to_pil(arr: Any) -> Image.Image:
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    # No ``mode=``: deprecated in Pillow 11.3 (removed in 13); uint8 HxWx3 is RGB.
    return Image.fromarray(a)


def _error(status: int, message: str) -> Response:
    return Response(content=json_numpy.dumps({"error": message}), status_code=status,
                    media_type="application/json")


def build_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="MolmoAct2-LIBERO server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "repo_id": policy.repo_id,
            "norm_tag": NORM_TAG,
            "action_horizon": policy.action_horizon,
            "device": policy.device,
            "dtype": str(policy.model.dtype),
        })

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act(request: Request) -> Response:
        try:
            payload = json_numpy.loads((await request.body()).decode("utf-8"))
        except Exception as e:
            return _error(400, f"failed to decode json_numpy body: {e}")
        try:
            image, wrist = payload["image"], payload["wrist_image"]
            instruction, state = str(payload["instruction"]), payload["state"]
        except KeyError as e:
            return _error(400, f"missing required field: {e}")
        num_steps = int(payload.get("num_steps", DEFAULT_NUM_STEPS))
        cuda_graph = bool(payload.get("enable_cuda_graph", policy.default_cuda_graph))

        t0 = time.perf_counter()
        try:
            # Off the event loop: a blocking GPU predict here would stall GET /act
            # and /healthz for the whole inference. Policy._lock still serializes.
            actions = await run_in_threadpool(
                policy.predict, image, wrist, instruction, state,
                num_steps=num_steps, enable_cuda_graph=cuda_graph,
            )
        except Exception as e:
            log.exception("inference failed")
            return _error(500, f"inference failed: {e}")
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return Response(content=json_numpy.dumps({"actions": actions, "dt_ms": dt_ms}),
                        media_type="application/json")

    return app


def warmup(policy: Policy) -> None:
    log.info("Warming up (cuda_graph=%s) ...", policy.default_cuda_graph)
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    try:
        policy.predict(dummy, dummy, "warmup", np.zeros(STATE_DIM, np.float32),
                       enable_cuda_graph=policy.default_cuda_graph)
    except Exception:
        log.exception("warmup inference failed (server will still start)")
        return
    log.info("Warmup OK (%.1f ms)", (time.perf_counter() - t0) * 1000.0)


def main() -> None:
    p = argparse.ArgumentParser(description="MolmoAct2-LIBERO inference server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--repo-id", default=REPO_ID)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="bfloat16 fits a 24 GB L4; the model card's reference is float32.")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--cuda-graph", action="store_true",
                   help="CUDA-graph the action expert (faster, ~2 GB more VRAM).")
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    policy = Policy(args.repo_id, args.device, dtype, enable_cuda_graph=args.cuda_graph)
    if not args.no_warmup:
        warmup(policy)

    import uvicorn

    log.info("Listening on %s:%d (action_horizon=%d)", args.host, args.port,
             policy.action_horizon)
    uvicorn.run(build_app(policy), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
