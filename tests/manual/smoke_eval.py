"""Standalone smoke test for the multi-agent EVAL pipeline.

Runs the full PlannedEvalRuntime (Gemma SPECIALIST + OpenVLA PILOT) against a
Robosuite benchmark WITHOUT a finetuned checkpoint or the training pipeline.
It uses the *base* ``openvla/openvla-7b`` — already trained on the OXE/Bridge
mixture — as the PILOT, so we can validate the genuinely-new multi-agent eval
path end-to-end while training is blocked:

    SPECIALIST plans  ->  PILOT acts per sub-instruction  ->  simulator judges

Mirrors ``RobosuiteRunner``'s episode loop (success = ``info["success"]`` on
done, falling back to ``reward > 0``).

Requirements:
  * robosuite + MuJoCo (headless: ``export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl``)
  * ~15.5 GB VRAM (OpenVLA bf16 ~14 GB + Gemma int4 ~1.5 GB)
  * HF auth (both models are gated)

Usage (odyssey repo root, venv active):
    python scripts/smoke_eval.py
    python scripts/smoke_eval.py --benchmark Lift --episodes 2 --max-steps 150
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="Lift")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--steps-per-phase", type=int, default=40)
    parser.add_argument(
        "--pilot",
        default="openvla/openvla-7b",
        help="HF id or local path for the PILOT (the base model works fine).",
    )
    parser.add_argument("--unnorm-key", default="bridge_orig")
    args = parser.parse_args()

    # engine-first import to dodge the pre-existing engine<->runners cycle
    import odyssey.engine  # noqa: F401
    from odyssey.runners.agents.planned import PhaseConfig, PlannedEvalRuntime
    from odyssey.runners.agents.planner import LLMPlanner
    from odyssey.runners.evals.robosuite import _make_eval_env
    from odyssey.runners.models.gemma import GemmaTextGenerator
    from odyssey.runners.models.openvla import _DEFAULT_INSTRUCTIONS, VLARuntime

    instruction = _DEFAULT_INSTRUCTIONS.get(args.benchmark, "complete the task")

    print("\n=== Loading SPECIALIST: google/gemma-2b-it (int4) ===", flush=True)
    planner = LLMPlanner(GemmaTextGenerator("google/gemma-2b-it", quantization="int4"))

    print(f"\n=== Loading PILOT: {args.pilot} ===", flush=True)
    pilot = VLARuntime(args.pilot, unnorm_key=args.unnorm_key)

    runtime = PlannedEvalRuntime(
        pilot=pilot,
        planner=planner,
        phase_config=PhaseConfig(steps_per_phase=args.steps_per_phase),
        fallback_instruction=instruction,
    )

    print(f"\n=== Building Robosuite env: {args.benchmark} (Panda) ===", flush=True)
    env = _make_eval_env(args.benchmark, "Panda", {"camera_names": "agentview"})

    successes = 0
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        plan = runtime.begin_episode(instruction)
        print(f"\n--- Episode {ep}: instruction={instruction!r} ---", flush=True)
        for i, step in enumerate(plan):
            print(f"    phase {i}: {step}")

        success = False
        last_reward = 0.0
        for _step in range(args.max_steps):
            image = obs.get("agentview_image") if isinstance(obs, dict) else obs
            action = runtime.get_action(image)
            obs, reward, done, info = env.step(action)
            last_reward = float(reward)
            if done:
                success = bool(
                    info.get("success", reward > 0)
                    if isinstance(info, dict)
                    else reward > 0
                )
                break

        successes += int(success)
        print(
            f"    -> episode {ep}: {'PASS' if success else 'FAIL'} "
            f"(last_reward={last_reward:.3f})",
            flush=True,
        )

    print(f"\n=== {successes}/{args.episodes} successful episode(s) ===\n")


if __name__ == "__main__":
    main()
