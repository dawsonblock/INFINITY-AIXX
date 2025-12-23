from __future__ import annotations

import argparse
import os
import sys

from infinity_dual_hybrid.train_cli import main as train_main


def smoke_main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test runner that forces tracing on and runs a tiny training loop.")
    ap.add_argument("--env", "--env-id", dest="env_id", default="CartPole-v1")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--total-timesteps", type=int, default=1024)
    ap.add_argument("--rollout-steps", type=int, default=128)
    ap.add_argument("--trace-every", type=int, default=10)
    ap.add_argument("--trace-cuda", action="store_true")
    ap.add_argument("--run-id", default="SMOKE")
    args = ap.parse_args()

    # Force tracer on for smoke mode
    os.environ["INFINITY_TRACE"] = "1"
    os.environ["INFINITY_TRACE_EVERY"] = str(args.trace_every)
    if args.trace_cuda:
        os.environ["INFINITY_TRACE_CUDA"] = "1"

    # Call the existing CLI by faking argv. This keeps behavior identical to normal runs.
    sys.argv = [
        sys.argv[0],
        "--env", args.env_id,
        "--device", args.device,
        "--total-timesteps", str(args.total_timesteps),
        "--rollout-steps", str(args.rollout_steps),
        "--run-id", args.run_id,
    ]
    train_main()


if __name__ == "__main__":
    smoke_main()
