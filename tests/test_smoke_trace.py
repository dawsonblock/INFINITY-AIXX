import os
import subprocess
import sys


def test_smoke_trace_cartpole():
    env = os.environ.copy()
    env["INFINITY_TRACE"] = "1"
    env["INFINITY_TRACE_EVERY"] = "5"

    cmd = [
        sys.executable,
        "-m",
        "infinity_dual_hybrid.smoke",
        "--env",
        "CartPole-v1",
        "--device",
        "cpu",
        "--total-timesteps",
        "256",
        "--rollout-steps",
        "64",
        "--trace-every",
        "5",
    ]
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    assert p.returncode == 0, p.stdout
    assert "[rollout] begin" in p.stdout
