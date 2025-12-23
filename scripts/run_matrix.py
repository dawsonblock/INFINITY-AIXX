import os
import subprocess
import sys
from pathlib import Path
import json
import time

COND = [
    ("baseline", {"agent.use_miras": "false", "agent.use_ltm": "false", "pseudo.enabled": "false",
                  "agent.use_world_model": "false", "agent.humanlike_enabled": "false"}),
    ("memory",   {"agent.use_miras": "true",  "agent.use_ltm": "true",  "pseudo.enabled": "false",
                  "agent.use_world_model": "false", "agent.humanlike_enabled": "false"}),
    ("pseudo_no_wm", {"agent.use_miras": "true", "agent.use_ltm": "true", "pseudo.enabled": "true",
                      "agent.use_world_model": "false", "agent.humanlike_enabled": "true"}),
    ("pseudo_wm", {"agent.use_miras": "true", "agent.use_ltm": "true", "pseudo.enabled": "true",
                   "agent.use_world_model": "true", "agent.humanlike_enabled": "true"}),
]

def run(cmd, env):
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, env=env)
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_matrix.py <config.yaml> [--seed N]")
        raise SystemExit(2)

    cfg = sys.argv[1]
    seed = 0
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])

    out_dir = Path("runs_matrix")
    out_dir.mkdir(exist_ok=True)

    base_env = os.environ.copy()
    base_env["INFINITY_TRACE"] = "1"
    base_env["INFINITY_TRACE_EVERY"] = base_env.get("INFINITY_TRACE_EVERY", "50")

    results = []
    start = time.time()

    for name, overrides in COND:
        run_id = f"{Path(cfg).stem}__{name}__seed{seed}"
        env = base_env.copy()
        env["INFINITY_RUN_NAME"] = run_id

        cmd = [sys.executable, "-m", "infinity_dual_hybrid.train_cli", "--config", cfg, "--seed", str(seed), "--run-id", run_id]
        for k, v in overrides.items():
            cmd += ["--set", f"{k}={v}"]

        try:
            run(cmd, env)
            results.append({"run": run_id, "ok": True, "overrides": overrides})
        except SystemExit as e:
            results.append({"run": run_id, "ok": False, "overrides": overrides, "error": str(e)})
            break

    (out_dir / f"{Path(cfg).stem}__seed{seed}.json").write_text(json.dumps({
        "config": cfg,
        "seed": seed,
        "seconds": time.time() - start,
        "results": results,
    }, indent=2))

    print("\nSaved summary to:", out_dir / f"{Path(cfg).stem}__seed{seed}.json")

if __name__ == "__main__":
    main()
