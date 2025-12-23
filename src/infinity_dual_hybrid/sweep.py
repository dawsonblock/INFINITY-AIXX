from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Any

from .train_cli import main as train_main

def _parse_kv_list(kvs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kv in kvs:
        if "=" not in kv:
            raise SystemExit(f"Invalid override '{kv}'. Expected key=value.")
        k, v = kv.split("=", 1)
        # best-effort type cast
        if v.lower() in {"true", "false"}:
            vv: Any = (v.lower() == "true")
        else:
            try:
                vv = int(v)
            except ValueError:
                try:
                    vv = float(v)
                except ValueError:
                    vv = v
        out[k] = vv
    return out

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Grid sweep runner for Infinity-human.")
    p.add_argument("--config", type=str, required=True, help="Path to base YAML config.")
    p.add_argument("--set", nargs="*", default=[], help="Fixed overrides key=value applied to every run.")
    p.add_argument("--grid", nargs="*", default=[], help="Grid overrides like key=a,b,c (comma-separated).")
    p.add_argument("--repeat", type=int, default=1, help="Repeat each config N times with different seeds.")
    args = p.parse_args(argv)

    base_cfg = Path(args.config)
    if not base_cfg.exists():
        raise SystemExit(f"Config not found: {base_cfg}")

    fixed = _parse_kv_list(args.set)

    grid_keys = []
    grid_vals = []
    for item in args.grid:
        if "=" not in item:
            raise SystemExit(f"Invalid grid '{item}'. Expected key=a,b,c")
        k, vs = item.split("=", 1)
        vals = [v for v in vs.split(",") if v != ""]
        if not vals:
            raise SystemExit(f"Grid '{item}' has no values.")
        grid_keys.append(k)
        grid_vals.append(vals)

    combos = list(itertools.product(*grid_vals)) if grid_vals else [()]

    sweep_dir = Path("runs") / "sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_id = f"sweep_{base_cfg.stem}_{len(combos)}x{args.repeat}"
    (sweep_dir / f"{sweep_id}.json").write_text(json.dumps({
        "config": str(base_cfg),
        "fixed": fixed,
        "grid": dict(zip(grid_keys, grid_vals)),
        "repeat": args.repeat,
    }, indent=2))

    rc = 0
    for combo in combos:
        combo_overrides = dict(zip(grid_keys, combo))
        for rep in range(args.repeat):
            # pass overrides via CLI --override (supported by train_cli)
            ov = dict(fixed)
            ov.update(combo_overrides)
            # seed bump if provided, else set
            if "seed" in ov:
                try:
                    ov["seed"] = int(ov["seed"]) + rep
                except Exception:
                    pass
            else:
                ov["seed"] = rep
            override_json = json.dumps(ov)
            run_argv = ["--config", str(base_cfg), "--override", override_json]
            rc = train_main(run_argv)
            if rc != 0:
                return rc
    return rc

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
