from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

_EVAL_KEYS = (
    "eval/mean_reward",
    "eval_mean_reward",
    "eval/return_mean",
    "eval_return_mean",
    "eval/episode_return_mean",
    "episode_return_mean",
)

def best_checkpoint_path(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    best = ckpt_dir / "best.pt"
    if best.exists():
        return best
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return latest
    # fallback: pick newest *.pt
    pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0] if pts else None

def best_eval_from_metrics_csv(metrics_csv: Path) -> Optional[Tuple[str, float]]:
    if not metrics_csv.exists():
        return None
    best_key = None
    best_val = None
    try:
        with metrics_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for k in _EVAL_KEYS:
                    if k in row and row[k] not in (None, ""):
                        try:
                            v = float(row[k])
                        except Exception:
                            continue
                        if best_val is None or v > best_val:
                            best_val = v
                            best_key = k
        if best_val is None or best_key is None:
            return None
        return best_key, float(best_val)
    except Exception:
        return None

def _find_metrics_csv(run_dir: Path) -> Optional[Path]:
    direct = run_dir / "metrics.csv"
    if direct.exists():
        return direct
    for sub in (run_dir / "logs").glob("**/metrics.csv"):
        return sub
    return None

def summarize_run(run_dir: Path, ckpt_root: Path) -> Dict[str, Any]:
    run_id = run_dir.name
    ckpt_dir = ckpt_root / run_id
    ckpt = best_checkpoint_path(ckpt_dir)
    metrics_csv = _find_metrics_csv(run_dir)
    best_eval = best_eval_from_metrics_csv(metrics_csv) if metrics_csv else None
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "ckpt_dir": str(ckpt_dir),
        "best_checkpoint": str(ckpt) if ckpt else None,
        "best_eval_key": best_eval[0] if best_eval else None,
        "best_eval": best_eval[1] if best_eval else None,
    }

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Resolve the best checkpoint for a run")
    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a path")
    args = ap.parse_args(argv)

    run_dir = Path(args.runs_dir) / args.run_id
    ckpt_root = Path(args.checkpoints_dir)
    info = summarize_run(run_dir, ckpt_root)

    if args.json:
        import json
        print(json.dumps(info, indent=2))
    else:
        if info["best_checkpoint"] is None:
            print("")
            return 2
        print(info["best_checkpoint"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
