from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .best_checkpoint import summarize_run, best_eval_from_metrics_csv

def _read_last_csv_row(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            last = None
            for row in reader:
                last = row
            return last
    except Exception:
        return None

def _read_meta(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "run_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _find_metrics_csv(run_dir: Path) -> Optional[Path]:
    direct = run_dir / "metrics.csv"
    if direct.exists():
        return direct
    logs = run_dir / "logs"
    if logs.exists():
        for sub in logs.glob("**/metrics.csv"):
            return sub
    return None

def _best_eval_from_eval_best_json(run_dir: Path) -> Optional[float]:
    p = run_dir / "eval_best.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        v = metrics.get("eval/return_mean")
        if isinstance(v, (int, float)):
            return float(v)
    stats = payload.get("stats")
    if isinstance(stats, dict):
        v = stats.get("eval_mean_reward")
        if isinstance(v, (int, float)):
            return float(v)
    return None

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize runs into a comparable ranked report.")
    p.add_argument("--runs_dir", type=str, default="runs", help="Runs directory (default: runs)")
    p.add_argument("--checkpoints_dir", type=str, default=None, help="Checkpoints directory (default: sibling 'checkpoints')")
    p.add_argument("--out", type=str, default="runs/report.md", help="Output markdown path")
    args = p.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    ckpt_root = Path(args.checkpoints_dir) if args.checkpoints_dir else (runs_dir.parent / "checkpoints")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    for run in sorted([p for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]):
        meta = _read_meta(run)
        mpath = _find_metrics_csv(run)
        last_metrics = _read_last_csv_row(mpath) if mpath else None
        info = summarize_run(run, ckpt_root)

        # best eval from full CSV if available; else fallback to last row values
        best_eval = None
        # Prefer fresh eval outputs if present
        best_eval = _best_eval_from_eval_best_json(run_dir)
        if best_eval is not None:
            best_key = "eval_best.json:eval/return_mean"
        best_key = None
        if mpath:
            be = best_eval_from_metrics_csv(mpath)
            if be:
                best_key, best_eval = be
        if best_eval is None and last_metrics:
            for k in ("eval/mean_reward","eval_mean_reward","eval/return_mean","eval_return_mean","episode_return_mean"):
                if k in last_metrics and last_metrics[k] not in ("", None):
                    try:
                        best_eval = float(last_metrics[k])
                        best_key = k
                        break
                    except Exception:
                        pass

        items.append({
            "run_id": run.name,
            "meta": meta,
            "last": last_metrics or {},
            "best_checkpoint": info.get("best_checkpoint"),
            "best_eval": best_eval,
            "best_eval_key": best_key,
        })

    def score(x: Dict[str, Any]) -> float:
        v = x.get("best_eval")
        return float(v) if v is not None else float("-inf")

    items_sorted = sorted(items, key=score, reverse=True)

    md: List[str] = []
    md.append("# Infinity Runs Report\n")
    md.append(f"Runs scanned: {len(items_sorted)}\n")
    md.append("\nRanked by best available eval metric (higher is better).\n")
    md.append("| rank | run_id | best_eval | metric | seed | encoder | miras | world_model | best_checkpoint |")
    md.append("|---:|---|---:|---|---:|---|---|---|---|")

    for i, it in enumerate(items_sorted, start=1):
        meta = it["meta"] or {}
        cfg = (meta.get("config") or {}) if isinstance(meta.get("config"), dict) else {}
        seed = cfg.get("seed", meta.get("seed", ""))
        encoder = cfg.get("encoder_type", cfg.get("encoder", meta.get("encoder","")))
        miras = cfg.get("use_miras", cfg.get("miras", meta.get("miras","")))
        wm = cfg.get("use_world_model", cfg.get("world_model", meta.get("world_model","")))
        be = it.get("best_eval")
        be_s = f"{be:.4f}" if isinstance(be,(int,float)) and be is not None else ""
        key = it.get("best_eval_key") or ""
        ckpt = it.get("best_checkpoint") or ""
        md.append(f"| {i} | {it['run_id']} | {be_s} | {key} | {seed} | {encoder} | {miras} | {wm} | {ckpt} |")

    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote report: {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())