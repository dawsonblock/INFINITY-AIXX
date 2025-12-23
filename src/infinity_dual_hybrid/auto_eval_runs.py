from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .best_checkpoint import best_checkpoint_path
from .eval import main as eval_main


def _iter_run_ids(runs_dir: Path) -> List[str]:
    # Prefer index.jsonl if present (created by run_registry)
    index = runs_dir / "index.jsonl"
    if index.exists():
        run_ids: List[str] = []
        for line in index.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rid = obj.get("run_id")
            if isinstance(rid, str) and rid:
                run_ids.append(rid)
        # Keep insertion order but unique
        seen = set()
        out: List[str] = []
        for rid in run_ids:
            if rid in seen:
                continue
            seen.add(rid)
            out.append(rid)
        return out

    # Fallback: directory scan
    if not runs_dir.exists():
        return []
    run_ids = [p.name for p in runs_dir.iterdir() if p.is_dir()]
    run_ids.sort()
    return run_ids


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def evaluate_best_for_run(
    run_id: str,
    runs_dir: Path,
    checkpoints_dir: Path,
    env_id: str,
    episodes: int,
    seed: int,
    device: str,
    force: bool,
) -> Tuple[str, Optional[float], Optional[Path], Optional[Path]]:
    run_dir = runs_dir / run_id
    ckpt_dir = checkpoints_dir / run_id
    ckpt = best_checkpoint_path(ckpt_dir)
    if ckpt is None:
        return run_id, None, None, None

    out_json = run_dir / "eval_best.json"
    if out_json.exists() and not force:
        payload = _load_json(out_json)
        if payload and isinstance(payload.get("stats"), dict):
            stats = payload["stats"]
            score = None
            metrics = payload.get("metrics")
            if isinstance(metrics, dict):
                score = metrics.get("eval/return_mean")
            if score is None:
                score = stats.get("eval_mean_reward")
            if isinstance(score, (int, float)):
                return run_id, float(score), ckpt, out_json

    run_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "--env",
        env_id,
        "--checkpoint",
        str(ckpt),
        "--episodes",
        str(int(episodes)),
        "--seed",
        str(int(seed)),
        "--device",
        device,
        "--out",
        str(out_json),
    ]
    rc = eval_main(argv)
    if rc != 0:
        return run_id, None, ckpt, out_json

    payload = _load_json(out_json)
    score: Optional[float] = None
    if payload and isinstance(payload.get("stats"), dict):
        s = payload["stats"]
        v = None
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            v = metrics.get("eval/return_mean")
        if v is None:
            v = s.get("eval_mean_reward")
        if isinstance(v, (int, float)):
            score = float(v)
    return run_id, score, ckpt, out_json


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate best checkpoint for each run and rank results")
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    ap.add_argument("--env", "--env-id", dest="env_id", type=str, default="CartPole-v1")
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--force", action="store_true", help="Re-evaluate even if eval_best.json exists")
    ap.add_argument("--limit", type=int, default=0, help="Max runs to evaluate (0 = no limit)")
    ap.add_argument("--out_csv", type=str, default="runs/eval_rank.csv")
    ap.add_argument("--out_md", type=str, default="runs/eval_rank.md")
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    checkpoints_dir = Path(args.checkpoints_dir)

    run_ids = _iter_run_ids(runs_dir)
    if args.limit and args.limit > 0:
        run_ids = run_ids[: int(args.limit)]

    rows: List[Dict[str, Any]] = []
    for rid in run_ids:
        run_id, score, ckpt, out_json = evaluate_best_for_run(
            rid,
            runs_dir=runs_dir,
            checkpoints_dir=checkpoints_dir,
            env_id=args.env_id,
            episodes=int(args.episodes),
            seed=int(args.seed),
            device=str(args.device),
            force=bool(args.force),
        )
        rows.append(
            {
                "run_id": run_id,
                "eval_mean_reward": score,
                "best_checkpoint": str(ckpt) if ckpt else None,
                "eval_json": str(out_json) if out_json else None,
            }
        )

    ranked = sorted(
        rows,
        key=lambda r: (r["eval_mean_reward"] is None, -(r["eval_mean_reward"] or 0.0)),
    )

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "eval_mean_reward", "best_checkpoint", "eval_json"])
        w.writeheader()
        for r in ranked:
            w.writerow(r)

    # Write Markdown
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Eval ranking")
    lines.append("")
    lines.append(f"- env: `{args.env_id}`")
    lines.append(f"- episodes: `{int(args.episodes)}`")
    lines.append(f"- seed: `{int(args.seed)}`")
    lines.append("")
    lines.append("| rank | run_id | eval_mean_reward | best_checkpoint | eval_json |")
    lines.append("|---:|---|---:|---|---|")
    for i, r in enumerate(ranked, start=1):
        score = r["eval_mean_reward"]
        score_s = "" if score is None else f"{float(score):.6f}"
        lines.append(f"| {i} | `{r['run_id']}` | {score_s} | `{r['best_checkpoint'] or ''}` | `{r['eval_json'] or ''}` |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
