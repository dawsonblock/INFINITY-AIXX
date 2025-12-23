from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class TransferSummary:
    time_to_threshold_steps: Dict[str, Optional[int]]
    forgetting: Dict[str, float]
    forward_transfer: Dict[str, Optional[float]]


def _read_eval_tasks_csv(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def compute_transfer_metrics(
    run_dir: Path,
    *,
    thresholds: Dict[str, Optional[float]],
    scratch_time_to_threshold: Optional[Dict[str, int]] = None,
) -> TransferSummary:
    """
    Compute Phase-3 metrics from runs/<run_id>/eval_tasks.csv.

    - time_to_threshold_steps: first global_step where return_mean >= threshold
    - forgetting: best_return - final_return (per task)
    - forward_transfer: scratch_steps / multitask_steps (if baselines provided)
    """
    eval_csv = run_dir / "eval_tasks.csv"
    rows = _read_eval_tasks_csv(eval_csv)

    by_task: Dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        tid = row.get("task_id", "")
        try:
            step = int(float(row.get("global_step", "0")))
            ret = float(row.get("return_mean", "nan"))
        except Exception:
            continue
        by_task.setdefault(tid, []).append((step, ret))

    time_to: Dict[str, Optional[int]] = {}
    forgetting: Dict[str, float] = {}
    ft: Dict[str, Optional[float]] = {}

    for tid, series in by_task.items():
        series.sort(key=lambda x: x[0])
        thr = thresholds.get(tid, None)

        # time-to-threshold
        tval: Optional[int] = None
        if thr is not None:
            for step, ret in series:
                if ret >= float(thr):
                    tval = int(step)
                    break
        time_to[tid] = tval

        # forgetting
        best = max(ret for _, ret in series) if series else float("nan")
        final = series[-1][1] if series else float("nan")
        forgetting[tid] = float(best - final) if (best == best and final == final) else float("nan")

        # forward transfer
        if scratch_time_to_threshold and tid in scratch_time_to_threshold and tval is not None and tval > 0:
            ft[tid] = float(scratch_time_to_threshold[tid]) / float(tval)
        else:
            ft[tid] = None

    return TransferSummary(time_to_threshold_steps=time_to, forgetting=forgetting, forward_transfer=ft)


def write_transfer_artifacts(run_dir: Path, summary: TransferSummary) -> None:
    (run_dir / "transfer.json").write_text(json.dumps({
        "time_to_threshold_steps": summary.time_to_threshold_steps,
        "forgetting": summary.forgetting,
        "forward_transfer": summary.forward_transfer,
    }, indent=2), encoding="utf-8")

    # compact markdown
    lines = ["# Transfer Metrics", ""]
    lines.append("## Time to threshold (steps)")
    for k, v in sorted(summary.time_to_threshold_steps.items()):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Forgetting (best - final)")
    for k, v in sorted(summary.forgetting.items()):
        lines.append(f"- {k}: {v:.3f}")
    lines.append("")
    lines.append("## Forward transfer (scratch_steps / multitask_steps)")
    for k, v in sorted(summary.forward_transfer.items()):
        lines.append(f"- {k}: {v if v is not None else 'n/a'}")
    (run_dir / "transfer.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
