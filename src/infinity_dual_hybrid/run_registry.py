from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _safe_git_rev(cwd: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None

def make_run_id(prefix: str = "run") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"

def ensure_dirs(root: Path, run_id: str) -> Dict[str, Path]:
    runs_dir = root / "runs"
    ckpt_dir = root / "checkpoints"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / run_id).mkdir(parents=True, exist_ok=True)
    return {
        "runs_dir": runs_dir,
        "checkpoints_dir": ckpt_dir,
        "run_dir": run_dir,
        "ckpt_dir": ckpt_dir / run_id,
    }

def write_run_meta(run_dir: Path, *, config: Any, extra: Optional[Dict[str, Any]] = None, repo_root: Optional[Path]=None) -> Path:
    meta: Dict[str, Any] = {
        "run_id": run_dir.name,
        "started_at": _utc_now_iso(),
    }
    if is_dataclass(config):
        meta["config"] = asdict(config)
    else:
        meta["config"] = config
    if repo_root is not None:
        meta["git_rev"] = _safe_git_rev(repo_root)
    if extra:
        meta.update(extra)
    path = run_dir / "run_meta.json"
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path

def append_run_index(runs_dir: Path, record: Dict[str, Any]) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / "index.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return path
