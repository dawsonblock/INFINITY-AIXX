from __future__ import annotations

from pathlib import Path
import json
from infinity_dual_hybrid.best_checkpoint import best_checkpoint_path, summarize_run

def test_best_checkpoint_prefers_best(tmp_path: Path):
    ckpt = tmp_path / "checkpoints" / "run_x"
    run = tmp_path / "runs" / "run_x"
    ckpt.mkdir(parents=True)
    run.mkdir(parents=True)
    (ckpt / "latest.pt").write_bytes(b"latest")
    (ckpt / "best.pt").write_bytes(b"best")
    info = summarize_run(run, tmp_path / "checkpoints")
    assert info["best_checkpoint"].endswith("best.pt")

def test_best_checkpoint_falls_back_latest(tmp_path: Path):
    ckpt = tmp_path / "checkpoints" / "run_y"
    ckpt.mkdir(parents=True)
    (ckpt / "latest.pt").write_bytes(b"latest")
    assert best_checkpoint_path(ckpt).name == "latest.pt"
