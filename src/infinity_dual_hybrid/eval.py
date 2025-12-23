from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import get_config_for_env
from .envs import make_envs
from .agent import build_agent
from .ppo_trainer import PPOTrainer


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore
            return "mps"
        return "cpu"
    return device


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Infinity Human evaluation")
    ap.add_argument("--env", "--env-id", dest="env_id", default="CartPole-v1")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--out", type=str, default=None, help="Write results JSON here")
    args = ap.parse_args(argv)

    device = _resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = get_config_for_env(args.env_id)
    envs = make_envs(args.env_id, num_envs=1, cfg=cfg)

    agent = build_agent(cfg.agent).to(device)  # type: ignore[attr-defined]
    trainer = PPOTrainer(agent, cfg.ppo, device=device, seed=args.seed)
    trainer.load(args.checkpoint)

    stats = trainer.evaluate(envs, num_episodes=args.episodes, deterministic=True)

    # Normalize evaluation metrics
    return_mean = float(stats.get("eval_mean_reward", stats.get("return_mean", 0.0)))
    return_std = float(stats.get("eval_std_reward", stats.get("return_std", 0.0)))
    length_mean = float(stats.get("eval_mean_length", stats.get("length_mean", 0.0)))
    metrics = {
        "eval/return_mean": return_mean,
        "eval/return_std": return_std,
        "eval/length_mean": length_mean,
    }
    
import sys as _sys
import platform as _platform
import subprocess as _subprocess

def _git_rev() -> Optional[str]:
    try:
        out = _subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=_subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None

payload = {
    "env_id": args.env_id,
    "checkpoint": args.checkpoint,
    "episodes": args.episodes,
    "seed": args.seed,
    "device": device,
    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    "argv": list(argv or []),
    "python": _sys.version.replace("\n", " "),
    "platform": _platform.platform(),
    "torch": getattr(torch, "__version__", None),
    "numpy": getattr(np, "__version__", None),
    "git_rev": _git_rev(),
    "metrics": metrics,
}

if args.out:

        Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))

    for env in envs:
        try:
            env.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
