"""VectorEnv smoke test runner.

Usage:
    python scripts/vectorenv_smoke.py --config configs/cartpole_ultra_fast_async_storage.yaml --iters 3
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataclasses import asdict

from infinity_dual_hybrid.config import get_config_for_env, TrainConfig
from infinity_dual_hybrid.envs import make_vector_env
from infinity_dual_hybrid.agent import build_agent
from infinity_dual_hybrid.ppo_trainer import PPOTrainer
from infinity_dual_hybrid.train_cli import _load_yaml, _deep_update, _resolve_device
from infinity_dual_hybrid.utils import set_global_seed


def apply_dict(obj, d):
    for k, v in d.items():
        if not hasattr(obj, k):
            continue
        cur = getattr(obj, k)
        if isinstance(v, dict) and hasattr(cur, "__dict__"):
            apply_dict(cur, v)
        else:
            setattr(obj, k, v)


def load_cfg(yaml_path: str) -> TrainConfig:
    y = _load_yaml(yaml_path)
    env_id = y.get("env_id") or y.get("env") or "CartPole-v1"
    cfg = get_config_for_env(env_id)

    # Apply top-level scalars
    for key in ["env_id", "seed", "device", "log_interval", "save_interval", "save_path"]:
        if key in y:
            setattr(cfg, key, y[key])

    # Agent overrides
    if isinstance(y.get("agent"), dict):
        agent_y = dict(y["agent"])
        # Support shorthand: agent.backbone: "gru"
        if isinstance(agent_y.get("backbone"), str):
            bb = agent_y.pop("backbone").strip().lower()
            if bb in ("gru", "rnn"):
                cfg.agent.backbone.use_mamba = False
                cfg.agent.backbone.use_attention = False
            elif bb in ("hybrid", "mamba", "mamba2"):
                cfg.agent.backbone.use_mamba = True
            # keep other fields as-is
        apply_dict(cfg.agent, agent_y)
        cfg.agent.sync_dims()

    # PPO overrides
    if isinstance(y.get("ppo"), dict):
        apply_dict(cfg.ppo, y["ppo"])

    cfg.device = _resolve_device(getattr(cfg, "device", "auto"))
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_global_seed(getattr(cfg, "seed", 0), deterministic=bool(getattr(cfg, "deterministic", False)))

    n_envs = int(getattr(cfg.ppo, "n_envs", getattr(cfg.ppo, "num_envs", 8)))
        try:
        envs = make_vector_env(
            cfg.env_id,
            num_envs=n_envs,
            cfg=cfg,
            async_vector=bool(getattr(cfg.ppo, "vector_async", False)),
            seed=getattr(cfg, "seed", None),
        )
    except Exception as e:
        print("VectorEnv not available:", e)
        print("Install deps: pip install -e '.[dev]'  (or at least: pip install gymnasium)")
        return 2


    agent = build_agent(cfg)
    trainer = PPOTrainer(cfg.ppo, agent.cfg, device=cfg.device, seed=getattr(cfg, "seed", None))
    trainer.agent = agent.to(trainer.device)

    t0 = time.time()
    for it in range(args.iters):
        rollouts = trainer.collect_rollouts(envs, steps=int(cfg.ppo.rollout_steps))
        assert rollouts.observations.ndim == 2
        assert rollouts.w.shape[0] == rollouts.observations.shape[0]
        metrics = trainer.train_step(rollouts)

        for k in ["policy_loss", "value_loss", "entropy", "mean_return", "mean_reward"]:
            v = float(metrics.get(k, 0.0))
            if not math.isfinite(v):
                raise RuntimeError(f"Non-finite metric {k}={v}")

        print(
            f"iter {it+1}/{args.iters} "
            f"policy_loss={metrics['policy_loss']:.4f} "
            f"value_loss={metrics['value_loss']:.4f} "
            f"mean_return={metrics['mean_return']:.2f} "
            f"episodes_finished={metrics.get('episodes/episodes_finished',0):.0f}"
        )

    dt = time.time() - t0
    steps = int(cfg.ppo.rollout_steps) * n_envs * args.iters
    print(f"OK: {steps} env-steps in {dt:.2f}s => {steps/dt:.0f} steps/s")

    try:
        envs.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())