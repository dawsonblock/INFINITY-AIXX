"""PPO rollout smoke test.

Imports kept inside the test to keep `pytest --collect-only` fast.
"""


def test_collect_rollouts_smoke():
    import torch
    import importlib
    import pytest
    if importlib.util.find_spec('gymnasium') is None and importlib.util.find_spec('gym') is None:
        pytest.skip('gymnasium/gym not installed')

    from infinity_dual_hybrid.config import get_config_for_env
    from infinity_dual_hybrid.envs import make_envs
    from infinity_dual_hybrid.agent import build_agent
    from infinity_dual_hybrid.ppo_trainer import PPOTrainer

    cfg = get_config_for_env("CartPole-v1")
    envs = make_envs("CartPole-v1", num_envs=2, cfg=cfg)
    agent = build_agent(cfg.agent)
    trainer = PPOTrainer(agent, cfg.ppo, device="cpu", seed=123)

    rollouts = trainer.collect_rollouts(envs, steps=8)
    assert rollouts.obs.shape[0] == 8
    assert rollouts.actions.shape[0] == 8
    assert torch.isfinite(rollouts.returns).all()

    for e in envs:
        try:
            e.close()
        except Exception:
            pass