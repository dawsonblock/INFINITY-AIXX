"""Checkpoint roundtrip.

Keep imports inside the test so `pytest --collect-only` stays fast.
"""

import os


def test_checkpoint_save_load_roundtrip(tmp_path):
    import torch
    from infinity_dual_hybrid.config import get_config_for_env
    from infinity_dual_hybrid.agent import build_agent
    from infinity_dual_hybrid.ppo_trainer import PPOTrainer

    cfg = get_config_for_env("CartPole-v1")
    agent = build_agent(cfg.agent)
    trainer = PPOTrainer(agent, cfg.ppo, device="cpu", seed=123)

    ckpt = tmp_path / "ckpt.pt"
    trainer.save(str(ckpt))

    # load into a fresh trainer
    agent2 = build_agent(cfg.agent)
    trainer2 = PPOTrainer(agent2, cfg.ppo, device="cpu", seed=123)
    trainer2.load(str(ckpt))

    # spot-check: parameter tensors match
    for (n1, p1), (n2, p2) in zip(trainer.agent.named_parameters(), trainer2.agent.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2)
