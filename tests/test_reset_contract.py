"""Reset contract for HumanLikeRouter.

Keep torch imports inside the test so pytest collection stays fast.
"""


def test_reset_contract_zeroes_states():
    import torch
    from humanlike_core.config import HumanLikeConfig
    from humanlike_core.decision_router import HumanLikeRouter

    cfg = HumanLikeConfig(z_persist_across_episodes=False, z_reset_on_env_reset=True, enabled=True)
    router = HumanLikeRouter(cfg, w_dim=32, r_dim=32, act_dim=2)

    B = 3
    device = torch.device("cpu")
    states = router.init_states(B, device)

    states["self"].z += 1.0
    states["emo"].e += 2.0

    done = torch.tensor([[True], [False], [True]])
    states2 = router.reset_on_env_reset(states, done)

    assert torch.all(states2["self"].z[0] == 0)
    assert torch.all(states2["self"].z[2] == 0)
    assert torch.all(states2["emo"].e[0] == 0)
    assert torch.all(states2["emo"].e[2] == 0)
    assert torch.any(states2["self"].z[1] != 0)
