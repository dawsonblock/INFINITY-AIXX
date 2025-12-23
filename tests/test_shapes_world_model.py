"""World model output shape contracts.

Imports kept inside the test to avoid slow collection.
"""


def test_world_model_shapes():
    import torch
    from infinity_dual_hybrid.world_model import WorldModel

    B = 4
    d = 16
    act_dim = 3
    wm = WorldModel(w_dim=d, act_dim=act_dim, hidden=64)

    w = torch.randn(B, d)
    a = torch.randint(0, act_dim, (B,))
    out = wm(w, a)

    assert out.w_next.shape == (B, d)
    assert out.r_hat.shape == (B, 1)
    assert out.done_logit.shape == (B, 1)
    assert out.u_hat.shape == (B, 1)
