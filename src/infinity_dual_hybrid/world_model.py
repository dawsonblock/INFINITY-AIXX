from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WorldModelOut:
    """
    Latent dynamics model outputs.

    We model:
      w_{t+1} ~ N(mu_w, diag(exp(logvar_w)))
      r_{t}   ~ N(mu_r, exp(logvar_r))
      done_t  ~ Bernoulli(sigmoid(done_logit))

    Uncertainty u_hat is a compact scalar summary (mean std) that can be used by routers.
    """
    w_next: torch.Tensor         # [B, D] mean (for convenience)
    w_next_mu: torch.Tensor      # [B, D]
    w_next_logvar: torch.Tensor  # [B, D]
    r_hat: torch.Tensor          # [B, 1] mean
    r_logvar: torch.Tensor       # [B, 1]
    done_logit: torch.Tensor     # [B, 1]
    u_hat: torch.Tensor          # [B, 1]


class WorldModel(nn.Module):
    """
    Learned latent world model for Infinity Dual Hybrid.

    Inputs:
        w: [B, D] fused latent/workspace
        a: [B] or [B,1] discrete action index (long/int) OR one-hot float [B, A]

    Outputs:
        mu/logvar for next latent, mu/logvar for reward, done logit, scalar uncertainty.

    Notes:
      - This is intentionally lightweight and CPU-friendly.
      - Uncertainty is heteroscedastic (learned predictive log-variance), not a heuristic.
    """

    def __init__(
        self,
        d_model: Optional[int] = None,
        act_dim: int = 0,
        hidden: int = 256,
        min_logvar: float = -8.0,
        max_logvar: float = 4.0,
        *,
        w_dim: Optional[int] = None,
    ):
        super().__init__()
        if d_model is None:
            d_model = w_dim
        if d_model is None:
            raise TypeError("WorldModel requires d_model (or w_dim) to be set")
        if act_dim <= 0:
            raise TypeError("WorldModel requires act_dim > 0")

        self.d_model = int(d_model)
        self.act_dim = int(act_dim)
        self.hidden = int(hidden)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        self.fc1 = nn.Linear(self.d_model + self.act_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.hidden)

        # next-latent distribution
        self.w_mu = nn.Linear(self.hidden, self.d_model)
        self.w_logvar = nn.Linear(self.hidden, self.d_model)

        # reward distribution
        self.r_mu = nn.Linear(self.hidden, 1)
        self.r_logvar = nn.Linear(self.hidden, 1)

        # termination
        self.done = nn.Linear(self.hidden, 1)

    def _action_features(self, a: torch.Tensor) -> torch.Tensor:
        # Accept [B], [B,1] integer actions or one-hot [B,A]
        if a.dim() == 2 and a.shape[-1] == self.act_dim and a.dtype.is_floating_point:
            return a
        if a.dim() == 2 and a.shape[-1] == 1:
            a = a.view(-1)
        if a.dim() == 1:
            a = a.long().clamp(0, self.act_dim - 1)
            return F.one_hot(a, num_classes=self.act_dim).float()
        raise ValueError(f"Unsupported action tensor shape for WorldModel: {tuple(a.shape)}")

    def forward(self, w: torch.Tensor, a: torch.Tensor) -> WorldModelOut:
        a_feat = self._action_features(a)
        x = torch.cat([w, a_feat], dim=-1)
        h = F.silu(self.fc1(x))
        h = F.silu(self.fc2(h))

        w_mu = w + self.w_mu(h)  # residual dynamics
        w_logvar = self.w_logvar(h).clamp(self.min_logvar, self.max_logvar)

        r_mu = self.r_mu(h)
        r_logvar = self.r_logvar(h).clamp(self.min_logvar, self.max_logvar)

        done_logit = self.done(h)

        # Compact scalar uncertainty: mean std of next-latent + reward std
        w_std = torch.exp(0.5 * w_logvar).mean(dim=-1, keepdim=True)
        r_std = torch.exp(0.5 * r_logvar)
        u_hat = 0.5 * (w_std + r_std)

        return WorldModelOut(
            w_next=w_mu,
            w_next_mu=w_mu,
            w_next_logvar=w_logvar,
            r_hat=r_mu,
            r_logvar=r_logvar,
            done_logit=done_logit,
            u_hat=u_hat,
        )

    @torch.no_grad()
    def step(self, w: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Back-compat step() used by earlier counterfactual code.
        Returns: (w_next_mean, r_mean, u_hat)
        """
        out = self.forward(w, a)
        return out.w_next, out.r_hat, out.u_hat

    @torch.no_grad()
    def model_step(self, w: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preferred interface for planners/routers.
        Returns: (w_next_mean, r_mean, u_hat, done_logit)
        """
        out = self.forward(w, a)
        return out.w_next, out.r_hat, out.u_hat, out.done_logit

    def loss(
        self,
        w: torch.Tensor,
        a: torch.Tensor,
        w_next_target: torch.Tensor,
        r_target: torch.Tensor,
        done_target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Heteroscedastic negative log-likelihood for latent and reward + BCE for done.

        mask: [B,1] optional weighting (e.g., 1-done) to avoid training across resets.
        """
        out = self.forward(w, a)

        # Gaussian NLL for next latent
        # nll = 0.5*(logvar + (x-mu)^2/exp(logvar))
        w_var = torch.exp(out.w_next_logvar)
        nll_w = 0.5 * (out.w_next_logvar + (w_next_target - out.w_next_mu) ** 2 / (w_var + 1e-8))
        nll_w = nll_w.mean(dim=-1, keepdim=True)  # [B,1]

        # Gaussian NLL for reward
        r_var = torch.exp(out.r_logvar)
        nll_r = 0.5 * (out.r_logvar + (r_target - out.r_hat) ** 2 / (r_var + 1e-8))  # [B,1]

        # BCE for done
        bce_done = F.binary_cross_entropy_with_logits(out.done_logit, done_target, reduction="none")  # [B,1]

        if mask is not None:
            nll_w = nll_w * mask
            nll_r = nll_r * mask
            bce_done = bce_done * torch.ones_like(bce_done)  # keep done supervised even when masked? choose no mask
            # We intentionally do NOT mask done: it is defined at boundaries.

        loss_w = nll_w.mean()
        loss_r = nll_r.mean()
        loss_done = bce_done.mean()
        loss_total = loss_w + loss_r + loss_done

        metrics = {
            "wm/loss_total": float(loss_total.detach().cpu().item()),
            "wm/loss_w_nll": float(loss_w.detach().cpu().item()),
            "wm/loss_r_nll": float(loss_r.detach().cpu().item()),
            "wm/loss_done_bce": float(loss_done.detach().cpu().item()),
            "wm/u_mean": float(out.u_hat.detach().mean().cpu().item()),
        }
        return loss_total, metrics