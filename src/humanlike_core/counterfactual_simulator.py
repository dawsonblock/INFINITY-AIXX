from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional

import torch

from .config import HumanLikeConfig

@dataclass
class CFResult:
    action: torch.Tensor        # [B]
    score: torch.Tensor         # [B,1]
    best_idx: torch.Tensor      # [B]

class CounterfactualSimulator:
    """Cheap lookahead over proposed actions.

    This stays model-agnostic by accepting callables:
      proposal_fn(w,z) -> a_candidates [K,B,1] (int actions)
      model_step_fn(latent, a) -> (latent_next, r_hat [B,1], u_hat [B,1])
      value_fn(latent) -> v_hat [B,1]
    """

    def __init__(self, cfg: HumanLikeConfig):
        self.cfg = cfg

    @torch.no_grad()
    def choose(
        self,
        latent: torch.Tensor,  # [B, d]
        proposal_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        model_step_fn: Callable[[torch.Tensor, torch.Tensor], tuple],
        value_fn: Callable[[torch.Tensor], torch.Tensor],
        w: torch.Tensor,
        z: torch.Tensor,
        cf_horizon: int = None,
        cf_lambda_uncertainty: float | None = None,
    ) -> CFResult:
        K = int(self.cfg.cf_k)
        horizon = int(self.cfg.cf_horizon) if cf_horizon is None else int(cf_horizon)
        lam_u = float(self.cfg.cf_lambda_uncertainty) if cf_lambda_uncertainty is None else float(cf_lambda_uncertainty)

        a_cand = proposal_fn(w, z)  # [K,B,1]
        if a_cand.ndim != 3:
            raise ValueError("proposal_fn must return [K,B,1]")

        B = a_cand.shape[1]
        scores = torch.zeros(K, B, 1, device=latent.device)

        for k in range(K):
            lat_k = latent
            a_k = a_cand[k]  # [B,1]
            total = torch.zeros(B, 1, device=latent.device)
            lam_done = float(getattr(self.cfg, "cf_lambda_done", 0.25))
            for _ in range(horizon):
                step_out = model_step_fn(lat_k, a_k)
                if isinstance(step_out, (tuple, list)) and len(step_out) == 4:
                    lat_k, r_hat, u_hat, done_logit = step_out
                    done_prob = torch.sigmoid(done_logit)
                else:
                    lat_k, r_hat, u_hat = step_out
                    done_prob = torch.zeros_like(r_hat)

                v_hat = value_fn(lat_k)
                # maximize reward/value, penalize uncertainty and predicted termination
                total = total + r_hat + v_hat - lam_u * u_hat - lam_done * done_prob
            scores[k] = total

        best_idx = scores.squeeze(-1).argmax(dim=0)  # [B]
        best_a = a_cand[best_idx, torch.arange(B, device=latent.device), 0]
        best_score = scores[best_idx, torch.arange(B, device=latent.device)]
        return CFResult(action=best_a, score=best_score, best_idx=best_idx)
