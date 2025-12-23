from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .config import HumanLikeConfig

@dataclass
class SelfModelState:
    z: torch.Tensor           # [B, z_dim]
    pause_budget: torch.Tensor # [B, 1] remaining pause steps

class SelfModel(nn.Module):
    """A small recurrent self-state that tracks 'what am I doing' and uncertainty signals.

    This is intentionally lightweight and stable:
    - GRUCell updates a latent z from (w, r, signals)
    - A head predicts an uncertainty scalar in [0,1]
    - A head predicts an identity-consistency scalar in [0,1]
    """

    def __init__(self, cfg: HumanLikeConfig, w_dim: int, r_dim: int, signals_dim: int):
        super().__init__()
        self.cfg = cfg
        self.in_dim = w_dim + r_dim + signals_dim

        self.gru = nn.GRUCell(self.in_dim, cfg.z_dim)
        self.unc_head = nn.Sequential(
            nn.Linear(cfg.z_dim, cfg.z_dim // 2),
            nn.Tanh(),
            nn.Linear(cfg.z_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.id_head = nn.Sequential(
            nn.Linear(cfg.z_dim, cfg.z_dim // 2),
            nn.Tanh(),
            nn.Linear(cfg.z_dim // 2, 1),
            nn.Sigmoid(),
        )

    def init_state(self, batch: int, device: torch.device) -> SelfModelState:
        z = torch.zeros(batch, self.cfg.z_dim, device=device)
        budget = torch.full((batch, 1), float(self.cfg.max_pause_steps), device=device)
        return SelfModelState(z=z, pause_budget=budget)

    def step(
        self,
        state: SelfModelState,
        w: torch.Tensor,   # [B, w_dim]
        r: torch.Tensor,   # [B, r_dim]
        signals: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        # signals -> [B, signals_dim]
        sig_list = []
        for k in sorted(signals.keys()):
            t = signals[k]
            if t.ndim == 1:
                t = t.unsqueeze(-1)
            sig_list.append(t)
        sig = torch.cat(sig_list, dim=-1) if sig_list else torch.zeros(w.shape[0], 0, device=w.device)

        x = torch.cat([w, r, sig], dim=-1)
        z_new = self.gru(x, state.z)

        uncertainty = self.unc_head(z_new)  # [B,1]
        identity = self.id_head(z_new)      # [B,1]

        return {
            "state": SelfModelState(z=z_new, pause_budget=state.pause_budget),
            "uncertainty": uncertainty,
            "identity": identity,
        }
