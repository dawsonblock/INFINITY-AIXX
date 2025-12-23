from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

import torch
import torch.nn as nn

from .config import HumanLikeConfig

@dataclass
class EmotionState:
    e: torch.Tensor            # [B, e_dim]
    paused_steps: torch.Tensor # [B, 1]

class EmotionController(nn.Module):
    """Maps signals + self-model outputs to an emotion/control vector.

    Outputs:
      - e: low-dim emotion embedding
      - entropy_coef: scalar exploration weight for PPO
    """

    def __init__(self, cfg: HumanLikeConfig, signals_dim: int):
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Sequential(
            nn.Linear(signals_dim + 2, 64),  # + uncertainty + identity
            nn.Tanh(),
            nn.Linear(64, cfg.e_dim),
        )
        self.entropy_head = nn.Sequential(
            nn.Linear(cfg.e_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def init_state(self, batch: int, device: torch.device) -> EmotionState:
        e = torch.zeros(batch, self.cfg.e_dim, device=device)
        paused = torch.zeros(batch, 1, device=device)
        return EmotionState(e=e, paused_steps=paused)

    def step(
        self,
        state: EmotionState,
        signals: Dict[str, torch.Tensor],
        uncertainty: torch.Tensor,  # [B,1]
        identity: torch.Tensor,     # [B,1]
    ) -> Dict[str, Any]:
        sig_list = []
        for k in sorted(signals.keys()):
            t = signals[k]
            if t.ndim == 1:
                t = t.unsqueeze(-1)
            sig_list.append(t)
        sig = torch.cat(sig_list, dim=-1) if sig_list else torch.zeros(uncertainty.shape[0], 0, device=uncertainty.device)

        x = torch.cat([sig, uncertainty, identity], dim=-1)
        e = self.fc(x)

        # map to entropy coef in [min,max] around base
        p = self.entropy_head(e)  # [B,1] in [0,1]
        ent = self.cfg.entropy_min + (self.cfg.entropy_max - self.cfg.entropy_min) * p
        # pull slightly toward base for stability
        ent = 0.5 * ent + 0.5 * float(self.cfg.entropy_base)

        return {"state": EmotionState(e=e, paused_steps=state.paused_steps), "e": e, "entropy_coef": ent}
