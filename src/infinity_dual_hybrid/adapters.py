from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdapterBankConfig:
    enabled: bool = False
    num_adapters: int = 4          # includes non-identity adapters (we always keep an identity path)
    hidden_dim: int = 256
    gate_hidden: int = 128
    entropy_coef: float = 0.001    # encourage non-collapse / exploration
    kl_anchor_coef: float = 0.0    # optional drift control
    anchor_ema: float = 0.995      # EMA for anchor probabilities


class ResidualAdapter(nn.Module):
    """Tiny residual adapter: y = x + f(x)."""
    def __init__(self, dim: int, bottleneck: int = 64):
        super().__init__()
        b = max(4, min(bottleneck, dim))
        self.net = nn.Sequential(
            nn.Linear(dim, b),
            nn.Tanh(),
            nn.Linear(b, dim),
        )
        # start near-identity: small outputs
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class GateNet(nn.Module):
    """Trained selector producing an adapter distribution."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # logits


class AdapterBank(nn.Module):
    """Adapter bank with a trained gate.

    - Always provides an identity path.
    - Selects among N residual adapters.
    - Returns (x_adapted, info) where info includes probs and entropy.

    NOTE: This is intentionally minimal and safe: if cfg.enabled is False,
    it returns x unchanged.
    """
    def __init__(self, dim: int, cfg: Optional[AdapterBankConfig] = None):
        super().__init__()
        self.dim = dim
        self.cfg = cfg or AdapterBankConfig(enabled=False, hidden_dim=dim)
        self.num = int(self.cfg.num_adapters)
        self.adapters = nn.ModuleList([ResidualAdapter(dim) for _ in range(self.num)])
        # gate inputs: task embedding + wm uncertainty + self confidence
        self.task_emb = nn.Embedding(128, dim)  # task_id hashed to <=127 by caller
        self.gate = GateNet(in_dim=dim + 2, hidden=self.cfg.gate_hidden, out_dim=self.num + 1)  # +1 for identity
        self.register_buffer("anchor_probs", torch.full((self.num + 1,), 1.0 / (self.num + 1)), persistent=False)

    def _task_to_index(self, task_id: Optional[str]) -> int:
        if task_id is None:
            return 0
        return (abs(hash(task_id)) % 127) + 1

    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[str] = None,
        wm_u: Optional[torch.Tensor] = None,
        self_conf: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not getattr(self.cfg, "enabled", False):
            return x, {"adapter_probs": None, "adapter_entropy": None, "adapter_choice": None}

        B = x.shape[0]
        tidx = self._task_to_index(task_id)
        t = self.task_emb(torch.tensor([tidx], device=x.device)).expand(B, -1)

        u = wm_u if wm_u is not None else torch.zeros((B, 1), device=x.device, dtype=x.dtype)
        c = self_conf if self_conf is not None else torch.zeros((B, 1), device=x.device, dtype=x.dtype)

        gate_in = torch.cat([t, u, c], dim=-1)
        logits = self.gate(gate_in)
        probs = F.softmax(logits, dim=-1)  # [B, num+1]
        entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1)  # [B]

        # mixture: identity + residual adapters
        out = probs[:, :1] * x
        for i, ad in enumerate(self.adapters):
            out = out + probs[:, i + 1:i + 2] * ad(x)

        # update anchor EMA (detached)
        with torch.no_grad():
            p_mean = probs.mean(dim=0).detach()
            self.anchor_probs.mul_(self.cfg.anchor_ema).add_((1.0 - self.cfg.anchor_ema) * p_mean)

        choice = torch.argmax(probs, dim=-1)  # [B]
        info = {
            "adapter_probs": probs,
            "adapter_entropy": entropy,
            "adapter_choice": choice,
        }
        return out, info

    def regularization_loss(self, probs: Optional[torch.Tensor]) -> torch.Tensor:
        """Optional drift/entropy regularizers. Safe if probs is None."""
        if probs is None:
            return torch.zeros((), device=self.anchor_probs.device)
        loss = torch.zeros((), device=probs.device)
        if getattr(self.cfg, "entropy_coef", 0.0) > 0:
            ent = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1).mean()
            loss = loss - float(self.cfg.entropy_coef) * ent
        if getattr(self.cfg, "kl_anchor_coef", 0.0) > 0:
            # KL(probs || anchor)
            anchor = self.anchor_probs.clamp_min(1e-8)
            kl = (probs * (probs.clamp_min(1e-8).log() - anchor.log())).sum(dim=-1).mean()
            loss = loss + float(self.cfg.kl_anchor_coef) * kl
        return loss
