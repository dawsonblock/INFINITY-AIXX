from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba2(nn.Module):
    """
    CPU-safe fallback for Mamba2 API.

    This is NOT the official Mamba2 implementation. It exists so the repo can run
    end-to-end on machines without Triton/CUDA while preserving the same call sites.

    Interface:
      - forward(x): x -> x
      - optional step(x, state): streaming-ish API used by HybridSSMAttentionBackbone
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: Optional[int] = None,
        **_: Any,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.expand = int(expand)
        hidden = self.d_model * self.expand

        self.in_proj = nn.Linear(self.d_model, 2 * hidden)
        # depthwise causal conv
        self.dwconv = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=int(d_conv),
            padding=int(d_conv) - 1,
            groups=hidden,
        )
        self.out_proj = nn.Linear(hidden, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        h = self.in_proj(x)  # [B,T,2H]
        u, gate = h.chunk(2, dim=-1)  # [B,T,H]

        # causal-ish depthwise conv over time
        u_t = u.transpose(1, 2)  # [B,H,T]
        u_t = self.dwconv(u_t)[..., :T]  # trim to original length
        u = u_t.transpose(1, 2)

        y = u * torch.sigmoid(gate)
        y = F.silu(y)
        return self.out_proj(y)

    def step(self, x: torch.Tensor, state: Optional[Any] = None) -> Tuple[torch.Tensor, Optional[Any]]:
        # For simplicity, treat the provided x as a full sequence chunk.
        return self.forward(x), state
