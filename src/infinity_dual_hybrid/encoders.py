"""encoders.py

Drop-in modality encoders used by the Infinity agent when observations are dict-like.

Design goals:
- Simple defaults (MLP / CNN / GRU)
- Optional pretrained text encoder via Hugging Face (transformers)
- Always output a fixed-size embedding: [B, d_model]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int, hidden: Optional[int] = None):
        super().__init__()
        h = hidden or max(d_model, 128)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, d_model),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class SimpleCNNEncoder(nn.Module):
    """Simple CNN for (B,C,H,W) uint8 or float images.

    This uses adaptive pooling to avoid hardcoding spatial dims.
    """
    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(64, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        z = self.conv(x)
        z = self.pool(z).flatten(1)
        return self.proj(z)


class TextGRUEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens.long())
        _, h = self.gru(x)
        return h.squeeze(0)  # [B, d_model]


class HFTextEncoder(nn.Module):
    """HuggingFace transformer text encoder (optional dependency).

    Returns CLS-like pooled embedding projected to d_model.
    """
    def __init__(self, model_id: str, d_model: int):
        super().__init__()
        try:
            from transformers import AutoModel
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "HFTextEncoder requires 'transformers'. Install with: pip install transformers"
            ) from e
        self.model = AutoModel.from_pretrained(model_id)
        hidden = getattr(self.model.config, "hidden_size", None)
        if hidden is None:
            raise ValueError("Could not infer hidden_size from HF model config.")
        self.proj = nn.Linear(hidden, d_model)

    def forward(self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.model(input_ids=tokens.long(), attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0, :]
        return self.proj(pooled)

    def unfreeze_last_n_layers(self, n: int) -> None:
        """Freeze everything except the last n transformer layers (n<=0 means unfreeze all)."""
        # common HF naming: encoder.layer, transformer.layer, etc.
        for p in self.model.parameters():
            p.requires_grad = False
        if n <= 0:
            for p in self.model.parameters():
                p.requires_grad = True
            return
        layers = []
        for name, module in self.model.named_modules():
            if name.endswith("encoder.layer") or name.endswith("transformer.layer"):
                # module is a ModuleList, grab children
                try:
                    layers = list(module)
                except Exception:
                    pass
        if not layers:
            # try common: model.encoder.layer
            try:
                layers = list(self.model.encoder.layer)
            except Exception:
                layers = []
        if not layers:
            # fallback: unfreeze all
            for p in self.model.parameters():
                p.requires_grad = True
            return
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True


class MultiModalEncoder(nn.Module):
    """Encodes a dict observation into a single embedding."""
    def __init__(self, d_model: int, specs: list[dict]):
        super().__init__()
        self.d_model = d_model
        self.encoders = nn.ModuleDict()
        self.specs = specs

        for spec in specs:
            name = spec["name"]
            typ = spec["type"]
            if typ == "mlp":
                self.encoders[name] = MLPEncoder(spec["input_dim"], d_model)
            elif typ == "cnn":
                self.encoders[name] = SimpleCNNEncoder(spec.get("in_ch", 3), d_model)
            elif typ == "text_gru":
                self.encoders[name] = TextGRUEncoder(spec.get("vocab_size", 30522), d_model)
            elif typ == "hf_text":
                self.encoders[name] = HFTextEncoder(spec.get("pretrained_id", "distilbert-base-uncased"), d_model)
            else:
                raise ValueError(f"Unknown encoder type: {typ}")

        self.fuse = nn.Linear(len(specs) * d_model, d_model)

    def forward(self, obs: Dict[str, Any], modality_dropout_p: float = 0.0) -> torch.Tensor:
        feats = []
        device = None
        for spec in self.specs:
            name = spec["name"]
            x = obs[name]
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            if device is None:
                device = x.device
            enc = self.encoders[name]
            # HFTextEncoder may accept attention_mask optionally if provided
            if isinstance(enc, HFTextEncoder) and isinstance(x, dict):
                z = enc(x["input_ids"], x.get("attention_mask", None))
            else:
                z = enc(x.to(next(enc.parameters()).device))
            if self.training and modality_dropout_p > 0.0:
                if torch.rand(()) < modality_dropout_p:
                    z = torch.zeros_like(z)
            feats.append(z)
        fused = self.fuse(torch.cat(feats, dim=-1))
        return fused

    def set_trainable(self, trainable: bool) -> None:
        for p in self.parameters():
            p.requires_grad = trainable
