"""Generic fusion head for any pair of modality encoders."""
from __future__ import annotations

import torch
from torch import nn


class TwoModalityFusion(nn.Module):
    """Encode two inputs, cross-attend, gated-fuse, and classify.

    Each encoder must return [batch, tokens, dim]. Encoders are injected by the
    dataset/model adapter; this module has no assumptions about audio, video,
    text, sensors, or the source dataset.
    """
    def __init__(self, encoder_1: nn.Module, encoder_2: nn.Module, num_classes: int,
                 dim: int = 128, heads: int = 4, freeze_encoders: bool = False) -> None:
        super().__init__()
        self.encoder_1, self.encoder_2 = encoder_1, encoder_2
        self.projection_1, self.projection_2 = nn.LazyLinear(dim), nn.LazyLinear(dim)
        self.freeze_encoders = freeze_encoders
        if freeze_encoders:
            self.encoder_1.requires_grad_(False)
            self.encoder_2.requires_grad_(False)
        self.missing_1 = nn.Parameter(torch.zeros(1, 1, dim))
        self.missing_2 = nn.Parameter(torch.zeros(1, 1, dim))
        self.context_1 = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.context_2 = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.2),
                                        nn.Linear(dim, num_classes))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoders:
            self.encoder_1.eval()
            self.encoder_2.eval()
        return self

    def encode(self, modality_1: torch.Tensor, modality_2: torch.Tensor,
               modality_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.projection_1(self.encoder_1(modality_1))
        x2 = self.projection_2(self.encoder_2(modality_2))
        if modality_mask is not None:
            mask = modality_mask.to(device=x1.device, dtype=torch.bool)
            if mask.ndim != 2 or mask.shape != (x1.shape[0], 2):
                raise ValueError("modality_mask must have shape [batch, 2]")
            x1 = torch.where(mask[:, 0, None, None], x1, self.missing_1.expand_as(x1))
            x2 = torch.where(mask[:, 1, None, None], x2, self.missing_2.expand_as(x2))
        return x1, x2

    def fused_features(self, modality_1: torch.Tensor, modality_2: torch.Tensor,
                       modality_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return fused and attended modality features for contribution estimators."""
        x1, x2 = self.encode(modality_1, modality_2, modality_mask)
        u1, _ = self.context_1(x1, x2, x2, need_weights=False)
        u2, _ = self.context_2(x2, x1, x1, need_weights=False)
        pooled_1, pooled_2 = u1.mean(dim=1), u2.mean(dim=1)
        gate = self.gate(torch.cat((pooled_1, pooled_2), dim=-1))
        fused = gate * pooled_1 + (1.0 - gate) * pooled_2
        # FCD uses unimodal projected representations, not cross-conditioned outputs.
        return fused, x1.mean(dim=1), x2.mean(dim=1)

    def forward(self, modality_1: torch.Tensor, modality_2: torch.Tensor,
                modality_mask: torch.Tensor | None = None) -> torch.Tensor:
        fused, _, _ = self.fused_features(modality_1, modality_2, modality_mask)
        return self.classifier(fused)
