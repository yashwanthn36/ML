"""InfoNCE lower-bound FCD estimator from the Utopia paper, Eqs. (7)-(9)."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class InfoNCEFCD(nn.Module):
    """Estimate per-example modality shares from normalized paired/fused features.

    Each input is mean-pooled across tokens, matching the architecture PDF's
    consistent temporal aggregation requirement. The batch supplies the negatives
    for the InfoNCE bound. The finite-batch lower bound may be negative, so
    softplus makes it nonnegative before FCD normalization for the KL loss.
    """
    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, fused: torch.Tensor, modality_1: torch.Tensor,
                modality_2: torch.Tensor) -> torch.Tensor:
        if fused.ndim != 2 or modality_1.ndim != 2 or modality_2.ndim != 2:
            raise ValueError("InfoNCEFCD inputs must be pooled [batch, dimension] vectors")
        batch_size = fused.shape[0]
        if batch_size < 2:
            raise ValueError("InfoNCE FCD needs at least two paired examples in each batch")
        z = F.normalize(fused, p=2, dim=-1, eps=self.epsilon)
        estimates = []
        for source in (modality_1, modality_2):
            x = F.normalize(source, p=2, dim=-1, eps=self.epsilon)
            scores = x @ z.transpose(0, 1)
            lower_bound = scores.diagonal() + torch.log(scores.new_tensor(float(batch_size))) \
                - torch.logsumexp(scores, dim=1)
            estimates.append(lower_bound)
        information = F.softplus(torch.stack(estimates, dim=-1)) + self.epsilon
        return information / information.sum(dim=-1, keepdim=True)
