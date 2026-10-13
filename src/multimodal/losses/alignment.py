"""UCD-to-FCD KL alignment helpers, target-first direction as in Eq. (6.4)."""
from __future__ import annotations

import torch


def normalized_distribution(values: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    values = values.clamp_min(0.0) + epsilon
    return values / values.sum(dim=-1, keepdim=True).clamp_min(epsilon)


def target_to_current_kl(target: torch.Tensor, current: torch.Tensor,
                         epsilon: float = 1e-8) -> torch.Tensor:
    target = normalized_distribution(target, epsilon)
    current = normalized_distribution(current, epsilon)
    return (target * (target.log() - current.log())).sum(dim=-1).mean()
