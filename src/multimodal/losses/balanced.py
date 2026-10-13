"""Class-balanced and hierarchical class/group-balanced CE objectives."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def class_balanced_ce(logits: torch.Tensor, labels: torch.Tensor,
                      class_counts: dict[int, int] | None = None) -> torch.Tensor:
    """Estimate Eq. (8.1) with training-split class frequencies."""
    losses = F.cross_entropy(logits, labels, reduction="none")
    if class_counts:
        weights = torch.tensor([1.0 / class_counts[int(label)] for label in labels],
                               dtype=losses.dtype, device=losses.device)
        return (losses * weights).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    classes = torch.unique(labels)
    return torch.stack([losses[labels == c].mean() for c in classes]).mean()


def group_balanced_ce(logits: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor,
                      group_counts: dict[tuple[int, int], int] | None = None,
                      groups_per_class: dict[int, int] | None = None) -> torch.Tensor:
    """Estimate Eq. (8.2) with fixed full-training group counts."""
    if group_ids.shape != labels.shape:
        raise ValueError("group_ids must have the same shape as labels")
    losses = F.cross_entropy(logits, labels, reduction="none")
    if group_counts and groups_per_class:
        weights = torch.tensor([
            1.0 / (groups_per_class[int(label)] * group_counts[(int(label), int(group))])
            for label, group in zip(labels, group_ids)
        ], dtype=losses.dtype, device=losses.device)
        return (losses * weights).sum() / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    class_losses = []
    for class_id in torch.unique(labels):
        class_mask = labels == class_id
        within_class = []
        for group_id in torch.unique(group_ids[class_mask]):
            mask = class_mask & (group_ids == group_id)
            within_class.append(losses[mask].mean())
        class_losses.append(torch.stack(within_class).mean())
    return torch.stack(class_losses).mean()
