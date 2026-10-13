"""Class-balanced and hierarchical class/group-balanced CE objectives."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def class_balanced_ce(logits: torch.Tensor, labels: torch.Tensor,
                      class_counts: dict[int, int] | None = None) -> torch.Tensor:
    """Estimate Eq. (8.1) using natural-sampling minibatches.

    Full-training counts give N/B-scaled inverse-frequency importance weights,
    whose minibatch mean is unbiased under uniform natural sampling.
    """
    losses = F.cross_entropy(logits, labels, reduction="none")
    if class_counts:
        total, classes = sum(class_counts.values()), len(class_counts)
        weights = torch.tensor([total / (classes * class_counts[int(label)]) for label in labels],
                               dtype=losses.dtype, device=losses.device)
        return (losses * weights).mean()
    classes = torch.unique(labels)
    return torch.stack([losses[labels == c].mean() for c in classes]).mean()


def group_balanced_ce(logits: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor,
                      group_counts: dict[tuple[int, int], int] | None = None,
                      groups_per_class: dict[int, int] | None = None) -> torch.Tensor:
    """Estimate Eq. (8.2) with full-training class/group importance weights.

    The N/B-scaled minibatch mean is unbiased for uniform natural sampling.
    Without full counts, fall back to means over scopes present in the batch.
    """
    if group_ids.shape != labels.shape:
        raise ValueError("group_ids must have the same shape as labels")
    losses = F.cross_entropy(logits, labels, reduction="none")
    if group_counts and groups_per_class:
        total = sum(group_counts.values())
        weights = []
        for label, group in zip(labels.tolist(), group_ids.tolist()):
            key = (int(label), int(group))
            if key not in group_counts or int(label) not in groups_per_class:
                raise ValueError(f"Missing full-training support for class/group {key}")
            weights.append(total / (len(groups_per_class) * groups_per_class[int(label)] * group_counts[key]))
        weights_tensor = torch.tensor(weights, dtype=losses.dtype, device=losses.device)
        return (losses * weights_tensor).mean()
    class_losses = []
    for class_id in torch.unique(labels):
        class_mask = labels == class_id
        within_class = []
        for group_id in torch.unique(group_ids[class_mask]):
            mask = class_mask & (group_ids == group_id)
            within_class.append(losses[mask].mean())
        class_losses.append(torch.stack(within_class).mean())
    return torch.stack(class_losses).mean()
