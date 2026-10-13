"""Differentiable InfoNCE estimates and scope-level FCD distributions.

Utopia Eqs. (7)-(9) define an expected MI estimate per modality and normalize
the modality estimates. InfoNCE can produce negative finite-sample lower bounds;
we apply softplus to the *scope-mean* bounds before normalization so they form a
valid KL target. This positivity transform is an explicit adaptation, not a
transform specified by Utopia.

Negatives are previous/current TRAIN examples only. A detached FIFO memory bank
increases the contrast set beyond the small supervised minibatch. Optional
same-source exclusion avoids treating sibling AVE segments as negatives; datasets
without source IDs fall back to sample-ID exclusion only.
"""
from __future__ import annotations

from collections import deque

import torch
from torch import nn
from torch.nn import functional as F


class InfoNCEFCD(nn.Module):
    """InfoNCE MI estimator with per-scope probability normalization.

    ``per_sample_information`` returns signed lower-bound estimates shaped
    ``[anchors, modalities]``. Call ``distribution`` on all anchors in a scope
    to average the estimates, apply softplus, and normalize to a two-element
    probability vector. Its gradients flow through current anchors and current
    minibatch candidates; queue entries are detached historical negatives.
    """

    def __init__(self, epsilon: float = 1e-8, queue_size: int = 1024,
                 exclude_same_source_negatives: bool = False) -> None:
        super().__init__()
        if not (epsilon > 0.0):
            raise ValueError("epsilon must be positive")
        if queue_size < 0:
            raise ValueError("queue_size must be nonnegative")
        self.epsilon = float(epsilon)
        self.queue_size = int(queue_size)
        self.exclude_same_source_negatives = bool(exclude_same_source_negatives)
        self._queue: deque[tuple[torch.Tensor, str | None, str | None]] = deque()
        self._queue_count = 0

    @property
    def queue_count(self) -> int:
        return self._queue_count

    def reset_queue(self) -> None:
        """Clear historical negatives, for example before a new sweep run."""
        self._queue.clear()
        self._queue_count = 0

    @staticmethod
    def _validate(fused: torch.Tensor, modality_1: torch.Tensor,
                  modality_2: torch.Tensor) -> None:
        names = ("fused", "modality_1", "modality_2")
        values = (fused, modality_1, modality_2)
        for name, value in zip(names, values):
            if value.ndim != 2:
                raise ValueError(f"{name} must be a [batch, dimension] tensor")
            if value.shape[0] == 0 or value.shape[1] == 0:
                raise ValueError(f"{name} cannot have an empty batch or feature dimension")
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if fused.shape[0] != modality_1.shape[0] or fused.shape[0] != modality_2.shape[0]:
            raise ValueError("All FCD inputs must have the same batch size")
        if fused.shape[1] != modality_1.shape[1] or fused.shape[1] != modality_2.shape[1]:
            raise ValueError("All FCD inputs must have the same feature dimension")
        if fused.device != modality_1.device or fused.device != modality_2.device:
            raise ValueError("All FCD inputs must be on the same device")

    @staticmethod
    def _normalized(value: torch.Tensor, epsilon: float) -> torch.Tensor:
        return F.normalize(value, p=2, dim=-1, eps=epsilon)

    def _remember(self, fused_values: torch.Tensor, sample_ids: list[str] | None = None,
                  source_ids: list[str] | None = None) -> None:
        if self.queue_size == 0:
            return
        detached = fused_values.detach()
        identities = sample_ids or [None] * detached.shape[0]
        sources = source_ids or [None] * detached.shape[0]
        if len(identities) != detached.shape[0] or len(sources) != detached.shape[0]:
            raise ValueError("sample_ids and source_ids must have one ID per FCD row when provided")
        for value, identity, source in zip(detached, identities, sources):
            self._queue.append((value.unsqueeze(0), None if identity is None else str(identity),
                                None if source is None else str(source)))
            self._queue_count += 1
        while self._queue_count > self.queue_size:
            self._queue.popleft()
            self._queue_count -= 1

    def per_sample_information(self, fused: torch.Tensor, modality_1: torch.Tensor,
                               modality_2: torch.Tensor, *,
                               update_queue: bool = False,
                               sample_ids: list[str] | None = None,
                               source_ids: list[str] | None = None) -> torch.Tensor:
        """Return signed InfoNCE lower bounds per anchor and modality.

        Each current example is the positive pair. Denominator candidates are
        current minibatch examples plus prior entries in the FIFO bank. The
        ``log(K)`` term uses the total candidate count K, including positives.
        """
        self._validate(fused, modality_1, modality_2)
        z = self._normalized(fused, self.epsilon)
        x1 = self._normalized(modality_1, self.epsilon)
        x2 = self._normalized(modality_2, self.epsilon)
        if sample_ids is not None and len(sample_ids) != z.shape[0]:
            raise ValueError("sample_ids must have one ID per FCD row")
        if source_ids is not None and len(source_ids) != z.shape[0]:
            raise ValueError("source_ids must have one ID per FCD row")
        result_rows = []
        for i in range(z.shape[0]):
            identity = None if sample_ids is None else str(sample_ids[i])
            source = None if source_ids is None else str(source_ids[i])
            batch_keep = [j for j in range(z.shape[0])
                          if j == i or identity is None or sample_ids is None
                          or str(sample_ids[j]) != identity]
            if self.exclude_same_source_negatives and source is not None and source_ids is not None:
                batch_keep = [j for j in batch_keep if j == i or str(source_ids[j]) != source]
            historical = [value.to(device=z.device, dtype=z.dtype)
                          for value, queue_id, queue_source in self._queue
                          if value.device == z.device and (identity is None or queue_id != identity)
                          and not (self.exclude_same_source_negatives and source is not None
                                   and queue_source == source)]
            candidates = torch.cat((z[batch_keep], *historical), dim=0) if historical else z[batch_keep]
            if candidates.shape[0] < 2:
                raise ValueError("InfoNCE has fewer than two valid candidates after configured sample/source "
                                 "exclusions; provide another independent example or queue negative")
            modality_bounds = []
            for current_x in (x1, x2):
                scores = current_x[i] @ candidates.transpose(0, 1)
                positive = (current_x[i] * z[i]).sum()
                modality_bounds.append(torch.log(scores.new_tensor(float(candidates.shape[0])))
                                       + positive - torch.logsumexp(scores, dim=-1))
            result_rows.append(torch.stack(modality_bounds))
        result = torch.stack(result_rows, dim=0)
        if not torch.isfinite(result).all():
            raise ValueError("InfoNCE produced non-finite lower-bound estimates")
        if update_queue:
            self._remember(z, sample_ids, source_ids)
        return result

    def distribution(self, information: torch.Tensor) -> torch.Tensor:
        """Convert scope-level lower bounds to a finite FCD distribution."""
        if information.ndim != 2 or information.shape[1] != 2:
            raise ValueError("scope information must have shape [samples, 2]")
        if information.shape[0] == 0:
            raise ValueError("Cannot estimate FCD for an empty scope")
        if not torch.isfinite(information).all():
            raise ValueError("Scope information contains non-finite values")
        mean_bound = information.mean(dim=0)
        positive_information = F.softplus(mean_bound) + self.epsilon
        total = positive_information.sum()
        if not torch.isfinite(total) or total <= 0:
            raise ValueError("FCD normalization received invalid information scores")
        return positive_information / total

    def forward(self, fused: torch.Tensor, modality_1: torch.Tensor,
                modality_2: torch.Tensor) -> torch.Tensor:
        """Estimate the aggregate FCD for all supplied anchors (shape ``[2]``)."""
        information = self.per_sample_information(fused, modality_1, modality_2)
        return self.distribution(information)
