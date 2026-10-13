"""Training-only grouped holdout utilities."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def grouped_reference_holdout(labels: list[int], sample_ids: list[str], *,
                              source_ids: list[str] | None = None,
                              estimation_fraction: float = 0.2,
                              seed: int = 17) -> tuple[list[int], list[int], dict[str, object]]:
    """Create a deterministic stratified holdout without splitting known sources."""
    if not 0.0 < estimation_fraction < 1.0:
        raise ValueError("estimation_fraction must be strictly between zero and one")
    if len(labels) != len(sample_ids) or not labels:
        raise ValueError("labels and sample_ids must have the same nonzero length")
    used_source_ids = source_ids if source_ids is not None else sample_ids
    if len(used_source_ids) != len(sample_ids):
        raise ValueError("source_ids must have the same length as sample_ids")
    n_sources = len(set(used_source_ids))
    n_splits = max(2, round(1.0 / estimation_fraction))
    n_splits = min(n_splits, n_sources)
    if n_splits < 2:
        raise ValueError("At least two unique source groups are required for a grouped holdout")
    if source_ids is None:
        warnings.warn("Dataset adapter has no source_ids; sample_ids are used as independent groups",
                      RuntimeWarning, stacklevel=2)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, estimate_idx = next(splitter.split(np.zeros(len(labels)), labels, used_source_ids))
    train_sources = {used_source_ids[i] for i in train_idx}
    estimate_sources = {used_source_ids[i] for i in estimate_idx}
    if train_sources & estimate_sources:
        raise RuntimeError("Grouped holdout split leaked source IDs")
    metadata: dict[str, object] = {
        "protocol": "independent_group_stratified_estimation_holdout",
        "grouping_unit": "source_id" if source_ids is not None else "sample_id_proxy",
        "seed": seed, "n_splits": n_splits,
        "requested_estimation_fraction": estimation_fraction,
        "actual_estimation_fraction": len(estimate_idx) / len(labels),
        "fit_sample_ids": [sample_ids[i] for i in train_idx],
        "estimation_sample_ids": [sample_ids[i] for i in estimate_idx],
        "fit_source_ids": sorted(train_sources),
        "estimation_source_ids": sorted(estimate_sources),
        "source_overlap": [],
        "limitations": ["StratifiedGroupKFold preserves source groups; class coverage is best-effort.",
                        "Independent estimation signatures fit profiles/UCD; held-out-trained reference "
                        "model is used to estimate assignments for remaining training samples."],
    }
    return train_idx.tolist(), estimate_idx.tolist(), metadata
