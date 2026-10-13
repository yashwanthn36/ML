"""Reproducible AVE source-video-grouped split utilities."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def source_video_disjoint_split(rows: list[dict[str, Any]], *, seed: int = 17,
                                n_splits: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign each source video to train/val/test using nested grouped stratification.

    One fold (approximately 1/K) becomes test; one fold of the remaining
    sources becomes validation. With K=5 this targets about 64/16/20 percent
    of segments. Group separation is strict; class coverage is best-effort.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not rows:
        raise ValueError("Cannot split an empty manifest")
    source_ids = [str(row.get("video_id", row.get("source_id", ""))) for row in rows]
    labels = [int(row.get("label_id", row.get("label"))) for row in rows]
    if any(not source_id for source_id in source_ids):
        raise ValueError("Every manifest row must have a nonempty source video ID")
    if len(set(source_ids)) < n_splits:
        raise ValueError(f"Need at least {n_splits} unique source videos for n_splits={n_splits}")
    indices = np.arange(len(rows))
    first = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    remaining_indices, test_indices = next(first.split(indices, labels, source_ids))
    remaining_rows = [rows[i] for i in remaining_indices]
    remaining_labels = [labels[i] for i in remaining_indices]
    remaining_sources = [source_ids[i] for i in remaining_indices]
    second = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1)
    train_local, val_local = next(second.split(np.arange(len(remaining_rows)), remaining_labels,
                                                   remaining_sources))
    assignment = {int(i): "test" for i in test_indices}
    for local in train_local:
        assignment[int(remaining_indices[local])] = "train"
    for local in val_local:
        assignment[int(remaining_indices[local])] = "val"
    result = []
    for i, row in enumerate(rows):
        output_row = dict(row)
        output_row["split"] = assignment[i]
        result.append(output_row)
    ids_by_split = {name: {source_ids[i] for i, split in assignment.items() if split == name}
                    for name in ("train", "val", "test")}
    overlaps = {f"{a}_{b}": sorted(ids_by_split[a] & ids_by_split[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlaps.values()):
        raise RuntimeError("Source-grouped split produced a source video overlap")
    classes = sorted(set(labels))
    stats = {}
    for name in ("train", "val", "test"):
        selected_labels = [labels[i] for i, split in assignment.items() if split == name]
        counts = Counter(selected_labels)
        stats[name] = {"segment_count": len(selected_labels),
                       "source_video_count": len(ids_by_split[name]),
                       "class_counts": {str(c): counts.get(c, 0) for c in classes},
                       "missing_classes": [c for c in classes if counts.get(c, 0) == 0]}
    summary = {"protocol": "source_video_disjoint", "seed": seed,
               "method": "nested StratifiedGroupKFold", "n_splits": n_splits,
               "target_approximate_segment_fractions": {"train": (n_splits - 1) ** 2 / n_splits**2,
                                                         "val": (n_splits - 1) / n_splits**2,
                                                         "test": 1 / n_splits},
               "split_summary": stats, "source_id_overlap": overlaps,
               "class_ids": classes,
               "notes": ["Source-video separation is exact; class stratification is best-effort.",
                         "Missing classes in a partition are reported, not repaired by splitting a source video."]}
    return result, summary
