"""Deterministic split generation for manifests without official assignments."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def generate_manifest_splits(input_path: str | Path, output_path: str | Path, *, seed: int = 17,
                             sample_column: str = "sample_id", label_column: str = "label",
                             source_column: str | None = "source_id", split_column: str = "split",
                             proportions: tuple[float, float, float] = (0.7, 0.15, 0.15),
                             require_source_disjoint: bool = True) -> dict[str, Any]:
    """Write a new manifest with explicit train/val/test assignments.

    Existing non-empty split assignments are never overwritten. Grouped splitting
    is deterministic but not stratified: source constraints take precedence, and
    the returned per-class counts report any missing-class limitations.
    """
    input_path, output_path = Path(input_path).resolve(), Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("Input and output manifests must be different; official assignments are preserved")
    if output_path.exists():
        raise FileExistsError(f"Split output already exists and will not be overwritten: {output_path}")
    if not np.isclose(sum(proportions), 1.0) or min(proportions) <= 0:
        raise ValueError("train/val/test proportions must be positive and sum to 1")
    with input_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or sample_column not in reader.fieldnames or label_column not in reader.fieldnames:
            raise ValueError(f"Manifest must have {sample_column!r} and {label_column!r} columns")
        fieldnames, rows = reader.fieldnames, list(reader)
    if not rows:
        raise ValueError("Cannot split an empty manifest")
    if split_column in fieldnames and any(row.get(split_column, "").strip() for row in rows):
        raise ValueError("Input already has split assignments; use a new unsplit source manifest")
    ids = [row[sample_column].strip() for row in rows]
    labels = np.asarray([row[label_column].strip() for row in rows])
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Sample IDs must be non-empty and unique before splitting")
    source_ids = [row.get(source_column, "").strip() for row in rows] if source_column and source_column in fieldnames else []
    has_groups = bool(source_ids) and all(source_ids)
    if require_source_disjoint and not has_groups:
        raise ValueError("Source-disjoint splitting was requested but usable source IDs are absent")
    rng_seed = int(seed)
    all_indices = np.arange(len(rows))
    limitations = []
    if has_groups and require_source_disjoint:
        groups = np.asarray(source_ids)
        # Split test first, then validation from the remaining source groups.
        first = GroupShuffleSplit(n_splits=1, test_size=proportions[2], random_state=rng_seed)
        train_val_idx, test_idx = next(first.split(all_indices, labels, groups))
        relative_val = proportions[1] / (proportions[0] + proportions[1])
        second = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=rng_seed + 1)
        train_rel, val_rel = next(second.split(train_val_idx, labels[train_val_idx], groups[train_val_idx]))
        train_idx, val_idx = train_val_idx[train_rel], train_val_idx[val_rel]
        limitations.append("Source-grouped split preserves groups; per-class proportions are not guaranteed.")
        if len(set(groups[train_idx]) & set(groups[val_idx]) | set(groups[train_idx]) & set(groups[test_idx]) |
               set(groups[val_idx]) & set(groups[test_idx])):
            raise RuntimeError("Internal error: generated source groups overlap across partitions")
    else:
        low_support = {label: int(np.sum(labels == label)) for label in sorted(set(labels.tolist()))
                       if int(np.sum(labels == label)) < 3}
        if low_support:
            raise ValueError(f"Sample-stratified three-way split requires at least 3 rows per class; "
                             f"insufficient support: {low_support}. No samples were discarded.")
        try:
            train_idx, remaining = train_test_split(all_indices, test_size=proportions[1] + proportions[2],
                random_state=rng_seed, stratify=labels)
            val_idx, test_idx = train_test_split(remaining,
                test_size=proportions[2] / (proportions[1] + proportions[2]),
                random_state=rng_seed + 1, stratify=labels[remaining])
        except ValueError as error:
            raise ValueError(f"Unable to create stratified sample-level partitions; inspect class support and "
                             f"requested proportions. No samples were discarded. {error}") from error
        limitations.append("Sample-level stratified split may share source recordings across partitions.")
    assignments = {}
    for name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for index in indices:
            assignments[int(index)] = name
    if len(assignments) != len(rows):
        raise RuntimeError("Split generation failed to assign every sample")
    for index, row in enumerate(rows):
        row[split_column] = assignments[index]
    if split_column not in fieldnames:
        fieldnames.append(split_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    partition_counts: dict[str, Counter] = {key: Counter() for key in ("train", "val", "test")}
    for row in rows:
        partition_counts[row[split_column]][row[label_column].strip()] += 1
    all_classes = sorted(set(labels.tolist()))
    missing_classes = {split: [label for label in all_classes if partition_counts[split][label] == 0]
                       for split in ("train", "val", "test")}
    for split, missing_labels in missing_classes.items():
        if missing_labels:
            limitations.append(f"{split} partition has no samples for classes: {missing_labels}")
    source_overlap = {}
    if has_groups:
        grouped = {name: {source_ids[i] for i in indices} for name, indices in
                   (("train", train_idx), ("val", val_idx), ("test", test_idx))}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            source_overlap[f"{left}_{right}"] = sorted(grouped[left] & grouped[right])
    summary = {"input_manifest": str(input_path), "output_manifest": str(output_path), "seed": rng_seed,
               "proportions_requested": {"train": proportions[0], "val": proportions[1], "test": proportions[2]},
               "sample_count": len(rows), "split_counts": {name: dict(sorted(counter.items()))
                   for name, counter in partition_counts.items()},
               "source_grouped": bool(has_groups and require_source_disjoint),
               "source_overlap": source_overlap, "missing_classes_by_split": missing_classes,
               "limitations": limitations}
    summary_path = output_path.with_suffix(output_path.suffix + ".split_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
