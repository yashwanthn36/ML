"""Manifest integrity checks that retain every source annotation row."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any


ANNOTATION_FIELDS = ("video_id", "label", "label_id", "time_start", "time_end")


def _number(row: dict[str, Any], key: str) -> float | None:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return None


def audit_manifest(rows: list[dict[str, Any]], *, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """Report duplicate/conflicting annotations and verify split summaries.

    Rows are never removed. Exact annotation duplicates are reported by sample ID;
    downstream users must choose an explicit exclusion policy.
    """
    by_exact: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    by_interval: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid = []
    missing_fields = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        for field in ANNOTATION_FIELDS:
            if row.get(field) in (None, ""):
                missing_fields.append({"sample_id": sample_id, "field": field})
        start, end = _number(row, "time_start"), _number(row, "time_end")
        if start is None or end is None or not math.isfinite(start) or not math.isfinite(end) or end <= start:
            invalid.append({"sample_id": sample_id, "video_id": row.get("video_id"),
                            "time_start": row.get("time_start"), "time_end": row.get("time_end")})
        key = (str(row.get("video_id", "")), str(row.get("label", row.get("label_id", ""))),
               row.get("time_start"), row.get("time_end"))
        by_interval[key].append(row)
        exact_key = (key, str(row.get("quality", "")))
        by_exact[exact_key].append(row)
        by_source[str(row.get("video_id", ""))].append(row)

    exact_duplicates, duplicate_combinations = [], []
    for items in by_exact.values():
        if len(items) > 1:
            exact_duplicates.append({"sample_ids": [str(r.get("sample_id", "")) for r in items],
                                     "annotations": [{field: r.get(field) for field in
                                                       ("video_id", "label", "label_id", "quality",
                                                        "time_start", "time_end")} for r in items]})
    for items in by_interval.values():
        if len(items) > 1:
            duplicate_combinations.append({"sample_ids": [str(r.get("sample_id", "")) for r in items],
                                           "annotations": [{field: r.get(field) for field in
                                                             ("video_id", "label", "label_id", "time_start",
                                                              "time_end")} for r in items]})

    label_conflicts, interval_pairs = [], []
    for source, items in by_source.items():
        labels = {str(row.get("label", row.get("label_id", ""))) for row in items}
        if len(labels) > 1:
            label_conflicts.append({"video_id": source, "labels": sorted(labels),
                                    "sample_ids": [str(r.get("sample_id", "")) for r in items],
                                    "annotations": [{field: r.get(field) for field in
                                                     ("sample_id", "label", "label_id", "time_start", "time_end")}
                                                    for r in items]})
        ordered = sorted(items, key=lambda r: (_number(r, "time_start") or 0.0,
                                                _number(r, "time_end") or 0.0))
        for i, left in enumerate(ordered):
            a, b = _number(left, "time_start"), _number(left, "time_end")
            if a is None or b is None or b <= a:
                continue
            for right in ordered[i + 1:]:
                c, d = _number(right, "time_start"), _number(right, "time_end")
                if c is None or d is None or d <= c:
                    continue
                if c >= b:
                    break
                if max(a, c) < min(b, d):
                    interval_pairs.append({"video_id": source,
                        "kind": "identical" if a == c and b == d else "overlap",
                        "sample_ids": [str(left.get("sample_id", "")), str(right.get("sample_id", ""))],
                        "annotations": [{field: r.get(field) for field in
                                         ("sample_id", "label", "label_id", "time_start", "time_end")}
                                        for r in (left, right)]})

    names = ("train", "val", "test")
    ids = {name: {str(row.get("video_id", "")) for row in rows if row.get("split") == name}
           for name in names}
    overlaps = {f"{a}_{b}": sorted(ids[a] & ids[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    actual_summary = {}
    disjoint_summary = summary is not None and summary.get("protocol") == "source_video_disjoint"
    class_field = "label_id" if disjoint_summary else "label"
    class_ids = sorted({str(row.get(class_field, "")) for row in rows})
    for name in names:
        split_rows = [row for row in rows if row.get("split") == name]
        counts = Counter(str(row.get(class_field, "")) for row in split_rows)
        actual_summary[name] = {"segment_count": len(split_rows),
                                "source_video_count": len({str(r.get("video_id", "")) for r in split_rows}),
                                "class_counts": {c: counts.get(c, 0) for c in class_ids},
                                "missing_classes": [c for c in class_ids if counts.get(c, 0) == 0]}
    mismatches = []
    if summary is not None:
        supplied = summary.get("split_summary", {})
        for split in names:
            got = supplied.get(split)
            if got is None:
                mismatches.append(f"summary missing split {split}")
                continue
            expected_values = {"segment_count": got.get("segment_count"),
                               "source_video_count": got.get("source_video_count", got.get("unique_video_count")),
                               "class_counts": {str(k): int(v) for k, v in got.get("class_counts", {}).items()},
                               "missing_classes": sorted(map(str, got.get(
                                   "missing_classes", actual_summary[split]["missing_classes"])))}
            if "source_video_count" not in got and "unique_video_count" not in got:
                mismatches.append(f"summary missing {split}.source_video_count/unique_video_count")
            for key in ("segment_count", "source_video_count", "class_counts", "missing_classes"):
                if expected_values[key] != actual_summary[split].get(key):
                    mismatches.append(f"{split}.{key}: summary={expected_values[key]!r}, actual={actual_summary[split].get(key)!r}")
        recorded_overlaps = summary.get("source_id_overlap", summary.get("video_id_overlap_between_splits", {}))
        if summary.get("source_id_overlap") is not None:
            if recorded_overlaps != overlaps:
                mismatches.append("source_id_overlap does not match actual manifest")
        else:
            normalized = {key: sorted(value.get("video_ids", [])) for key, value in recorded_overlaps.items()}
            if normalized != overlaps:
                mismatches.append("video_id_overlap_between_splits does not match actual manifest")
    return {
        "row_count": len(rows), "exact_duplicate_annotations": exact_duplicates,
        "duplicate_source_label_interval_combinations": duplicate_combinations,
        "cross_label_source_conflicts": label_conflicts,
        "overlapping_or_identical_intervals": interval_pairs,
        "missing_annotation_fields": missing_fields, "invalid_intervals": invalid,
        "source_ids_by_split": {k: sorted(v) for k, v in ids.items()},
        "source_id_overlap": overlaps, "actual_split_summary": actual_summary,
        "summary_consistent": not mismatches, "summary_mismatches": mismatches,
        "duplicate_training_policy": "No rows are removed by manifest generation or the AVE loader; all detected exact duplicates remain eligible for training and are reported for explicit study-level handling.",
    }
