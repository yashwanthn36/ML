"""Leakage-safe multiclass reporting helpers shared by evaluation and reports."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)


def identify_minority_classes(training_labels: list[int], *, class_names: list[str],
                              bottom_quantile: float = 0.25) -> dict:
    """Select classes at or below the quantile of TRAINING counts (ties included)."""
    if not 0 < bottom_quantile <= 1:
        raise ValueError("minority bottom_quantile must be in (0, 1]")
    num_classes = len(class_names)
    labels = np.asarray(training_labels, dtype=np.int64)
    if not len(labels):
        raise ValueError("Cannot identify minority classes from an empty training label list")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(f"Training labels must be in [0, {num_classes})")
    counts = np.bincount(labels, minlength=num_classes)
    threshold = float(np.quantile(counts, bottom_quantile))
    selected = [int(i) for i, count in enumerate(counts) if count <= threshold]
    return {
        "rule": "classes with training count at or below the bottom quantile; ties included",
        "bottom_quantile": float(bottom_quantile), "quantile_threshold": threshold,
        "training_class_counts": {str(i): int(counts[i]) for i in range(num_classes)},
        "class_names": {str(i): class_names[i] for i in range(num_classes)},
        "minority_class_ids": selected,
        "minority_class_names": [class_names[i] for i in selected],
    }


def validate_probabilities(probabilities: np.ndarray, *, num_classes: int,
                           atol: float = 1e-5) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
        raise ValueError(f"Expected probabilities with shape [samples, {num_classes}], got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted class probabilities contain non-finite values")
    if np.any(probabilities < -atol) or np.any(probabilities > 1.0 + atol):
        raise ValueError("Predicted class probabilities must lie in [0, 1]")
    if len(probabilities) and not np.allclose(probabilities.sum(axis=1), 1.0, atol=atol, rtol=0.0):
        raise ValueError("Predicted class probabilities must sum to one for every sample")
    return np.clip(probabilities, 0.0, 1.0)


def classification_metrics(y_true: list[int] | np.ndarray, probabilities: np.ndarray, *,
                           class_names: list[str], minority_class_ids: list[int]) -> dict:
    """Return complete metrics; AP for classes without positives is explicitly null."""
    y_true = np.asarray(y_true, dtype=np.int64)
    num_classes = len(class_names)
    if not len(y_true):
        raise ValueError("Cannot calculate classification metrics on an empty test set")
    if np.any(y_true < 0) or np.any(y_true >= num_classes):
        raise ValueError(f"True labels must be in [0, {num_classes})")
    probs = validate_probabilities(probabilities, num_classes=num_classes)
    if len(probs) != len(y_true):
        raise ValueError("Probability row count does not match the number of true labels")
    predicted = probs.argmax(axis=1)
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predicted, labels=labels, zero_division=0)
    ap: list[float | None] = []
    for class_id in labels:
        positives = y_true == class_id
        if not positives.any():
            ap.append(None)
        else:
            ap.append(float(average_precision_score(positives.astype(np.int8), probs[:, class_id])))
    defined_ap = [value for value in ap if value is not None]
    minority = [class_id for class_id in minority_class_ids if 0 <= class_id < num_classes]
    minority_f1 = float(np.mean(f1[minority])) if minority else None
    minority_recall = float(np.mean(recall[minority])) if minority else None
    matrix = confusion_matrix(y_true, predicted, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=np.float64),
                                  where=row_totals != 0)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)) if support.sum() else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "minority_macro_f1": minority_f1,
        "minority_macro_recall": minority_recall,
        "per_class": [{"class_id": class_id, "class_name": class_names[class_id],
                       "precision": float(precision[class_id]), "recall": float(recall[class_id]),
                       "f1": float(f1[class_id]), "support": int(support[class_id]),
                       "average_precision": ap[class_id]} for class_id in labels],
        "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(), "per_class_support": support.tolist(),
        "per_class_average_precision": ap,
        "macro_average_precision": float(np.mean(defined_ap)) if defined_ap else None,
        "average_precision_defined_class_ids": [i for i, value in enumerate(ap) if value is not None],
        "class_labels": [{"class_id": i, "class_name": class_names[i]} for i in labels],
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_row_normalized": normalized_matrix.tolist(),
    }
