"""Canonical contract and validation for paired-modality datasets."""
from __future__ import annotations

from typing import Any


REQUIRED_DATASET_ATTRIBUTES = ("labels", "sample_ids", "class_names", "class_to_id", "num_classes")


def validate_dataset(dataset: Any, *, name: str = "dataset", require_media: bool = False) -> dict[str, Any]:
    """Validate the adapter metadata contract and optionally decode its first sample.

    A dataset sample must be a mapping with ``modality_1``, ``modality_2``,
    ``label`` and ``sample_id``. Tensor shape is intentionally encoder-specific.
    """
    missing = [field for field in REQUIRED_DATASET_ATTRIBUTES if not hasattr(dataset, field)]
    if missing:
        raise ValueError(f"{name} adapter is missing required attributes: {', '.join(missing)}")
    n = len(dataset)
    ids, labels = list(dataset.sample_ids), list(dataset.labels)
    if n <= 0:
        raise ValueError(f"{name} adapter has no usable samples")
    if len(ids) != n or len(labels) != n:
        raise ValueError(f"{name} sample_ids/labels lengths must equal len(dataset) ({n})")
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError(f"{name} sample IDs must be non-empty strings")
    if len(set(ids)) != n:
        raise ValueError(f"{name} contains duplicate sample IDs")
    num_classes = int(dataset.num_classes)
    names = list(map(str, dataset.class_names))
    mapping = dict(dataset.class_to_id)
    if num_classes < 2 or len(names) != num_classes:
        raise ValueError(f"{name} must expose at least two classes and one name per class")
    if len(mapping) != num_classes or set(mapping.values()) != set(range(num_classes)):
        raise ValueError(f"{name} class_to_id must map every class to contiguous IDs [0, {num_classes})")
    if [next((label for label, class_id in mapping.items() if class_id == i), None)
            for i in range(num_classes)] != names:
        raise ValueError(f"{name} class_names order must match class_to_id")
    invalid = sorted({int(label) for label in labels if int(label) < 0 or int(label) >= num_classes})
    if invalid:
        raise ValueError(f"{name} contains labels outside [0, {num_classes}): {invalid}")
    sources = getattr(dataset, "source_ids", None)
    if sources is not None and len(sources) != n:
        raise ValueError(f"{name} source_ids length must equal len(dataset)")
    result = {"num_samples": n, "num_classes": num_classes, "class_names": names,
              "class_to_id": {str(key): int(value) for key, value in mapping.items()},
              "split": getattr(dataset, "split", None),
              "split_protocol": getattr(dataset, "split_protocol", None),
              "source_ids_available": sources is not None,
              "metadata": dict(getattr(dataset, "metadata", {}))}
    if require_media:
        sample = dataset[0]
        if not isinstance(sample, dict):
            raise ValueError(f"{name}[0] must return a mapping, got {type(sample).__name__}")
        missing_sample = {"modality_1", "modality_2", "label", "sample_id"} - sample.keys()
        if missing_sample:
            raise ValueError(f"{name}[0] is missing sample fields: {sorted(missing_sample)}")
    return result


def validate_split_datasets(train: Any, validation: Any, test: Any | None = None,
                            *, require_source_disjoint: bool = False) -> dict[str, Any]:
    """Check stable labels and sample/source separation for a set of partitions."""
    datasets = {"train": train, "validation": validation}
    if test is not None:
        datasets["test"] = test
    contracts = {key: validate_dataset(value, name=key) for key, value in datasets.items()}
    reference = contracts["train"]
    for name, contract in contracts.items():
        if contract["class_names"] != reference["class_names"] or contract["class_to_id"] != reference["class_to_id"]:
            raise ValueError(f"Class mapping differs between train and {name} datasets")
        if contract["split_protocol"] != reference["split_protocol"]:
            raise ValueError(f"Split protocol differs between train and {name} datasets")
    ids = {key: set(value.sample_ids) for key, value in datasets.items()}
    sources = {key: set(value.source_ids) for key, value in datasets.items()
               if getattr(value, "source_ids", None) is not None}
    for left_i, left in enumerate(datasets):
        for right in list(datasets)[left_i + 1:]:
            overlap = ids[left] & ids[right]
            if overlap:
                raise ValueError(f"Sample IDs overlap between {left} and {right}: {sorted(overlap)[:5]}")
            if require_source_disjoint and left in sources and right in sources:
                common = sources[left] & sources[right]
                if common:
                    raise ValueError(f"Source IDs overlap between {left} and {right}: {sorted(common)[:5]}")
    return {"partitions": contracts, "source_disjoint_required": require_source_disjoint}
