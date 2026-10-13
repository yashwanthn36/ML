"""Dataset adapter for on-disk frozen-encoder feature caches."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    def __init__(self, index_path: str | Path, *, split_protocol: str | None = None) -> None:
        self.index_path = Path(index_path).resolve()
        with self.index_path.open("r", newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"No cached features listed in {self.index_path}")
        if "class_to_id" not in self.rows[0] or "split_protocol" not in self.rows[0]:
            raise ValueError(f"Cached-feature index lacks class/protocol metadata: {self.index_path}")
        self.class_to_id = {str(key): int(value) for key, value in json.loads(self.rows[0]["class_to_id"]).items()}
        self.class_names = [name for name, _ in sorted(self.class_to_id.items(), key=lambda pair: pair[1])]
        self.classes = self.class_names
        self.num_classes = len(self.class_names)
        self.labels = [int(row["label"]) for row in self.rows]
        self.sample_ids = [row["sample_id"] for row in self.rows]
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError(f"Duplicate sample IDs in cached feature index: {self.index_path}")
        self.source_ids = ([row.get("source_id", "") for row in self.rows]
                           if all(row.get("source_id", "") for row in self.rows) else None)
        self.split = self.rows[0].get("split", "")
        self.split_protocol = self.rows[0]["split_protocol"]
        if split_protocol is not None and self.split_protocol != split_protocol:
            raise ValueError(f"Cached index protocol is {self.split_protocol!r}, config expects {split_protocol!r}")
        if any(row["split_protocol"] != self.split_protocol for row in self.rows):
            raise ValueError(f"Mixed split protocols in cached feature index: {self.index_path}")
        self.metadata = {"dataset_name": self.rows[0].get("dataset_name", "cached_features"),
                         "adapter": "multimodal.data_adapters.cached_features:CachedFeatureDataset",
                         "manifest_sha256": hashlib.sha256(self.index_path.read_bytes()).hexdigest(),
                         "class_to_id": self.class_to_id}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = self.index_path.parent / feature_path
        features = torch.load(feature_path, map_location="cpu", weights_only=True)
        label = int(row["label"])
        name = row.get("label_name", self.class_names[label])
        if self.class_to_id.get(name) != label:
            raise ValueError(f"Cached feature label/name mismatch in {self.index_path}: {row['sample_id']}")
        item = {"modality_1": features["modality_1"], "modality_2": features["modality_2"],
                "label": torch.tensor(label, dtype=torch.long),
                "sample_id": row["sample_id"],
                "label_name": name, "split": row.get("split", ""),
                "split_protocol": self.split_protocol}
        if self.source_ids is not None:
            item["source_id"] = row["source_id"]
        return item


def build_datasets(train_index: str, validation_index: str, *, split_protocol: str | None = None):
    return (CachedFeatureDataset(train_index, split_protocol=split_protocol),
            CachedFeatureDataset(validation_index, split_protocol=split_protocol))


def build_test_dataset(index_path: str, *, split_protocol: str | None = None) -> CachedFeatureDataset:
    return CachedFeatureDataset(index_path, split_protocol=split_protocol)
