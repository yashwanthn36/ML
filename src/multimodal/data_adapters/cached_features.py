"""Dataset adapter for on-disk frozen-encoder feature caches."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path).resolve()
        with self.index_path.open("r", newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"No cached features listed in {self.index_path}")
        self.num_classes = max(int(row["label"]) for row in self.rows) + 1
        self.labels = [int(row["label"]) for row in self.rows]
        self.sample_ids = [row["sample_id"] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = self.index_path.parent / feature_path
        features = torch.load(feature_path, map_location="cpu", weights_only=True)
        return {"modality_1": features["modality_1"], "modality_2": features["modality_2"],
                "label": torch.tensor(int(row["label"]), dtype=torch.long),
                "sample_id": row["sample_id"], "source_id": row.get("source_id", ""),
                "label_name": row.get("label_name", ""), "split": row.get("split", "")}


def build_datasets(train_index: str, validation_index: str) -> tuple[CachedFeatureDataset, CachedFeatureDataset]:
    return CachedFeatureDataset(train_index), CachedFeatureDataset(validation_index)


def build_test_dataset(index_path: str) -> CachedFeatureDataset:
    return CachedFeatureDataset(index_path)
