"""Configurable manifest-backed adapter for paired WAV/video classification.

Audio supports uncompressed PCM WAV (8/16/24/32-bit) and is converted to mono,
linearly resampled, and padded/truncated to a fixed waveform length. Video
decoding uses OpenCV-supported containers (commonly MP4, AVI, MOV and MKV) and
returns RGB float frames in [T, C, H, W]. Compressed audio and non-video
modalities require a dataset-specific adapter.
"""
from __future__ import annotations

import csv
import hashlib
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from multimodal.data_adapters.media import read_video_frames


DEFAULT_COLUMNS = {"sample_id": "sample_id", "audio_path": "audio_path", "video_path": "video_path",
                   "label": "label", "source_id": "source_id", "split": "split"}


class GenericAudioVideoDataset(Dataset):
    """A single split of a generic manifest with a stable dataset-wide class map."""

    def __init__(self, manifest_path: str | Path, split: str, *, media_root: str | Path = ".",
                 columns: dict[str, str] | None = None, class_mapping: dict[str, int] | None = None,
                 num_frames: int = 8, image_size: int = 224, sample_rate: int = 16_000,
                 audio_samples: int = 160_000, min_audio_coverage: float = 0.5,
                 split_protocol: str = "manifest", manifest_relative_to: str = "manifest",
                 dataset_name: str = "generic_av") -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Dataset manifest does not exist: {self.manifest_path}")
        if manifest_relative_to not in {"manifest", "media_root"}:
            raise ValueError("manifest_relative_to must be 'manifest' or 'media_root'")
        if num_frames <= 0 or image_size <= 0 or sample_rate <= 0 or audio_samples <= 0:
            raise ValueError("num_frames, image_size, sample_rate and audio_samples must be positive")
        if not 0 <= min_audio_coverage <= 1:
            raise ValueError("min_audio_coverage must be in [0, 1]")
        self.media_root = Path(media_root).expanduser()
        if not self.media_root.is_absolute():
            self.media_root = (self.manifest_path.parent / self.media_root).resolve()
        self.columns = {**DEFAULT_COLUMNS, **(columns or {})}
        self.num_frames, self.image_size = int(num_frames), int(image_size)
        self.sample_rate, self.audio_samples = int(sample_rate), int(audio_samples)
        self.min_audio_coverage = float(min_audio_coverage)
        self.split_protocol, self.split = str(split_protocol), str(split)
        self.dataset_name = dataset_name
        with self.manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"Manifest has no header: {self.manifest_path}")
            self.manifest_columns = set(reader.fieldnames)
            required = ("sample_id", "audio_path", "video_path", "label")
            absent = [field for field in required if self.columns[field] not in self.manifest_columns]
            if absent:
                raise ValueError(f"Manifest {self.manifest_path} lacks configured columns for: {absent}")
            all_rows = list(reader)
        if not all_rows:
            raise ValueError(f"Manifest contains no records: {self.manifest_path}")
        if self.columns["split"] in self.manifest_columns:
            assigned = [row for row in all_rows if row[self.columns["split"]].strip() == self.split]
        else:
            raise ValueError("Manifest has no split column; run split_manifest.py to generate explicit assignments")
        if not assigned:
            raise ValueError(f"No samples assigned to split {split!r} in {self.manifest_path}")
        labels = sorted({row[self.columns["label"]].strip() for row in all_rows})
        if any(not label for label in labels):
            raise ValueError("Manifest contains an empty label")
        if class_mapping is None:
            class_mapping = {name: index for index, name in enumerate(labels)}
        else:
            class_mapping = {str(name): int(index) for name, index in class_mapping.items()}
            if set(class_mapping) != set(labels) or set(class_mapping.values()) != set(range(len(class_mapping))):
                raise ValueError("Configured class_mapping must cover manifest labels and use contiguous IDs from 0")
        self.class_to_id = class_mapping
        self.class_names = [name for name, _ in sorted(class_mapping.items(), key=lambda pair: pair[1])]
        self.classes = self.class_names
        self.num_classes = len(self.class_names)
        self.all_rows = all_rows
        self.rows = assigned
        all_ids = [str(row.get(self.columns["sample_id"], "")).strip() for row in all_rows]
        if any(not sample_id for sample_id in all_ids) or len(set(all_ids)) != len(all_ids):
            raise ValueError(f"Sample IDs must be non-empty and globally unique in {self.manifest_path}")
        self.sample_ids = [self._value(row, "sample_id") for row in assigned]
        self.labels = [class_mapping[self._value(row, "label")] for row in assigned]
        source_column = self.columns["source_id"]
        self.source_ids = ([row.get(source_column, "").strip() for row in assigned]
                           if source_column in self.manifest_columns and
                           all(row.get(source_column, "").strip() for row in assigned) else None)
        self.metadata = {"dataset_name": dataset_name, "adapter": "generic_audio_video",
                         "manifest_path": str(self.manifest_path),
                         "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                         "media_root": str(self.media_root), "columns": self.columns,
                         "class_to_id": self.class_to_id,
                         "preprocessing": {"audio_format": "PCM WAV", "sample_rate": self.sample_rate,
                                           "audio_samples": self.audio_samples, "num_frames": self.num_frames,
                                           "image_size": self.image_size},
                         "usable_samples": len(assigned)}

    def _value(self, row: dict[str, str], key: str) -> str:
        return str(row.get(self.columns[key], "")).strip()

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        # Relative paths are rooted at media_root. Absolute paths remain absolute.
        return (self.media_root / path).resolve()

    def _read_audio(self, path: Path) -> tuple[torch.Tensor, float]:
        try:
            with wave.open(str(path), "rb") as wav:
                channels, rate, width = wav.getnchannels(), wav.getframerate(), wav.getsampwidth()
                frames = wav.readframes(wav.getnframes())
        except (wave.Error, OSError) as exc:
            raise OSError(f"Could not read PCM WAV audio {path}: {exc}") from exc
        if channels <= 0 or rate <= 0 or width not in (1, 2, 3, 4):
            raise ValueError(f"Unsupported WAV encoding in {path}; expected PCM 8/16/24/32-bit")
        if width == 1:
            signal = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128
        elif width == 2:
            signal = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
        elif width == 3:
            raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
            values = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
            values = (values ^ 0x800000) - 0x800000
            signal = values.astype(np.float32) / 8388608
        else:
            signal = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648
        if channels > 1:
            signal = signal.reshape(-1, channels).mean(axis=1)
        if rate != self.sample_rate:
            old_positions = np.arange(len(signal), dtype=np.float64) / rate
            new_count = max(1, int(round(len(signal) * self.sample_rate / rate)))
            signal = np.interp(np.arange(new_count, dtype=np.float64) / self.sample_rate,
                               old_positions, signal).astype(np.float32)
        if not len(signal) or not np.isfinite(signal).all():
            raise ValueError(f"Audio has no valid samples: {path}")
        output = np.zeros(self.audio_samples, dtype=np.float32)
        count = min(len(signal), self.audio_samples)
        output[:count] = signal[:count]
        coverage = count / self.audio_samples
        if coverage < self.min_audio_coverage:
            raise ValueError(f"Audio {path} fills {coverage:.1%} of configured window; minimum is "
                             f"{self.min_audio_coverage:.1%}")
        return torch.from_numpy(output), float(coverage)

    def _read_video(self, path: Path) -> torch.Tensor:
        return read_video_frames(path, start=None, end=None,
                                 num_frames=self.num_frames, image_size=self.image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        sample_id = self._value(row, "sample_id")
        audio_path, video_path = self._resolve(self._value(row, "audio_path")), self._resolve(self._value(row, "video_path"))
        audio, coverage = self._read_audio(audio_path)
        video = self._read_video(video_path)
        label_name = self._value(row, "label")
        item = {"modality_1": audio, "modality_2": video,
                "label": torch.tensor(self.class_to_id[label_name], dtype=torch.long),
                "sample_id": sample_id, "label_name": label_name, "split": self.split,
                "split_protocol": self.split_protocol, "audio_valid_fraction": coverage,
                "audio_path": str(audio_path), "video_path": str(video_path)}
        source_id = self._value(row, "source_id")
        if source_id:
            item["source_id"] = source_id
        return item


def _factory_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in {"train_split", "validation_split", "test_split"}}


def build_datasets(manifest_path: str, *, train_split: str = "train", validation_split: str = "val",
                   **kwargs: Any) -> tuple[GenericAudioVideoDataset, GenericAudioVideoDataset]:
    common = _factory_kwargs(kwargs)
    return (GenericAudioVideoDataset(manifest_path, train_split, **common),
            GenericAudioVideoDataset(manifest_path, validation_split, **common))


def build_test_dataset(manifest_path: str, *, test_split: str = "test", **kwargs: Any) -> GenericAudioVideoDataset:
    return GenericAudioVideoDataset(manifest_path, test_split, **_factory_kwargs(kwargs))
