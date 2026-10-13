"""Dataset-specific AVE segment loader implementing the paired-modality contract."""
from __future__ import annotations

import csv
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from multimodal.data_adapters.media import read_video_frames
from multimodal.training.provenance import file_sha256


class AVEDataset(Dataset):
    """Read AVE event windows; modality_1 is audio and modality_2 is video."""

    modality_names = ("audio", "video")

    def __init__(self, manifest_path: str | Path, split: str, *, root: str | Path | None = None,
                 num_frames: int = 8, image_size: int = 112, sample_rate: int = 16_000,
                 audio_samples: int = 160_000, split_protocol: str = "official",
                 temporal_window_policy: str = "full_clip",
                 min_audio_coverage: float = 0.5) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        if split_protocol not in {"official", "source_video_disjoint"}:
            raise ValueError("split_protocol must be 'official' or 'source_video_disjoint'")
        if split_protocol == "source_video_disjoint":
            if self.manifest_path.stem.endswith("_source_video_disjoint"):
                alternative = self.manifest_path
            else:
                alternative = self.manifest_path.with_name(
                    f"{self.manifest_path.stem}_source_video_disjoint{self.manifest_path.suffix}"
                )
            if not alternative.exists():
                raise FileNotFoundError(f"Source-video-disjoint manifest not found: {alternative}; "
                                        "run prepare_ave_manifest.py --include-source-video-disjoint")
            self.manifest_path = alternative
        self.root = Path(root).resolve() if root else self.manifest_path.parents[4]
        self.num_frames, self.image_size = num_frames, image_size
        self.sample_rate, self.audio_samples = sample_rate, audio_samples
        self.split_protocol = split_protocol
        if temporal_window_policy not in {"full_clip", "annotated_event"}:
            raise ValueError("temporal_window_policy must be 'full_clip' or 'annotated_event'")
        if audio_samples <= 0 or not (0.0 <= min_audio_coverage <= 1.0):
            raise ValueError("audio_samples must be positive and min_audio_coverage must be in [0, 1]")
        self.temporal_window_policy = temporal_window_policy
        self.min_audio_coverage = min_audio_coverage
        with self.manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split={split!r} in {self.manifest_path}")
        self.split = split
        label_to_id: dict[str, int] = {}
        id_to_label: dict[int, str] = {}
        for row in rows:
            name, class_id = row["label"], int(row["label_id"])
            if name in label_to_id and label_to_id[name] != class_id:
                raise ValueError(f"Class label {name!r} maps to multiple IDs in {self.manifest_path}")
            if class_id in id_to_label and id_to_label[class_id] != name:
                raise ValueError(f"Class ID {class_id} maps to multiple labels in {self.manifest_path}")
            label_to_id[name], id_to_label[class_id] = class_id, name
        self.num_classes = max(id_to_label) + 1
        if set(id_to_label) != set(range(self.num_classes)):
            raise ValueError(f"Class IDs in {self.manifest_path} must be contiguous from zero")
        self.class_names = [id_to_label[i] for i in range(self.num_classes)]
        self.class_to_id = dict(label_to_id)
        self.classes = list(self.class_names)
        self.labels = [int(row["label_id"]) for row in self.rows]
        self.sample_ids = [row["sample_id"] for row in self.rows]
        self.source_ids = [row["video_id"] for row in self.rows]
        self.metadata = {"dataset_name": "ave", "adapter": "ave_mm.dataset:AVEDataset",
                         "manifest_path": str(self.manifest_path), "manifest_sha256": file_sha256(self.manifest_path),
                         "split": split, "split_protocol": self.split_protocol,
                         "class_to_id": self.class_to_id,
                         "preprocessing": {"sample_rate": self.sample_rate, "audio_samples": self.audio_samples,
                                           "num_frames": self.num_frames, "image_size": self.image_size,
                                           "temporal_window_policy": self.temporal_window_policy}}

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else self.root / path

    def _read_audio(self, path: Path, start: float, end: float) -> tuple[torch.Tensor, float]:
        with wave.open(str(path), "rb") as wav:
            channels, rate, width = wav.getnchannels(), wav.getframerate(), wav.getsampwidth()
            if width != 2 or rate != self.sample_rate:
                raise ValueError(f"Expected 16-bit {self.sample_rate} Hz PCM at {path}")
            frames = wav.readframes(wav.getnframes())
        if channels < 1:
            raise ValueError(f"Audio file has no channels: {path}")
        signal = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            signal = signal.reshape(-1, channels).mean(axis=1)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError(f"Invalid audio interval [{start}, {end}) for {path}")
        first = max(0, min(round(start * rate), len(signal)))
        last = max(first, min(round(end * rate), len(signal)))
        segment = signal[first:last]
        if not len(segment):
            raise ValueError(f"Audio interval [{start}, {end}) contains no samples in {path}")
        output = np.zeros(self.audio_samples, dtype=np.float32)
        count = min(len(segment), self.audio_samples)
        if count:
            output[:count] = segment[:count]
        coverage = count / self.audio_samples
        if coverage < self.min_audio_coverage:
            raise ValueError(f"Audio interval in {path} fills only {coverage:.1%} of the configured "
                             f"{self.audio_samples}-sample window (minimum {self.min_audio_coverage:.1%}); "
                             "use full_clip or adjust the window policy/length")
        return torch.from_numpy(output), coverage

    def _read_video(self, path: Path, start: float | None, end: float | None) -> torch.Tensor:
        return read_video_frames(path, start=start, end=end,
                                 num_frames=self.num_frames, image_size=self.image_size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio_path = self._path(row["audio_path"])
        if self.temporal_window_policy == "full_clip":
            # This policy samples the available clip; AVE annotations are metadata
            # and can be invalid (the published manifest includes a zero-length row).
            with wave.open(str(audio_path), "rb") as wav:
                clip_end = wav.getnframes() / wav.getframerate()
            start, end = 0.0, clip_end
            video_start, video_end = None, None
        else:
            try:
                start, end = float(row["time_start"]), float(row["time_end"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid annotated event interval for sample {row.get('sample_id', '<unknown>')}: "
                                 f"time_start={row.get('time_start')!r}, time_end={row.get('time_end')!r}; "
                                 "expected finite numeric values with time_end > time_start") from exc
            if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end <= start:
                raise ValueError(f"Invalid annotated event interval for sample {row.get('sample_id', '<unknown>')}: "
                                 f"[{start}, {end}); expected finite values, 0 <= time_start, and time_end > time_start")
            video_start, video_end = start, end
        audio, coverage = self._read_audio(audio_path, start, end)
        video_path = self._path(row["video_path"])
        video = self._read_video(video_path, video_start, video_end)
        return {
            "modality_1": audio,
            "modality_2": video,
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "sample_id": row["sample_id"], "source_id": row["video_id"],
            "label_name": row["label"], "split": row["split"],
            "split_protocol": self.split_protocol,
            "temporal_window_policy": self.temporal_window_policy,
            "audio_valid_fraction": coverage,
            "annotation_time_start": row.get("time_start"),
            "annotation_time_end": row.get("time_end"),
            "time_start": start, "time_end": end,
        }


def build_datasets(manifest_path: str, root: str = ".", split_protocol: str = "official",
                   **kwargs: Any) -> tuple[AVEDataset, AVEDataset]:
    """Factory consumed by the dataset-agnostic training entry point."""
    return (AVEDataset(manifest_path, "train", root=root, split_protocol=split_protocol, **kwargs),
            AVEDataset(manifest_path, "val", root=root, split_protocol=split_protocol, **kwargs))


def build_test_dataset(manifest_path: str, root: str = ".", split_protocol: str = "official",
                       **kwargs: Any) -> AVEDataset:
    """Build the untouched official test split for post-selection evaluation only."""
    return AVEDataset(manifest_path, "test", root=root, split_protocol=split_protocol, **kwargs)
