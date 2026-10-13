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


class AVEDataset(Dataset):
    """Read AVE event windows; modality_1 is audio and modality_2 is video."""

    modality_names = ("audio", "video")

    def __init__(self, manifest_path: str | Path, split: str, *, root: str | Path | None = None,
                 num_frames: int = 8, image_size: int = 112, sample_rate: int = 16_000,
                 audio_samples: int = 160_000) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = Path(root).resolve() if root else self.manifest_path.parents[4]
        self.num_frames, self.image_size = num_frames, image_size
        self.sample_rate, self.audio_samples = sample_rate, audio_samples
        with self.manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split={split!r} in {self.manifest_path}")
        self.split = split
        self.classes = sorted({row["label"] for row in rows})
        self.num_classes = max(int(row["label_id"]) for row in rows) + 1
        self.labels = [int(row["label_id"]) for row in self.rows]
        self.sample_ids = [row["sample_id"] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else self.root / path

    def _read_audio(self, path: Path, start: float, end: float) -> torch.Tensor:
        with wave.open(str(path), "rb") as wav:
            channels, rate, width = wav.getnchannels(), wav.getframerate(), wav.getsampwidth()
            if width != 2 or rate != self.sample_rate:
                raise ValueError(f"Expected 16-bit {self.sample_rate} Hz PCM at {path}")
            frames = wav.readframes(wav.getnframes())
        signal = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            signal = signal.reshape(-1, channels).mean(axis=1)
        first = max(0, min(round(start * rate), len(signal)))
        last = max(first, min(round(end * rate), len(signal)))
        segment = signal[first:last]
        output = np.zeros(self.audio_samples, dtype=np.float32)
        count = min(len(segment), self.audio_samples)
        if count:
            output[:count] = segment[:count]
        return torch.from_numpy(output)

    def _read_video(self, path: Path, start: float, end: float) -> torch.Tensor:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise OSError(f"Could not open video: {path}")
        fps, frame_count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else max(end, 0.1)
        start = max(0.0, min(start, max(duration - 0.01, 0.0)))
        end = max(start + 0.01, min(end, duration))
        times = np.linspace(start, end, self.num_frames, endpoint=False, dtype=np.float64)
        times += max((end - start) / (2 * self.num_frames), 0.005)
        times = np.minimum(times, max(duration - 0.01, 0.0))
        frames, last_good = [], None
        for second in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(second * 1000.0))
            ok, frame = capture.read()
            if ok:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_good = cv2.resize(frame, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            elif last_good is not None:
                frame = last_good
            else:
                continue
            frames.append(frame)
        capture.release()
        if not frames:
            raise OSError(f"Could not decode any frames from {path}")
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        array = np.stack(frames[:self.num_frames]).astype(np.float32) / 255.0
        return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        start, end = float(row["time_start"]), float(row["time_end"])
        if end <= start:
            end = start + 0.1
        return {
            "modality_1": self._read_audio(self._path(row["audio_path"]), start, end),
            "modality_2": self._read_video(self._path(row["video_path"]), start, end),
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "sample_id": row["sample_id"], "source_id": row["video_id"],
            "label_name": row["label"], "split": row["split"],
            "time_start": start, "time_end": end,
        }


def build_datasets(manifest_path: str, root: str = ".", **kwargs: Any) -> tuple[AVEDataset, AVEDataset]:
    """Factory consumed by the dataset-agnostic training entry point."""
    return (AVEDataset(manifest_path, "train", root=root, **kwargs),
            AVEDataset(manifest_path, "val", root=root, **kwargs))


def build_test_dataset(manifest_path: str, root: str = ".", **kwargs: Any) -> AVEDataset:
    """Build the untouched official test split for post-selection evaluation only."""
    return AVEDataset(manifest_path, "test", root=root, **kwargs)
