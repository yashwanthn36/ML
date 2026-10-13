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
        self.classes = sorted({row["label"] for row in rows})
        self.num_classes = max(int(row["label_id"]) for row in rows) + 1
        self.labels = [int(row["label_id"]) for row in self.rows]
        self.sample_ids = [row["sample_id"] for row in self.rows]
        self.source_ids = [row["video_id"] for row in self.rows]

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

    def _read_video(self, path: Path, start: float, end: float) -> torch.Tensor:
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError(f"Invalid video interval [{start}, {end}) for {path}")
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
        try:
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
        finally:
            capture.release()
        if not frames:
            raise OSError(f"Could not decode any frames from {path}")
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        array = np.stack(frames[:self.num_frames]).astype(np.float32) / 255.0
        return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        event_start, event_end = float(row["time_start"]), float(row["time_end"])
        if not np.isfinite(event_start) or not np.isfinite(event_end) or event_end <= event_start:
            raise ValueError(f"Invalid annotated event interval for sample {row['sample_id']}: "
                             f"[{event_start}, {event_end})")
        audio_path = self._path(row["audio_path"])
        if self.temporal_window_policy == "full_clip":
            with wave.open(str(audio_path), "rb") as wav:
                clip_end = wav.getnframes() / wav.getframerate()
            start, end = 0.0, clip_end
        else:
            start, end = event_start, event_end
        audio, coverage = self._read_audio(audio_path, start, end)
        video_path = self._path(row["video_path"])
        return {
            "modality_1": audio,
            "modality_2": self._read_video(video_path, start, end),
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "sample_id": row["sample_id"], "source_id": row["video_id"],
            "label_name": row["label"], "split": row["split"],
            "split_protocol": self.split_protocol,
            "temporal_window_policy": self.temporal_window_policy,
            "audio_valid_fraction": coverage,
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
