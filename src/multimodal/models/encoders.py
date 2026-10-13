"""Reusable encoder building blocks and a small extensible encoder registry.

Encoders adapt modality-specific inputs to [batch, tokens, dimension]. Dataset
preprocessing stays in dataset adapters; encoder selection belongs in config.
"""
from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class AudioCNNEncoder(nn.Module):
    """Trainable log-spectrum encoder for mono waveform batches [B, samples]."""
    def __init__(self, dim: int, tokens: int = 16) -> None:
        super().__init__()
        self.tokens = tokens
        self.register_buffer("window", torch.hann_window(400), persistent=False)
        self.conv = nn.Sequential(
            nn.Conv1d(201, 96, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, dim, 3, stride=2, padding=1, bias=False), nn.BatchNorm1d(dim), nn.GELU(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(waveform, n_fft=400, hop_length=160, win_length=400,
                              window=self.window, center=True, return_complex=True)
        features = self.conv(spectrum.abs().clamp_min(1e-6).log())
        return self.norm(F.adaptive_avg_pool1d(features, self.tokens).transpose(1, 2))


class PANNsCnn14AudioEncoder(nn.Module):
    """AudioSet-pretrained PANNs CNN14 waveform encoder (one pooled token)."""
    CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

    def __init__(self, dim: int, *, input_sample_rate: int = 16_000,
                 pretrained: bool = True, freeze: bool = True,
                 checkpoint_path: str | None = None) -> None:
        super().__init__()
        try:
            from panns_inference.models import Cnn14
        except ImportError as exc:
            raise ImportError("panns_cnn14 requires panns-inference and its torchlibrosa dependency") from exc
        self.backbone = Cnn14(sample_rate=32_000, window_size=1024, hop_size=320,
                              mel_bins=64, fmin=50, fmax=14_000, classes_num=527)
        if pretrained:
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            else:
                checkpoint = torch.hub.load_state_dict_from_url(
                    self.CHECKPOINT_URL, map_location="cpu", file_name="Cnn14_mAP=0.431.pth"
                )
            self.backbone.load_state_dict(checkpoint["model"])
        self.input_sample_rate = int(input_sample_rate)
        self.freeze_backbone = freeze
        if freeze:
            self.backbone.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.input_sample_rate != 32_000:
            try:
                import torchaudio.functional as audio_functional
            except ImportError as exc:
                raise ImportError("PANNs resampling requires torchaudio matching the installed PyTorch") from exc
            waveform = audio_functional.resample(waveform, self.input_sample_rate, 32_000)
        output = self.backbone(waveform, mixup_lambda=None)["embedding"]
        return output.unsqueeze(1)


class FrameCNNEncoder(nn.Module):
    """Trainable per-frame CNN for video tensors [B, frames, 3, height, width]."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, dim), nn.LayerNorm(dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = frames.shape
        values = self.backbone(frames.reshape(batch * time, channels, height, width))
        return values.reshape(batch, time, -1)


class ResNet18FrameEncoder(nn.Module):
    """Torchvision ResNet-18 applied independently to frames, returning tokens."""
    def __init__(self, dim: int, *, pretrained: bool = True, freeze: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise ImportError("resnet18 encoder requires torchvision; install the matching torchvision build") from exc
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.freeze_backbone = freeze
        if freeze:
            self.backbone.requires_grad_(False)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = frames.shape
        frames = (frames - self.mean) / self.std
        features = self.backbone(frames.reshape(batch * time, channels, height, width))
        return features.reshape(batch, time, -1)


_ENCODERS = {
    "audio_cnn": AudioCNNEncoder,
    "panns_cnn14": PANNsCnn14AudioEncoder,
    "frame_cnn": FrameCNNEncoder,
    "resnet18": ResNet18FrameEncoder,
}


def build_encoder(name: str, *, dim: int, kwargs: dict[str, Any] | None = None,
                  factory: str | None = None) -> nn.Module:
    """Build a registered encoder or a custom ``module:callable`` encoder factory."""
    if factory:
        module_name, callable_name = factory.split(":", maxsplit=1)
        constructor = getattr(importlib.import_module(module_name), callable_name)
        return constructor(dim=dim, **(kwargs or {}))
    try:
        encoder = _ENCODERS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown encoder {name!r}; available: {', '.join(sorted(_ENCODERS))}") from exc
    return encoder(dim=dim, **(kwargs or {}))
