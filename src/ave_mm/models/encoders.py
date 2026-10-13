"""Backward-compatible aliases; reusable encoders live in multimodal.models."""
from multimodal.models.encoders import AudioCNNEncoder as AudioEncoder
from multimodal.models.encoders import FrameCNNEncoder as VideoEncoder

__all__ = ["AudioEncoder", "VideoEncoder"]
