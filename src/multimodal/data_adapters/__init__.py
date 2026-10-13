"""Reusable dataset adapters for the shared training contract."""

from .contract import validate_dataset, validate_split_datasets
from .generic_audio_video import GenericAudioVideoDataset

__all__ = ["GenericAudioVideoDataset", "validate_dataset", "validate_split_datasets"]
