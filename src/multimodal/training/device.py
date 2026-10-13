"""Explicit device selection and one-time run diagnostics."""
from __future__ import annotations

import torch


def log_device(device: torch.device) -> None:
    available = torch.cuda.is_available()
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        name = torch.cuda.get_device_name(index)
        print(f"device={device} gpu={name} cuda_available={available} pytorch_cuda={torch.version.cuda}")
    else:
        print(f"device=cpu cuda_available={available}; running on CPU")


def select_device(requested: str = "auto", *, log: bool = True) -> torch.device:
    requested = str(requested).lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, cuda, cpu")
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was explicitly requested but torch.cuda.is_available() is false; "
                           f"PyTorch CUDA build: {torch.version.cuda!r}")
    device = torch.device("cuda" if (requested == "cuda" or (requested == "auto" and available)) else "cpu")
    if log:
        log_device(device)
    return device


def assert_module_device(module: torch.nn.Module, device: torch.device) -> None:
    for parameter in module.parameters():
        if parameter.device.type != device.type or (device.index is not None and parameter.device.index != device.index):
            raise RuntimeError(f"Model parameter is on {parameter.device}; selected device is {device}")


def assert_batch_device(batch: dict, device: torch.device) -> None:
    for key in ("modality_1", "modality_2", "label"):
        if key in batch and isinstance(batch[key], torch.Tensor) and (
            batch[key].device.type != device.type or
            (device.index is not None and batch[key].device.index != device.index)
        ):
            raise RuntimeError(f"Batch tensor {key!r} is on {batch[key].device}; selected device is {device}")


def on_device(tensor: torch.Tensor, device: torch.device) -> bool:
    return tensor.device.type == device.type and (device.index is None or tensor.device.index == device.index)
