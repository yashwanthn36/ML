"""Early validation for experiment configurations before dataset/media loading."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any


def load_symbol(spec: str):
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(f"Factory must use module:callable syntax, got {spec!r}")
    module_name, name = spec.split(":", 1)
    try:
        symbol = getattr(import_module(module_name), name)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"Cannot load configured factory {spec!r}: {error}") from error
    if not callable(symbol):
        raise ValueError(f"Configured factory {spec!r} is not callable")
    return symbol


def validate_pipeline_config(config: dict[str, Any], *, path: str | Path = "<config>") -> dict[str, Any]:
    """Check the shared schema and importable factories before constructing loaders."""
    for key in ("seed", "dataset", "test_dataset", "model", "training", "output_dir"):
        if key not in config:
            raise ValueError(f"{path}: required top-level field {key!r} is missing")
    if not isinstance(config["seed"], int):
        raise ValueError(f"{path}: seed must be an integer")
    for section_name in ("dataset", "test_dataset"):
        section = config[section_name]
        for key in ("name", "adapter", "factory", "split_protocol", "kwargs"):
            if key not in section:
                raise ValueError(f"{path}: {section_name}.{key} is required")
        if not isinstance(section["kwargs"], dict):
            raise ValueError(f"{path}: {section_name}.kwargs must be a mapping")
        load_symbol(section["factory"])
    if config["dataset"]["split_protocol"] != config["test_dataset"]["split_protocol"]:
        raise ValueError(f"{path}: training and test split protocols must match")
    model = config["model"]
    if "factory" not in model or "kwargs" not in model:
        raise ValueError(f"{path}: model.factory and model.kwargs are required")
    load_symbol(model["factory"])
    model_kwargs = model["kwargs"]
    if model["factory"] == "multimodal.models:build_model" or any(
            key in model_kwargs for key in ("encoder_1", "encoder_2")):
        for role in ("encoder_1", "encoder_2"):
            encoder = model_kwargs.get(role)
            if not isinstance(encoder, dict) or not (encoder.get("name") or encoder.get("factory")):
                raise ValueError(f"{path}: model.kwargs.{role} must declare an encoder name or factory")
            if encoder.get("factory"):
                load_symbol(encoder["factory"])
            else:
                from multimodal.models.encoders import _ENCODERS
                if encoder["name"] not in _ENCODERS:
                    raise ValueError(f"{path}: unknown {role} encoder {encoder['name']!r}; available: {sorted(_ENCODERS)}")
    training = config["training"]
    for key in ("batch_size", "epochs", "learning_rate", "weight_decay"):
        if key not in training:
            raise ValueError(f"{path}: training.{key} is required")
    if int(training["batch_size"]) <= 0 or int(training["epochs"]) <= 0 or float(training["learning_rate"]) <= 0:
        raise ValueError(f"{path}: batch_size, epochs and learning_rate must be positive")
    if not isinstance(config["output_dir"], str) or not config["output_dir"].strip():
        raise ValueError(f"{path}: output_dir must be a non-empty project-relative or absolute path")
    return config
