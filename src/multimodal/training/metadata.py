"""Small config-derived reproducibility metadata helpers."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any


def experiment_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Capture configured encoder identity, preprocessing, freezing, and cache policy."""
    model = config.get("model", {})
    model_kwargs = model.get("kwargs", {})
    dataset = config.get("dataset", {})
    test_dataset = config.get("test_dataset", {})
    encoders = {}
    for role, modality in (("encoder_1", "modality_1"), ("encoder_2", "modality_2")):
        encoder = model_kwargs.get(role, {})
        name = encoder.get("name", "custom_or_unspecified") if isinstance(encoder, dict) else "custom_or_unspecified"
        kwargs = encoder.get("kwargs", {}) if isinstance(encoder, dict) else {}
        if name == "panns_cnn14":
            checkpoint_id = kwargs.get("checkpoint_path", "PANNs Cnn14_mAP=0.431 (AudioSet; official URL)")
            preprocessing = {"input_sample_rate": kwargs.get("input_sample_rate", 16000),
                             "backbone_sample_rate": 32000, "backbone": "PANNs Cnn14 log-mel frontend"}
        elif name == "resnet18":
            checkpoint_id = "torchvision ResNet18_Weights.DEFAULT" if kwargs.get("pretrained", True) else "random initialization"
            preprocessing = {"input": "RGB float frames in [0,1]", "resize": dataset.get("kwargs", {}).get("image_size"),
                             "normalization": {"mean": [0.485, 0.456, 0.406],
                                               "std": [0.229, 0.224, 0.225]}}
        else:
            checkpoint_id = kwargs.get("checkpoint_path", "config-defined or random initialization")
            preprocessing = {"adapter_config": dataset.get("kwargs", {})}
        encoders[role] = {"modality": modality, "name": name, "kwargs": kwargs,
                          "checkpoint_identity": checkpoint_id,
                          "frozen": bool(kwargs.get("freeze", model_kwargs.get("freeze_encoders", False))),
                          "preprocessing": preprocessing}
    packages = {}
    for package in ("torch", "torchvision", "panns-inference"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "model_factory": model.get("factory"), "encoders": encoders,
        "dataset_factory": dataset.get("factory"),
        "dataset_preprocessing_config": dataset.get("kwargs", {}),
        "test_preprocessing_config": test_dataset.get("kwargs", {}),
        "feature_cache": config.get("feature_cache", model_kwargs.get("feature_cache", {
            "enabled": "cached_features" in str(dataset.get("factory", "")),
            "dataset_factory": dataset.get("factory")
        })),
        "packages": packages,
        "identity_note": "Default weight aliases resolve according to the package version; no weight-file hash is implied.",
    }
