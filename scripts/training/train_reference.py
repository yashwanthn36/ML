"""Generic entry point; dataset and model adapters are selected by config."""
from __future__ import annotations
import argparse
import importlib
import json
import sys
from pathlib import Path
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from multimodal.training import set_seed, train_reference


def load_factory(spec: str):
    module_name, function_name = spec.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), function_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a paired-modality reference model")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    set_seed(seed)

    dataset_cfg = config["dataset"]
    train_set, validation_set = load_factory(dataset_cfg["factory"])(**dataset_cfg.get("kwargs", {}))
    training_cfg = config["training"]
    estimation_fraction = float(training_cfg.get("contribution_estimation_fraction", 0.2))
    if not 0.0 <= estimation_fraction < 1.0:
        raise ValueError("contribution_estimation_fraction must be in [0, 1)")
    if estimation_fraction:
        if not hasattr(train_set, "labels") or not hasattr(train_set, "sample_ids"):
            raise ValueError("Dataset adapter must expose training-only labels and sample_ids for the holdout split")
        reference_indices, estimation_indices = train_test_split(
            list(range(len(train_set))), test_size=estimation_fraction, random_state=seed,
            stratify=train_set.labels,
        )
        reference_train = Subset(train_set, reference_indices)
        estimation_ids = [train_set.sample_ids[i] for i in estimation_indices]
    else:
        reference_train = train_set
        estimation_ids = []
        estimation_indices = []
    loader_options = {"batch_size": int(training_cfg["batch_size"]),
                      "num_workers": int(training_cfg["num_workers"]),
                      "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(reference_train, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_options)

    model_cfg = config["model"]
    model = load_factory(model_cfg["factory"])(
        num_classes=train_set.num_classes, **model_cfg.get("kwargs", {})
    )
    output = PROJECT_ROOT / config["output_dir"] / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_estimation_ids.json").write_text(
        json.dumps({"indices": estimation_indices, "sample_ids": estimation_ids}, indent=2),
        encoding="utf-8",
    )
    checkpoint = train_reference(model, train_loader, validation_loader,
                                 training=training_cfg, run_dir=output, seed=seed, config=config)
    print(f"selected checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
