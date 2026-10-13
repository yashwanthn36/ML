"""Extract and cache frozen encoder tokens for each configured dataset split."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from multimodal.models.encoders import build_encoder
from multimodal.training.device import assert_module_device, select_device, on_device
from multimodal.training.metadata import experiment_metadata
from multimodal.training.provenance import (configured_split_protocol, effective_split_protocol,
                                            ensure_protocol_output, requires_source_disjoint)
from multimodal.training.configuration import validate_pipeline_config
from multimodal.data_adapters.contract import validate_split_datasets
from multimodal.training.reference import seed_data_worker, set_seed


def load_factory(spec: str):
    module_name, callable_name = spec.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), callable_name)


def cache_split(dataset, encoder_1, encoder_2, name: str, output_dir: Path,
                *, batch_size: int, num_workers: int, device: torch.device, seed: int) -> Path:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        worker_init_fn=seed_data_worker, generator=torch.Generator().manual_seed(seed))
    index_path = output_dir / f"{name}_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", "source_id", "label", "label_name", "split", "split_protocol",
                  "dataset_name", "class_to_id", "feature_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for batch_number, batch in enumerate(loader):
            raw_1 = batch["modality_1"].to(device)
            raw_2 = batch["modality_2"].to(device)
            if not on_device(raw_1, device) or not on_device(raw_2, device):
                raise RuntimeError(f"Feature cache batch is not on selected device {device}")
            features_1, features_2 = encoder_1(raw_1), encoder_2(raw_2)
            for offset, sample_id in enumerate(batch["sample_id"]):
                item_number = batch_number * batch_size + offset
                relative_feature = Path("features") / f"{name}_{item_number:08d}.pt"
                torch.save({"modality_1": features_1[offset].detach().cpu().contiguous(),
                            "modality_2": features_2[offset].detach().cpu().contiguous()},
                           output_dir / relative_feature)
                writer.writerow({"sample_id": sample_id, "source_id": batch.get("source_id", [""] * len(batch["sample_id"]))[offset],
                                 "label": int(batch["label"][offset]),
                                 "label_name": batch["label_name"][offset],
                                 "split": batch["split"][offset],
                                 "split_protocol": dataset.split_protocol,
                                 "dataset_name": getattr(dataset, "dataset_name",
                                     getattr(dataset, "metadata", {}).get("dataset_name", "dataset")),
                                 "class_to_id": json.dumps(dataset.class_to_id, sort_keys=True),
                                 "feature_path": relative_feature.as_posix()})
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_pipeline_config(config, path=config_path)
    configured_protocol = configured_split_protocol(config)
    if args.device:
        config["device"] = args.device
    set_seed(int(config["seed"]))
    device = select_device(str(config.get("device", "auto")))
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    dataset_cfg = config["dataset"]
    dataset_kwargs = dict(dataset_cfg.get("kwargs", {}))
    if "split_protocol" in dataset_cfg:
        dataset_kwargs["split_protocol"] = dataset_cfg["split_protocol"]
    train_set, validation_set = load_factory(dataset_cfg["factory"])(**dataset_kwargs)
    split_protocol = effective_split_protocol(train_set, configured_protocol)
    validate_split_datasets(train_set, validation_set,
        require_source_disjoint=requires_source_disjoint(config, split_protocol))
    output_dir = ensure_protocol_output(output_dir, split_protocol=split_protocol, seed=int(config["seed"]))
    (output_dir / "features").mkdir(parents=True, exist_ok=True)
    model_cfg, training_cfg = config["model"]["kwargs"], config["training"]
    encoders = []
    for key in ("encoder_1", "encoder_2"):
        spec = model_cfg[key]
        encoder = build_encoder(spec.get("name", "custom"), dim=int(model_cfg.get("dim", 128)),
                                kwargs=spec.get("kwargs"), factory=spec.get("factory"))
        encoder.eval().requires_grad_(False)
        encoders.append(encoder)
    encoder_1, encoder_2 = (encoder.to(device) for encoder in encoders)
    assert_module_device(encoder_1, device)
    assert_module_device(encoder_2, device)
    metadata = experiment_metadata(config)
    metadata["device"] = str(device)
    (output_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "cache_metadata.json").write_text(json.dumps({
        "split_protocol": split_protocol, "device": str(device),
        "encoder_configs": {key: model_cfg[key] for key in ("encoder_1", "encoder_2")}}, indent=2),
        encoding="utf-8")
    with torch.inference_mode():
        train_index = cache_split(train_set, encoder_1, encoder_2, "train", output_dir,
                                  batch_size=int(training_cfg["batch_size"]),
                                  num_workers=int(training_cfg["num_workers"]), device=device,
                                  seed=int(config["seed"]))
        validation_index = cache_split(validation_set, encoder_1, encoder_2, "validation", output_dir,
                                       batch_size=int(training_cfg["batch_size"]),
                                       num_workers=int(training_cfg["num_workers"]), device=device,
                                       seed=int(config["seed"]) + 1)
        test_cfg = config.get("test_dataset")
        if test_cfg:
            test_kwargs = dict(test_cfg.get("kwargs", {}))
            if "split_protocol" in test_cfg:
                test_kwargs["split_protocol"] = test_cfg["split_protocol"]
            test_set = load_factory(test_cfg["factory"])(**test_kwargs)
            effective_split_protocol(test_set, configured_protocol)
            validate_split_datasets(train_set, validation_set, test_set,
                require_source_disjoint=requires_source_disjoint(config, split_protocol))
            test_index = cache_split(test_set, encoder_1, encoder_2, "test", output_dir,
                                     batch_size=int(training_cfg["batch_size"]),
                                     num_workers=int(training_cfg["num_workers"]), device=device,
                                     seed=int(config["seed"]) + 2)
        else:
            test_index = None
    print(f"train_index: {train_index}\nvalidation_index: {validation_index}")
    if test_index:
        print(f"test_index: {test_index}")


if __name__ == "__main__":
    main()
