"""Extract and cache frozen encoder tokens for each configured dataset split."""
from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from multimodal.models.encoders import build_encoder


def load_factory(spec: str):
    module_name, callable_name = spec.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), callable_name)


def cache_split(dataset, encoder_1, encoder_2, name: str, output_dir: Path,
                *, batch_size: int, num_workers: int, device: torch.device) -> Path:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    index_path = output_dir / f"{name}_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", "source_id", "label", "label_name", "split", "feature_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for batch_number, batch in enumerate(loader):
            raw_1 = batch["modality_1"].to(device)
            raw_2 = batch["modality_2"].to(device)
            features_1, features_2 = encoder_1(raw_1), encoder_2(raw_2)
            for offset, sample_id in enumerate(batch["sample_id"]):
                item_number = batch_number * batch_size + offset
                relative_feature = Path("features") / f"{name}_{item_number:08d}.pt"
                torch.save({"modality_1": features_1[offset].detach().cpu().contiguous(),
                            "modality_2": features_2[offset].detach().cpu().contiguous()},
                           output_dir / relative_feature)
                writer.writerow({"sample_id": sample_id, "source_id": batch["source_id"][offset],
                                 "label": int(batch["label"][offset]),
                                 "label_name": batch["label_name"][offset],
                                 "split": batch["split"][offset],
                                 "feature_path": relative_feature.as_posix()})
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    (output_dir / "features").mkdir(parents=True, exist_ok=True)
    train_set, validation_set = load_factory(config["dataset"]["factory"])(
        **config["dataset"].get("kwargs", {}))
    model_cfg, training_cfg = config["model"]["kwargs"], config["training"]
    encoders = []
    for key in ("encoder_1", "encoder_2"):
        spec = model_cfg[key]
        encoder = build_encoder(spec.get("name", "custom"), dim=int(model_cfg.get("dim", 128)),
                                kwargs=spec.get("kwargs"), factory=spec.get("factory"))
        encoder.eval().requires_grad_(False)
        encoders.append(encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_1, encoder_2 = (encoder.to(device) for encoder in encoders)
    with torch.inference_mode():
        train_index = cache_split(train_set, encoder_1, encoder_2, "train", output_dir,
                                  batch_size=int(training_cfg["batch_size"]),
                                  num_workers=int(training_cfg["num_workers"]), device=device)
        validation_index = cache_split(validation_set, encoder_1, encoder_2, "validation", output_dir,
                                       batch_size=int(training_cfg["batch_size"]),
                                       num_workers=int(training_cfg["num_workers"]), device=device)
        test_cfg = config.get("test_dataset")
        if test_cfg:
            test_set = load_factory(test_cfg["factory"])(**test_cfg.get("kwargs", {}))
            test_index = cache_split(test_set, encoder_1, encoder_2, "test", output_dir,
                                    batch_size=int(training_cfg["batch_size"]),
                                    num_workers=int(training_cfg["num_workers"]), device=device)
        else:
            test_index = None
    print(f"train_index: {train_index}\nvalidation_index: {validation_index}")
    if test_index:
        print(f"test_index: {test_index}")


if __name__ == "__main__":
    main()
