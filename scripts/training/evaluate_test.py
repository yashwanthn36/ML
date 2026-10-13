"""Evaluate one selected checkpoint on a dataset adapter's untouched test split."""
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
from sklearn.metrics import accuracy_score, f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from multimodal.contribution.fcd import InfoNCEFCD
from multimodal.losses.alignment import target_to_current_kl


def load_factory(spec: str):
    module_name, callable_name = spec.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), callable_name)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_cfg = config.get("test_dataset", {
        "factory": "ave_mm.dataset:build_test_dataset", "kwargs": config["dataset"].get("kwargs", {})
    })
    dataset_factory = load_factory(dataset_cfg["factory"])
    test_set = dataset_factory(**dataset_cfg.get("kwargs", {}))
    if isinstance(test_set, tuple):
        raise ValueError("Test factory must return a single test dataset, not train/validation datasets")
    model_cfg = config["model"]
    model = load_factory(model_cfg["factory"])(num_classes=test_set.num_classes,
                                                **model_cfg.get("kwargs", {}))
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    batch_size = int(config["training"].get("evaluation_batch_size", config["training"]["batch_size"]))
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                        num_workers=int(config["training"]["num_workers"]), pin_memory=torch.cuda.is_available())
    conditions = {"audio_video": None,
                  "audio_only": torch.tensor([1.0, 0.0], device=device),
                  "video_only": torch.tensor([0.0, 1.0], device=device)}
    targets, predictions = {key: [] for key in conditions}, {key: [] for key in conditions}
    full_modality_1, full_modality_2, fused_features = [], [], []
    output_dir = args.output_dir or (PROJECT_ROOT / config["output_dir"] / f"seed_{config['seed']}" / "test")
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "test_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_id", "source_id", "label", "label_name", "prediction_audio_video",
                      "prediction_audio_only", "prediction_video_only"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for batch in loader:
            x1, x2 = batch["modality_1"].to(device), batch["modality_2"].to(device)
            labels = batch["label"].to(device)
            fused, one, two = model.fused_features(x1, x2)
            fused_features.append(fused.cpu())
            full_modality_1.append(one.cpu())
            full_modality_2.append(two.cpu())
            batch_predictions = {}
            for name, mask in conditions.items():
                logits = (model.classifier(fused) if mask is None else
                          model(x1, x2, modality_mask=mask.expand(labels.size(0), -1)))
                pred = logits.argmax(dim=1)
                targets[name].extend(labels.cpu().tolist())
                predictions[name].extend(pred.cpu().tolist())
                batch_predictions[name] = pred.cpu().tolist()
            for i, sample_id in enumerate(batch["sample_id"]):
                writer.writerow({"sample_id": sample_id, "source_id": batch["source_id"][i],
                                 "label": int(labels[i]), "label_name": batch["label_name"][i],
                                 "prediction_audio_video": batch_predictions["audio_video"][i],
                                 "prediction_audio_only": batch_predictions["audio_only"][i],
                                 "prediction_video_only": batch_predictions["video_only"][i]})
    num_classes = test_set.num_classes
    metrics = {}
    for name in conditions:
        recalls = recall_score(targets[name], predictions[name], average=None,
                               labels=list(range(num_classes)), zero_division=0)
        metrics[name] = {"accuracy": float(accuracy_score(targets[name], predictions[name])),
                         "macro_f1": float(f1_score(targets[name], predictions[name], average="macro",
                                                     labels=list(range(num_classes)), zero_division=0)),
                         "balanced_accuracy": float(recalls.mean()),
                         "per_class_recall": recalls.tolist()}
    # Dataset-level InfoNCE negatives avoid a batch-size-dependent test estimate.
    source_1 = torch.cat(full_modality_1, dim=0)
    source_2 = torch.cat(full_modality_2, dim=0)
    fused_all = torch.cat(fused_features, dim=0)
    fcd = InfoNCEFCD(float(config.get("contribution", {}).get("epsilon", 1e-8)))(
        fused_all, source_1, source_2).mean(dim=0)
    metrics["fcd_test"] = fcd.tolist()
    if args.targets:
        targets_path = args.targets if args.targets.is_absolute() else PROJECT_ROOT / args.targets
        ucd_data = json.loads(targets_path.read_text(encoding="utf-8"))
        ucd = torch.tensor(ucd_data["global"]["target"], dtype=fcd.dtype)
        metrics["ucd_minus_fcd_kl_test"] = float(target_to_current_kl(ucd, fcd).cpu())
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["test_size"] = len(test_set)
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved final test results under {output_dir}")


if __name__ == "__main__":
    main()
