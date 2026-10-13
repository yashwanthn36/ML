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
from multimodal.training.device import assert_module_device, select_device, on_device
from multimodal.training.provenance import (effective_split_protocol, validate_checkpoint,
    validate_downstream_artifact, file_sha256, configured_split_protocol, ensure_protocol_output)


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
    parser.add_argument("--allow-legacy-artifacts", action="store_true",
                        help="Explicitly accept older checkpoints/targets without complete provenance")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured_protocol = configured_split_protocol(config)
    if args.device:
        config["device"] = args.device
    device = select_device(str(config.get("device", "auto")))
    dataset_cfg = config.get("test_dataset", {
        "factory": "ave_mm.dataset:build_test_dataset", "kwargs": config["dataset"].get("kwargs", {})
    })
    dataset_factory = load_factory(dataset_cfg["factory"])
    dataset_kwargs = dict(dataset_cfg.get("kwargs", {}))
    if "split_protocol" in dataset_cfg:
        dataset_kwargs["split_protocol"] = dataset_cfg["split_protocol"]
    test_set = dataset_factory(**dataset_kwargs)
    if isinstance(test_set, tuple):
        raise ValueError("Test factory must return a single test dataset, not train/validation datasets")
    model_cfg = config["model"]
    model = load_factory(model_cfg["factory"])(num_classes=test_set.num_classes,
                                                **model_cfg.get("kwargs", {}))
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    split_protocol = effective_split_protocol(test_set, configured_protocol)
    checkpoint_provenance = validate_checkpoint(checkpoint, split_protocol=split_protocol,
        seed=int(config["seed"]), config=config, allow_legacy=args.allow_legacy_artifacts,
        artifact="selected checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    assert_module_device(model, device)
    batch_size = int(config["training"].get("evaluation_batch_size", config["training"]["batch_size"]))
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                        num_workers=int(config["training"]["num_workers"]), pin_memory=device.type == "cuda")
    conditions = {"audio_video": None,
                  "audio_only": torch.tensor([1.0, 0.0], device=device),
                  "video_only": torch.tensor([0.0, 1.0], device=device)}
    targets, predictions = {key: [] for key in conditions}, {key: [] for key in conditions}
    full_modality_1, full_modality_2, fused_features = [], [], []
    all_sample_ids, all_source_ids = [], []
    output_dir = args.output_dir or (PROJECT_ROOT / config["output_dir"] / f"seed_{config['seed']}" / "test")
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = ensure_protocol_output(output_dir, split_protocol=split_protocol, seed=int(config["seed"]))
    prediction_path = output_dir / "test_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_id", "source_id", "label", "label_name", "prediction_audio_video",
                      "prediction_audio_only", "prediction_video_only"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for batch in loader:
            x1, x2 = batch["modality_1"].to(device), batch["modality_2"].to(device)
            labels = batch["label"].to(device)
            if not all(on_device(tensor, device) for tensor in (x1, x2, labels)):
                raise RuntimeError(f"Evaluation batch is not on selected device {device}")
            fused, one, two = model.fused_features(x1, x2)
            fused_features.append(fused.cpu())
            full_modality_1.append(one.cpu())
            full_modality_2.append(two.cpu())
            all_sample_ids.extend(map(str, batch.get("sample_id", [])))
            all_source_ids.extend(map(str, batch.get("source_id", [])))
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
    fcd_estimator = InfoNCEFCD(float(config.get("contribution", {}).get("epsilon", 1e-8)),
                               **{**config.get("contribution", {}).get("fcd_kwargs", {}), "queue_size": 0})
    fcd_information = fcd_estimator.per_sample_information(
        fused_all, source_1, source_2,
        sample_ids=all_sample_ids if len(all_sample_ids) == len(fused_all) else None,
        source_ids=all_source_ids if len(all_source_ids) == len(fused_all) else None,
    )
    fcd = fcd_estimator.distribution(fcd_information)
    metrics["fcd_test"] = fcd.tolist()
    if args.targets:
        targets_path = args.targets if args.targets.is_absolute() else PROJECT_ROOT / args.targets
        ucd_data = json.loads(targets_path.read_text(encoding="utf-8"))
        ref_sha = (checkpoint_provenance or {}).get("reference_checkpoint_sha256", file_sha256(checkpoint_path))
        validate_downstream_artifact(ucd_data.get("provenance"),
            reference_provenance=checkpoint_provenance, reference_sha256=ref_sha,
            split_protocol=split_protocol, seed=int(config["seed"]), artifact="UCD target file",
            allow_legacy=args.allow_legacy_artifacts)
        ucd = torch.tensor(ucd_data["global"]["target"], dtype=fcd.dtype)
        metrics["ucd_minus_fcd_kl_test"] = float(target_to_current_kl(ucd, fcd).cpu())
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["test_size"] = len(test_set)
    metrics["split_protocol"] = split_protocol
    metrics["checkpoint_split_protocol"] = checkpoint.get("split_protocol", "unknown")
    metrics["device"] = str(device)
    metrics["run_provenance"] = checkpoint_provenance
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved final test results under {output_dir}")


if __name__ == "__main__":
    main()
