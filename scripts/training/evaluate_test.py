"""Evaluate one selected checkpoint on an untouched test split."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from multimodal.contribution.fcd import InfoNCEFCD
from multimodal.losses.alignment import target_to_current_kl
from multimodal.training.classification_metrics import (classification_metrics,
    identify_minority_classes, validate_probabilities)
from multimodal.training.device import assert_module_device, select_device, on_device
from multimodal.training.provenance import (effective_split_protocol, validate_checkpoint,
    validate_downstream_artifact, file_sha256, config_hash, configured_split_protocol,
    ensure_protocol_output)
from multimodal.data_adapters.contract import validate_split_datasets
from multimodal.training.reference import seed_data_worker


def load_factory(spec: str):
    module_name, callable_name = spec.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), callable_name)


def _class_names(dataset) -> list[str]:
    names = getattr(dataset, "class_names", None) or getattr(dataset, "classes", None)
    if names is None:
        names = [str(index) for index in range(int(dataset.num_classes))]
    names = list(map(str, names))
    if len(names) != int(dataset.num_classes):
        raise ValueError(f"Dataset exposes {len(names)} class names for {dataset.num_classes} model outputs")
    return names


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

    # Read labels/metadata only. No training media is loaded for minority selection.
    train_cfg = config["dataset"]
    train_kwargs = dict(train_cfg.get("kwargs", {}))
    if "split_protocol" in train_cfg:
        train_kwargs["split_protocol"] = train_cfg["split_protocol"]
    train_set, validation_set = load_factory(train_cfg["factory"])(**train_kwargs)
    training_protocol = effective_split_protocol(train_set, configured_protocol)
    if not hasattr(train_set, "labels") or not hasattr(validation_set, "labels"):
        raise ValueError("Training and validation datasets must expose labels for metadata and minority selection")

    test_cfg = config.get("test_dataset", {
        "factory": "ave_mm.dataset:build_test_dataset", "kwargs": train_cfg.get("kwargs", {})
    })
    test_kwargs = dict(test_cfg.get("kwargs", {}))
    if "split_protocol" in test_cfg:
        test_kwargs["split_protocol"] = test_cfg["split_protocol"]
    test_set = load_factory(test_cfg["factory"])(**test_kwargs)
    if isinstance(test_set, tuple):
        raise ValueError("Test factory must return a single test dataset, not train/validation datasets")
    split_protocol = effective_split_protocol(test_set, configured_protocol)
    if split_protocol != training_protocol:
        raise ValueError(f"Training/test protocol mismatch: training uses {training_protocol!r}, test uses {split_protocol!r}")
    validate_split_datasets(train_set, validation_set, test_set,
        require_source_disjoint=(split_protocol == "source_video_disjoint" or split_protocol.endswith("_source_disjoint")))
    if int(train_set.num_classes) != int(test_set.num_classes):
        raise ValueError(f"Training/test class-count mismatch: {train_set.num_classes} vs {test_set.num_classes}")
    class_names = _class_names(test_set)
    if _class_names(train_set) != class_names or _class_names(validation_set) != class_names:
        raise ValueError("Training/validation/test class-to-label mapping mismatch")
    num_classes = int(test_set.num_classes)
    evaluation_cfg = config.get("evaluation", {})
    minority = identify_minority_classes(list(map(int, train_set.labels)), class_names=class_names,
        bottom_quantile=float(evaluation_cfg.get("minority_bottom_quantile", 0.25)))
    training_counts = np.bincount(np.asarray(train_set.labels, dtype=np.int64), minlength=num_classes)
    validation_counts = np.bincount(np.asarray(validation_set.labels, dtype=np.int64), minlength=num_classes)

    model_cfg = config["model"]
    model = load_factory(model_cfg["factory"])(num_classes=num_classes, **model_cfg.get("kwargs", {}))
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_provenance = validate_checkpoint(checkpoint, split_protocol=split_protocol,
        seed=int(config["seed"]), config=config, training_sample_ids=list(train_set.sample_ids),
        dataset_metadata=getattr(train_set, "metadata", {}), class_to_id=train_set.class_to_id,
        allow_legacy=args.allow_legacy_artifacts,
        artifact="selected checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    assert_module_device(model, device)
    batch_size = int(config["training"].get("evaluation_batch_size", config["training"]["batch_size"]))
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                        num_workers=int(config["training"]["num_workers"]), pin_memory=device.type == "cuda",
                        worker_init_fn=seed_data_worker,
                        generator=torch.Generator().manual_seed(int(config["seed"])))
    conditions = {"audio_video": None,
                  "audio_only": torch.tensor([1.0, 0.0], device=device),
                  "video_only": torch.tensor([0.0, 1.0], device=device)}
    true_labels: list[int] = []
    ids: list[str] = []
    sources: list[str] = []
    labels_text: list[str] = []
    probabilities: dict[str, list[np.ndarray]] = {key: [] for key in conditions}
    fused_rows, modality_1_rows, modality_2_rows = [], [], []
    output_dir = args.output_dir or (PROJECT_ROOT / config["output_dir"] / f"seed_{config['seed']}" / "test")
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = ensure_protocol_output(output_dir, split_protocol=split_protocol, seed=int(config["seed"]))
    with torch.inference_mode():
        for batch in loader:
            x1, x2 = batch["modality_1"].to(device), batch["modality_2"].to(device)
            y = batch["label"].to(device)
            if not all(on_device(tensor, device) for tensor in (x1, x2, y)):
                raise RuntimeError(f"Evaluation batch is not on selected device {device}")
            fused, one, two = model.fused_features(x1, x2)
            fused_rows.append(fused.cpu())
            modality_1_rows.append(one.cpu())
            modality_2_rows.append(two.cpu())
            true_labels.extend(map(int, y.cpu().tolist()))
            ids.extend(map(str, batch["sample_id"]))
            sources.extend(map(str, batch.get("source_id", [""] * len(y))))
            labels_text.extend(map(str, batch.get("label_name", [class_names[i] for i in y.cpu().tolist()])))
            for condition, mask in conditions.items():
                logits = model.classifier(fused) if mask is None else model(
                    x1, x2, modality_mask=mask.expand(y.size(0), -1))
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probabilities[condition].append(probs)

    if not true_labels:
        raise ValueError("Test dataset contains no samples")
    for sample_id, class_id, class_name in zip(ids, true_labels, labels_text):
        if class_names[class_id] != class_name:
            raise ValueError(f"Test class name/ID mismatch for sample {sample_id}: "
                             f"ID {class_id} maps to {class_names[class_id]!r}, got {class_name!r}")
    probs_all = {key: validate_probabilities(np.concatenate(rows), num_classes=num_classes)
                 for key, rows in probabilities.items()}
    metrics = {}
    for condition, probs in probs_all.items():
        values = classification_metrics(true_labels, probs, class_names=class_names,
                                        minority_class_ids=minority["minority_class_ids"])
        # Retain legacy metric keys used by prior downstream code.
        values["per_class_recall"] = [row["recall"] for row in values["per_class"]]
        metrics[condition] = values

    class_counts_test = np.bincount(np.asarray(true_labels), minlength=num_classes)
    prediction_path = output_dir / "test_predictions.csv"
    fields = ["sample_id", "source_id", "label", "label_name", "true_class_id", "true_class_name"]
    fields += [f"prediction_{condition}" for condition in conditions]
    fields += [f"prob_{condition}_class_{class_id}" for condition in conditions for class_id in range(num_classes)]
    with prediction_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index, sample_id in enumerate(ids):
            row = {"sample_id": sample_id, "source_id": sources[row_index],
                   "label": true_labels[row_index], "label_name": labels_text[row_index],
                   "true_class_id": true_labels[row_index], "true_class_name": labels_text[row_index]}
            for condition, probs in probs_all.items():
                prediction = int(probs[row_index].argmax())
                row[f"prediction_{condition}"] = prediction
                if prediction != int(np.argmax(probs[row_index])):
                    raise RuntimeError(f"Saved prediction/probability mismatch for {sample_id} ({condition})")
                for class_id, prob in enumerate(probs[row_index]):
                    row[f"prob_{condition}_class_{class_id}"] = float(prob)
            writer.writerow(row)

    source_1, source_2 = torch.cat(modality_1_rows), torch.cat(modality_2_rows)
    fused_all = torch.cat(fused_rows)
    fcd_estimator = InfoNCEFCD(float(config.get("contribution", {}).get("epsilon", 1e-8)),
                               **{**config.get("contribution", {}).get("fcd_kwargs", {}), "queue_size": 0})
    fcd_information = fcd_estimator.per_sample_information(fused_all, source_1, source_2,
        sample_ids=ids if len(ids) == len(fused_all) else None,
        source_ids=sources if len(sources) == len(fused_all) else None)
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

    metrics.update({
        "seed": int(config["seed"]), "experiment_seed": int(config["seed"]),
        "split_protocol": split_protocol,
        "model_condition": checkpoint.get("model_condition",
            "proposed_ci_mi" if "optimization_mode" in checkpoint else "reference"),
        "checkpoint_split_protocol": checkpoint.get("split_protocol", "unknown"),
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_identity": f"{checkpoint_path.name}:{file_sha256(checkpoint_path)}",
        "test_size": len(test_set), "test_sample_count": len(test_set),
        "class_names": class_names,
        "class_counts": {"train": {str(i): int(training_counts[i]) for i in range(num_classes)},
                         "validation": {str(i): int(validation_counts[i]) for i in range(num_classes)},
                         "test": {str(i): int(class_counts_test[i]) for i in range(num_classes)}},
        "minority_class_definition": minority,
        "device": str(device), "run_provenance": checkpoint_provenance,
        "experiment_config_sha256": config_hash(config),
        "dataset_name": getattr(train_set, "metadata", {}).get("dataset_name", config["dataset"].get("name")),
        "dataset_adapter": getattr(train_set, "metadata", {}).get("adapter", config["dataset"].get("adapter")),
        "dataset_metadata": getattr(train_set, "metadata", {}),
        "class_to_id": {str(key): int(value) for key, value in train_set.class_to_id.items()},
    })
    (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved final test results under {output_dir}")


if __name__ == "__main__":
    main()
