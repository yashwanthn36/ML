"""Estimate training-only signatures, contribution-profile groups and UCD targets."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from multimodal.contribution import build_ucd_targets, estimate_reference_fcd, estimate_signatures
from multimodal.grouping import cluster_profiles
from multimodal.training.device import select_device
from multimodal.training.provenance import (checkpoint_provenance, effective_split_protocol,
    file_sha256, validate_checkpoint, validate_provenance, configured_split_protocol,
    ensure_protocol_output)
from multimodal.data_adapters.contract import validate_dataset
from multimodal.training.reference import seed_data_worker


def load_factory(spec: str):
    module_name, function_name = spec.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), function_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--estimation-ids", type=Path)
    parser.add_argument("--allow-legacy-artifacts", action="store_true",
                        help="Explicitly accept older artifacts without complete run provenance")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured_protocol = configured_split_protocol(config)
    if args.device:
        config["device"] = args.device
    device = select_device(str(config.get("device", "auto")))
    dataset_cfg, model_cfg, training_cfg = config["dataset"], config["model"], config["training"]
    dataset_kwargs = dict(dataset_cfg.get("kwargs", {}))
    if "split_protocol" in dataset_cfg:
        dataset_kwargs["split_protocol"] = dataset_cfg["split_protocol"]
    train_set, _ = load_factory(dataset_cfg["factory"])(**dataset_kwargs)
    split_protocol = effective_split_protocol(train_set, configured_protocol)
    validate_dataset(train_set, name="training")
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ref_provenance = validate_checkpoint(checkpoint, split_protocol=split_protocol,
        seed=int(config["seed"]), config=config, training_sample_ids=list(train_set.sample_ids),
        dataset_metadata=getattr(train_set, "metadata", {}), class_to_id=train_set.class_to_id,
        allow_legacy=args.allow_legacy_artifacts)
    model = load_factory(model_cfg["factory"])(num_classes=train_set.num_classes,
                                                **model_cfg.get("kwargs", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    out = args.output_dir or (PROJECT_ROOT / config["output_dir"] / f"seed_{config['seed']}" / "contributions")
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out = ensure_protocol_output(out, split_protocol=split_protocol, seed=int(config["seed"]))
    estimation_ids_path = args.estimation_ids or (checkpoint_path.parent / "training_estimation_ids.json")
    if not estimation_ids_path.is_absolute():
        estimation_ids_path = PROJECT_ROOT / estimation_ids_path
    if estimation_ids_path.exists():
        estimation_metadata = json.loads(estimation_ids_path.read_text(encoding="utf-8"))
        if ref_provenance is not None:
            validate_provenance(ref_provenance, estimation_metadata.get("provenance"),
                                artifact="training_estimation_ids.json", allow_legacy=args.allow_legacy_artifacts)
        elif not args.allow_legacy_artifacts:
            raise ValueError("Reference checkpoint has no run provenance; pass --allow-legacy-artifacts explicitly")
        estimation_ids = set(estimation_metadata["sample_ids"])
        training_ids = set(train_set.sample_ids)
        unknown_ids = sorted(estimation_ids - training_ids)
        if unknown_ids:
            raise ValueError(f"Estimation IDs are not in the current training partition: {unknown_ids[:5]}")
        if len(train_set.sample_ids) != len(training_ids):
            raise ValueError("Training dataset sample IDs must be unique for leakage-safe contribution fitting")
        estimation_indices = [i for i, sample_id in enumerate(train_set.sample_ids) if sample_id in estimation_ids]
        if not estimation_indices:
            raise ValueError(f"No training samples match the estimation ID file: {estimation_ids_path}")
        estimation_set = Subset(train_set, estimation_indices)
    else:
        raise FileNotFoundError("Reference checkpoint has no training_estimation_ids.json; refusing to "
                                "silently estimate contributions on all reference-training examples")
    if set(estimation_ids) != {train_set.sample_ids[i] for i in estimation_indices}:
        raise ValueError("Estimation ID metadata does not exactly match the current training dataset")
    loader_args = {"batch_size": int(training_cfg["batch_size"]), "shuffle": False,
                   "num_workers": int(training_cfg["num_workers"]), "pin_memory": device.type == "cuda"}
    estimation_loader = DataLoader(estimation_set, worker_init_fn=seed_data_worker,
                                   generator=torch.Generator().manual_seed(int(config["seed"])), **loader_args)
    signatures_path = estimate_signatures(model, estimation_loader, out / "estimation_signatures.csv", device=device)
    fcd_spec = config.get("contribution", {}).get("fcd_factory")
    if not fcd_spec:
        raise ValueError("Set contribution.fcd_factory to the configured mutual-information estimator")
    fcd_module, fcd_name = fcd_spec.split(":", maxsplit=1)
    fcd_estimator = getattr(importlib.import_module(fcd_module), fcd_name)(
        **config.get("contribution", {}).get("fcd_kwargs", {})
    )
    reference_fcd = estimate_reference_fcd(model, estimation_loader, fcd_estimator, device=device)
    (out / "reference_fcd.json").write_text(json.dumps(reference_fcd, indent=2), encoding="utf-8")
    all_signatures_path = None
    if len(estimation_set) != len(train_set):
        all_loader = DataLoader(train_set, worker_init_fn=seed_data_worker,
                                generator=torch.Generator().manual_seed(int(config["seed"]) + 1), **loader_args)
        all_signatures_path = estimate_signatures(model, all_loader, out / "all_training_signatures.csv", device=device)
    contribution_cfg = config.get("contribution", {})
    grouping = cluster_profiles(
        signatures_path, out / "profile_groups.json",
        min_support=contribution_cfg.get("min_group_support"),
        min_support_floor=int(contribution_cfg.get("min_group_support_floor", 20)),
        max_groups=int(contribution_cfg.get("max_groups_per_class", 3)),
        assignment_signatures_csv=all_signatures_path,
    )
    targets = build_ucd_targets(
        signatures_path, out / "ucd_targets.json", class_ids=list(range(train_set.num_classes)),
        group_ids=grouping["assignments"],
        min_support=int(contribution_cfg.get("target_min_support", 20)),
        epsilon=float(contribution_cfg.get("epsilon", 1e-8)),
        bootstrap_replicates=int(contribution_cfg.get("bootstrap_replicates", 200)),
        seed=int(config["seed"]),
        max_bootstrap_std=contribution_cfg.get("max_bootstrap_std", 0.15),
    )
    training_ids_hash = (ref_provenance or {}).get("training_sample_ids_sha256")
    if training_ids_hash is None:
        from multimodal.training.provenance import ids_hash
        training_ids_hash = ids_hash(list(train_set.sample_ids))
    artifact_provenance = {
        **(ref_provenance or {"split_protocol": split_protocol, "seed": int(config["seed"]),
                              "run_id": f"legacy-{file_sha256(checkpoint_path)[:16]}",
                              "training_sample_ids_sha256": training_ids_hash,
                              "estimation_sample_ids_sha256": "legacy-unverified",
                              "experiment_config_sha256": "legacy-unverified"}),
        "reference_checkpoint_sha256": file_sha256(checkpoint_path),
    }
    grouping["provenance"] = artifact_provenance
    targets["provenance"] = artifact_provenance
    (out / "profile_groups.json").write_text(json.dumps(grouping, indent=2), encoding="utf-8")
    (out / "ucd_targets.json").write_text(json.dumps(targets, indent=2), encoding="utf-8")
    protocol = {
        "protocol": estimation_metadata.get("protocol", "unspecified"),
        "provenance": artifact_provenance,
        "split_protocol": split_protocol,
        "estimation_sample_ids": sorted(estimation_ids),
        "reference_fit_sample_ids": estimation_metadata.get("fit_sample_ids", []),
        "reference_fit_source_ids": estimation_metadata.get("fit_source_ids", []),
        "estimation_source_ids": estimation_metadata.get("estimation_source_ids", []),
        "assignment_sample_ids": list(train_set.sample_ids),
        "assignment_source_ids": list(getattr(train_set, "source_ids", [])),
        "assignment_signatures_include_reference_fit_examples": all_signatures_path is not None,
        "validation_or_test_used": False,
        "device": str(device),
        "limitations": estimation_metadata.get("limitations", []),
    }
    (out / "contribution_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out / "analysis_summary.json").write_text(json.dumps({
        "checkpoint": str(checkpoint_path), "signatures": str(signatures_path),
        "reference_fcd": str(out / "reference_fcd.json"),
        "group_count": len(grouping["group_statistics"]),
        "ucd_scopes": {"classes": len(targets["classes"]), "groups": len(targets["groups"])},
        "fallbacks": targets["fallbacks"],
    }, indent=2), encoding="utf-8")
    print(f"saved signatures, groups and UCD targets under {out}")


if __name__ == "__main__":
    main()
