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


def load_factory(spec: str):
    module_name, function_name = spec.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), function_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--estimation-ids", type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_cfg, model_cfg, training_cfg = config["dataset"], config["model"], config["training"]
    train_set, _ = load_factory(dataset_cfg["factory"])(**dataset_cfg.get("kwargs", {}))
    model = load_factory(model_cfg["factory"])(num_classes=train_set.num_classes,
                                                **model_cfg.get("kwargs", {}))
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    out = args.output_dir or (PROJECT_ROOT / config["output_dir"] / f"seed_{config['seed']}" / "contributions")
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    estimation_ids_path = args.estimation_ids or (checkpoint_path.parent / "training_estimation_ids.json")
    if not estimation_ids_path.is_absolute():
        estimation_ids_path = PROJECT_ROOT / estimation_ids_path
    if estimation_ids_path.exists():
        estimation_ids = set(json.loads(estimation_ids_path.read_text(encoding="utf-8"))["sample_ids"])
        estimation_indices = [i for i, sample_id in enumerate(train_set.sample_ids) if sample_id in estimation_ids]
        if not estimation_indices:
            raise ValueError(f"No training samples match the estimation ID file: {estimation_ids_path}")
        estimation_set = Subset(train_set, estimation_indices)
    else:
        estimation_set = train_set
    loader_args = {"batch_size": int(training_cfg["batch_size"]), "shuffle": False,
                   "num_workers": int(training_cfg["num_workers"]), "pin_memory": torch.cuda.is_available()}
    estimation_loader = DataLoader(estimation_set, **loader_args)
    signatures_path = estimate_signatures(model, estimation_loader, out / "estimation_signatures.csv")
    fcd_spec = config.get("contribution", {}).get("fcd_factory")
    if not fcd_spec:
        raise ValueError("Set contribution.fcd_factory to the configured mutual-information estimator")
    fcd_module, fcd_name = fcd_spec.split(":", maxsplit=1)
    fcd_estimator = getattr(importlib.import_module(fcd_module), fcd_name)(
        **config.get("contribution", {}).get("fcd_kwargs", {})
    )
    reference_fcd = estimate_reference_fcd(model, estimation_loader, fcd_estimator)
    (out / "reference_fcd.json").write_text(json.dumps(reference_fcd, indent=2), encoding="utf-8")
    all_signatures_path = None
    if len(estimation_set) != len(train_set):
        all_loader = DataLoader(train_set, **loader_args)
        all_signatures_path = estimate_signatures(model, all_loader, out / "all_training_signatures.csv")
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
