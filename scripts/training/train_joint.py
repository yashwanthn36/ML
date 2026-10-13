"""Config-driven second-stage class/group and contribution-alignment training."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from multimodal.training import set_seed
from multimodal.training.joint import train_joint
from multimodal.training.device import select_device
from multimodal.training.provenance import (effective_split_protocol, file_sha256, ids_hash,
    validate_checkpoint, validate_downstream_artifact, configured_split_protocol,
    ensure_protocol_output)


def load_factory(spec: str):
    module_name, callable_name = spec.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), callable_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--groups", type=Path)
    parser.add_argument("--allow-legacy-artifacts", action="store_true",
                        help="Explicitly accept older checkpoints/targets/groups without complete provenance")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured_protocol = configured_split_protocol(config)
    if args.device:
        config["device"] = args.device
    device = select_device(str(config.get("device", "auto")), log=False)
    set_seed(int(config["seed"]))
    dataset_cfg, model_cfg, training_cfg = config["dataset"], config["model"], config["training"]
    dataset_factory = load_factory(dataset_cfg["factory"])
    dataset_kwargs = dict(dataset_cfg.get("kwargs", {}))
    if "split_protocol" in dataset_cfg:
        dataset_kwargs["split_protocol"] = dataset_cfg["split_protocol"]
    train_set, validation_set = dataset_factory(**dataset_kwargs)
    split_protocol = effective_split_protocol(train_set, configured_protocol)
    loader_args = {"batch_size": int(training_cfg["batch_size"]),
                   "num_workers": int(training_cfg["num_workers"]),
                   "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, shuffle=True, drop_last=int(training_cfg["batch_size"]) > 1,
                              **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    contribution_cfg = config.get("contribution", {})
    factory_spec = contribution_cfg.get("fcd_factory")
    if not factory_spec:
        raise ValueError("Set contribution.fcd_factory to the verified Utopia MI estimator factory")
    fcd_estimator = load_factory(factory_spec)(**contribution_cfg.get("fcd_kwargs", {}))
    reference_path = args.reference_checkpoint if args.reference_checkpoint.is_absolute() else PROJECT_ROOT / args.reference_checkpoint
    reference_checkpoint = torch.load(reference_path, map_location="cpu", weights_only=False)
    reference_provenance = validate_checkpoint(reference_checkpoint, split_protocol=split_protocol,
        seed=int(config["seed"]), config=config, training_sample_ids=list(train_set.sample_ids),
        allow_legacy=args.allow_legacy_artifacts)
    reference_sha256 = file_sha256(reference_path)
    targets_path = args.targets if args.targets.is_absolute() else PROJECT_ROOT / args.targets
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    validate_downstream_artifact(targets.get("provenance"), reference_provenance=reference_provenance,
        reference_sha256=reference_sha256, split_protocol=split_protocol, seed=int(config["seed"]),
        artifact="UCD target file", allow_legacy=args.allow_legacy_artifacts)
    target_ids_hash = targets.get("provenance", {}).get("training_sample_ids_sha256")
    if target_ids_hash is not None and target_ids_hash != ids_hash(list(train_set.sample_ids)):
        raise ValueError("UCD target training sample IDs do not match the configured training dataset")
    assignments = None
    if args.groups:
        groups_path = args.groups if args.groups.is_absolute() else PROJECT_ROOT / args.groups
        group_data = json.loads(groups_path.read_text(encoding="utf-8"))
        validate_downstream_artifact(group_data.get("provenance"), reference_provenance=reference_provenance,
            reference_sha256=reference_sha256, split_protocol=split_protocol, seed=int(config["seed"]),
            artifact="profile group file", allow_legacy=args.allow_legacy_artifacts)
        group_ids_hash = group_data.get("provenance", {}).get("training_sample_ids_sha256")
        if group_ids_hash is not None and group_ids_hash != ids_hash(list(train_set.sample_ids)):
            raise ValueError("Profile group training sample IDs do not match the configured training dataset")
        assignments = group_data["assignments"]
        expected_ids = set(map(str, train_set.sample_ids))
        actual_ids = set(map(str, assignments))
        if expected_ids != actual_ids:
            raise ValueError(f"Profile group assignment coverage mismatch: expected {len(expected_ids)} "
                             f"training IDs, got {len(actual_ids)}; missing={sorted(expected_ids-actual_ids)[:5]}, "
                             f"extra={sorted(actual_ids-expected_ids)[:5]}")
    seed_dir = ensure_protocol_output(PROJECT_ROOT / config["output_dir"] / f"seed_{config['seed']}",
                                      split_protocol=split_protocol, seed=int(config["seed"]))
    base_dir = seed_dir / "joint"
    modes = [float(x) for x in training_cfg.get("lambda_mi_grid", [])] \
        if training_cfg.get("optimization_mode", "fixed_weight") == "fixed_weight" else [None]
    if not modes:
        modes = [float(training_cfg.get("lambda_mi", 0.1))]
    candidates = []
    for lambda_mi in modes:
        set_seed(int(config["seed"]))
        model = load_factory(model_cfg["factory"])(num_classes=train_set.num_classes,
                                                    **model_cfg.get("kwargs", {}))
        run_training = copy.deepcopy(training_cfg)
        run_training.pop("lambda_mi_grid", None)
        if lambda_mi is not None:
            run_training["lambda_mi"] = lambda_mi
        run_dir = base_dir if lambda_mi is None else base_dir / f"lambda_{lambda_mi:g}"
        ensure_protocol_output(run_dir, split_protocol=split_protocol, seed=int(config["seed"]))
        checkpoint_path = train_joint(model, train_loader, validation_loader,
                                      training=run_training, targets=targets,
                                      group_assignments=assignments,
                                      fcd_estimator=fcd_estimator, run_dir=run_dir,
                                      seed=int(config["seed"]), config=config,
                                      reference_checkpoint=reference_checkpoint,
                                      reference_checkpoint_sha256=reference_sha256, device=device)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        candidates.append({"lambda_mi": lambda_mi, "checkpoint": str(checkpoint_path),
                           "validation_macro_f1": payload["validation_metrics"]["macro_f1"]})
    selected = max(candidates, key=lambda row: row["validation_macro_f1"])
    summary_path = base_dir / "lambda_grid_selection.json"
    base_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"selection_metric": "validation_macro_f1",
                                       "candidates": candidates, "selected": selected}, indent=2),
                            encoding="utf-8")
    print(f"selected checkpoint: {selected['checkpoint']}")


if __name__ == "__main__":
    main()
