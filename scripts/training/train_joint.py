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


def load_factory(spec: str):
    module_name, callable_name = spec.split(":", maxsplit=1)
    return getattr(importlib.import_module(module_name), callable_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--groups", type=Path)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    set_seed(int(config["seed"]))
    dataset_cfg, model_cfg, training_cfg = config["dataset"], config["model"], config["training"]
    dataset_factory = load_factory(dataset_cfg["factory"])
    dataset_kwargs = dict(dataset_cfg.get("kwargs", {}))
    if "split_protocol" in dataset_cfg:
        dataset_kwargs["split_protocol"] = dataset_cfg["split_protocol"]
    train_set, validation_set = dataset_factory(**dataset_kwargs)
    loader_args = {"batch_size": int(training_cfg["batch_size"]),
                   "num_workers": int(training_cfg["num_workers"]),
                   "pin_memory": torch.cuda.is_available()}
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
    targets_path = args.targets if args.targets.is_absolute() else PROJECT_ROOT / args.targets
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    assignments = None
    if args.groups:
        groups_path = args.groups if args.groups.is_absolute() else PROJECT_ROOT / args.groups
        assignments = json.loads(groups_path.read_text(encoding="utf-8"))["assignments"]
    base_dir = PROJECT_ROOT / config["output_dir"] / f"seed_{config['seed']}" / "joint"
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
        checkpoint_path = train_joint(model, train_loader, validation_loader,
                                      training=run_training, targets=targets,
                                      group_assignments=assignments,
                                      fcd_estimator=fcd_estimator, run_dir=run_dir,
                                      seed=int(config["seed"]), config=config,
                                      reference_checkpoint=reference_checkpoint)
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
