"""Second-stage CI/MI optimization with fixed-weight or minimum-norm MOO."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from multimodal.losses.alignment import target_to_current_kl
from multimodal.losses.balanced import class_balanced_ce, group_balanced_ce
from multimodal.optim import min_norm_combine, gradient_diagnostics


def _scope_alignment(current: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor | None,
                     targets: dict[str, Any], granularity: str, epsilon: float) -> torch.Tensor:
    if granularity == "global":
        scopes = [(torch.ones_like(labels, dtype=torch.bool), targets["global"]["target"])]
    elif granularity == "class":
        scopes = [(labels == label, targets["classes"][str(int(label))]["target"])
                  for label in torch.unique(labels)]
    elif granularity == "group":
        if group_ids is None:
            raise ValueError("group granularity requires group IDs for every batch item")
        scopes = []
        for label in torch.unique(labels):
            class_mask = labels == label
            for group_id in torch.unique(group_ids[class_mask]):
                mask = class_mask & (group_ids == group_id)
                key = f"{int(label)}:{int(group_id)}"
                scoped = targets["groups"].get(key, targets["classes"][str(int(label))])
                scopes.append((mask, scoped["target"]))
    else:
        raise ValueError(f"Unsupported target granularity: {granularity}")
    losses = []
    for mask, target in scopes:
        if mask.any():
            current_mean = current[mask].mean(dim=0)
            target_tensor = torch.as_tensor(target, dtype=current.dtype, device=current.device)
            losses.append(target_to_current_kl(target_tensor, current_mean, epsilon))
    return torch.stack(losses).mean()


def train_joint(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader,
                *, training: dict[str, Any], targets: dict[str, Any],
                group_assignments: dict[str, int] | None,
                fcd_estimator: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
                run_dir: Path, seed: int, config: dict[str, Any],
                reference_checkpoint: dict[str, Any] | None = None) -> Path:
    """Optimize selected CI/MI objectives; the FCD implementation is injected explicitly."""
    from multimodal.training.reference import evaluate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if reference_checkpoint is None:
        raise ValueError("Second-stage optimization must initialize from the reference checkpoint")
    model.load_state_dict(reference_checkpoint["model_state_dict"])
    # The architecture's second stage optimizes projections, attention, fusion,
    # and the classifier while keeping both reference encoders fixed.
    if not hasattr(model, "encoder_1") or not hasattr(model, "encoder_2"):
        raise ValueError("The paired-modality model must expose encoder_1 and encoder_2 for stage-two freezing")
    model.encoder_1.requires_grad_(False)
    model.encoder_2.requires_grad_(False)
    model.freeze_encoders = True
    model.encoder_1.eval()
    model.encoder_2.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable fusion/projection parameters remain for stage-two optimization")
    optimizer = torch.optim.AdamW(parameters, lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    mode = str(training.get("optimization_mode", "fixed_weight"))
    ci_mode = str(training.get("ci_objective", "class_balanced"))
    granularity = str(config.get("contribution", {}).get("target_granularity", "global"))
    epsilon = float(config.get("contribution", {}).get("epsilon", 1e-8))
    lambda_mi = float(training.get("lambda_mi", 0.1))
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "joint_best.pt"
    best_f1, stale = -1.0, 0
    group_assignments = group_assignments or {}
    dataset = train_loader.dataset
    class_counts: dict[int, int] = {}
    group_counts: dict[tuple[int, int], int] = {}
    groups_per_class: dict[int, int] = {}
    if hasattr(dataset, "labels"):
        for label in dataset.labels:
            class_counts[int(label)] = class_counts.get(int(label), 0) + 1
        if group_assignments and hasattr(dataset, "sample_ids"):
            for label, sample_id in zip(dataset.labels, dataset.sample_ids):
                group_id = group_assignments.get(str(sample_id))
                if group_id is not None:
                    key = (int(label), int(group_id))
                    group_counts[key] = group_counts.get(key, 0) + 1
            for label, _ in group_counts:
                groups_per_class[label] = groups_per_class.get(label, 0) + 1
    history_path = run_dir / "training_log.jsonl"
    with history_path.open("w", encoding="utf-8") as history:
        for epoch in range(1, int(training["epochs"]) + 1):
            model.train()
            sums = {"loss_ci": 0.0, "loss_mi": 0.0, "steps": 0, "negative_cosine": 0,
                    "gradient_cosine": 0.0, "gradient_norm_ci": 0.0, "gradient_norm_mi": 0.0}
            for batch in train_loader:
                x1 = batch["modality_1"].to(device, non_blocking=True)
                x2 = batch["modality_2"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                fused, modality_1, modality_2 = model.fused_features(x1, x2)
                logits = model.classifier(fused)
                ids = batch.get("sample_id", [])
                group_ids = None
                if ci_mode == "group_balanced" or granularity == "group":
                    if not ids or not group_assignments:
                        raise ValueError("Group objectives require saved training group assignments")
                    group_ids = torch.tensor([group_assignments[str(sample_id)] for sample_id in ids],
                                             device=device, dtype=torch.long)
                if ci_mode == "group_balanced":
                    loss_ci = group_balanced_ce(logits, labels, group_ids,
                                                group_counts or None, groups_per_class or None)
                elif ci_mode == "class_balanced":
                    loss_ci = class_balanced_ce(logits, labels, class_counts or None)
                else:
                    loss_ci = nn.functional.cross_entropy(logits, labels)
                factual = fcd_estimator(fused, modality_1, modality_2)
                if factual.shape != (labels.shape[0], 2):
                    raise ValueError("FCD estimator must return nonnegative [batch, 2] contributions")
                loss_mi = _scope_alignment(factual, labels, group_ids, targets, granularity, epsilon)
                optimizer.zero_grad(set_to_none=True)
                diagnostics = {}
                if mode == "fixed_weight":
                    (loss_ci + lambda_mi * loss_mi).backward()
                elif mode == "moo":
                    gradients_ci = torch.autograd.grad(loss_ci, parameters, retain_graph=True, allow_unused=True)
                    gradients_mi = torch.autograd.grad(loss_mi, parameters, retain_graph=False, allow_unused=True)
                    diagnostics = gradient_diagnostics(gradients_ci, gradients_mi)
                    sums["gradient_cosine"] += diagnostics["cosine"]
                    sums["gradient_norm_ci"] += diagnostics["norm_ci"]
                    sums["gradient_norm_mi"] += diagnostics["norm_mi"]
                    _, combined = min_norm_combine(gradients_ci, gradients_mi)
                    for parameter, gradient in zip(parameters, combined):
                        parameter.grad = gradient
                    sums["negative_cosine"] += int(diagnostics["cosine"] < 0.0)
                else:
                    raise ValueError("optimization_mode must be 'fixed_weight' or 'moo'")
                torch.nn.utils.clip_grad_norm_(parameters, float(training["grad_clip"]))
                optimizer.step()
                sums["loss_ci"] += float(loss_ci.detach())
                sums["loss_mi"] += float(loss_mi.detach())
                sums["steps"] += 1
            metrics = evaluate(model, validation_loader, device)
            record = {"epoch": epoch, "mode": mode, "ci_objective": ci_mode,
                      "granularity": granularity, "lambda_mi": lambda_mi,
                      "train_loss_ci": sums["loss_ci"] / max(sums["steps"], 1),
                      "train_loss_mi": sums["loss_mi"] / max(sums["steps"], 1),
                      "moo_gradient_cosine_mean": sums["gradient_cosine"] / max(sums["steps"], 1),
                      "moo_gradient_norm_ci_mean": sums["gradient_norm_ci"] / max(sums["steps"], 1),
                      "moo_gradient_norm_mi_mean": sums["gradient_norm_mi"] / max(sums["steps"], 1),
                      "moo_negative_cosine_fraction": sums["negative_cosine"] / max(sums["steps"], 1),
                      **{f"val_{key}": value for key, value in metrics.items()}}
            history.write(json.dumps(record) + "\n")
            history.flush()
            print(json.dumps(record))
            if metrics["macro_f1"] > best_f1:
                best_f1, stale = metrics["macro_f1"], 0
                torch.save({"model_state_dict": model.state_dict(), "config": config,
                            "seed": seed, "epoch": epoch, "validation_metrics": metrics,
                            "optimization_mode": mode}, checkpoint_path)
            else:
                stale += 1
                if stale >= int(training["early_stopping_patience"]):
                    break
    return checkpoint_path
