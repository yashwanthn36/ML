"""Dataset-independent reference-stage trainer for paired modalities."""
from __future__ import annotations
import random
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch import nn
from torch.utils.data import DataLoader

from multimodal.training.metadata import experiment_metadata


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_modality_mask(batch_size: int, probability: float, device: torch.device) -> torch.Tensor:
    """Randomly remove at most one modality per example for missing-path training."""
    mask = torch.ones(batch_size, 2, device=device)
    dropped = torch.rand(batch_size, device=device) < probability
    which = torch.randint(0, 2, (batch_size,), device=device)
    mask[dropped, which[dropped]] = 0.0
    return mask


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    num_classes = 0
    loss_sum, count = 0.0, 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for batch in loader:
            x1 = batch["modality_1"].to(device, non_blocking=True)
            x2 = batch["modality_2"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(x1, x2)
            num_classes = logits.shape[1]
            loss_sum += float(loss_fn(logits, labels))
            count += labels.numel()
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    recalls = recall_score(targets, predictions, average=None, labels=list(range(num_classes)),
                           zero_division=0)
    return {"loss": loss_sum / max(count, 1),
            "macro_f1": float(f1_score(targets, predictions, average="macro", labels=list(range(num_classes)),
                                       zero_division=0)),
            "balanced_accuracy": float(recalls.mean()),
            "accuracy": float(accuracy_score(targets, predictions)),
            "per_class_recall": recalls.tolist()}


def train_reference(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader,
                    *, training: dict[str, Any], run_dir: Path, seed: int,
                    config: dict[str, Any],
                    data_metadata: dict[str, Any] | None = None) -> Path:
    """Natural-sampling CE reference training with validation checkpointing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss()
    run_dir.mkdir(parents=True, exist_ok=True)
    best_f1, stale = -1.0, 0
    checkpoint_path = run_dir / "reference_best.pt"
    protocol = ((data_metadata or {}).get("split_protocol")
                or config.get("dataset", {}).get("split_protocol", "unspecified"))
    history_path = run_dir / "reference_training_log.jsonl"
    history = history_path.open("w", encoding="utf-8")
    (run_dir / "reference_data_protocol.json").write_text(
        json.dumps(data_metadata or {}, indent=2), encoding="utf-8"
    )
    run_metadata = experiment_metadata(config)
    (run_dir / "experiment_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    try:
        for epoch in range(1, int(training["epochs"]) + 1):
            model.train()
            loss_sum, count = 0.0, 0
            for batch in train_loader:
                x1 = batch["modality_1"].to(device, non_blocking=True)
                x2 = batch["modality_2"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                mask = sample_modality_mask(labels.shape[0], float(training["modality_dropout"]), device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x1, x2, modality_mask=mask), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip"]))
                optimizer.step()
                loss_sum += float(loss.detach()) * labels.numel()
                count += labels.numel()
            metrics = evaluate(model, validation_loader, device)
            record = {"epoch": epoch, "split_protocol": protocol,
                      "train_loss": loss_sum / max(count, 1),
                      **{f"val_{key}": value for key, value in metrics.items()}}
            history.write(json.dumps(record) + "\n")
            history.flush()
            print(f"epoch={epoch:03d} train_loss={loss_sum / max(count, 1):.4f} "
                  f"val_loss={metrics['loss']:.4f} val_macro_f1={metrics['macro_f1']:.4f} "
                  f"val_balanced_acc={metrics['balanced_accuracy']:.4f}")
            if metrics["macro_f1"] > best_f1:
                best_f1, stale = metrics["macro_f1"], 0
                torch.save({"model_state_dict": model.state_dict(), "config": config,
                            "seed": seed, "epoch": epoch, "validation_metrics": metrics,
                            "split_protocol": protocol, "data_metadata": data_metadata or {},
                            "experiment_metadata": run_metadata},
                           checkpoint_path)
            else:
                stale += 1
                if stale >= int(training["early_stopping_patience"]):
                    print(f"early stopping after {epoch} epochs")
                    break
    finally:
        history.close()
    return checkpoint_path
