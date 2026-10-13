"""Training-only signed modality-removal loss signatures."""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from multimodal.training.device import assert_module_device, on_device, select_device


@torch.inference_mode()
def estimate_signatures(model: nn.Module, loader: DataLoader, output_csv: str | Path,
                        *, device: torch.device | None = None) -> Path:
    """Save per-example (loss_without_1 - full_loss, loss_without_2 - full_loss)."""
    device = device or select_device("auto")
    model = model.to(device)
    assert_module_device(model, device)
    was_training = model.training
    model.eval()
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["sample_id", "source_id", "label", "label_name", "split",
               "full_loss", "loss_without_1", "loss_without_2", "delta_1", "delta_2"]
    try:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for batch in loader:
                x1, x2 = batch["modality_1"].to(device), batch["modality_2"].to(device)
                labels = batch["label"].to(device)
                if not all(on_device(tensor, device) for tensor in (x1, x2, labels)):
                    raise RuntimeError(f"Contribution batch is not on selected device {device}")
                full = nn.functional.cross_entropy(model(x1, x2), labels, reduction="none")
                only_2 = nn.functional.cross_entropy(
                    model(x1, x2, modality_mask=torch.tensor([0., 1.], device=device).expand(labels.size(0), -1)),
                    labels, reduction="none")
                only_1 = nn.functional.cross_entropy(
                    model(x1, x2, modality_mask=torch.tensor([1., 0.], device=device).expand(labels.size(0), -1)),
                    labels, reduction="none")
                for i in range(labels.size(0)):
                    writer.writerow({
                        "sample_id": batch.get("sample_id", [""] * labels.size(0))[i],
                        "source_id": batch.get("source_id", [""] * labels.size(0))[i],
                        "label": int(labels[i]),
                        "label_name": batch.get("label_name", [""] * labels.size(0))[i],
                        "split": batch.get("split", [""] * labels.size(0))[i],
                        "full_loss": float(full[i]),
                        # Mask [0, 1] keeps only modality 2, hence modality 1 is removed.
                        "loss_without_1": float(only_2[i]),
                        # Mask [1, 0] keeps only modality 1, hence modality 2 is removed.
                        "loss_without_2": float(only_1[i]),
                        "delta_1": float(only_2[i] - full[i]),
                        "delta_2": float(only_1[i] - full[i]),
                    })
    finally:
        model.train(was_training)
    return output


@torch.inference_mode()
def estimate_reference_fcd(model: nn.Module, loader: DataLoader, estimator: nn.Module,
                           *, device: torch.device | None = None) -> dict[str, object]:
    """Estimate reference FCD on the training-only contribution subset.

    Features are collected first so the InfoNCE negatives come from the complete
    estimation set rather than from arbitrary loader batches.
    """
    device = device or select_device("auto")
    model = model.to(device).eval()
    assert_module_device(model, device)
    fused_rows, modality_1_rows, modality_2_rows = [], [], []
    sample_ids: list[str] = []
    source_ids: list[str] = []
    for batch in loader:
        batch_x1, batch_x2 = batch["modality_1"].to(device), batch["modality_2"].to(device)
        if not on_device(batch_x1, device) or not on_device(batch_x2, device):
            raise RuntimeError(f"Reference FCD batch is not on selected device {device}")
        fused, modality_1, modality_2 = model.fused_features(
            batch_x1, batch_x2
        )
        fused_rows.append(fused)
        modality_1_rows.append(modality_1)
        modality_2_rows.append(modality_2)
        sample_ids.extend(map(str, batch.get("sample_id", [])))
        source_ids.extend(map(str, batch.get("source_id", [])))
    if not fused_rows:
        raise ValueError("Cannot estimate reference FCD from an empty loader")
    if sum(row.shape[0] for row in fused_rows) < 2:
        raise ValueError("Reference FCD needs at least two paired examples")
    information = estimator.per_sample_information(
        torch.cat(fused_rows), torch.cat(modality_1_rows), torch.cat(modality_2_rows),
        sample_ids=sample_ids if len(sample_ids) == sum(row.shape[0] for row in fused_rows) else None,
        source_ids=source_ids if len(source_ids) == sum(row.shape[0] for row in fused_rows) else None,
    )
    distribution = estimator.distribution(information)
    return {"distribution": distribution.cpu().tolist(),
            "mean_information_lower_bound": information.mean(dim=0).cpu().tolist(),
            "sample_count": int(information.shape[0]), "negative_pool": "entire_estimation_subset",
            "estimator": type(estimator).__name__}
