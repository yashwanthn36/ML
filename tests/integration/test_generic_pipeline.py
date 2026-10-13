"""Offline end-to-end run of the generic adapter and all three conditions."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _make_media(root: Path, index: int) -> tuple[str, str]:
    audio_dir, video_dir = root / "audio", root / "video"
    audio_dir.mkdir(parents=True, exist_ok=True); video_dir.mkdir(parents=True, exist_ok=True)
    audio_name, video_name = f"waveform_{index:02}.wav", f"clip_{index:02}.avi"
    with wave.open(str(audio_dir / audio_name), "wb") as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(16_000)
        signal = (np.sin(np.arange(2000) * (.03 + .003 * (index % 5))) * 10000).astype("<i2")
        output.writeframes(signal.tobytes())
    writer = cv2.VideoWriter(str(video_dir / video_name), cv2.VideoWriter_fourcc(*"MJPG"), 5, (16, 16))
    if not writer.isOpened():
        pytest.skip("OpenCV build has no local MJPG encoder for synthetic integration fixtures")
    for frame in range(3):
        image = np.full((16, 16, 3), (index * 11 + frame * 7) % 255, dtype=np.uint8)
        writer.write(image)
    writer.release()
    return f"audio/{audio_name}", f"video/{video_name}"


def _make_fixture(tmp_path: Path) -> Path:
    media_root = tmp_path / "media"
    manifest = tmp_path / "manifest.csv"
    mapping = {"maple": 0, "oak": 1, "pine": 2}
    split_labels = {"train": ["maple"] * 4 + ["oak"] * 4 + ["pine"] * 2,
                    "val": list(mapping), "test": list(mapping)}
    rows = []
    index = 0
    for split, labels in split_labels.items():
        for local, label in enumerate(labels):
            audio_path, video_path = _make_media(media_root, index)
            source = "shared_validation_source" if split == "val" and local < 2 else f"source_{split}_{local}"
            rows.append({"sample_id": f"sample_{split}_{local:02}", "audio_path": audio_path,
                         "video_path": video_path, "label": label, "source_id": source, "split": split})
            index += 1
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return manifest


def _run(script: str, *args: str, expected_success: bool = True) -> subprocess.CompletedProcess:
    command = [sys.executable, str(ROOT / script), *map(str, args)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
    if expected_success and result.returncode != 0:
        pytest.fail(f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}\n{result.stderr}")
    return result


def test_generic_manifest_short_full_pipeline(tmp_path):
    manifest = _make_fixture(tmp_path)
    config_path = tmp_path / "toy.yaml"
    output_root = tmp_path / "outputs" / "toy" / "explicit"
    dataset_common = {"manifest_path": str(manifest), "media_root": str(tmp_path / "media"),
        "media_paths_relative_to": "media_root", "dataset_name": "toy_av",
        "class_mapping": {"maple": 0, "oak": 1, "pine": 2},
        "columns": {"sample_id": "sample_id", "audio_path": "audio_path", "video_path": "video_path",
                    "label": "label", "source_id": "source_id", "split": "split"},
        "num_frames": 1, "image_size": 16, "sample_rate": 16_000,
        "audio_samples": 1600, "min_audio_coverage": .1}
    config = {"seed": 41, "device": "cpu", "output_dir": str(output_root),
        "dataset": {"name": "toy_av", "adapter": "generic_audio_video",
            "factory": "multimodal.data_adapters.generic_audio_video:build_datasets",
            "split_protocol": "toy_explicit", "kwargs": dataset_common},
        "test_dataset": {"name": "toy_av", "adapter": "generic_audio_video",
            "factory": "multimodal.data_adapters.generic_audio_video:build_test_dataset",
            "split_protocol": "toy_explicit", "kwargs": dataset_common},
        "model": {"factory": "multimodal.models:build_model", "kwargs": {
            "dim": 8, "heads": 2, "encoder_1": {"name": "audio_cnn", "kwargs": {"tokens": 2}},
            "encoder_2": {"name": "frame_cnn"}, "freeze_encoders": False}},
        "contribution": {"fcd_factory": "multimodal.contribution.fcd:InfoNCEFCD",
            "target_granularity": "global", "epsilon": 1e-8,
            "fcd_kwargs": {"queue_size": 4, "exclude_same_source_negatives": False},
            "bootstrap_replicates": 0, "target_min_support": 1,
            "min_group_support_floor": 1, "max_groups_per_class": 2},
        "evaluation": {"minority_bottom_quantile": .25},
        "training": {"batch_size": 2, "num_workers": 0, "epochs": 1,
            "early_stopping_patience": 1, "learning_rate": .001, "weight_decay": 0.,
            "modality_dropout": .1, "contribution_estimation_fraction": .5,
            "optimization_mode": "fixed_weight", "ci_objective": "class_balanced",
            "lambda_mi_grid": [.1], "grad_clip": 1.}}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    from multimodal.data_adapters.generic_audio_video import build_datasets, build_test_dataset
    from multimodal.data_adapters.contract import validate_split_datasets
    train, val = build_datasets(**dataset_common, split_protocol="toy_explicit")
    test = build_test_dataset(**dataset_common, split_protocol="toy_explicit")
    contract = validate_split_datasets(train, val, test)
    assert train.class_names == ["maple", "oak", "pine"]
    assert train.num_classes == 3 and len(train) == 10
    assert contract["partitions"]["test"]["num_samples"] == 3
    assert "modality_1" in train[0] and "modality_2" in train[0]
    assert len(set(train.sample_ids + val.sample_ids + test.sample_ids)) == 16
    assert val.source_ids[0] == val.source_ids[1]  # repeated segments of one source are represented.

    _run("scripts/training/train_reference.py", "--config", str(config_path))
    seed_root = output_root / "seed_41"
    reference = seed_root / "reference_best.pt"
    _run("scripts/training/estimate_contributions.py", "--config", str(config_path), "--checkpoint", str(reference))
    _run("scripts/training/train_joint.py", "--config", str(config_path), "--condition", "ci_only",
         "--reference-checkpoint", str(reference))
    _run("scripts/training/train_joint.py", "--config", str(config_path), "--condition", "proposed_ci_mi",
         "--reference-checkpoint", str(reference), "--targets", str(seed_root / "contributions" / "ucd_targets.json"),
         "--groups", str(seed_root / "contributions" / "profile_groups.json"))
    selection = json.loads((seed_root / "proposed_ci_mi" / "lambda_grid_selection.json").read_text())
    proposed_checkpoint = Path(selection["selected"]["checkpoint"])
    ci_checkpoint = seed_root / "ci_only" / "joint_best.pt"
    for condition, checkpoint, targets in (
        ("reference", reference, None), ("ci_only", ci_checkpoint, None),
        ("proposed_ci_mi", proposed_checkpoint, seed_root / "contributions" / "ucd_targets.json")):
        args = ["--config", str(config_path), "--checkpoint", str(checkpoint), "--output-dir",
                str(seed_root / "evaluations" / condition)]
        if targets is not None:
            args += ["--targets", str(targets)]
        _run("scripts/training/evaluate_test.py", *args)
    report_args = ["--protocol", "toy_explicit", "--config", str(config_path), "--seed", "41",
        "--targets", str(seed_root / "contributions" / "ucd_targets.json")]
    for model in ("reference", "ci_only", "proposed_ci_mi"):
        report_args += ["--result", f"{model}={seed_root / 'evaluations' / model / 'test_metrics.json'}"]
    _run("scripts/analysis/plot_results.py", *report_args)
    report = seed_root / "reports"
    assert (report / "summary_metrics.csv").is_file()
    assert (report / "summary_metrics.md").is_file()
    assert (report / "per_class_metrics.csv").is_file()
    assert (report / "summary_metrics.json").is_file()
    assert len(list((report / "figures").glob("*.png"))) == 7
    assert len(list((report / "figures").glob("*.pdf"))) == 7
    result = _run("scripts/analysis/plot_results.py", "--protocol", "wrong_protocol", "--config", str(config_path),
        "--seed", "41", *sum((["--result", f"{model}={seed_root / 'evaluations' / model / 'test_metrics.json'}"]
                               for model in ("reference", "ci_only", "proposed_ci_mi")), []),
        expected_success=False)
    assert result.returncode != 0 and "protocols" in result.stderr.lower()


def test_deterministic_split_generation_preserves_source_groups_and_varies_by_seed(tmp_path):
    source = tmp_path / "unsplit.csv"
    rows = [{"sample_id": f"s{label}_{i}", "label": f"class{label}", "source_id": f"v{label}_{i}"}
            for label in range(3) for i in range(10)]
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    from multimodal.data_adapters.splits import generate_manifest_splits
    one, two, different = (tmp_path / "split1.csv", tmp_path / "split2.csv", tmp_path / "split3.csv")
    summary1 = generate_manifest_splits(source, one, seed=8)
    summary2 = generate_manifest_splits(source, two, seed=8)
    summary3 = generate_manifest_splits(source, different, seed=9)
    assert one.read_bytes() == two.read_bytes()
    assert summary1["source_overlap"] == {"train_val": [], "train_test": [], "val_test": []}
    assignments1 = {row["sample_id"]: row["split"] for row in csv.DictReader(one.open(encoding="utf-8"))}
    assignments3 = {row["sample_id"]: row["split"] for row in csv.DictReader(different.open(encoding="utf-8"))}
    assert assignments1 != assignments3
    assert summary2["split_counts"] == summary1["split_counts"]
    assert summary3["seed"] == 9
