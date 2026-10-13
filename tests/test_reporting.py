import csv
import json

import numpy as np
import pytest
import yaml

from multimodal.training.classification_metrics import classification_metrics, identify_minority_classes
from multimodal.training.provenance import config_hash, file_sha256, ids_hash
from scripts.analysis import plot_results


def _make_synthetic_report_inputs(tmp_path, protocol="official"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    names = ["class a", "class b", "class c"]
    train_labels = [0] * 4 + [1] * 2 + [2]
    minority = identify_minority_classes(train_labels, class_names=names)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "label_id", "label", "split"])
        writer.writeheader()
        for split, labels in (("train", train_labels), ("val", [0, 1, 2]), ("test", [0, 1, 0, 2])):
            for i, class_id in enumerate(labels):
                writer.writerow({"sample_id": f"{split}_{i}", "label_id": class_id,
                                 "label": names[class_id], "split": split})
    if protocol == "source_video_disjoint":
        manifest.with_name("manifest_source_video_disjoint.csv").write_text(manifest.read_text(encoding="utf-8"),
                                                                                encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    output_name = "official" if protocol == "official" else "source_video_disjoint"
    config_path.write_text(yaml.safe_dump({"seed": 3, "output_dir": str(tmp_path / "out" / output_name),
        "dataset": {"split_protocol": protocol, "kwargs": {"manifest_path": str(manifest)}}}))
    targets_path = tmp_path / "ucd_targets.json"
    targets_path.write_text(json.dumps({"global": {"target": [.4, .6]}}))
    config = yaml.safe_load(config_path.read_text())
    reference_run = {"split_protocol": protocol, "seed": 3, "run_id": f"reference-{protocol}-3",
        "training_sample_ids_sha256": ids_hash([f"train-{i}" for i in range(7)]),
        "experiment_config_sha256": config_hash(config)}
    ref_dir = tmp_path / "reference" / "seed_3"
    ref_dir.mkdir(parents=True)
    reference_checkpoint = ref_dir / "reference_best.pt"
    reference_checkpoint.write_bytes(b"synthetic reference checkpoint")
    reference_sha = file_sha256(reference_checkpoint)
    model_paths = {}
    true = np.asarray([0, 1, 0, 2])
    base = np.asarray([[.7, .2, .1], [.2, .7, .1], [.5, .3, .2], [.1, .2, .7]])
    for model_index, model in enumerate(plot_results.MODELS):
        run_dir = tmp_path / model / "seed_3"
        run_dir.mkdir(parents=True, exist_ok=True)
        if model == "reference":
            checkpoint = reference_checkpoint
            log_name = "reference_training_log.jsonl"
            log = {"epoch": 1, "train_loss": .8, "val_loss": .7, "val_accuracy": .6,
                   "val_macro_f1": .5, "val_balanced_accuracy": .6}
        else:
            checkpoint = run_dir / "joint_best.pt"
            checkpoint.write_bytes(f"synthetic {model} checkpoint".encode())
            log_name = "training_log.jsonl"
            log = {"epoch": 1, "train_loss_ci": .8, "train_loss_mi": .2 if model_index == 2 else None,
                   "val_loss": .7, "val_accuracy": .6, "val_macro_f1": .5,
                   "val_balanced_accuracy": .6}
        (run_dir / log_name).write_text(json.dumps(log) + "\n")
        condition_metrics = {}
        prediction_data = {}
        for condition_index, condition in enumerate(plot_results.CONDITIONS):
            probabilities = base.copy()
            probabilities[:, (model_index + condition_index) % 3] += .02
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            prediction_data[condition] = probabilities
            condition_metrics[condition] = classification_metrics(
                true, probabilities, class_names=names, minority_class_ids=minority["minority_class_ids"])
        test_dir = run_dir / "test"
        test_dir.mkdir()
        metrics = {"seed": 3, "experiment_seed": 3, "split_protocol": protocol,
            "model_condition": model, "checkpoint": str(checkpoint),
            "checkpoint_sha256": reference_sha if model == "reference" else file_sha256(checkpoint),
            "run_provenance": (reference_run if model == "reference" else
                               {**reference_run, "reference_checkpoint_sha256": reference_sha}),
            "test_size": len(true), "test_sample_count": len(true), "class_names": names,
            "minority_class_definition": minority,
            "class_counts": {"train": {str(i): train_labels.count(i) for i in range(3)},
                "validation": {"0": 1, "1": 1, "2": 1}, "test": {"0": 2, "1": 1, "2": 1}},
            "fcd_test": [.5, .5], "ucd_minus_fcd_kl_test": .1, **condition_metrics}
        metric_path = test_dir / "test_metrics.json"
        metric_path.write_text(json.dumps(metrics))
        prediction_path = test_dir / "test_predictions.csv"
        fields = ["sample_id", "source_id", "label", "label_name", "true_class_id", "true_class_name"]
        fields += [f"prediction_{condition}" for condition in plot_results.CONDITIONS]
        fields += [f"prob_{condition}_class_{i}" for condition in plot_results.CONDITIONS for i in range(3)]
        with prediction_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for row_index, label in enumerate(true):
                row = {"sample_id": f"test_{row_index}", "source_id": f"video_{row_index}",
                       "label": int(label), "label_name": names[label],
                       "true_class_id": int(label), "true_class_name": names[label]}
                for condition in plot_results.CONDITIONS:
                    probs = prediction_data[condition][row_index]
                    row[f"prediction_{condition}"] = int(probs.argmax())
                    for class_id, prob in enumerate(probs):
                        row[f"prob_{condition}_class_{class_id}"] = float(prob)
                writer.writerow(row)
        model_paths[model] = metric_path
    return config_path, targets_path, model_paths


def test_plot_results_synthetic_end_to_end_writes_tables_and_seven_figures_per_protocol(tmp_path):
    report_dirs = {}
    for protocol in ("official", "source_video_disjoint"):
        config, targets, model_paths = _make_synthetic_report_inputs(tmp_path / protocol, protocol)
        args = ["--protocol", protocol, "--seed", "3", "--config", str(config), "--targets", str(targets)]
        for model, path in model_paths.items():
            args += ["--result", f"{model}={path}"]
        out = plot_results.main(args)
        report_dirs[protocol] = out
        assert (out / "summary_metrics.csv").is_file()
        assert (out / "summary_metrics.md").is_file()
        assert (out / "per_class_metrics.csv").is_file()
        summary = json.loads((out / "summary_metrics.json").read_text())
        assert summary["aggregation"] == "single_seed"
        assert summary["figure7_generated"] is True
        for figure_id in range(1, 8):
            assert len(list((out / "figures").glob(f"figure{figure_id}_*.png"))) == 1
            assert len(list((out / "figures").glob(f"figure{figure_id}_*.pdf"))) == 1
    assert report_dirs["official"] != report_dirs["source_video_disjoint"]


def test_plot_results_rejects_missing_and_mismatched_protocol_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required evaluation result is missing"):
        plot_results.parse_results(["reference=missing.json"], "official", 1)
    config, _, model_paths = _make_synthetic_report_inputs(tmp_path)
    with pytest.raises(ValueError, match="requested 'source_video_disjoint'"):
        plot_results.parse_results([f"reference={model_paths['reference']}"], "source_video_disjoint", 3)


def test_plot_results_rejects_model_initialized_from_different_reference(tmp_path):
    _, _, paths = _make_synthetic_report_inputs(tmp_path)
    entries = [f"{model}={path}" for model, path in paths.items()]
    results = plot_results.parse_results(entries, "official", 3)
    results["ci_only"][0]["metrics"]["run_provenance"]["reference_checkpoint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="not initialized from the evaluated reference checkpoint"):
        plot_results.check_comparable(results, "official")


def test_multi_seed_aggregate_uses_sample_standard_deviation_and_single_seed_is_labeled():
    assert plot_results.mean_std([.5]) == (.5, None)
    mean, std = plot_results.mean_std([.5, .7])
    assert mean == pytest.approx(.6)
    assert std == pytest.approx(np.std([.5, .7], ddof=1))


def test_result_loading_supports_matching_multiple_seeds(tmp_path):
    config_path, _, paths = _make_synthetic_report_inputs(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["seed"] = 4
    config_sha = config_hash(config)
    entries = []
    for model, source in paths.items():
        metrics = json.loads(source.read_text())
        metrics["seed"] = metrics["experiment_seed"] = 4
        provenance = dict(metrics["run_provenance"])
        provenance.update({"seed": 4, "run_id": f"{model}-seed4",
                           "experiment_config_sha256": config_sha})
        if model == "reference":
            target_dir = tmp_path / "seed4" / "reference"
            checkpoint = target_dir / "reference_best.pt"
            target_dir.mkdir(parents=True)
            checkpoint.write_bytes(b"synthetic seed 4 reference")
            metrics["checkpoint"] = str(checkpoint)
            metrics["checkpoint_sha256"] = file_sha256(checkpoint)
            metrics["run_provenance"] = provenance
            (target_dir / "reference_training_log.jsonl").write_text(json.dumps({"epoch": 1}) + "\n")
        else:
            ref_digest = json.loads(paths["reference"].read_text())["checkpoint_sha256"]
            provenance["reference_checkpoint_sha256"] = ref_digest
            target_dir = tmp_path / "seed4" / model
            checkpoint = target_dir / "joint_best.pt"
            target_dir.mkdir(parents=True)
            checkpoint.write_bytes(f"synthetic {model} seed 4".encode())
            metrics["checkpoint"] = str(checkpoint)
            metrics["checkpoint_sha256"] = file_sha256(checkpoint)
            metrics["run_provenance"] = provenance
            (target_dir / "training_log.jsonl").write_text(json.dumps({"epoch": 1}) + "\n")
        target_dir = target_dir / "test"
        target_dir.mkdir(parents=True)
        destination = target_dir / "test_metrics.json"
        destination.write_text(json.dumps(metrics))
        (target_dir / "test_predictions.csv").write_bytes(source.with_name("test_predictions.csv").read_bytes())
        entries.extend([f"{model}={source}", f"{model}={destination}"])
    results = plot_results.parse_results(entries, "official", None)
    assert {item["seed"] for item in results["reference"]} == {3, 4}
    mean, std = plot_results.aggregate_metrics(results["reference"], "audio_video")["accuracy"]
    assert std == pytest.approx(0.0)
