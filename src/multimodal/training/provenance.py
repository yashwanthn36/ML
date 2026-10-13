"""Stable hashes and explicit artifact compatibility checks."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ids_hash(sample_ids: list[str]) -> str:
    return _digest(sorted(map(str, sample_ids)))


def config_hash(config: dict[str, Any]) -> str:
    # Output paths can legitimately differ between repeated copies of the same run.
    normalized = json.loads(json.dumps(config, sort_keys=True, default=str))
    normalized.pop("output_dir", None)
    normalized.pop("device", None)
    return _digest(normalized)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_provenance(expected: dict[str, Any], actual: dict[str, Any] | None, *,
                        artifact: str, allow_legacy: bool = False,
                        required_keys: tuple[str, ...] | None = None) -> None:
    required = required_keys or tuple(expected)
    if not actual or any(key not in actual for key in required):
        missing = [key for key in required if not actual or key not in actual]
        message = f"{artifact} is missing required provenance fields: {missing}"
        if allow_legacy:
            warnings.warn(message + "; explicit legacy compatibility enabled", RuntimeWarning, stacklevel=2)
            print(f"WARNING: {message}; accepting because --allow-legacy-artifacts was supplied")
            return
        raise ValueError(message + ". Recreate the artifact or pass --allow-legacy-artifacts explicitly.")
    for key in required:
        if expected.get(key) != actual.get(key):
            raise ValueError(f"Incompatible {artifact} provenance for {key}: expected "
                             f"{expected.get(key)!r}, got {actual.get(key)!r}")


def checkpoint_provenance(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    value = checkpoint.get("run_provenance")
    return value if isinstance(value, dict) else None


def make_run_provenance(*, split_protocol: str, seed: int, run_id: str,
                        training_sample_ids: list[str], config: dict[str, Any],
                        estimation_sample_ids: list[str] | None = None,
                        reference_fit_sample_ids: list[str] | None = None,
                        dataset_metadata: dict[str, Any] | None = None,
                        class_to_id: dict[str, int] | None = None) -> dict[str, Any]:
    dataset_metadata = dataset_metadata or {}
    return {
        "split_protocol": str(split_protocol), "seed": int(seed), "run_id": str(run_id),
        "training_sample_ids_sha256": ids_hash(training_sample_ids),
        "estimation_sample_ids_sha256": ids_hash(estimation_sample_ids or []),
        "reference_fit_sample_ids_sha256": ids_hash(reference_fit_sample_ids or []),
        "experiment_config_sha256": config_hash(config),
        "dataset_name": dataset_metadata.get("dataset_name", config.get("dataset", {}).get("name")),
        "dataset_adapter": dataset_metadata.get("adapter", config.get("dataset", {}).get("adapter")),
        "dataset_manifest_sha256": dataset_metadata.get("manifest_sha256"),
        "split_assignments_sha256": dataset_metadata.get("manifest_sha256"),
        "class_to_id": ({str(key): int(value) for key, value in (class_to_id or dataset_metadata.get("class_to_id", {})).items()}
                        if class_to_id or dataset_metadata.get("class_to_id") else None),
    }


def validate_checkpoint(checkpoint: dict[str, Any], *, split_protocol: str, seed: int,
                        config: dict[str, Any], training_sample_ids: list[str] | None = None,
                        dataset_metadata: dict[str, Any] | None = None,
                        class_to_id: dict[str, int] | None = None,
                        allow_legacy: bool = False, artifact: str = "reference checkpoint") -> dict[str, Any] | None:
    provenance = checkpoint_provenance(checkpoint)
    expected = {"split_protocol": split_protocol, "seed": int(seed),
                "experiment_config_sha256": config_hash(config)}
    if training_sample_ids is not None:
        expected["training_sample_ids_sha256"] = ids_hash(training_sample_ids)
    if dataset_metadata is not None:
        expected.update({"dataset_manifest_sha256": dataset_metadata.get("manifest_sha256"),
                         "split_assignments_sha256": dataset_metadata.get("manifest_sha256"),
                         "dataset_name": dataset_metadata.get("dataset_name"),
                         "dataset_adapter": dataset_metadata.get("adapter")})
    if class_to_id is not None:
        expected["class_to_id"] = class_to_id
    validate_provenance(expected, provenance, artifact=artifact, allow_legacy=allow_legacy,
                        required_keys=tuple(expected))
    return provenance


def validate_downstream_artifact(provenance: dict[str, Any] | None, *,
                                 reference_provenance: dict[str, Any] | None,
                                 reference_sha256: str, split_protocol: str, seed: int, artifact: str,
                                 allow_legacy: bool = False) -> None:
    if not reference_provenance:
        expected = {"split_protocol": split_protocol, "seed": int(seed),
                    "reference_checkpoint_sha256": reference_sha256}
        validate_provenance(expected, provenance, artifact=artifact, allow_legacy=allow_legacy,
                            required_keys=("split_protocol", "seed", "reference_checkpoint_sha256"))
        return
    expected = {key: reference_provenance.get(key) for key in
                ("split_protocol", "seed", "run_id", "training_sample_ids_sha256",
                 "estimation_sample_ids_sha256", "experiment_config_sha256",
                 "dataset_name", "dataset_adapter", "dataset_manifest_sha256",
                 "split_assignments_sha256", "class_to_id")}
    expected["reference_checkpoint_sha256"] = reference_sha256
    validate_provenance(expected, provenance, artifact=artifact, allow_legacy=allow_legacy,
                        required_keys=tuple(expected))


def effective_split_protocol(dataset: Any, configured: str | None) -> str:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    actual = getattr(dataset, "split_protocol", None)
    expected = configured or actual
    if not expected:
        raise ValueError("No split protocol is configured or exposed by the dataset adapter")
    if actual is not None and str(actual) != str(expected):
        raise ValueError(f"Dataset split protocol mismatch: expected {expected!r}, got {actual!r}")
    return str(expected)


def configured_split_protocol(config: dict[str, Any]) -> str:
    dataset_protocol = config.get("dataset", {}).get("split_protocol")
    test_protocol = config.get("test_dataset", {}).get("split_protocol", dataset_protocol)
    if not dataset_protocol:
        raise ValueError("Config must declare dataset.split_protocol")
    if test_protocol != dataset_protocol:
        raise ValueError(f"Config split-protocol mismatch: dataset.split_protocol expects "
                         f"{dataset_protocol!r}, test_dataset.split_protocol is {test_protocol!r}")
    return str(dataset_protocol)


def requires_source_disjoint(config: dict[str, Any], protocol: str | None = None) -> bool:
    """Resolve source independence from explicit config, with AVE legacy fallback."""
    dataset = config.get("dataset", {})
    if "require_source_disjoint" in dataset:
        return bool(dataset["require_source_disjoint"])
    actual = protocol or dataset.get("split_protocol")
    return actual == "source_video_disjoint"


def ensure_protocol_output(directory: str | Path, *, split_protocol: str, seed: int) -> Path:
    """Prevent reuse of a populated output directory by another protocol."""
    directory = Path(directory)
    marker = directory / "protocol_output.json"
    if marker.exists():
        actual = json.loads(marker.read_text(encoding="utf-8"))
        for key, expected in (("split_protocol", split_protocol), ("seed", int(seed))):
            if actual.get(key) != expected:
                raise ValueError(f"Output directory {directory} belongs to {key}={actual.get(key)!r}; "
                                 f"current run expects {expected!r}. Choose the matching protocol output root.")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Output directory {directory} already contains files but has no protocol marker; "
                         "use a new protocol-specific output root or archive the legacy outputs first")
    directory.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        marker.write_text(json.dumps({"split_protocol": split_protocol, "seed": int(seed)}, indent=2),
                          encoding="utf-8")
    return directory
