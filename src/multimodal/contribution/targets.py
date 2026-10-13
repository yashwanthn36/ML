"""Risk-based UCD targets with support-aware parent-scope fallback."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def risk_based_distribution(deltas: np.ndarray, *, epsilon: float = 1e-8) -> np.ndarray:
    """Normalize positive mean removal-risk increases using the PDF's Eq. (6.1-6.2)."""
    increases = np.maximum(np.asarray(deltas, dtype=np.float64).mean(axis=0), 0.0)
    weights = increases + epsilon
    return weights / weights.sum()


def build_ucd_targets(signatures_csv: str | Path, output_json: str | Path, *,
                      class_ids: list[int] | None = None, group_ids: dict[str, int] | None = None,
                      min_support: int = 20, epsilon: float = 1e-8,
                      bootstrap_replicates: int = 200, seed: int = 17,
                      max_bootstrap_std: float | None = 0.15) -> dict[str, Any]:
    """Estimate global/class/group targets; unreliable scopes inherit parent targets."""
    import json

    with Path(signatures_csv).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("No training signatures were supplied")
    labels = np.asarray([int(row["label"]) for row in rows])
    deltas = np.asarray([[float(row["delta_1"]), float(row["delta_2"])] for row in rows])
    sample_ids = [row["sample_id"] for row in rows]
    rng = np.random.default_rng(seed)

    def estimate(scope_values: np.ndarray) -> dict[str, Any]:
        target = risk_based_distribution(scope_values, epsilon=epsilon)
        if bootstrap_replicates <= 0:
            uncertainty = None
        else:
            samples = np.empty((bootstrap_replicates, 2), dtype=np.float64)
            for b in range(bootstrap_replicates):
                sample_indices = rng.integers(0, len(scope_values), size=len(scope_values))
                samples[b] = risk_based_distribution(scope_values[sample_indices], epsilon=epsilon)
            uncertainty = {"bootstrap_std": samples.std(axis=0).tolist(),
                           "ci_95_low": np.quantile(samples, 0.025, axis=0).tolist(),
                           "ci_95_high": np.quantile(samples, 0.975, axis=0).tolist(),
                           "replicates": bootstrap_replicates}
        stable = (uncertainty is None or max(uncertainty["bootstrap_std"]) <= max_bootstrap_std
                  if max_bootstrap_std is not None else True)
        return {"target": target.tolist(), "support": int(len(scope_values)),
                "uncertainty": uncertainty, "stable": bool(stable)}

    targets: dict[str, Any] = {"global": estimate(deltas),
       "classes": {}, "groups": {}, "sample_group": group_ids or {},
       "epsilon": epsilon, "min_support": min_support,
       "max_bootstrap_std": max_bootstrap_std, "fallbacks": []}
    ids = sorted(set(class_ids if class_ids is not None else labels.tolist()))
    for class_id in ids:
        mask = labels == class_id
        key = str(class_id)
        class_estimate = estimate(deltas[mask]) if int(mask.sum()) else None
        if class_estimate and int(mask.sum()) >= min_support and class_estimate["stable"]:
            targets["classes"][key] = {**class_estimate, "fallback": None}
        else:
            reason = "insufficient_support" if int(mask.sum()) < min_support else "unstable_bootstrap"
            targets["classes"][key] = {"target": targets["global"]["target"],
                                        "support": int(mask.sum()), "fallback": "global",
                                        "uncertainty": class_estimate["uncertainty"] if class_estimate else None,
                                        "stable": False}
            targets["fallbacks"].append({"scope": f"class:{key}", "to": "global", "reason": reason})
        if not group_ids:
            continue
        memberships = np.asarray([group_ids.get(sample_id, -1) for sample_id in sample_ids])
        for group_id in sorted(set(memberships[mask].tolist()) - {-1}):
            group_mask = mask & (memberships == group_id)
            group_key = f"{class_id}:{group_id}"
            group_estimate = estimate(deltas[group_mask]) if int(group_mask.sum()) else None
            if group_estimate and int(group_mask.sum()) >= min_support and group_estimate["stable"]:
                targets["groups"][group_key] = {**group_estimate, "fallback": None}
            else:
                reason = "insufficient_support" if int(group_mask.sum()) < min_support else "unstable_bootstrap"
                targets["groups"][group_key] = {
                    "target": targets["classes"][key]["target"],
                    "support": int(group_mask.sum()), "fallback": f"class:{key}",
                    "uncertainty": group_estimate["uncertainty"] if group_estimate else None,
                    "stable": False,
                }
                targets["fallbacks"].append({"scope": f"group:{group_key}", "to": f"class:{key}",
                                              "reason": reason})
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(targets, indent=2), encoding="utf-8")
    return targets
