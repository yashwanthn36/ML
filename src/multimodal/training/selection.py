"""Checkpoint/model selection helpers that restrict decisions to validation data."""
from __future__ import annotations


def select_validation_macro_f1(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("Cannot select a model from an empty candidate list")
    if any("validation_macro_f1" not in item for item in candidates):
        raise ValueError("Each candidate must include validation_macro_f1")
    return max(candidates, key=lambda item: float(item["validation_macro_f1"]))
