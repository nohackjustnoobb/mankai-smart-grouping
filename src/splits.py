"""Deterministic source-level split assignment shared by generation and training."""

from __future__ import annotations

import random
from collections.abc import Sequence

SPLIT_NAMES = ("training", "validation", "calibration", "test")


def validate_split_fractions(
    validation_fraction: float,
    calibration_fraction: float,
    test_fraction: float,
) -> None:
    fractions = {
        "validation": validation_fraction,
        "calibration": calibration_fraction,
        "test": test_fraction,
    }
    for name, fraction in fractions.items():
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"{name}_fraction must be strictly between 0 and 1")
    if sum(fractions.values()) >= 1.0:
        raise ValueError(
            "validation, calibration, and test fractions must sum to less than 1"
        )


def assign_source_splits(
    source_ids: Sequence[str],
    *,
    validation_fraction: float,
    calibration_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    """Assign every unique source to one reproducible experiment partition."""

    validate_split_fractions(validation_fraction, calibration_fraction, test_fraction)
    fractions = {
        "validation": validation_fraction,
        "calibration": calibration_fraction,
        "test": test_fraction,
    }

    unique_sources = sorted(set(source_ids))
    if len(unique_sources) != len(source_ids):
        raise ValueError("source IDs must be unique")
    if len(unique_sources) < 8:
        raise ValueError("at least eight sources are required for four binary splits")

    shuffled = list(unique_sources)
    random.Random(seed).shuffle(shuffled)
    counts = {
        name: max(2, round(len(shuffled) * fraction))
        for name, fraction in fractions.items()
    }
    training_count = len(shuffled) - sum(counts.values())
    if training_count < 2:
        raise ValueError(
            "split fractions leave fewer than two training sources. Use more sources "
            "or smaller holdout fractions"
        )

    assignments: dict[str, str] = {}
    cursor = 0
    for name in ("validation", "calibration", "test"):
        next_cursor = cursor + counts[name]
        assignments.update(
            (source_id, name) for source_id in shuffled[cursor:next_cursor]
        )
        cursor = next_cursor
    assignments.update((source_id, "training") for source_id in shuffled[cursor:])
    return assignments
