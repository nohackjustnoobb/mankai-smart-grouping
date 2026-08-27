"""Pair-manifest loading, leakage-safe splitting, and image transforms."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast, overload

import numpy as np
import polars as pl
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from .splits import SPLIT_NAMES

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MANIFEST_COLUMNS = (
    "sample_id",
    "input_1",
    "input_2",
    "label",
    "source_1",
    "source_2",
    "split",
)
REQUIRED_COLUMNS = set(MANIFEST_COLUMNS)


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    input_1: Path
    input_2: Path
    label: int
    source_1: str
    source_2: str
    split: str


class PairRecords(Sequence[PairRecord]):
    """Columnar pair records backed by a Polars DataFrame."""

    def __init__(self, frame: pl.DataFrame, dataset_root: Path):
        self.frame = frame.select(MANIFEST_COLUMNS)
        self.dataset_root = dataset_root

    def __len__(self) -> int:
        return self.frame.height

    @overload
    def __getitem__(self, index: int) -> PairRecord: ...

    @overload
    def __getitem__(self, index: slice) -> list[PairRecord]: ...

    def __getitem__(self, index: int | slice) -> PairRecord | list[PairRecord]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        (
            sample_id,
            input_1,
            input_2,
            label,
            source_1,
            source_2,
            split,
        ) = self.frame.row(index)
        return PairRecord(
            sample_id=sample_id,
            input_1=self.dataset_root / input_1,
            input_2=self.dataset_root / input_2,
            label=label,
            source_1=source_1,
            source_2=source_2,
            split=split,
        )

    @property
    def labels(self) -> np.ndarray:
        return self.frame.get_column("label").to_numpy().copy()

    def filter(self, predicate: pl.Expr) -> PairRecords:
        return PairRecords(self.frame.filter(predicate), self.dataset_root)


@dataclass(frozen=True)
class ExperimentSplits:
    training: Sequence[PairRecord]
    validation: Sequence[PairRecord]
    calibration: Sequence[PairRecord]
    test: Sequence[PairRecord]


def sample_experiment_splits(
    splits: ExperimentSplits,
    *,
    fraction: float,
    seed: int,
) -> ExperimentSplits:
    """Return deterministic, stratified fractional subsets of every split."""

    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise TypeError("fraction must be a number")
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be finite and in (0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    return ExperimentSplits(
        training=_sample_records(splits.training, fraction=fraction, seed=seed),
        validation=_sample_records(splits.validation, fraction=fraction, seed=seed + 1),
        calibration=_sample_records(
            splits.calibration, fraction=fraction, seed=seed + 2
        ),
        test=_sample_records(splits.test, fraction=fraction, seed=seed + 3),
    )


def _sample_records(
    records: Sequence[PairRecord],
    *,
    fraction: float,
    seed: int,
) -> Sequence[PairRecord]:
    target_count = min(
        len(records),
        max(2, math.ceil(len(records) * fraction)),
    )
    if target_count >= len(records):
        return records

    positive_total = (
        int(records.frame.get_column("label").sum())
        if isinstance(records, PairRecords)
        else sum(record.label for record in records)
    )
    negative_total = len(records) - positive_total
    minimum_positives = max(1, target_count - negative_total)
    maximum_positives = min(positive_total, target_count - 1)
    proportional_positives = round(positive_total * target_count / len(records))
    positive_count = min(
        max(proportional_positives, minimum_positives), maximum_positives
    )
    negative_count = target_count - positive_count

    if isinstance(records, PairRecords):
        positives = records.frame.filter(pl.col("label") == 1)
        negatives = records.frame.filter(pl.col("label") == 0)
        subset = pl.concat(
            [
                negatives.sample(n=negative_count, shuffle=True, seed=seed),
                positives.sample(n=positive_count, shuffle=True, seed=seed + 1),
            ]
        ).sample(fraction=1.0, shuffle=True, seed=seed + 2)
        return PairRecords(subset, records.dataset_root)

    by_label = {
        0: [record for record in records if record.label == 0],
        1: [record for record in records if record.label == 1],
    }
    rng = random.Random(seed)
    rng.shuffle(by_label[0])
    rng.shuffle(by_label[1])
    subset = by_label[0][:negative_count] + by_label[1][:positive_count]
    rng.shuffle(subset)
    return subset


def limit_experiment_splits(
    splits: ExperimentSplits,
    *,
    max_samples: int,
    seed: int,
) -> ExperimentSplits:
    """Return deterministic subsets with both classes retained in every split."""

    if isinstance(max_samples, bool) or not isinstance(max_samples, int):
        raise TypeError("max_samples must be an integer")
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    return ExperimentSplits(
        training=_limit_records(splits.training, max_samples=max_samples, seed=seed),
        validation=_limit_records(
            splits.validation, max_samples=max_samples, seed=seed + 1
        ),
        calibration=_limit_records(
            splits.calibration, max_samples=max_samples, seed=seed + 2
        ),
        test=_limit_records(splits.test, max_samples=max_samples, seed=seed + 3),
    )


def _limit_records(
    records: Sequence[PairRecord],
    *,
    max_samples: int,
    seed: int,
) -> Sequence[PairRecord]:
    if len(records) <= max_samples:
        return records

    if isinstance(records, PairRecords):
        positives = records.frame.filter(pl.col("label") == 1)
        negatives = records.frame.filter(pl.col("label") == 0)
        positive_count = min(positives.height, max_samples // 2)
        negative_count = min(negatives.height, max_samples - positive_count)
        remaining = max_samples - positive_count - negative_count
        if remaining:
            extra_positives = min(remaining, positives.height - positive_count)
            positive_count += extra_positives
            remaining -= extra_positives
        if remaining:
            negative_count += min(remaining, negatives.height - negative_count)
        subset = pl.concat(
            [
                negatives.sample(n=negative_count, shuffle=True, seed=seed),
                positives.sample(n=positive_count, shuffle=True, seed=seed + 1),
            ]
        ).sample(fraction=1.0, shuffle=True, seed=seed + 2)
        return PairRecords(subset, records.dataset_root)

    by_label = {
        0: [record for record in records if record.label == 0],
        1: [record for record in records if record.label == 1],
    }
    rng = random.Random(seed)
    rng.shuffle(by_label[0])
    rng.shuffle(by_label[1])

    positive_count = min(len(by_label[1]), max_samples // 2)
    negative_count = min(len(by_label[0]), max_samples - positive_count)
    remaining = max_samples - positive_count - negative_count
    if remaining:
        extra_positives = min(remaining, len(by_label[1]) - positive_count)
        positive_count += extra_positives
        remaining -= extra_positives
    if remaining:
        negative_count += min(remaining, len(by_label[0]) - negative_count)

    subset = by_label[0][:negative_count] + by_label[1][:positive_count]
    rng.shuffle(subset)
    return subset


class PairDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, records: Sequence[PairRecord], transform: transforms.Compose):
        if not records:
            raise ValueError("pair dataset cannot be empty")
        self.transform = transform
        if isinstance(records, PairRecords):
            self._frame = records.frame.select("input_1", "input_2", "label")
            self._dataset_root = records.dataset_root
            self._records: list[PairRecord] | None = None
            self.labels = records.labels
        else:
            self._frame = None
            self._dataset_root = None
            self._records = list(records)
            self.labels = np.fromiter(
                (record.label for record in self._records),
                dtype=np.int8,
                count=len(self._records),
            )

    def __len__(self) -> int:
        return (
            self._frame.height if self._frame is not None else len(self._records or ())
        )

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if self._frame is not None:
            if self._dataset_root is None:
                raise RuntimeError("columnar pair dataset has no dataset root")
            input_1, input_2, label = self._frame.row(index)
            path_1 = self._dataset_root / input_1
            path_2 = self._dataset_root / input_2
        else:
            if self._records is None:
                raise RuntimeError("pair dataset has no record storage")
            record = self._records[index]
            path_1 = record.input_1
            path_2 = record.input_2
            label = record.label
        with Image.open(path_1) as opened:
            image_1 = opened.convert("RGB")
        with Image.open(path_2) as opened:
            image_2 = opened.convert("RGB")
        return (
            self.transform(image_1),
            self.transform(image_2),
            torch.tensor(label, dtype=torch.float32),
        )


def build_transform(training: bool) -> transforms.Compose:
    """Build deterministic preprocessing shared by training and evaluation."""

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _first_invalid_row(frame: pl.DataFrame, predicate: pl.Expr) -> int | None:
    invalid = frame.filter(predicate).get_column("_row_number")
    return None if invalid.is_empty() else int(invalid[0])


def read_manifest(manifest_path: Path) -> PairRecords:
    """Read and validate a manifest with Polars' native multithreaded parser."""

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")

    schema = pl.scan_csv(manifest_path, infer_schema=False).collect_schema()
    missing_columns = REQUIRED_COLUMNS - set(schema.names())
    if missing_columns:
        raise ValueError(
            f"manifest is missing columns: {', '.join(sorted(missing_columns))}"
        )

    frame = pl.read_csv(
        manifest_path,
        columns=MANIFEST_COLUMNS,
        infer_schema=False,
        row_index_name="_row_number",
        row_index_offset=2,
        low_memory=False,
        rechunk=False,
    ).with_columns(
        pl.col(
            [column for column in MANIFEST_COLUMNS if column != "label"]
        ).str.strip_chars(),
        pl.col("label").str.strip_chars().cast(pl.Int8, strict=False).alias("label"),
    )
    if frame.is_empty():
        raise ValueError(f"manifest has no samples: {manifest_path}")

    for columns, message in (
        (("sample_id",), "empty sample_id"),
        (("input_1", "input_2"), "empty input path"),
        (("source_1", "source_2"), "empty source ID"),
    ):
        row_number = _first_invalid_row(
            frame,
            pl.any_horizontal(
                [
                    pl.col(column).is_null() | (pl.col(column) == "")
                    for column in columns
                ]
            ),
        )
        if row_number is not None:
            raise ValueError(f"{message} on CSV row {row_number}")

    row_number = _first_invalid_row(
        frame, pl.col("label").is_null() | ~pl.col("label").is_in([0, 1])
    )
    if row_number is not None:
        raise ValueError(f"label must be 0 or 1 on CSV row {row_number}")

    row_number = _first_invalid_row(
        frame, pl.col("split").is_null() | ~pl.col("split").is_in(SPLIT_NAMES)
    )
    if row_number is not None:
        expected = ", ".join(SPLIT_NAMES)
        raise ValueError(f"invalid split on CSV row {row_number}. Expected {expected}")

    row_number = _first_invalid_row(
        frame,
        ((pl.col("label") == 1) & (pl.col("source_1") != pl.col("source_2")))
        | ((pl.col("label") == 0) & (pl.col("source_1") == pl.col("source_2"))),
    )
    if row_number is not None:
        raise ValueError(
            f"pair source identity does not match label on CSV row {row_number}"
        )

    if frame.get_column("sample_id").n_unique() != frame.height:
        raise ValueError("manifest contains duplicate sample_id values")

    source_splits = pl.concat(
        [
            frame.select(pl.col("source_1").alias("source_id"), "split").unique(),
            frame.select(pl.col("source_2").alias("source_id"), "split").unique(),
        ]
    ).unique()
    leaked_source = (
        source_splits.group_by("source_id")
        .len()
        .filter(pl.col("len") > 1)
        .get_column("source_id")
    )
    if not leaked_source.is_empty():
        raise ValueError(f"source {leaked_source[0]} appears in multiple splits")

    return PairRecords(frame.drop("_row_number"), manifest_path.parent)


def split_records_for_experiment(
    records: Sequence[PairRecord],
) -> ExperimentSplits:
    """Bucket records using their required source-disjoint manifest split."""

    if isinstance(records, PairRecords):
        partitioned = records.frame.partition_by("split", maintain_order=True)
        frames_by_split = {
            str(frame.get_column("split")[0]): frame for frame in partitioned
        }
        buckets = {
            name: PairRecords(
                frames_by_split.get(name, records.frame.head(0)),
                records.dataset_root,
            )
            for name in SPLIT_NAMES
        }
        for name, bucket in buckets.items():
            _validate_binary_split(name, bucket)
        return ExperimentSplits(
            training=buckets["training"],
            validation=buckets["validation"],
            calibration=buckets["calibration"],
            test=buckets["test"],
        )

    buckets: dict[str, list[PairRecord]] = {name: [] for name in SPLIT_NAMES}
    source_assignments: dict[str, str] = {}
    sample_ids: set[str] = set()
    for record in records:
        _validate_pair_identity(
            record.sample_id,
            record.label,
            record.source_1,
            record.source_2,
            context=f"record {record.sample_id or '<empty>'}",
        )
        if record.sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {record.sample_id}")
        sample_ids.add(record.sample_id)
        if record.split not in SPLIT_NAMES:
            raise ValueError(f"invalid declared split: {record.split}")
        for source_id in (record.source_1, record.source_2):
            previous = source_assignments.setdefault(source_id, record.split)
            if previous != record.split:
                raise ValueError(
                    f"source {source_id} appears in both {previous} and {record.split}"
                )
        buckets[record.split].append(record)

    for name in SPLIT_NAMES:
        _validate_binary_split(name, buckets[name])
    return ExperimentSplits(
        training=buckets["training"],
        validation=buckets["validation"],
        calibration=buckets["calibration"],
        test=buckets["test"],
    )


def _validate_pair_identity(
    sample_id: str,
    label: int,
    source_1: str,
    source_2: str,
    *,
    context: str,
) -> None:
    if not sample_id:
        raise ValueError(f"empty sample_id in {context}")
    if not source_1 or not source_2:
        raise ValueError(f"empty source ID in {context}")
    if label not in (0, 1):
        raise ValueError(f"label must be 0 or 1 in {context}")
    if label == 1 and source_1 != source_2:
        raise ValueError(f"positive pair must use one source in {context}")
    if label == 0 and source_1 == source_2:
        raise ValueError(f"negative pair must use different sources in {context}")


def _validate_binary_split(name: str, records: Sequence[PairRecord]) -> None:
    if isinstance(records, PairRecords):
        counts = class_counts(records)
        if not counts["negative"] or not counts["positive"]:
            raise ValueError(
                f"{name} split needs both classes. Found "
                f"{counts['positive']} positive and {counts['negative']} negative "
                "samples. Generate more negative pairs or use more source images."
            )
        return

    counts = {0: 0, 1: 0}
    for record in records:
        counts[record.label] += 1
    if not counts[0] or not counts[1]:
        raise ValueError(
            f"{name} split needs both classes. Found "
            f"{counts[1]} positive and {counts[0]} negative samples. "
            "Generate more negative pairs or use more source images."
        )


def make_balanced_sampler(
    labels: Sequence[int] | np.ndarray | Tensor,
    *,
    positive_fraction: float,
    seed: int,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Sample a chosen class mix without throwing away abundant negatives."""

    if not 0.0 < positive_fraction < 1.0:
        raise ValueError("positive_fraction must be strictly between 0 and 1")
    label_tensor = torch.as_tensor(labels, dtype=torch.float64)
    positive_count = int(label_tensor.sum().item())
    negative_count = label_tensor.numel() - positive_count
    if not positive_count or not negative_count:
        raise ValueError("sampler requires both positive and negative samples")
    if num_samples is not None and num_samples <= 0:
        raise ValueError("num_samples must be positive")

    positive_weight = positive_fraction / positive_count
    negative_weight = (1.0 - positive_fraction) / negative_count
    weights = cast(
        Sequence[float],
        torch.where(
            label_tensor > 0,
            torch.tensor(positive_weight, dtype=torch.float64),
            torch.tensor(negative_weight, dtype=torch.float64),
        ),
    )
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=len(labels) if num_samples is None else num_samples,
        replacement=True,
        generator=generator,
    )


def class_counts(records: Sequence[PairRecord]) -> dict[str, int]:
    positives = (
        int(records.frame.get_column("label").sum())
        if isinstance(records, PairRecords)
        else sum(record.label for record in records)
    )
    return {
        "total": len(records),
        "positive": positives,
        "negative": len(records) - positives,
    }
