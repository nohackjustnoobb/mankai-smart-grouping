"""Pair-manifest loading, leakage-safe splitting, and image transforms."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .splits import SPLIT_NAMES

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REQUIRED_COLUMNS = {
    "sample_id",
    "input_1",
    "input_2",
    "label",
    "source_1",
    "source_2",
    "split",
}


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    input_1: Path
    input_2: Path
    label: int
    source_1: str
    source_2: str
    split: str


@dataclass(frozen=True)
class ExperimentSplits:
    training: list[PairRecord]
    validation: list[PairRecord]
    calibration: list[PairRecord]
    test: list[PairRecord]


class PairDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, records: Sequence[PairRecord], transform: transforms.Compose):
        if not records:
            raise ValueError("pair dataset cannot be empty")
        self.records = list(records)
        self.transform = transform
        self.labels = [record.label for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        record = self.records[index]
        with Image.open(record.input_1) as opened:
            image_1 = opened.convert("RGB")
        with Image.open(record.input_2) as opened:
            image_2 = opened.convert("RGB")
        return (
            self.transform(image_1),
            self.transform(image_2),
            torch.tensor(record.label, dtype=torch.float32),
        )


def build_transform(training: bool) -> transforms.Compose:
    """Build transforms without flips that would reverse boundary direction."""

    operations: list[object] = [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    ]
    if training:
        operations.extend(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.15,
                            contrast=0.15,
                            saturation=0.1,
                            hue=0.02,
                        )
                    ],
                    p=0.6,
                ),
                transforms.RandomGrayscale(p=0.03),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(operations)


def read_manifest(manifest_path: Path, *, check_files: bool = True) -> list[PairRecord]:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")

    records: list[PairRecord] = []
    missing_paths: list[Path] = []
    checked_paths: set[Path] = set()
    sample_ids: set[str] = set()
    declared_source_splits: dict[str, str] = {}
    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            raise ValueError(
                f"manifest is missing columns: {', '.join(sorted(missing_columns))}"
            )
        for row_number, row in enumerate(reader, start=2):
            sample_id = (row.get("sample_id") or "").strip()
            source_1 = (row.get("source_1") or "").strip()
            source_2 = (row.get("source_2") or "").strip()
            input_1 = (row.get("input_1") or "").strip()
            input_2 = (row.get("input_2") or "").strip()
            if not input_1 or not input_2:
                raise ValueError(f"empty input path on CSV row {row_number}")
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid label on CSV row {row_number}") from error
            if label not in (0, 1):
                raise ValueError(f"label must be 0 or 1 on CSV row {row_number}")
            _validate_pair_identity(
                sample_id,
                label,
                source_1,
                source_2,
                context=f"CSV row {row_number}",
            )
            if sample_id in sample_ids:
                raise ValueError(f"duplicate sample_id on CSV row {row_number}")
            sample_ids.add(sample_id)

            path_1 = (manifest_path.parent / input_1).resolve()
            path_2 = (manifest_path.parent / input_2).resolve()
            if check_files:
                for path in (path_1, path_2):
                    if path not in checked_paths:
                        checked_paths.add(path)
                        if not path.is_file() and len(missing_paths) < 10:
                            missing_paths.append(path)

            split = (row.get("split") or "").strip()
            if split not in SPLIT_NAMES:
                expected = ", ".join(SPLIT_NAMES)
                raise ValueError(
                    f"invalid split on CSV row {row_number}. Expected {expected}"
                )
            for source_id in (source_1, source_2):
                previous = declared_source_splits.setdefault(source_id, split)
                if previous != split:
                    raise ValueError(
                        f"source {source_id} appears in both {previous} and {split}"
                    )

            records.append(
                PairRecord(
                    sample_id=sample_id,
                    input_1=path_1,
                    input_2=path_2,
                    label=label,
                    source_1=source_1,
                    source_2=source_2,
                    split=split,
                )
            )

    if not records:
        raise ValueError(f"manifest has no samples: {manifest_path}")
    if missing_paths:
        examples = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"manifest references missing images:\n{examples}")
    return records


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file's exact bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_sha256(
    manifest_path: Path,
    records: Sequence[PairRecord],
    *,
    expected_manifest_sha256: str | None = None,
) -> str:
    """Bind manifest bytes, referenced paths, and referenced image bytes."""

    manifest_path = manifest_path.expanduser().resolve()
    observed_manifest_sha256 = file_sha256(manifest_path)
    if (
        expected_manifest_sha256 is not None
        and observed_manifest_sha256 != expected_manifest_sha256
    ):
        raise ValueError("manifest changed while the dataset was being inspected")
    dataset_root = manifest_path.parent
    referenced_paths = {record.input_1 for record in records} | {
        record.input_2 for record in records
    }
    path_tokens: list[tuple[str, Path]] = []
    for path in referenced_paths:
        try:
            token = f"relative:{path.relative_to(dataset_root).as_posix()}"
        except ValueError:
            token = f"absolute:{path.as_posix()}"
        path_tokens.append((token, path))

    digest = hashlib.sha256()
    digest.update(b"mankai-dataset\0")
    digest.update(bytes.fromhex(observed_manifest_sha256))
    for token, path in sorted(path_tokens):
        encoded_token = token.encode("utf-8")
        digest.update(len(encoded_token).to_bytes(8, "big"))
        digest.update(encoded_token)
        digest.update(bytes.fromhex(file_sha256(path)))
    if file_sha256(manifest_path) != observed_manifest_sha256:
        raise ValueError("manifest changed while the dataset was being fingerprinted")
    return digest.hexdigest()


def split_records_for_experiment(
    records: Sequence[PairRecord],
) -> ExperimentSplits:
    """Bucket records using their required source-disjoint manifest split."""

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
    labels: Sequence[int],
    *,
    positive_fraction: float,
    seed: int,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Sample a chosen class mix without throwing away abundant negatives."""

    if not 0.0 < positive_fraction < 1.0:
        raise ValueError("positive_fraction must be strictly between 0 and 1")
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        raise ValueError("sampler requires both positive and negative samples")
    if num_samples is not None and num_samples <= 0:
        raise ValueError("num_samples must be positive")

    positive_weight = positive_fraction / positive_count
    negative_weight = (1.0 - positive_fraction) / negative_count
    weights = [positive_weight if label else negative_weight for label in labels]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=len(labels) if num_samples is None else num_samples,
        replacement=True,
        generator=generator,
    )


def class_counts(records: Sequence[PairRecord]) -> dict[str, int]:
    positives = sum(record.label for record in records)
    return {
        "total": len(records),
        "positive": positives,
        "negative": len(records) - positives,
    }
