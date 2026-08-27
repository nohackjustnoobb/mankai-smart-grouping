"""Generate paired image strips for the smart-grouping model efficiently.

For every accepted source image, the generator creates four 224-pixel-wide
strips after resizing the source to a fixed height:

    part 1: the left edge
    part 2: the 224 pixels immediately left of the horizontal centre
    part 3: the 224 pixels immediately right of the horizontal centre
    part 4: the right edge

Parts 2 and 3 from the same image form a positive pair. Part 4 from one image
and part 1 from a different image in the same source-disjoint experiment split
form a negative pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.splits import (  # noqa: E402
    SPLIT_NAMES,
    assign_source_splits,
    validate_split_fractions,
)

STRIP_WIDTH = 224
OUTPUT_SIZE = (224, 224)
EXIF_ORIENTATION = 274
DEFAULT_WORKERS = (os.cpu_count() or 2) * 2
DEFAULT_NEGATIVES_PER_IMAGE = 16
DEFAULT_HARD_NEGATIVE_FRACTION = 0.5
DEFAULT_HARD_NEGATIVE_CANDIDATES = 128
DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_CALIBRATION_FRACTION = 0.1
DEFAULT_TEST_FRACTION = 0.1
GENERATED_FILENAMES = {
    "dataset.csv",
    "sources.csv",
    "rejected.csv",
    "config.json",
}


@dataclass(frozen=True)
class SourceImage:
    source_id: str
    path: Path
    relative_path: str
    original_width: int
    original_height: int
    normalized_width: int
    perceptual_hash: int


@dataclass(frozen=True)
class InspectedImage:
    path: Path
    relative_path: str
    original_width: int
    original_height: int
    normalized_width: int
    perceptual_hash: int


@dataclass(frozen=True)
class RejectedImage:
    path: str
    reason: str
    similar_to: str = ""


class _BKNode:
    """Node in a BK-tree indexed by Hamming distance."""

    def __init__(self, value: int, source: SourceImage) -> None:
        self.value = value
        self.source = source
        self.children: dict[int, _BKNode] = {}


class PerceptualHashIndex:
    """Efficiently find an accepted image within a Hamming-distance limit."""

    def __init__(self) -> None:
        self._root: _BKNode | None = None

    @staticmethod
    def _distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def find(self, value: int, max_distance: int) -> SourceImage | None:
        if self._root is None:
            return None

        pending = [self._root]
        while pending:
            node = pending.pop()
            distance = self._distance(value, node.value)
            if distance <= max_distance:
                return node.source

            lower = distance - max_distance
            upper = distance + max_distance
            pending.extend(
                child
                for edge_distance, child in node.children.items()
                if lower <= edge_distance <= upper
            )
        return None

    def add(self, value: int, source: SourceImage) -> None:
        if self._root is None:
            self._root = _BKNode(value, source)
            return

        node = self._root
        while True:
            distance = self._distance(value, node.value)
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value, source)
                return
            node = child


def perceptual_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Return a difference hash that places similar images close together."""

    grayscale = image.convert("L").resize(
        (hash_size + 1, hash_size), Image.Resampling.LANCZOS
    )
    pixels = grayscale.tobytes()
    value = 0
    row_width = hash_size + 1
    for row in range(hash_size):
        row_start = row * row_width
        for column in range(hash_size):
            value = (value << 1) | int(
                pixels[row_start + column] > pixels[row_start + column + 1]
            )
    return value


def _display_path(path: Path, input_dir: Path) -> str:
    try:
        return path.relative_to(input_dir).as_posix()
    except ValueError:
        return str(path)


def discover_files(
    input_dir: Path, output_dir: Path, recursive: bool
) -> Iterable[Path]:
    """Yield all regular files, letting Pillow decide which ones are images."""

    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    for path in sorted(iterator):
        if not path.is_file():
            continue
        try:
            path.relative_to(output_dir)
        except ValueError:
            yield path


def _bounded_parallel_map[T, R](
    function: Callable[[T], R], items: Iterable[T], workers: int
) -> Iterator[R]:
    """Map in input order without queuing the entire dataset in memory."""

    if workers == 1:
        for item in items:
            yield function(item)
        return

    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: deque[Future[R]] = deque()
        for _ in range(workers * 2):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break

        while pending:
            future = pending.popleft()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass
            yield future.result()


def _orientation_swaps_axes(orientation: int) -> bool:
    return orientation in {5, 6, 7, 8}


def _inspect_file(
    path: Path, input_dir: Path, fixed_height: int
) -> InspectedImage | RejectedImage:
    """Read dimensions and a tiny perceptual hash without a full-size resize."""

    relative_path = _display_path(path, input_dir)
    try:
        with Image.open(path) as opened:
            raw_width, raw_height = opened.size
            orientation = int(opened.getexif().get(EXIF_ORIENTATION, 1))
            if _orientation_swaps_axes(orientation):
                width, height = raw_height, raw_width
            else:
                width, height = raw_width, raw_height

            if height < fixed_height:
                return RejectedImage(
                    relative_path,
                    f"height {height} is smaller than fixed height {fixed_height}",
                )

            normalized_width = round(width * fixed_height / height)
            if normalized_width < 2 * STRIP_WIDTH:
                return RejectedImage(
                    relative_path,
                    f"normalized width {normalized_width} is smaller than "
                    f"{2 * STRIP_WIDTH}",
                )

            # Draft mode reduces JPEG decoding work and is ignored by other formats.
            opened.draft("L", (32, 32))
            preview = ImageOps.exif_transpose(opened)
            image_hash = perceptual_dhash(preview)

        return InspectedImage(
            path=path,
            relative_path=relative_path,
            original_width=width,
            original_height=height,
            normalized_width=normalized_width,
            perceptual_hash=image_hash,
        )
    except (
        OSError,
        UnidentifiedImageError,
        ValueError,
        Image.DecompressionBombError,
    ) as error:
        return RejectedImage(relative_path, f"unreadable image: {error}")


def scan_sources(
    input_dir: Path,
    output_dir: Path,
    fixed_height: int,
    similarity_threshold: int,
    recursive: bool,
    workers: int,
    progress_every: int,
) -> tuple[list[SourceImage], list[RejectedImage]]:
    """Inspect sources in parallel and deterministically remove near-duplicates."""

    accepted: list[SourceImage] = []
    rejected: list[RejectedImage] = []
    hash_index = PerceptualHashIndex()
    files = list(discover_files(input_dir, output_dir, recursive))
    print(f"[info] inspecting {len(files):,} files with {workers} workers")

    inspected = _bounded_parallel_map(
        lambda path: _inspect_file(path, input_dir, fixed_height), files, workers
    )
    for completed, result in enumerate(inspected, start=1):
        if isinstance(result, RejectedImage):
            rejected.append(result)
            if progress_every and completed % progress_every == 0:
                print(f"[info] inspected {completed:,}/{len(files):,} files")
            continue

        similar = hash_index.find(result.perceptual_hash, similarity_threshold)
        if similar is not None:
            rejected.append(
                RejectedImage(
                    result.relative_path,
                    "exact or near-duplicate image",
                    similar.relative_path,
                )
            )
        else:
            source = SourceImage(
                source_id=f"source_{len(accepted):06d}",
                path=result.path,
                relative_path=result.relative_path,
                original_width=result.original_width,
                original_height=result.original_height,
                normalized_width=result.normalized_width,
                perceptual_hash=result.perceptual_hash,
            )
            accepted.append(source)
            hash_index.add(result.perceptual_hash, source)

        if progress_every and completed % progress_every == 0:
            print(f"[info] inspected {completed:,}/{len(files):,} files")

    return accepted, rejected


def extract_parts(source: SourceImage, fixed_height: int) -> list[Image.Image]:
    """Decoder-scale, height-normalize, and crop four 224-square parts."""

    with Image.open(source.path) as opened:
        orientation = int(opened.getexif().get(EXIF_ORIENTATION, 1))
        target_size = (source.normalized_width, fixed_height)
        if _orientation_swaps_axes(orientation):
            target_size = (fixed_height, source.normalized_width)
        # Draft mode reduces JPEG work before the exact LANCZOS resize.
        opened.draft("RGB", target_size)
        image = ImageOps.exif_transpose(opened).convert("RGB")

    normalized = image.resize(
        (source.normalized_width, fixed_height), Image.Resampling.LANCZOS
    )
    width = normalized.width
    middle = width // 2
    boxes = [
        (0, 0, STRIP_WIDTH, fixed_height),
        (middle - STRIP_WIDTH, 0, middle, fixed_height),
        (middle, 0, middle + STRIP_WIDTH, fixed_height),
        (width - STRIP_WIDTH, 0, width, fixed_height),
    ]
    parts = [normalized.crop(box) for box in boxes]
    return [
        part
        if part.size == OUTPUT_SIZE
        else part.resize(OUTPUT_SIZE, Image.Resampling.LANCZOS)
        for part in parts
    ]


def _save_source_parts(
    source: SourceImage,
    fixed_height: int,
    parts_dir: Path,
    jpeg_quality: int,
) -> None:
    for part_number, part in enumerate(extract_parts(source, fixed_height), start=1):
        part.save(
            parts_dir / f"{source.source_id}_part{part_number}.jpg",
            format="JPEG",
            quality=jpeg_quality,
            optimize=False,
        )


def _prepare_output(output_dir: Path, overwrite: bool) -> Path:
    """Create a safe output directory and clear only known generated content."""

    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output_dir}")

    existing_generated = [
        output_dir / name
        for name in (*GENERATED_FILENAMES, "parts")
        if (output_dir / name).exists()
    ]
    if existing_generated and not overwrite:
        raise FileExistsError(
            f"generated files already exist in {output_dir}. Use --overwrite"
        )

    if overwrite:
        parts_dir = output_dir / "parts"
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        for name in GENERATED_FILENAMES:
            generated_file = output_dir / name
            if generated_file.is_file() or generated_file.is_symlink():
                generated_file.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir()
    return parts_dir


def _write_sources(
    output_dir: Path,
    sources: Sequence[SourceImage],
    source_splits: Mapping[str, str],
) -> None:
    with (output_dir / "sources.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_id",
                "path",
                "original_width",
                "original_height",
                "normalized_width",
                "perceptual_hash",
                "split",
            ],
        )
        writer.writeheader()
        for source in sources:
            writer.writerow(
                {
                    "source_id": source.source_id,
                    "path": source.relative_path,
                    "original_width": source.original_width,
                    "original_height": source.original_height,
                    "normalized_width": source.normalized_width,
                    "perceptual_hash": f"{source.perceptual_hash:016x}",
                    "split": source_splits[source.source_id],
                }
            )


def _write_rejections(output_dir: Path, rejected: Sequence[RejectedImage]) -> None:
    with (output_dir / "rejected.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "reason", "similar_to"])
        writer.writeheader()
        writer.writerows(asdict(item) for item in rejected)


def _write_dataset_csv(
    output_dir: Path,
    sources: Sequence[SourceImage],
    negatives_per_image: int,
    seed: int,
    source_splits: Mapping[str, str],
    hard_negative_fraction: float = DEFAULT_HARD_NEGATIVE_FRACTION,
    hard_negative_candidates: int = DEFAULT_HARD_NEGATIVE_CANDIDATES,
) -> int:
    source_ids = {source.source_id for source in sources}
    if len(source_ids) != len(sources):
        raise ValueError("source IDs must be unique")
    if set(source_splits) != source_ids:
        raise ValueError("source_splits must contain every source ID exactly once")
    invalid_splits = set(source_splits.values()) - set(SPLIT_NAMES)
    if invalid_splits:
        raise ValueError(f"invalid source split: {', '.join(sorted(invalid_splits))}")
    if not 0.0 <= hard_negative_fraction <= 1.0:
        raise ValueError("hard negative fraction must be between 0 and 1")
    if hard_negative_candidates <= 0:
        raise ValueError("hard negative candidates must be positive")
    groups = {
        split: [
            source for source in sources if source_splits.get(source.source_id) == split
        ]
        for split in SPLIT_NAMES
    }
    groups = {split: group for split, group in groups.items() if group}
    for split, group in groups.items():
        if negatives_per_image >= len(group):
            raise ValueError(
                f"split {split} has {len(group)} sources, but "
                f"negatives_per_image is {negatives_per_image}. Use at most "
                f"{len(group) - 1}"
            )
    row_count = 0

    with (output_dir / "dataset.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "sample_id",
            "input_1",
            "input_2",
            "label",
            "source_1",
            "source_2",
            "split",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for source in sources:
            writer.writerow(
                {
                    "sample_id": f"true_{source.source_id}",
                    "input_1": f"parts/{source.source_id}_part2.jpg",
                    "input_2": f"parts/{source.source_id}_part3.jpg",
                    "label": 1,
                    "source_1": source.source_id,
                    "source_2": source.source_id,
                    "split": source_splits[source.source_id],
                }
            )
            row_count += 1

        for split, group in groups.items():
            for source_index, source in enumerate(group):
                partner_choices = _select_negative_partners(
                    source,
                    group,
                    source_index=source_index,
                    count=negatives_per_image,
                    seed=seed,
                    split=split,
                    hard_negative_fraction=(
                        hard_negative_fraction if split == "training" else 0.0
                    ),
                    hard_negative_candidates=hard_negative_candidates,
                )
                for partner_number, (other, sampling_kind) in enumerate(
                    partner_choices
                ):
                    writer.writerow(
                        {
                            "sample_id": (
                                f"false_{split}_{sampling_kind}_"
                                f"{partner_number:03d}_{source.source_id}"
                            ),
                            "input_1": f"parts/{source.source_id}_part4.jpg",
                            "input_2": f"parts/{other.source_id}_part1.jpg",
                            "label": 0,
                            "source_1": source.source_id,
                            "source_2": other.source_id,
                            "split": split,
                        }
                    )
                    row_count += 1

    return row_count


def _select_negative_partners(
    source: SourceImage,
    group: Sequence[SourceImage],
    *,
    source_index: int,
    count: int,
    seed: int,
    split: str,
    hard_negative_fraction: float,
    hard_negative_candidates: int,
) -> list[tuple[SourceImage, str]]:
    """Select unique partners, optionally mining visually similar training rows."""

    rng = random.Random(f"{seed}:{split}:{source.source_id}")
    if hard_negative_fraction > 0.0:
        hard_count = min(count, max(1, round(count * hard_negative_fraction)))
        candidate_count = min(
            len(group) - 1,
            max(count, hard_negative_candidates),
        )
    else:
        hard_count = 0
        candidate_count = count

    raw_indices = rng.sample(range(len(group) - 1), candidate_count)
    candidates = [group[index + int(index >= source_index)] for index in raw_indices]
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            (source.perceptual_hash ^ candidate.perceptual_hash).bit_count(),
            candidate.source_id,
        ),
    )
    hard = ranked[:hard_count]
    hard_ids = {candidate.source_id for candidate in hard}
    random_pool = [
        candidate for candidate in candidates if candidate.source_id not in hard_ids
    ]
    random_partners = rng.sample(random_pool, count - hard_count)
    return [
        *((candidate, "hard") for candidate in hard),
        *((candidate, "random") for candidate in random_partners),
    ]


def generate_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    fixed_height: int = 720,
    similarity_threshold: int = 5,
    negatives_per_image: int = DEFAULT_NEGATIVES_PER_IMAGE,
    hard_negative_fraction: float = DEFAULT_HARD_NEGATIVE_FRACTION,
    hard_negative_candidates: int = DEFAULT_HARD_NEGATIVE_CANDIDATES,
    seed: int = 42,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    recursive: bool = True,
    overwrite: bool = False,
    workers: int = DEFAULT_WORKERS,
    jpeg_quality: int = 92,
    progress_every: int = 1000,
) -> dict[str, int]:
    """Generate the processed parts and pair manifest, returning summary counts."""

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("input and output directories must be different")
    try:
        input_dir.relative_to(output_dir)
    except ValueError:
        pass
    else:
        raise ValueError("output directory cannot contain the input directory")
    if fixed_height <= 0:
        raise ValueError("fixed height must be positive")
    if not 0 <= similarity_threshold <= 64:
        raise ValueError("similarity threshold must be between 0 and 64")
    if negatives_per_image <= 0:
        raise ValueError("negatives per image must be positive")
    if not 0.0 <= hard_negative_fraction <= 1.0:
        raise ValueError("hard negative fraction must be between 0 and 1")
    if hard_negative_candidates <= 0:
        raise ValueError("hard negative candidates must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be between 1 and 100")
    if progress_every < 0:
        raise ValueError("progress interval cannot be negative")
    validate_split_fractions(validation_fraction, calibration_fraction, test_fraction)

    sources, rejected = scan_sources(
        input_dir,
        output_dir,
        fixed_height,
        similarity_threshold,
        recursive,
        workers,
        progress_every,
    )
    if len(sources) < 2:
        raise ValueError(
            "at least two valid, non-similar images are required to create "
            f"negative pairs (found {len(sources)})"
        )
    source_splits = assign_source_splits(
        [source.source_id for source in sources],
        validation_fraction=validation_fraction,
        calibration_fraction=calibration_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    source_counts_by_split = Counter(source_splits.values())
    smallest_split, smallest_count = min(
        source_counts_by_split.items(), key=lambda item: item[1]
    )
    if negatives_per_image >= smallest_count:
        raise ValueError(
            f"split {smallest_split} has {smallest_count} sources, but "
            f"negatives_per_image is {negatives_per_image}. Use at most "
            f"{smallest_count - 1}"
        )

    parts_dir = _prepare_output(output_dir, overwrite)
    print(f"[info] writing {len(sources) * 4:,} image parts with {workers} workers")
    saved = _bounded_parallel_map(
        lambda source: _save_source_parts(
            source, fixed_height, parts_dir, jpeg_quality
        ),
        sources,
        workers,
    )
    for completed, _ in enumerate(saved, start=1):
        if progress_every and completed % progress_every == 0:
            print(f"[info] processed {completed:,}/{len(sources):,} accepted images")

    _write_sources(output_dir, sources, source_splits)
    _write_rejections(output_dir, rejected)
    sample_count = _write_dataset_csv(
        output_dir,
        sources,
        negatives_per_image,
        seed,
        source_splits,
        hard_negative_fraction,
        hard_negative_candidates,
    )
    config = {
        "input_dir": str(input_dir),
        "fixed_height": fixed_height,
        "strip_width": STRIP_WIDTH,
        "output_size": list(OUTPUT_SIZE),
        "similarity_threshold": similarity_threshold,
        "negatives_per_image": negatives_per_image,
        "negative_sampling": {
            "training": "per_source_unique_hard_and_uniform_candidate_mix",
            "validation": "per_source_unique_uniform",
            "calibration": "per_source_unique_uniform",
            "test": "per_source_unique_uniform",
        },
        "hard_negative_fraction": hard_negative_fraction,
        "hard_negative_candidates": hard_negative_candidates,
        "hard_negative_splits": ["training"],
        "seed": seed,
        "split_strategy": "source_disjoint_before_pairing",
        "split_fractions": {
            "validation": validation_fraction,
            "calibration": calibration_fraction,
            "test": test_fraction,
        },
        "source_counts_by_split": dict(sorted(source_counts_by_split.items())),
        "recursive": recursive,
        "workers": workers,
        "image_format": "JPEG",
        "jpeg_quality": jpeg_quality,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "accepted_images": len(sources),
        "rejected_files": len(rejected),
        "samples": sample_count,
        "positive_samples": len(sources),
        "negative_samples": len(sources) * negatives_per_image,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate 224x224 positive and negative image-strip pairs."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("raw"),
        help="directory containing raw images (default: raw)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("data"),
        help="directory for the dataset (default: data)",
    )
    parser.add_argument(
        "--fixed-height",
        type=int,
        default=720,
        help="height used before extracting strips (default: 720)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=int,
        default=5,
        metavar="BITS",
        help="reject dHashes within this Hamming distance, 0-64 (default: 5)",
    )
    parser.add_argument(
        "--negatives-per-image",
        type=int,
        default=DEFAULT_NEGATIVES_PER_IMAGE,
        help=(
            "unique random negative partners per image and split "
            f"(default: {DEFAULT_NEGATIVES_PER_IMAGE})"
        ),
    )
    parser.add_argument(
        "--hard-negative-fraction",
        type=float,
        default=DEFAULT_HARD_NEGATIVE_FRACTION,
        help=(
            "fraction of training negatives mined by dHash similarity, 0-1 "
            f"(default: {DEFAULT_HARD_NEGATIVE_FRACTION})"
        ),
    )
    parser.add_argument(
        "--hard-negative-candidates",
        type=int,
        default=DEFAULT_HARD_NEGATIVE_CANDIDATES,
        help=(
            "random candidates considered per training source for hard-negative "
            f"mining (default: {DEFAULT_HARD_NEGATIVE_CANDIDATES})"
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help=f"source fraction used for epoch selection (default: {DEFAULT_VALIDATION_FRACTION})",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
        help=f"source fraction used for threshold selection (default: {DEFAULT_CALIBRATION_FRACTION})",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=DEFAULT_TEST_FRACTION,
        help=f"source fraction reserved for final evaluation (default: {DEFAULT_TEST_FRACTION})",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="do not scan subdirectories",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace previously generated files in the output directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"parallel image workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        metavar="N",
        help="output JPEG quality, 1-100 (default: 92)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        metavar="N",
        help="print progress every N images. 0 disables it (default: 1000)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = generate_dataset(
            args.input_dir,
            args.output_dir,
            fixed_height=args.fixed_height,
            similarity_threshold=args.similarity_threshold,
            negatives_per_image=args.negatives_per_image,
            hard_negative_fraction=args.hard_negative_fraction,
            hard_negative_candidates=args.hard_negative_candidates,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            calibration_fraction=args.calibration_fraction,
            test_fraction=args.test_fraction,
            recursive=not args.no_recursive,
            overwrite=args.overwrite,
            workers=args.workers,
            jpeg_quality=args.jpeg_quality,
            progress_every=args.progress_every,
        )
    except (OSError, ValueError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1

    print(f"[info] dataset written to: {args.output_dir.expanduser().resolve()}")
    print(
        f"[info] accepted images: {summary['accepted_images']} | "
        f"rejected files: {summary['rejected_files']}"
    )
    print(
        f"[info] samples: {summary['samples']} | "
        f"positive: {summary['positive_samples']} | "
        f"negative: {summary['negative_samples']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
