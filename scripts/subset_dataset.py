"""Create a self-contained sampled dataset for training trials."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (  # noqa: E402
    MANIFEST_COLUMNS,
    PairRecords,
    read_manifest,
    sample_experiment_splits,
    split_records_for_experiment,
)
from src.splits import SPLIT_NAMES  # noqa: E402

DEFAULT_WORKERS = min(32, max(4, (os.cpu_count() or 2) * 2))
OUTPUT_MANIFEST_NAME = "dataset.csv"
OUTPUT_CONFIG_NAME = "subset_config.json"


def _bounded_parallel_map(
    function: Callable[[str], int],
    items: Iterable[str],
    workers: int,
) -> Iterator[int]:
    """Apply a file operation without queuing every path at once."""

    if workers == 1:
        for item in items:
            yield function(item)
        return

    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: deque[Future[int]] = deque()
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


def _sampled_frame(
    manifest_path: Path, fraction: float, seed: int
) -> tuple[pl.DataFrame, int, int]:
    records = read_manifest(manifest_path)
    splits = sample_experiment_splits(
        split_records_for_experiment(records),
        fraction=fraction,
        seed=seed,
    )

    frames: list[pl.DataFrame] = []
    for split_name in SPLIT_NAMES:
        split_records = getattr(splits, split_name)
        if not isinstance(split_records, PairRecords):
            raise TypeError("sampled manifest splits must use columnar pair records")
        frames.append(split_records.frame)

    sampled = pl.concat(frames).select(MANIFEST_COLUMNS)
    source_part_count = _referenced_paths(records.frame).len()
    return sampled, len(records), source_part_count


def _referenced_paths(frame: pl.DataFrame) -> pl.Series:
    return (
        pl.concat(
            [
                frame.select(pl.col("input_1").alias("path")),
                frame.select(pl.col("input_2").alias("path")),
            ]
        )
        .unique()
        .sort("path")
        .get_column("path")
    )


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"input path must stay inside the dataset: {value}")
    if path in {Path(OUTPUT_MANIFEST_NAME), Path(OUTPUT_CONFIG_NAME)}:
        raise ValueError(f"input path conflicts with subset metadata: {value}")
    return path


def _split_counts(frame: pl.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split_name in SPLIT_NAMES:
        split_frame = frame.filter(pl.col("split") == split_name)
        positives = int(split_frame.get_column("label").sum())
        result[split_name] = {
            "total": split_frame.height,
            "positive": positives,
            "negative": split_frame.height - positives,
        }
    return result


def create_subset(
    manifest_path: Path,
    output_dir: Path,
    *,
    fraction: float = 0.1,
    seed: int = 42,
    workers: int = DEFAULT_WORKERS,
    progress_every: int = 10_000,
) -> dict[str, object]:
    """Sample manifest rows and materialize exactly the parts they reference."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    if progress_every < 0:
        raise ValueError("progress interval cannot be negative")

    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    print(f"[info] reading and sampling manifest: {manifest_path}")
    sampled, source_sample_count, source_part_count = _sampled_frame(
        manifest_path,
        fraction,
        seed,
    )
    referenced_paths = [str(path) for path in _referenced_paths(sampled)]
    print(
        f"[info] selected {sampled.height:,}/{source_sample_count:,} samples and "
        f"{len(referenced_paths):,}/{source_part_count:,} referenced part files"
    )

    source_root = manifest_path.parent
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )

    def materialize(value: str) -> int:
        relative_path = _safe_relative_path(value)
        source_path = source_root / relative_path
        if not source_path.is_file():
            raise FileNotFoundError(f"referenced part does not exist: {source_path}")
        destination_path = staging_dir / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        return source_path.stat().st_size

    try:
        part_bytes = 0
        for completed, size in enumerate(
            _bounded_parallel_map(materialize, referenced_paths, workers),
            start=1,
        ):
            part_bytes += size
            if progress_every and completed % progress_every == 0:
                print(
                    f"[info] materialized {completed:,}/{len(referenced_paths):,} "
                    "part files"
                )

        sampled.write_csv(staging_dir / OUTPUT_MANIFEST_NAME)
        summary: dict[str, object] = {
            "source_manifest": str(manifest_path),
            "fraction": float(fraction),
            "seed": seed,
            "source_samples": source_sample_count,
            "samples": sampled.height,
            "source_referenced_parts": source_part_count,
            "referenced_parts": len(referenced_paths),
            "part_bytes": part_bytes,
            "splits": _split_counts(sampled),
        }
        (staging_dir / OUTPUT_CONFIG_NAME).write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_dir.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(f"[info] subset written to: {output_dir}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("data/dataset.csv"),
        help="source pair manifest (default: data/dataset.csv)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("data_trial"),
        help="new self-contained dataset directory (default: data_trial)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="stratified fraction sampled from every split (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"parallel file workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        metavar="N",
        help="print progress every N files. 0 disables it (default: 10000)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        create_subset(
            args.manifest,
            args.output_dir,
            fraction=args.fraction,
            seed=args.seed,
            workers=args.workers,
            progress_every=args.progress_every,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
