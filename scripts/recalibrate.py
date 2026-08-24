"""Recalibrate deployment policy without changing the exported models."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    PairDataset,
    PairRecord,
    build_transform,
    dataset_sha256,
    file_sha256,
    read_manifest,
    split_records_for_experiment,
)
from src.evaluate import (
    compute_metrics,
    print_metrics,
    resolve_device,
    select_threshold,
)
from src.policy import (
    CALIBRATION_THRESHOLD_SOURCE,
    DeploymentPolicy,
    deployment_probability,
    validate_classifier_contract,
    validate_metadata_contract,
    validated_probability,
)


def _atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _artifact_path(model_dir: Path, metadata: Mapping[str, Any], *, key: str) -> Path:
    filename = metadata.get(key)
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError(f"metadata {key} must be a plain filename")
    path = model_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing deployment artifact: {path}")
    return path


def _validate_dataset(
    manifest_path: Path, policy: DeploymentPolicy
) -> tuple[list[PairRecord], str, str]:
    manifest_sha256 = file_sha256(manifest_path)
    if manifest_sha256 != policy.manifest_sha256:
        raise ValueError(
            "manifest does not match the deployed model: "
            f"expected SHA-256 {policy.manifest_sha256}, got {manifest_sha256}"
        )
    records = read_manifest(manifest_path)
    observed_dataset_sha256 = dataset_sha256(
        manifest_path,
        records,
        expected_manifest_sha256=manifest_sha256,
    )
    if observed_dataset_sha256 != policy.dataset_sha256:
        raise ValueError(
            "dataset image content does not match the deployed model: "
            f"expected SHA-256 {policy.dataset_sha256}, got "
            f"{observed_dataset_sha256}"
        )
    return records, manifest_sha256, observed_dataset_sha256


def _build_loader(
    records: Sequence[PairRecord],
    *,
    batch_size: int,
    workers: int,
    pin_memory: bool,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    return DataLoader(
        PairDataset(records, build_transform(training=False)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )


@torch.inference_mode()
def _predict(
    encoder: Any,
    classifier: Any,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
    *,
    training_positive_fraction: float,
    deployment_positive_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for image_1, image_2, batch_labels in loader:
        image_1 = image_1.to(device, non_blocking=True)
        image_2 = image_2.to(device, non_blocking=True)
        embedding_1 = encoder(image_1)
        embedding_2 = encoder(image_2)
        logits = classifier(embedding_1, embedding_2)
        batch_probabilities = deployment_probability(
            logits,
            training_positive_fraction,
            deployment_positive_fraction,
        )
        labels.append(batch_labels.numpy().astype(np.int64))
        probabilities.append(batch_probabilities.cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def _warn_if_small_negative_split(name: str, records: Sequence[PairRecord]) -> None:
    negative_count = sum(record.label == 0 for record in records)
    if negative_count < 10_000:
        print(
            f"[warning] {name} has only {negative_count:,} negatives. "
            "Rare false-positive estimates will be coarse"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("data/dataset.csv"),
    )
    parser.add_argument("--model-dir", type=Path, default=Path("output/deploy"))
    parser.add_argument(
        "--deployment-positive-fraction",
        type=float,
        required=True,
        help="new expected positive fraction for deployment",
    )
    parser.add_argument(
        "--minimum-precision",
        type=float,
        help="maximize recall subject to this deployment precision",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="report path (default: MODEL_DIR/recalibration.json)",
    )
    parser.add_argument(
        "--no-update-metadata",
        action="store_true",
        help="write the report without replacing metadata.json policy fields",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    deployment_positive_fraction = validated_probability(
        args.deployment_positive_fraction,
        name="deployment_positive_fraction",
    )
    if args.minimum_precision is not None and not 0.0 < args.minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be in (0, 1]")

    model_dir = args.model_dir.expanduser().resolve()
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing deployment metadata: {metadata_path}")
    loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(loaded_metadata, dict):
        raise TypeError("metadata.json must contain a JSON object")
    metadata: dict[str, Any] = loaded_metadata
    policy = validate_metadata_contract(metadata)

    manifest_path = args.manifest.expanduser().resolve()
    records, manifest_sha256, observed_dataset_sha256 = _validate_dataset(
        manifest_path, policy
    )
    splits = split_records_for_experiment(records)
    _warn_if_small_negative_split("calibration", splits.calibration)
    _warn_if_small_negative_split("test", splits.test)

    device = resolve_device(args.device)
    encoder_path = _artifact_path(model_dir, metadata, key="encoder")
    classifier_path = _artifact_path(model_dir, metadata, key="classifier")
    encoder = torch.jit.load(str(encoder_path), map_location=device).eval()
    classifier = torch.jit.load(
        str(classifier_path),
        map_location=device,
    ).eval()
    validate_classifier_contract(classifier)

    loader_options = {
        "batch_size": args.batch_size,
        "workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    calibration_loader = _build_loader(splits.calibration, **loader_options)
    test_loader = _build_loader(splits.test, **loader_options)

    calibration_labels, calibration_probabilities = _predict(
        encoder,
        classifier,
        calibration_loader,
        device,
        training_positive_fraction=policy.training_positive_fraction,
        deployment_positive_fraction=deployment_positive_fraction,
    )
    recommended_threshold, threshold_reason = select_threshold(
        calibration_labels,
        calibration_probabilities,
        deployment_positive_fraction=deployment_positive_fraction,
        minimum_precision=args.minimum_precision,
    )
    calibration_metrics = compute_metrics(
        calibration_labels,
        calibration_probabilities,
        threshold=recommended_threshold,
        deployment_positive_fraction=deployment_positive_fraction,
    )

    test_labels, test_probabilities = _predict(
        encoder,
        classifier,
        test_loader,
        device,
        training_positive_fraction=policy.training_positive_fraction,
        deployment_positive_fraction=deployment_positive_fraction,
    )
    test_metrics = compute_metrics(
        test_labels,
        test_probabilities,
        threshold=recommended_threshold,
        deployment_positive_fraction=deployment_positive_fraction,
    )

    completed_at = datetime.now(UTC).isoformat()
    report: dict[str, Any] = {
        "completed_at": completed_at,
        "deployment_positive_fraction": deployment_positive_fraction,
        "training_positive_fraction": policy.training_positive_fraction,
        "minimum_precision": args.minimum_precision,
        "recommended_threshold": recommended_threshold,
        "threshold_reason": threshold_reason,
        "threshold_source": CALIBRATION_THRESHOLD_SOURCE,
        "manifest_sha256": manifest_sha256,
        "dataset_sha256": observed_dataset_sha256,
        "calibration_metrics": calibration_metrics,
        "test_metrics": test_metrics,
    }
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else model_dir / "recalibration.json"
    )
    protected_paths = {metadata_path, encoder_path, classifier_path}
    if output_json in protected_paths:
        raise ValueError(
            "output_json cannot replace deployment metadata or model files"
        )
    _atomic_write_json(report, output_json)

    if not args.no_update_metadata:
        metadata.update(
            {
                "recommended_threshold_deployment_positive_fraction": (
                    deployment_positive_fraction
                ),
                "recommended_threshold": recommended_threshold,
                "threshold_reason": threshold_reason,
                "threshold_source": CALIBRATION_THRESHOLD_SOURCE,
                "metrics": test_metrics,
                "metrics_split": "test",
                "recalibration": {
                    "completed_at": completed_at,
                    "minimum_precision": args.minimum_precision,
                    "calibration_metrics": calibration_metrics,
                    "test_metrics": test_metrics,
                },
            }
        )
        _atomic_write_json(metadata, metadata_path)

    print("[info] calibration operating point")
    print_metrics(calibration_metrics)
    print("[info] independent test result")
    print_metrics(test_metrics)
    print(f"[info] recalibration report: {output_json}")
    if args.no_update_metadata:
        print("[info] deployment metadata was not changed")
    else:
        print(f"[info] updated deployment metadata: {metadata_path}")
    print("[info] TorchScript model files were not changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
