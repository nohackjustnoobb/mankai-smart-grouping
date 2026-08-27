"""Evaluate a checkpoint with metrics suitable for a rare positive class."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import (
    PairDataset,
    build_transform,
    class_counts,
    limit_experiment_splits,
    read_manifest,
    sample_experiment_splits,
    split_records_for_experiment,
)
from .model import SiameseNetwork, build_model_from_checkpoint
from .policy import deployment_probability


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def predict(
    model: SiameseNetwork,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
    *,
    training_positive_fraction: float,
    deployment_positive_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for image_1, image_2, batch_labels in loader:
        image_1 = image_1.to(device, non_blocking=True)
        image_2 = image_2.to(device, non_blocking=True)
        logits = model(image_1, image_2)
        batch_probabilities = deployment_probability(
            logits,
            training_positive_fraction,
            deployment_positive_fraction,
        )
        labels.append(batch_labels.numpy().astype(np.int64))
        probabilities.append(batch_probabilities.cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    deployment_positive_fraction: float,
    minimum_precision: float | None = None,
) -> tuple[float, str]:
    """Select for deployment F1, or maximum recall at a precision floor."""

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        labels, probabilities, drop_intermediate=False
    )
    finite = np.isfinite(thresholds)
    false_positive_rate = false_positive_rate[finite]
    true_positive_rate = true_positive_rate[finite]
    thresholds = thresholds[finite]
    precision = _precision_at_prevalence(
        true_positive_rate, false_positive_rate, deployment_positive_fraction
    )

    if minimum_precision is not None:
        if not 0.0 < minimum_precision <= 1.0:
            raise ValueError("minimum_precision must be in (0, 1]")
        eligible = np.flatnonzero(precision >= minimum_precision)
        if not len(eligible):
            # Use an all-negative decision when no threshold meets the precision floor.
            return 1.0, f"no threshold reached precision {minimum_precision:.3f}"
        recalls = true_positive_rate[eligible]
        best_recall = recalls.max()
        candidates = eligible[recalls == best_recall]
        best = candidates[np.argmax(precision[candidates])]
        return float(thresholds[best]), "maximum recall at precision floor"

    f1 = np.divide(
        2.0 * precision * true_positive_rate,
        precision + true_positive_rate,
        out=np.zeros_like(precision),
        where=(precision + true_positive_rate) > 0,
    )
    return float(thresholds[int(np.argmax(f1))]), "maximum deployment-adjusted F1"


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    deployment_positive_fraction: float,
) -> dict[str, float | int]:
    threshold = _validated_threshold(threshold, source="metrics")
    predictions = probabilities >= threshold
    positives = labels == 1
    negatives = ~positives
    true_positive = int(np.sum(predictions & positives))
    false_positive = int(np.sum(predictions & negatives))
    true_negative = int(np.sum(~predictions & negatives))
    false_negative = int(np.sum(~predictions & positives))

    recall = _safe_divide(true_positive, true_positive + false_negative)
    false_positive_rate = _safe_divide(false_positive, false_positive + true_negative)
    sample_precision = _safe_divide(true_positive, true_positive + false_positive)
    deployment_precision = float(
        _precision_at_prevalence(
            np.asarray([recall]),
            np.asarray([false_positive_rate]),
            deployment_positive_fraction,
        )[0]
    )
    deployment_f1 = _safe_divide(
        2.0 * deployment_precision * recall, deployment_precision + recall
    )

    metrics = {
        "threshold": float(threshold),
        "samples": len(labels),
        "positives": int(np.sum(positives)),
        "negatives": int(np.sum(negatives)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "specificity": 1.0 - false_positive_rate,
        "sample_precision": sample_precision,
        "deployment_precision": deployment_precision,
        "deployment_f1": deployment_f1,
    }
    metrics.update(
        compute_ranking_metrics(
            labels,
            probabilities,
            deployment_positive_fraction=deployment_positive_fraction,
        )
    )
    return metrics


def compute_ranking_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    deployment_positive_fraction: float,
) -> dict[str, float | int]:
    """Compute threshold-free metrics suitable for model selection."""

    positives = labels == 1
    return {
        "samples": len(labels),
        "positives": int(np.sum(positives)),
        "negatives": int(np.sum(~positives)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "sample_average_precision": float(
            average_precision_score(labels, probabilities)
        ),
        "deployment_pr_auc": deployment_pr_auc(
            labels, probabilities, deployment_positive_fraction
        ),
        "deployment_positive_fraction": deployment_positive_fraction,
    }


def deployment_pr_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
    deployment_positive_fraction: float,
) -> float:
    false_positive_rate, recall, _ = roc_curve(
        labels, probabilities, drop_intermediate=False
    )
    precision = _precision_at_prevalence(
        recall, false_positive_rate, deployment_positive_fraction
    )
    # Repeated recall values contribute zero area.
    return float(np.trapezoid(precision, recall))


def _precision_at_prevalence(
    true_positive_rate: np.ndarray,
    false_positive_rate: np.ndarray,
    prevalence: float,
) -> np.ndarray:
    if not 0.0 < prevalence < 1.0:
        raise ValueError("deployment_positive_fraction must be in (0, 1)")
    numerator = prevalence * true_positive_rate
    denominator = numerator + (1.0 - prevalence) * false_positive_rate
    return np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _validated_threshold(value: Any, *, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{source} threshold must be a number")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{source} threshold must be finite and between 0 and 1")
    return threshold


def print_metrics(metrics: dict[str, float | int]) -> None:
    print(
        " | ".join(
            [
                f"[metrics] roc_auc={metrics['roc_auc']:.4f}",
                f"deployment_pr_auc={metrics['deployment_pr_auc']:.4f}",
                f"deployment_precision={metrics['deployment_precision']:.4f}",
                f"recall={metrics['recall']:.4f}",
                f"deployment_f1={metrics['deployment_f1']:.4f}",
                f"false_positive_rate={metrics['false_positive_rate']:.6f}",
                f"threshold={metrics['threshold']:.6f}",
            ]
        )
    )


def print_ranking_metrics(metrics: dict[str, float | int]) -> None:
    print(
        " | ".join(
            [
                f"[metrics] roc_auc={metrics['roc_auc']:.4f}",
                f"deployment_pr_auc={metrics['deployment_pr_auc']:.4f}",
                f"samples={metrics['samples']}",
            ]
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=Path("output/best_checkpoint.pt"),
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("data/dataset.csv"),
    )
    parser.add_argument(
        "--split",
        choices=("training", "validation", "calibration", "test", "all"),
        help="dataset split (default: independent test)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/evaluation.json"),
        help="metrics JSON path (default: output/evaluation.json)",
    )
    parser.add_argument("--predictions-csv", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    device = resolve_device(args.device)
    loaded_checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(loaded_checkpoint, dict):
        raise TypeError("checkpoint must contain a mapping")
    checkpoint: dict[str, Any] = loaded_checkpoint
    checkpoint_stage = checkpoint.get("stage")
    if checkpoint_stage not in {"training", "finalized"}:
        raise ValueError("checkpoint stage must be training or finalized")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("checkpoint config must be a mapping")

    recommendation = checkpoint.get("recommended_threshold")
    recommendation_source = checkpoint.get("threshold_source")
    recommendation_reason = checkpoint.get("threshold_reason")
    validated_recommendation: float | None = None
    if recommendation is None:
        if checkpoint_stage == "finalized":
            raise ValueError("finalized checkpoint requires recommended_threshold")
        finalized_metrics = {"calibration_metrics", "test_metrics"}.intersection(
            checkpoint
        )
        if finalized_metrics:
            raise ValueError(
                "training checkpoint cannot contain finalized metrics: "
                f"{', '.join(sorted(finalized_metrics))}"
            )
        if recommendation_source is not None or recommendation_reason is not None:
            raise ValueError(
                "threshold_source and threshold_reason require recommended_threshold"
            )
    else:
        if checkpoint_stage != "finalized":
            raise ValueError(
                "recommended_threshold is only valid on a finalized checkpoint"
            )
        validated_recommendation = _validated_threshold(
            recommendation, source="checkpoint recommendation"
        )
        if recommendation_source != "calibration":
            raise ValueError(
                "recommended_threshold requires threshold_source='calibration'"
            )
        if (
            not isinstance(recommendation_reason, str)
            or not recommendation_reason.strip()
        ):
            raise ValueError(
                "recommended_threshold requires a non-empty threshold_reason"
            )
        for metrics_name in ("calibration_metrics", "test_metrics"):
            stored_metrics = checkpoint.get(metrics_name)
            if stored_metrics is None:
                raise ValueError(f"finalized checkpoint requires {metrics_name}")
            if not isinstance(stored_metrics, Mapping):
                raise TypeError(f"{metrics_name} must be a mapping")
            stored_threshold = _validated_threshold(
                stored_metrics.get("threshold"),
                source=f"{metrics_name}",
            )
            if stored_threshold != validated_recommendation:
                raise ValueError(
                    f"{metrics_name}.threshold must match recommended_threshold"
                )

    manifest_path = args.manifest.expanduser().resolve()
    if args.threshold is not None:
        threshold = _validated_threshold(args.threshold, source="command-line")
        threshold_source = "command_line"
    elif validated_recommendation is not None:
        threshold = validated_recommendation
        threshold_source = "calibration"
    else:
        raise ValueError("checkpoint has no recommended threshold. Pass --threshold")

    data_fraction = config.get("data_fraction")
    if data_fraction is not None and (
        isinstance(data_fraction, bool)
        or not isinstance(data_fraction, (int, float))
        or not math.isfinite(data_fraction)
        or not 0.0 < data_fraction <= 1.0
    ):
        raise ValueError("checkpoint data_fraction must be finite and in (0, 1]")
    max_samples_per_split = config.get("max_samples_per_split")
    if max_samples_per_split is not None and (
        isinstance(max_samples_per_split, bool)
        or not isinstance(max_samples_per_split, int)
        or max_samples_per_split < 2
    ):
        raise ValueError("checkpoint max_samples_per_split must be an integer >= 2")
    if data_fraction is not None and max_samples_per_split is not None:
        raise ValueError(
            "checkpoint cannot set both data_fraction and max_samples_per_split"
        )
    subset_seed = config.get("seed")
    if (data_fraction is not None or max_samples_per_split is not None) and (
        isinstance(subset_seed, bool) or not isinstance(subset_seed, int)
    ):
        raise ValueError("subset checkpoint seed must be an integer")

    records = read_manifest(manifest_path)
    splits = split_records_for_experiment(records)
    if data_fraction is not None:
        splits = sample_experiment_splits(
            splits,
            fraction=float(data_fraction),
            seed=subset_seed,
        )
    elif max_samples_per_split is not None:
        splits = limit_experiment_splits(
            splits,
            max_samples=max_samples_per_split,
            seed=subset_seed,
        )
    split_name = args.split or "test"
    records = (
        [
            *splits.training,
            *splits.validation,
            *splits.calibration,
            *splits.test,
        ]
        if split_name == "all"
        else getattr(splits, split_name)
    )

    negative_count = class_counts(records)["negative"]
    if negative_count < 10_000:
        print(
            f"[warning] {split_name} has only {negative_count:,} negatives. "
            "Rare false-positive estimates will be coarse"
        )

    dataset = PairDataset(records, build_transform(training=False))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model_from_checkpoint(checkpoint).to(device)
    labels, probabilities = predict(
        model,
        loader,
        device,
        training_positive_fraction=config["sample_positive_fraction"],
        deployment_positive_fraction=config["deployment_positive_fraction"],
    )
    metrics = compute_metrics(
        labels,
        probabilities,
        threshold=threshold,
        deployment_positive_fraction=config["deployment_positive_fraction"],
    )
    print_metrics(metrics)
    report: dict[str, Any] = {
        **metrics,
        "split": split_name,
        "threshold_source": threshold_source,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    if args.predictions_csv:
        _write_predictions(
            args.predictions_csv, records, labels, probabilities, threshold
        )
    return 0


def _write_predictions(
    path: Path,
    records: Sequence[Any],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=("sample_id", "label", "probability", "prediction")
        )
        writer.writeheader()
        for record, label, probability in zip(records, labels, probabilities):
            writer.writerow(
                {
                    "sample_id": record.sample_id,
                    "label": int(label),
                    "probability": float(probability),
                    "prediction": int(probability >= threshold),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
