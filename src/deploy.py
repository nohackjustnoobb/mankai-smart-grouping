"""Export a threshold-free TorchScript deployment bundle from a checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .model import build_model_from_checkpoint, export_torchscript_models
from .policy import CLASSIFIER_FORWARD_OUTPUT

CALIBRATION_THRESHOLD_SOURCE = "calibration"
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")

_RESERVED_METADATA_KEYS = frozenset(
    {
        "encoder",
        "classifier",
        "classifier_forward_output",
        "threshold_embedded",
        "training_positive_fraction",
        "recommended_threshold_deployment_positive_fraction",
        "manifest_sha256",
        "dataset_sha256",
        "input_shape",
        "normalization_mean",
        "normalization_std",
        "metrics",
        "metrics_split",
        "analysis_run_id",
        "recommended_threshold",
        "threshold_reason",
        "threshold_source",
        "decision_threshold",
    }
)


def _validated_probability(value: Any, *, name: str, strict: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    probability = float(value)
    lower_comparison = 0.0 < probability if strict else 0.0 <= probability
    upper_comparison = probability < 1.0 if strict else probability <= 1.0
    if not math.isfinite(probability) or not (lower_comparison and upper_comparison):
        interval = "(0, 1)" if strict else "[0, 1]"
        raise ValueError(f"{name} must be finite and in {interval}")
    return probability


def _validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
) -> tuple[Mapping[str, Any], float, float, float | None, str | None]:
    checkpoint_stage = checkpoint.get("stage")
    if checkpoint_stage not in {"training", "finalized"}:
        raise ValueError("checkpoint stage must be training or finalized")

    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("checkpoint config must be a mapping")
    training_positive_fraction = _validated_probability(
        config.get("sample_positive_fraction"),
        name="sample_positive_fraction",
        strict=True,
    )
    deployment_positive_fraction = _validated_probability(
        config.get("deployment_positive_fraction"),
        name="deployment_positive_fraction",
        strict=True,
    )
    manifest_sha256 = config.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or not _SHA256_PATTERN.fullmatch(
        manifest_sha256
    ):
        raise ValueError(
            "checkpoint config manifest_sha256 must be a 64-character hex digest"
        )
    dataset_sha256 = config.get("dataset_sha256")
    if not isinstance(dataset_sha256, str) or not _SHA256_PATTERN.fullmatch(
        dataset_sha256
    ):
        raise ValueError(
            "checkpoint config dataset_sha256 must be a 64-character hex digest"
        )

    recommended_threshold = checkpoint.get("recommended_threshold")
    threshold_source = checkpoint.get("threshold_source")
    threshold_reason = checkpoint.get("threshold_reason")
    if recommended_threshold is None:
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
        if threshold_source is not None or threshold_reason is not None:
            raise ValueError(
                "threshold_source and threshold_reason require recommended_threshold"
            )
        return (
            config,
            training_positive_fraction,
            deployment_positive_fraction,
            None,
            None,
        )

    threshold = _validated_probability(
        recommended_threshold,
        name="recommended_threshold",
        strict=False,
    )
    if checkpoint_stage != "finalized":
        raise ValueError(
            "recommended_threshold is only valid on a finalized checkpoint"
        )
    if threshold_source != CALIBRATION_THRESHOLD_SOURCE:
        raise ValueError(
            "recommended_threshold requires threshold_source='calibration'"
        )
    if not isinstance(threshold_reason, str) or not threshold_reason.strip():
        raise ValueError("recommended_threshold requires a non-empty threshold_reason")
    for metrics_name in ("calibration_metrics", "test_metrics"):
        metrics = checkpoint.get(metrics_name)
        if metrics is None:
            raise ValueError(f"finalized checkpoint requires {metrics_name}")
        if not isinstance(metrics, Mapping):
            raise TypeError(f"{metrics_name} must be a mapping")
        metrics_threshold = _validated_probability(
            metrics.get("threshold"),
            name=f"{metrics_name}.threshold",
            strict=False,
        )
        if metrics_threshold != threshold:
            raise ValueError(
                f"{metrics_name}.threshold must match recommended_threshold"
            )
    return (
        config,
        training_positive_fraction,
        deployment_positive_fraction,
        threshold,
        threshold_reason,
    )


def export_checkpoint_bundle(
    checkpoint: Mapping[str, Any],
    output_dir: Path,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    """Export model files and external decision-policy metadata."""

    if extra_metadata:
        reserved = _RESERVED_METADATA_KEYS.intersection(extra_metadata)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"extra_metadata cannot override reserved fields: {names}")

    (
        _config,
        training_positive_fraction,
        deployment_positive_fraction,
        recommended_threshold,
        threshold_reason,
    ) = _validate_checkpoint_contract(checkpoint)
    model = build_model_from_checkpoint(dict(checkpoint))
    encoder_path, classifier_path = export_torchscript_models(model, output_dir)

    metrics = checkpoint.get("test_metrics", checkpoint.get("metrics"))
    if "test_metrics" in checkpoint:
        metrics_split = "test"
    else:
        metrics_split = str(checkpoint.get("metrics_split") or "validation")
    metadata: dict[str, Any] = {
        "encoder": encoder_path.name,
        "classifier": classifier_path.name,
        "classifier_forward_output": CLASSIFIER_FORWARD_OUTPUT,
        "threshold_embedded": False,
        "training_positive_fraction": training_positive_fraction,
        "recommended_threshold_deployment_positive_fraction": (
            deployment_positive_fraction
        ),
        "manifest_sha256": _config["manifest_sha256"],
        "dataset_sha256": _config["dataset_sha256"],
        "input_shape": ["batch", 3, 224, 224],
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "metrics": metrics,
        "metrics_split": metrics_split,
        "analysis_run_id": checkpoint.get("run_id"),
    }
    if recommended_threshold is not None:
        metadata.update(
            {
                "recommended_threshold": float(recommended_threshold),
                "threshold_reason": threshold_reason,
                "threshold_source": CALIBRATION_THRESHOLD_SOURCE,
            }
        )
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = output_dir / "metadata.json"
    temporary_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temporary_path.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary_path.replace(metadata_path)
    return encoder_path, classifier_path, metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=Path("output/best_checkpoint.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/deploy"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint: dict[str, Any] = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    encoder_path, classifier_path, metadata_path = export_checkpoint_bundle(
        checkpoint, args.output_dir
    )
    print(f"[info] exported encoder: {encoder_path}")
    print(f"[info] exported classifier: {classifier_path}")
    print(f"[info] exported metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
