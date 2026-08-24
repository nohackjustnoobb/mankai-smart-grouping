"""External probability adjustment and decision-policy validation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

CALIBRATION_THRESHOLD_SOURCE = "calibration"
CLASSIFIER_FORWARD_OUTPUT = "float training-distribution logit"
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


@dataclass(frozen=True)
class DeploymentPolicy:
    training_positive_fraction: float
    threshold_deployment_positive_fraction: float
    recommended_threshold: float | None
    threshold_source: str | None
    manifest_sha256: str
    dataset_sha256: str


def validated_probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1")
    return probability


def validated_threshold(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return threshold


def validate_metadata_contract(metadata: Mapping[str, Any]) -> DeploymentPolicy:
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata.json must contain a JSON object")
    if metadata.get("threshold_embedded") is not False:
        raise ValueError("metadata must declare threshold_embedded=false")
    if metadata.get("classifier_forward_output") != CLASSIFIER_FORWARD_OUTPUT:
        raise ValueError("metadata has an unsupported classifier_forward_output")

    manifest_sha256 = metadata.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or not _SHA256_PATTERN.fullmatch(
        manifest_sha256
    ):
        raise ValueError("metadata manifest_sha256 must be a 64-character hex digest")
    dataset_sha256 = metadata.get("dataset_sha256")
    if not isinstance(dataset_sha256, str) or not _SHA256_PATTERN.fullmatch(
        dataset_sha256
    ):
        raise ValueError("metadata dataset_sha256 must be a 64-character hex digest")

    training_positive_fraction = validated_probability(
        metadata.get("training_positive_fraction"),
        name="training_positive_fraction",
    )
    threshold_deployment_positive_fraction = validated_probability(
        metadata.get("recommended_threshold_deployment_positive_fraction"),
        name="recommended_threshold_deployment_positive_fraction",
    )

    if "recommended_threshold" not in metadata:
        if metadata.get("threshold_source") is not None:
            raise ValueError("threshold_source requires recommended_threshold")
        if metadata.get("threshold_reason") is not None:
            raise ValueError("threshold_reason requires recommended_threshold")
        return DeploymentPolicy(
            training_positive_fraction=training_positive_fraction,
            threshold_deployment_positive_fraction=(
                threshold_deployment_positive_fraction
            ),
            recommended_threshold=None,
            threshold_source=None,
            manifest_sha256=manifest_sha256,
            dataset_sha256=dataset_sha256,
        )

    recommended_threshold = validated_threshold(
        metadata["recommended_threshold"], name="recommended_threshold"
    )
    threshold_source = metadata.get("threshold_source")
    if threshold_source != CALIBRATION_THRESHOLD_SOURCE:
        raise ValueError(
            "recommended_threshold requires threshold_source='calibration'"
        )
    threshold_reason = metadata.get("threshold_reason")
    if not isinstance(threshold_reason, str) or not threshold_reason.strip():
        raise ValueError("recommended_threshold requires a non-empty threshold_reason")
    return DeploymentPolicy(
        training_positive_fraction=training_positive_fraction,
        threshold_deployment_positive_fraction=threshold_deployment_positive_fraction,
        recommended_threshold=recommended_threshold,
        threshold_source=threshold_source,
        manifest_sha256=manifest_sha256,
        dataset_sha256=dataset_sha256,
    )


def resolve_threshold(
    cli_threshold: float | None,
    policy: DeploymentPolicy,
    deployment_positive_fraction: float,
) -> tuple[float, str]:
    """Resolve a decision threshold compatible with the requested prior."""

    if cli_threshold is not None:
        return (
            validated_threshold(cli_threshold, name="decision threshold"),
            "command line",
        )
    if policy.recommended_threshold is None or policy.threshold_source is None:
        raise ValueError(
            "metadata has no recommended_threshold. Pass --threshold explicitly"
        )
    if not math.isclose(
        deployment_positive_fraction,
        policy.threshold_deployment_positive_fraction,
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise ValueError(
            "the recommended threshold was calibrated for deployment_positive_fraction "
            f"{policy.threshold_deployment_positive_fraction}. Pass --threshold "
            "explicitly when using a different deployment fraction"
        )
    return policy.recommended_threshold, policy.threshold_source


def resolve_deployment_positive_fraction(
    cli_value: float | None, policy: DeploymentPolicy
) -> tuple[float, str]:
    """Use an explicit deployment prior or the metadata calibration prior."""

    if cli_value is None:
        return policy.threshold_deployment_positive_fraction, "metadata"
    return (
        validated_probability(cli_value, name="deployment_positive_fraction"),
        "command line",
    )


def validate_classifier_contract(classifier: Any) -> None:
    """Require a raw-logit classifier with no embedded deployment policy."""

    if hasattr(classifier, "decision_threshold"):
        raise ValueError("classifier cannot contain decision_threshold")
    if hasattr(classifier, "prior_logit_bias") or hasattr(classifier, "probability"):
        raise ValueError("classifier must return raw logits without prior adjustment")


def deployment_probability(
    logits: Tensor,
    training_positive_fraction: float,
    deployment_positive_fraction: float,
) -> Tensor:
    """Apply deployment-prior correction outside the model artifact."""

    training_prior = validated_probability(
        training_positive_fraction, name="training_positive_fraction"
    )
    deployment_prior = validated_probability(
        deployment_positive_fraction, name="deployment_positive_fraction"
    )
    bias = math.log(deployment_prior / (1.0 - deployment_prior)) - math.log(
        training_prior / (1.0 - training_prior)
    )
    return torch.sigmoid(logits + bias)
