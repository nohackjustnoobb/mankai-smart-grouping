"""Run inference on a pair of raw images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.policy import (
    deployment_probability,
    resolve_deployment_positive_fraction,
    resolve_threshold,
    validate_classifier_contract,
    validate_metadata_contract,
)

STRIP_WIDTH = 224
DEFAULT_FIXED_HEIGHT = 720

PREPROCESS = transforms.Compose(
    [
        transforms.Resize(
            (STRIP_WIDTH, STRIP_WIDTH),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


def load_boundary(path: Path, *, side: str, fixed_height: int) -> Tensor:
    """Normalize a raw image's size, crop one boundary, and preprocess it."""

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    if image.height < fixed_height:
        raise ValueError(
            f"{path} is only {image.height}px high. Expected at least {fixed_height}px"
        )

    normalized_width = round(image.width * fixed_height / image.height)
    if normalized_width < STRIP_WIDTH:
        raise ValueError(
            f"{path} is too narrow after height normalization ({normalized_width}px)"
        )

    normalized = image.resize(
        (normalized_width, fixed_height), Image.Resampling.LANCZOS
    )
    left = 0 if side == "left" else normalized.width - STRIP_WIDTH
    boundary = normalized.crop((left, 0, left + STRIP_WIDTH, fixed_height))
    return PREPROCESS(boundary).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_1", type=Path)
    parser.add_argument("image_2", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("output/deploy"))
    parser.add_argument("--fixed-height", type=int, default=DEFAULT_FIXED_HEIGHT)
    parser.add_argument(
        "--deployment-positive-fraction",
        type=float,
        help=(
            "expected positive fraction for this deployment environment "
            "(defaults to metadata calibration value)"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="external decision threshold (defaults to metadata recommendation)",
    )
    args = parser.parse_args()

    metadata_path = args.model_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing deployment metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    policy = validate_metadata_contract(metadata)
    deployment_positive_fraction, deployment_fraction_source = (
        resolve_deployment_positive_fraction(
            args.deployment_positive_fraction,
            policy,
        )
    )
    threshold, threshold_source = resolve_threshold(
        args.threshold,
        policy,
        deployment_positive_fraction,
    )

    encoder = torch.jit.load(str(args.model_dir / "image_encoder.pt")).eval()
    classifier = torch.jit.load(str(args.model_dir / "pair_classifier.pt")).eval()
    validate_classifier_contract(classifier)

    with torch.inference_mode():
        embedding_1 = encoder(
            load_boundary(args.image_1, side="right", fixed_height=args.fixed_height)
        )
        embedding_2 = encoder(
            load_boundary(args.image_2, side="left", fixed_height=args.fixed_height)
        )
        logits = classifier(embedding_1, embedding_2)
        probability = deployment_probability(
            logits,
            policy.training_positive_fraction,
            deployment_positive_fraction,
        )
        label = int(probability.item() >= threshold)

    print(f"[result] label: {label}")
    print(f"[result] probability: {probability.item():.6f}")
    print(
        "[result] deployment positive fraction: "
        f"{deployment_positive_fraction:.6f} ({deployment_fraction_source})"
    )
    print(f"[result] threshold: {threshold:.6f} ({threshold_source})")


if __name__ == "__main__":
    main()
