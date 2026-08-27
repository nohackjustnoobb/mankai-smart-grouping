"""Run TorchScript or Core ML inference on prepared or raw image pairs."""

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

from src.policy import (  # noqa: E402
    deployment_probability,
    resolve_deployment_positive_fraction,
    resolve_threshold,
    validate_classifier_contract,
    validate_metadata_contract,
)

STRIP_WIDTH = 224
DEFAULT_FIXED_HEIGHT = 720
DEFAULT_PREPROCESSED_DIR = Path("output/inference_crops")
COREML_ENCODER_NAME = "image_encoder.mlpackage"
COREML_CLASSIFIER_NAME = "pair_classifier.mlpackage"

RESIZE_BOUNDARY = transforms.Resize(
    (STRIP_WIDTH, STRIP_WIDTH),
    interpolation=InterpolationMode.BICUBIC,
    antialias=True,
)
TENSOR_PREPROCESS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


def load_image(path: Path) -> Image.Image:
    """Load an image with its EXIF orientation applied."""
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def load_direct_image(path: Path) -> Image.Image:
    """Load a model-ready 224x224 image without resizing or cropping it."""

    image = load_image(path)
    expected_size = (STRIP_WIDTH, STRIP_WIDTH)
    if image.size != expected_size:
        raise ValueError(
            f"{path} must be {STRIP_WIDTH}x{STRIP_WIDTH}px in direct mode; "
            f"got {image.width}x{image.height}px"
        )
    return image


def load_pair_direct_tensors(
    image_1_path: Path,
    image_2_path: Path,
) -> tuple[Tensor, Tensor]:
    """Load and normalize two model-ready images for TorchScript inference."""

    return (
        TENSOR_PREPROCESS(load_direct_image(image_1_path)).unsqueeze(0),
        TENSOR_PREPROCESS(load_direct_image(image_2_path)).unsqueeze(0),
    )


def load_pair_direct_images(
    image_1_path: Path,
    image_2_path: Path,
) -> tuple[Image.Image, Image.Image]:
    """Load two model-ready images for Core ML inference."""

    return load_direct_image(image_1_path), load_direct_image(image_2_path)


def preprocess_boundary(
    image: Image.Image,
    *,
    source: str | Path,
    side: str,
    fixed_height: int,
) -> Tensor:
    """Normalize an image's size, crop one boundary, and preprocess it."""

    return TENSOR_PREPROCESS(
        preprocess_boundary_image(
            image,
            source=source,
            side=side,
            fixed_height=fixed_height,
        )
    ).unsqueeze(0)


def preprocess_boundary_image(
    image: Image.Image,
    *,
    source: str | Path,
    side: str,
    fixed_height: int,
) -> Image.Image:
    """Crop and resize one boundary for the Core ML image input."""

    if image.height < fixed_height:
        raise ValueError(
            f"{source} is only {image.height}px high. Expected at least {fixed_height}px"
        )

    normalized_width = round(image.width * fixed_height / image.height)
    if normalized_width < STRIP_WIDTH:
        raise ValueError(
            f"{source} is too narrow after height normalization ({normalized_width}px)"
        )

    normalized = image.resize(
        (normalized_width, fixed_height), Image.Resampling.LANCZOS
    )
    left = 0 if side == "left" else normalized.width - STRIP_WIDTH
    boundary = normalized.crop((left, 0, left + STRIP_WIDTH, fixed_height))
    return RESIZE_BOUNDARY(boundary)


def load_boundary(path: Path, *, side: str, fixed_height: int) -> Tensor:
    """Load an image and preprocess one of its boundaries."""

    return preprocess_boundary(
        load_image(path),
        source=path,
        side=side,
        fixed_height=fixed_height,
    )


def load_pair_boundaries(
    image_1_path: Path,
    image_2_path: Path | None,
    *,
    fixed_height: int,
) -> tuple[Tensor, Tensor]:
    """Load two image boundaries, splitting a single input down the middle."""

    if image_2_path is not None:
        return (
            load_boundary(image_1_path, side="right", fixed_height=fixed_height),
            load_boundary(image_2_path, side="left", fixed_height=fixed_height),
        )

    image = load_image(image_1_path)
    if image.width < 2:
        raise ValueError(f"{image_1_path} is too narrow to split into two halves")

    midpoint = image.width // 2
    left_half = image.crop((0, 0, midpoint, image.height))
    right_half = image.crop((midpoint, 0, image.width, image.height))
    return (
        preprocess_boundary(
            left_half,
            source=f"{image_1_path} (left half)",
            side="right",
            fixed_height=fixed_height,
        ),
        preprocess_boundary(
            right_half,
            source=f"{image_1_path} (right half)",
            side="left",
            fixed_height=fixed_height,
        ),
    )


def load_pair_boundary_images(
    image_1_path: Path,
    image_2_path: Path | None,
    *,
    fixed_height: int,
) -> tuple[Image.Image, Image.Image]:
    """Load two resized boundary images for Core ML inference."""

    if image_2_path is not None:
        return (
            preprocess_boundary_image(
                load_image(image_1_path),
                source=image_1_path,
                side="right",
                fixed_height=fixed_height,
            ),
            preprocess_boundary_image(
                load_image(image_2_path),
                source=image_2_path,
                side="left",
                fixed_height=fixed_height,
            ),
        )

    image = load_image(image_1_path)
    if image.width < 2:
        raise ValueError(f"{image_1_path} is too narrow to split into two halves")

    midpoint = image.width // 2
    left_half = image.crop((0, 0, midpoint, image.height))
    right_half = image.crop((midpoint, 0, image.width, image.height))
    return (
        preprocess_boundary_image(
            left_half,
            source=f"{image_1_path} (left half)",
            side="right",
            fixed_height=fixed_height,
        ),
        preprocess_boundary_image(
            right_half,
            source=f"{image_1_path} (right half)",
            side="left",
            fixed_height=fixed_height,
        ),
    )


def save_preprocessed_images(
    boundary_1: Image.Image,
    boundary_2: Image.Image,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save the final 224x224 boundary crops used for regular inference."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = (
        output_dir / "boundary_1_224x224.png",
        output_dir / "boundary_2_224x224.png",
    )
    boundary_1.save(output_paths[0], format="PNG")
    boundary_2.save(output_paths[1], format="PNG")
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image_1",
        type=Path,
        help="first image, or a combined image to split vertically when used alone",
    )
    parser.add_argument("image_2", type=Path, nargs="?", help="optional second image")
    parser.add_argument("--model-dir", type=Path, default=Path("output/deploy"))
    parser.add_argument(
        "--coreml",
        "--use-coreml",
        dest="coreml",
        action="store_true",
        help="use the Core ML .mlpackage models instead of TorchScript",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="use image_1 and image_2 directly as model-ready 224x224 images",
    )
    parser.add_argument(
        "--save-preprocessed",
        type=Path,
        nargs="?",
        const=DEFAULT_PREPROCESSED_DIR,
        metavar="DIR",
        help=(
            "save regular-mode 224x224 crops as PNGs "
            f"(default directory: {DEFAULT_PREPROCESSED_DIR})"
        ),
    )
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
    if args.direct and args.image_2 is None:
        parser.error("--direct requires both image_1 and image_2")
    if args.direct and args.save_preprocessed is not None:
        parser.error("--save-preprocessed is only available for regular image mode")

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

    if args.coreml:
        print("[info] backend: Core ML")
        import coremltools as ct

        encoder_path = args.model_dir / COREML_ENCODER_NAME
        classifier_path = args.model_dir / COREML_CLASSIFIER_NAME
        for path in (encoder_path, classifier_path):
            if not path.exists():
                raise FileNotFoundError(f"missing Core ML model: {path}")

        encoder = ct.models.MLModel(str(encoder_path))
        classifier = ct.models.MLModel(str(classifier_path))
        if args.direct:
            boundary_1, boundary_2 = load_pair_direct_images(
                args.image_1,
                args.image_2,
            )
        else:
            boundary_1, boundary_2 = load_pair_boundary_images(
                args.image_1,
                args.image_2,
                fixed_height=args.fixed_height,
            )
            if args.save_preprocessed is not None:
                saved_paths = save_preprocessed_images(
                    boundary_1,
                    boundary_2,
                    args.save_preprocessed,
                )
                print(f"[info] saved preprocessed image: {saved_paths[0]}")
                print(f"[info] saved preprocessed image: {saved_paths[1]}")
        embedding_1 = encoder.predict({"image": boundary_1})["embedding"]
        embedding_2 = encoder.predict({"image": boundary_2})["embedding"]
        logits = torch.as_tensor(
            classifier.predict(
                {
                    "embedding_1": embedding_1,
                    "embedding_2": embedding_2,
                }
            )["logit"]
        )
    else:
        print("[info] backend: TorchScript")
        encoder = torch.jit.load(str(args.model_dir / "image_encoder.pt")).eval()
        classifier = torch.jit.load(
            str(args.model_dir / "pair_classifier.pt")
        ).eval()
        validate_classifier_contract(classifier)

        if args.direct:
            boundary_1, boundary_2 = load_pair_direct_tensors(
                args.image_1,
                args.image_2,
            )
        else:
            if args.save_preprocessed is None:
                boundary_1, boundary_2 = load_pair_boundaries(
                    args.image_1,
                    args.image_2,
                    fixed_height=args.fixed_height,
                )
            else:
                boundary_image_1, boundary_image_2 = load_pair_boundary_images(
                    args.image_1,
                    args.image_2,
                    fixed_height=args.fixed_height,
                )
                saved_paths = save_preprocessed_images(
                    boundary_image_1,
                    boundary_image_2,
                    args.save_preprocessed,
                )
                print(f"[info] saved preprocessed image: {saved_paths[0]}")
                print(f"[info] saved preprocessed image: {saved_paths[1]}")
                boundary_1 = TENSOR_PREPROCESS(boundary_image_1).unsqueeze(0)
                boundary_2 = TENSOR_PREPROCESS(boundary_image_2).unsqueeze(0)

        with torch.inference_mode():
            embedding_1 = encoder(boundary_1)
            embedding_2 = encoder(boundary_2)
            logits = classifier(embedding_1, embedding_2)

    with torch.inference_mode():
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
