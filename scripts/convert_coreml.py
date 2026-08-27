"""Convert the final TorchScript deployment models to Core ML packages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import coremltools as ct
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (  # noqa: E402
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transform,
)

IMAGE_SHAPE = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
DEFAULT_MODEL_DIR = Path("output/deploy")
ENCODER_PACKAGE_NAME = "image_encoder.mlpackage"
CLASSIFIER_PACKAGE_NAME = "pair_classifier.mlpackage"
DEPLOYMENT_TARGETS = ("iOS15", "iOS16", "iOS17", "iOS18", "iOS26")
VERIFICATION_SPLITS = ("training", "validation", "calibration", "test")


@dataclass(frozen=True)
class VerificationPair:
    sample_id: str
    image_1: Path
    image_2: Path
    label: int


@dataclass(frozen=True)
class ErrorStats:
    mean_absolute: float
    max_absolute: float
    root_mean_square: float


class _CoreMLImageEncoder(torch.nn.Module):
    """Apply the channel-wise part of ImageNet normalization in the graph."""

    def __init__(self, encoder: torch.jit.ScriptModule) -> None:
        super().__init__()
        self.encoder = encoder
        standard_deviation = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("standard_deviation", standard_deviation)

    def forward(self, centered_image: torch.Tensor) -> torch.Tensor:
        standard_deviation = cast(torch.Tensor, self.standard_deviation)
        return self.encoder(centered_image / standard_deviation)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_metadata(model_dir: Path) -> dict[str, Any]:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing deployment metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("metadata.json must contain a JSON object")
    return metadata


def _source_model_path(model_dir: Path, metadata: dict[str, Any], key: str) -> Path:
    filename = metadata.get(key)
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"metadata.json must contain a non-empty {key!r} filename")
    path = model_dir / filename
    if path.parent.resolve() != model_dir.resolve():
        raise ValueError(f"metadata {key!r} must name a file directly in {model_dir}")
    if not path.is_file():
        raise FileNotFoundError(f"missing {key} model: {path}")
    return path


def _sample_verification_pairs(
    manifest_path: Path,
    *,
    split: str,
    sample_count: int,
    seed: int,
) -> list[VerificationPair]:
    """Reservoir-sample a balanced subset without loading the full manifest."""

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"verification manifest does not exist: {manifest_path}"
        )
    if sample_count < 2:
        raise ValueError("verification sample count must be at least 2")

    quotas = {0: (sample_count + 1) // 2, 1: sample_count // 2}
    seen = {0: 0, 1: 0}
    reservoirs: dict[int, list[VerificationPair]] = {0: [], 1: []}
    generator = random.Random(seed)
    required_columns = {"sample_id", "input_1", "input_2", "label", "split"}

    with manifest_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing_columns = required_columns - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "verification manifest is missing columns: "
                f"{', '.join(sorted(missing_columns))}"
            )
        for row_number, row in enumerate(reader, start=2):
            if (row.get("split") or "").strip() != split:
                continue
            try:
                label = int(row["label"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid label on verification manifest row {row_number}"
                ) from error
            if label not in reservoirs:
                raise ValueError(
                    f"label must be 0 or 1 on verification manifest row {row_number}"
                )

            sample_id = (row.get("sample_id") or "").strip()
            input_1 = (row.get("input_1") or "").strip()
            input_2 = (row.get("input_2") or "").strip()
            if not sample_id or not input_1 or not input_2:
                raise ValueError(
                    f"empty verification field on manifest row {row_number}"
                )
            pair = VerificationPair(
                sample_id=sample_id,
                image_1=(manifest_path.parent / input_1).resolve(),
                image_2=(manifest_path.parent / input_2).resolve(),
                label=label,
            )
            seen[label] += 1
            reservoir = reservoirs[label]
            if len(reservoir) < quotas[label]:
                reservoir.append(pair)
            else:
                replacement_index = generator.randrange(seen[label])
                if replacement_index < quotas[label]:
                    reservoir[replacement_index] = pair

    for label, quota in quotas.items():
        if len(reservoirs[label]) != quota:
            raise ValueError(
                f"verification split {split!r} has only {seen[label]} samples with "
                f"label {label}. {quota} required"
            )
    pairs = reservoirs[0] + reservoirs[1]
    generator.shuffle(pairs)
    for pair in pairs:
        for path in (pair.image_1, pair.image_2):
            if not path.is_file():
                raise FileNotFoundError(
                    f"verification sample {pair.sample_id} is missing image: {path}"
                )
    return pairs


def _set_model_metadata(
    model: ct.models.MLModel,
    *,
    description: str,
    source_path: Path,
    contract: dict[str, str],
) -> None:
    model.author = "Mankai"
    model.short_description = description
    model.user_defined_metadata.update(
        {
            "source_artifact": source_path.name,
            "source_sha256": _sha256(source_path),
            **contract,
        }
    )


def _convert_encoder(
    encoder: torch.jit.ScriptModule,
    *,
    source_path: Path,
    target: Any,
    precision: Any,
) -> ct.models.MLModel:
    image_encoder = _CoreMLImageEncoder(encoder).eval()
    traced_image_encoder = torch.jit.trace(
        image_encoder,
        torch.zeros(IMAGE_SHAPE),
        strict=True,
    )
    model = ct.convert(
        traced_image_encoder,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[
            ct.ImageType(
                name="image",
                shape=IMAGE_SHAPE,
                scale=1.0 / 255.0,
                bias=[-value for value in IMAGENET_MEAN],
                color_layout=ct.colorlayout.RGB,
            ),
        ],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        minimum_deployment_target=target,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    _set_model_metadata(
        model,
        description="Encodes one raw 224x224 RGB image into a unit-length embedding.",
        source_path=source_path,
        contract={
            "input_normalization": "embedded exact ImageNet mean/std",
            "encoder_forward_output": "unit-length image embedding",
        },
    )
    model.input_description["image"] = (
        "Raw 224x224 RGB image. ImageNet normalization is embedded."
    )
    model.output_description["embedding"] = "Unit-length image embedding."
    return model


def _convert_classifier(
    classifier: torch.jit.ScriptModule,
    embedding_shape: tuple[int, ...],
    *,
    source_path: Path,
    target: Any,
    precision: Any,
) -> ct.models.MLModel:
    model = ct.convert(
        classifier,
        source="pytorch",
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="embedding_1", shape=embedding_shape, dtype=np.float32),
            ct.TensorType(name="embedding_2", shape=embedding_shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logit", dtype=np.float32)],
        minimum_deployment_target=target,
        compute_precision=precision,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    _set_model_metadata(
        model,
        description=(
            "Returns a raw training-distribution logit for two image embeddings."
        ),
        source_path=source_path,
        contract={
            "classifier_forward_output": "float training-distribution logit",
            "threshold_embedded": "false",
        },
    )
    model.input_description["embedding_1"] = "First unit-length image embedding."
    model.input_description["embedding_2"] = "Second unit-length image embedding."
    model.output_description["logit"] = (
        "Raw logit. Deployment-prior correction and thresholding are external."
    )
    return model


def _load_verification_image(
    path: Path, transform: Any
) -> tuple[torch.Tensor, Image.Image]:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    resized_image = transform.transforms[0](image)
    normalized_tensor: Any = resized_image
    for operation in transform.transforms[1:]:
        normalized_tensor = operation(normalized_tensor)
    return normalized_tensor.unsqueeze(0), resized_image


def _error_stats(actual: np.ndarray, expected: np.ndarray) -> ErrorStats:
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    return ErrorStats(
        mean_absolute=float(np.mean(np.abs(difference))),
        max_absolute=float(np.max(np.abs(difference))),
        root_mean_square=float(np.sqrt(np.mean(np.square(difference)))),
    )


def _assert_close(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    message: str,
) -> None:
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        err_msg=message,
    )


def _verify_conversion(
    encoder: torch.jit.ScriptModule,
    classifier: torch.jit.ScriptModule,
    coreml_encoder: ct.models.MLModel,
    coreml_classifier: ct.models.MLModel,
    pairs: Sequence[VerificationPair],
    *,
    precision_name: str,
) -> tuple[ErrorStats, ErrorStats, ErrorStats]:
    transform = build_transform(training=False)
    expected_embeddings: list[np.ndarray] = []
    actual_embeddings: list[np.ndarray] = []
    expected_logits: list[np.ndarray] = []
    actual_classifier_logits: list[np.ndarray] = []
    actual_pipeline_logits: list[np.ndarray] = []

    for pair in pairs:
        image_1, coreml_image_1 = _load_verification_image(pair.image_1, transform)
        image_2, coreml_image_2 = _load_verification_image(pair.image_2, transform)
        with torch.inference_mode():
            expected_embedding_1 = encoder(image_1).cpu().numpy()
            expected_embedding_2 = encoder(image_2).cpu().numpy()
            expected_logit = (
                classifier(
                    torch.from_numpy(expected_embedding_1),
                    torch.from_numpy(expected_embedding_2),
                )
                .cpu()
                .numpy()
            )

        actual_embedding_1 = coreml_encoder.predict({"image": coreml_image_1})[
            "embedding"
        ]
        actual_embedding_2 = coreml_encoder.predict({"image": coreml_image_2})[
            "embedding"
        ]
        actual_classifier_logit = coreml_classifier.predict(
            {
                "embedding_1": expected_embedding_1,
                "embedding_2": expected_embedding_2,
            }
        )["logit"]
        actual_pipeline_logit = coreml_classifier.predict(
            {
                "embedding_1": actual_embedding_1,
                "embedding_2": actual_embedding_2,
            }
        )["logit"]

        expected_embeddings.extend((expected_embedding_1, expected_embedding_2))
        actual_embeddings.extend((actual_embedding_1, actual_embedding_2))
        expected_logits.append(expected_logit)
        actual_classifier_logits.append(actual_classifier_logit)
        actual_pipeline_logits.append(actual_pipeline_logit)

    expected_embedding_array = np.concatenate(expected_embeddings, axis=0)
    actual_embedding_array = np.concatenate(actual_embeddings, axis=0)
    expected_logit_array = np.concatenate(expected_logits, axis=0)
    actual_classifier_logit_array = np.concatenate(actual_classifier_logits, axis=0)
    actual_pipeline_logit_array = np.concatenate(actual_pipeline_logits, axis=0)

    if precision_name == "float16":
        relative_tolerance, absolute_tolerance = 2e-2, 2e-2
    else:
        relative_tolerance, absolute_tolerance = 1e-4, 1e-5
    _assert_close(
        actual_embedding_array,
        expected_embedding_array,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        message="Core ML encoder output differs from TorchScript",
    )
    _assert_close(
        actual_classifier_logit_array,
        expected_logit_array,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        message="Core ML classifier output differs from TorchScript",
    )
    _assert_close(
        actual_pipeline_logit_array,
        expected_logit_array,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        message="Core ML end-to-end output differs from TorchScript",
    )
    return (
        _error_stats(actual_embedding_array, expected_embedding_array),
        _error_stats(actual_classifier_logit_array, expected_logit_array),
        _error_stats(actual_pipeline_logit_array, expected_logit_array),
    )


def _print_error_stats(name: str, stats: ErrorStats) -> None:
    print(
        f"[metrics] {name}: mean_absolute_error={stats.mean_absolute:.6g}, "
        f"max_abs={stats.max_absolute:.6g}, rmse={stats.root_mean_square:.6g}"
    )


def _replace_package(source: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"destination already exists: {destination}. Pass --overwrite"
            )
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    source.replace(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/dataset.csv"),
        help="pair manifest used for numerical verification",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="destination directory (default: MODEL_DIR)",
    )
    parser.add_argument(
        "--minimum-deployment-target",
        choices=DEPLOYMENT_TARGETS,
        default="iOS15",
    )
    parser.add_argument(
        "--precision",
        choices=("float16", "float32"),
        default="float16",
        help="Core ML internal compute precision (default: float16)",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="skip numerical comparison using the local Core ML runtime",
    )
    parser.add_argument(
        "--verification-samples",
        type=int,
        default=8,
        metavar="N",
        help="balanced pair samples used for verification (default: 8)",
    )
    parser.add_argument(
        "--verification-split",
        choices=VERIFICATION_SPLITS,
        default="test",
        help="manifest split sampled for verification (default: test)",
    )
    parser.add_argument(
        "--verification-seed",
        type=int,
        default=0,
        help="deterministic verification sampling seed (default: 0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing .mlpackage destinations",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = (args.output_dir or model_dir).expanduser().resolve()
    metadata = _load_metadata(model_dir)
    encoder_path = _source_model_path(model_dir, metadata, "encoder")
    classifier_path = _source_model_path(model_dir, metadata, "classifier")
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_destination = output_dir / ENCODER_PACKAGE_NAME
    classifier_destination = output_dir / CLASSIFIER_PACKAGE_NAME
    for destination in (encoder_destination, classifier_destination):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"destination already exists: {destination}. Pass --overwrite"
            )

    encoder = torch.jit.load(str(encoder_path), map_location="cpu").eval()
    classifier = torch.jit.load(str(classifier_path), map_location="cpu").eval()

    with torch.inference_mode():
        example_embedding = encoder(torch.zeros(IMAGE_SHAPE))
    if example_embedding.ndim != 2 or example_embedding.shape[0] != 1:
        raise ValueError(
            "encoder must return one rank-2 embedding for a batch-1 image. "
            f"received {tuple(example_embedding.shape)}"
        )

    target = getattr(ct.target, args.minimum_deployment_target)
    precision = (
        ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
    )
    print(f"[info] converting encoder: {encoder_path}")
    coreml_encoder = _convert_encoder(
        encoder,
        source_path=encoder_path,
        target=target,
        precision=precision,
    )
    print(f"[info] converting classifier: {classifier_path}")
    coreml_classifier = _convert_classifier(
        classifier,
        tuple(example_embedding.shape),
        source_path=classifier_path,
        target=target,
        precision=precision,
    )

    with tempfile.TemporaryDirectory(
        prefix=".coreml-conversion-", dir=output_dir
    ) as temporary_directory:
        temporary_dir = Path(temporary_directory)
        temporary_encoder = temporary_dir / ENCODER_PACKAGE_NAME
        temporary_classifier = temporary_dir / CLASSIFIER_PACKAGE_NAME
        coreml_encoder.save(str(temporary_encoder))
        coreml_classifier.save(str(temporary_classifier))

        if not args.skip_verification:
            verification_pairs = _sample_verification_pairs(
                args.manifest,
                split=args.verification_split,
                sample_count=args.verification_samples,
                seed=args.verification_seed,
            )
            saved_coreml_encoder = ct.models.MLModel(
                str(temporary_encoder), compute_units=ct.ComputeUnit.CPU_ONLY
            )
            saved_coreml_classifier = ct.models.MLModel(
                str(temporary_classifier), compute_units=ct.ComputeUnit.CPU_ONLY
            )
            encoder_error, classifier_error, pipeline_error = _verify_conversion(
                encoder,
                classifier,
                saved_coreml_encoder,
                saved_coreml_classifier,
                verification_pairs,
                precision_name=args.precision,
            )
            positive_count = sum(pair.label for pair in verification_pairs)
            print(
                f"[info] verified {len(verification_pairs)} {args.verification_split} "
                f"pairs ({positive_count} positive, "
                f"{len(verification_pairs) - positive_count} negative)"
            )
            _print_error_stats("encoder embeddings", encoder_error)
            _print_error_stats("classifier logits", classifier_error)
            _print_error_stats("end-to-end logits", pipeline_error)

        _replace_package(
            temporary_encoder, encoder_destination, overwrite=args.overwrite
        )
        _replace_package(
            temporary_classifier, classifier_destination, overwrite=args.overwrite
        )

    print(f"[info] exported encoder: {encoder_destination}")
    print(f"[info] exported classifier: {classifier_destination}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1) from error
