"""Convert PyTorch models to Apple Core ML MLPackage format.

This script converts trained PyTorch adjacency detection models (either
Siamese network or merged classifier) to the Core ML MLPackage format for
deployment on Apple platforms (iOS/macOS). It supports both single-file and
recursive batch conversion, optional ImageType input optimization for iOS,
float16 precision, and post-conversion validation.

Usage:
    # Convert a single model
    python scripts/convert_to_mlpackage.py --model_path results/merged/model.pth

    # Recursively convert all models with iOS optimization
    python scripts/convert_to_mlpackage.py --model_path results/ -r --optimize --validate
"""

import argparse
import json
import os

# Import the SiameseNetwork from the source
import sys
from typing import List, Optional

import numpy as np
import coremltools as ct
import torch
import torch.nn as nn
from PIL import Image

sys.path.append("src")
from siamese_network import SiameseNetwork
from merged_classifier import MergedClassifier

# Supported model types
MODEL_TYPE_SIAMESE = "siamese"
MODEL_TYPE_MERGED = "merged"


class SiameseNetworkWrapper(nn.Module):
    """TorchScript-compatible wrapper for the SiameseNetwork model.

    Wraps the SiameseNetwork to provide a clean forward signature
    suitable for tracing with torch.jit.trace.

    Attributes:
        siamese_model: The underlying SiameseNetwork instance.
    """

    def __init__(self, siamese_model: SiameseNetwork):
        """Initialize the wrapper.

        Args:
            siamese_model: A trained SiameseNetwork instance.
        """
        super().__init__()
        self.siamese_model = siamese_model

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Siamese network.

        Args:
            input1: First image tensor of shape (1, 3, H, W).
            input2: Second image tensor of shape (1, 3, H, W).

        Returns:
            Adjacency probability tensor of shape (1, 1).
        """
        return self.siamese_model(input1, input2)


class MergedClassifierWrapper(nn.Module):
    """TorchScript-compatible wrapper for the MergedClassifier model.

    Wraps the MergedClassifier to provide a clean forward signature
    suitable for tracing with torch.jit.trace.

    Attributes:
        merged_model: The underlying MergedClassifier instance.
    """

    def __init__(self, merged_model: MergedClassifier):
        """Initialize the wrapper.

        Args:
            merged_model: A trained MergedClassifier instance.
        """
        super().__init__()
        self.merged_model = merged_model

    def forward(self, input1: torch.Tensor) -> torch.Tensor:
        """Forward pass through the merged classifier.

        Args:
            input1: Merged image tensor of shape (1, 3, H, W).

        Returns:
            Adjacency probability tensor of shape (1, 1).
        """
        return self.merged_model(input1)


def get_model_name_from_metrics(model_dir: str) -> Optional[str]:
    """Read the backbone model name from a metrics.json file.

    Args:
        model_dir: Directory containing the metrics.json file.

    Returns:
        The model name string if found, or None otherwise.
    """
    metrics_path = os.path.join(model_dir, "metrics.json")

    if not os.path.exists(metrics_path):
        print(f"Warning: metrics.json not found at {metrics_path}")
        return None

    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            model_name = metrics.get("parameters", {}).get("model")
            if model_name:
                print(f"Found model name in metrics.json: {model_name}")
                return model_name
            else:
                print("Warning: 'parameters.model' not found in metrics.json")
                return None
    except Exception as e:
        print(f"Warning: Failed to read metrics.json: {e}")
        return None


def load_pytorch_model(
    model_path: str,
    model_name: str = "resnet18",
    model_type: str = MODEL_TYPE_SIAMESE,
) -> nn.Module:
    """Load a trained PyTorch model from a .pth file.

    Args:
        model_path: Path to the saved model weights (.pth file).
        model_name: Name of the timm backbone architecture.
        model_type: Either 'siamese' or 'merged' to determine the model class.

    Returns:
        The loaded model in eval mode.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}")
    print(f"Using backbone: {model_name}")
    print(f"Model type: {model_type}")

    if model_type == MODEL_TYPE_MERGED:
        model = MergedClassifier(model_name=model_name)
    else:
        model = SiameseNetwork(model_name=model_name)

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully!")
    return model


def convert_to_mlpackage(
    pytorch_model: nn.Module,
    output_path: str,
    use_image_type: bool = False,
    compute_precision: Optional[str] = "float32",
    model_type: str = MODEL_TYPE_SIAMESE,
) -> None:
    """Convert a PyTorch model to Apple Core ML MLPackage format.

    Traces the model, converts it using coremltools, adds metadata, and
    saves the result as an MLPackage.

    Args:
        pytorch_model: The trained PyTorch model to convert.
        output_path: File path for the output .mlpackage.
        use_image_type: If True, use ct.ImageType inputs (optimized for iOS
            apps that pass PIL images directly).
        compute_precision: Either 'float32' or 'float16' for the Core ML
            compute precision.
        model_type: Either 'siamese' or 'merged' to determine input/output
            configuration.

    Returns:
        The converted coremltools MLModel.
    """
    print("Starting conversion to MLPackage...")
    is_merged = model_type == MODEL_TYPE_MERGED

    # Wrap the model
    if is_merged:
        wrapped_model = MergedClassifierWrapper(pytorch_model)
    else:
        wrapped_model = SiameseNetworkWrapper(pytorch_model)
    wrapped_model.eval()

    # Auto-detect input shape from the model's attributes
    input_shape = (
        pytorch_model.input_channels,
        pytorch_model.input_size[0],
        pytorch_model.input_size[1],
    )
    print(f"Auto-detected input shape: {input_shape}")

    # Create example inputs
    batch_size = 1
    example_input1 = torch.randn(batch_size, *input_shape)

    if is_merged:
        print(f"Input shape: {example_input1.shape}")

        # Test the model works
        with torch.no_grad():
            test_output = wrapped_model(example_input1)
            print(f"Test output shape: {test_output.shape}")
            print(f"Test output value: {test_output.item():.4f}")

        # Trace the model
        print("Tracing the model...")
        traced_model = torch.jit.trace(wrapped_model, (example_input1,))
    else:
        example_input2 = torch.randn(batch_size, *input_shape)
        print(f"Input shapes: {example_input1.shape}, {example_input2.shape}")

        # Test the model works
        with torch.no_grad():
            test_output = wrapped_model(example_input1, example_input2)
            print(f"Test output shape: {test_output.shape}")
            print(f"Test output value: {test_output.item():.4f}")

        # Trace the model
        print("Tracing the model...")
        traced_model = torch.jit.trace(wrapped_model, (example_input1, example_input2))

    # Convert to Core ML
    print("Converting to Core ML format...")

    # Define input descriptions
    if use_image_type:
        avg_std = 0.226
        scale = 1.0 / (avg_std * 255.0)

        # Per-channel bias
        bias = [
            -0.485 / 0.229,
            -0.456 / 0.224,
            -0.406 / 0.225,
        ]

        if is_merged:
            input_descriptions = [
                ct.ImageType(
                    name="image", shape=example_input1.shape, scale=scale, bias=bias
                ),
            ]
        else:
            input_descriptions = [
                ct.ImageType(
                    name="image1", shape=example_input1.shape, scale=scale, bias=bias
                ),
                ct.ImageType(
                    name="image2", shape=example_input2.shape, scale=scale, bias=bias
                ),
            ]
        print("Using ImageType for inputs (expecting PIL images in app)")
    else:
        if is_merged:
            input_descriptions = [
                ct.TensorType(name="image", shape=example_input1.shape),
            ]
        else:
            input_descriptions = [
                ct.TensorType(name="image1", shape=example_input1.shape),
                ct.TensorType(name="image2", shape=example_input2.shape),
            ]

    # Define output description
    output_descriptions = [ct.TensorType(name="adjacency_score")]

    # Set compute precision
    ct_precision = (
        ct.precision.FLOAT32 if compute_precision == "float32" else ct.precision.FLOAT16
    )
    print(f"Using compute precision: {compute_precision}")

    # Convert the model
    mlmodel = ct.convert(
        traced_model,
        inputs=input_descriptions,
        outputs=output_descriptions,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        compute_precision=ct_precision,
    )

    # Add metadata
    mlmodel.author = "Mankai"
    mlmodel.license = "MIT"
    if is_merged:
        mlmodel.short_description = "Merged Classifier for image adjacency detection"
        mlmodel.input_description["image"] = (
            "Merged image patch (two patches side-by-side)"
        )
    else:
        mlmodel.short_description = "Siamese Network for image similarity"
        mlmodel.input_description["image1"] = "First image patch (left or reference)"
        mlmodel.input_description["image2"] = "Second image patch (right or comparison)"
    mlmodel.version = "1.0.0"

    # Add output description
    mlmodel.output_description["adjacency_score"] = (
        "Probability that the two patches are adjacent (0.0-1.0)"
    )

    # Save the model
    print(f"Saving MLPackage to {output_path}")
    mlmodel.save(output_path)
    print("Conversion completed successfully!")

    # Print model information
    print("\nModel Information:")
    print(f"Model type: {type(mlmodel)}")
    spec = mlmodel.get_spec()
    input_names = [inp.name for inp in spec.description.input]
    output_names = [out.name for out in spec.description.output]
    print(f"Inputs: {input_names}")
    print(f"Outputs: {output_names}")

    return mlmodel


def validate_mlpackage(
    mlpackage_path: str,
    pytorch_model: nn.Module,
    model_type: str = MODEL_TYPE_SIAMESE,
) -> None:
    """Validate a converted MLPackage by comparing its output to PyTorch.

    Generates random input(s) and feeds them through both the PyTorch model
    and the Core ML model, then reports the numerical difference.

    Args:
        mlpackage_path: Path to the saved .mlpackage file.
        pytorch_model: The original PyTorch model for comparison.
        model_type: Either 'siamese' or 'merged'.

    Returns:
        The absolute difference between the PyTorch and Core ML outputs.
    """
    print("\nValidating converted model...")
    is_merged = model_type == MODEL_TYPE_MERGED

    # Load the MLPackage
    mlmodel = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY)

    # Check if inputs are ImageType
    spec = mlmodel.get_spec()
    input_type_spec = spec.description.input[0].type.WhichOneof("Type")
    is_image_input = input_type_spec == "imageType"

    pytorch_model.eval()

    if is_image_input:
        print("Model expects image inputs. Generating random images...")
        height = int(spec.description.input[0].type.imageType.height)
        width = int(spec.description.input[0].type.imageType.width)

        # Standard ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).float()
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).float()

        if is_merged:
            img_np = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            pil_img = Image.fromarray(img_np)
            coreml_input = {"image": pil_img}

            input_tensor = (
                torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            )
            test_input1 = (input_tensor - mean) / std
        else:
            img1_np = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            img2_np = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            pil_img1 = Image.fromarray(img1_np)
            pil_img2 = Image.fromarray(img2_np)
            coreml_input = {"image1": pil_img1, "image2": pil_img2}

            input1_tensor = (
                torch.from_numpy(img1_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            )
            input2_tensor = (
                torch.from_numpy(img2_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            )
            test_input1 = (input1_tensor - mean) / std
            test_input2 = (input2_tensor - mean) / std

    else:
        print("Model expects tensor inputs. Generating random tensors...")
        test_input1 = torch.randn(1, 3, 224, 224)

        if is_merged:
            coreml_input = {"image": test_input1.numpy()}
        else:
            test_input2 = torch.randn(1, 3, 224, 224)
            coreml_input = {
                "image1": test_input1.numpy(),
                "image2": test_input2.numpy(),
            }

    # Get PyTorch prediction
    with torch.no_grad():
        if is_merged:
            pytorch_output = pytorch_model(test_input1)
        else:
            pytorch_output = pytorch_model(test_input1, test_input2)
        pytorch_score = pytorch_output.item()

    # Get Core ML prediction
    coreml_output = mlmodel.predict(coreml_input)

    # Extract score from output dictionary
    if "adjacency_score" in coreml_output:
        coreml_score_raw = coreml_output["adjacency_score"]
        # Handle scalar (0-d array) or 1-element array
        if hasattr(coreml_score_raw, "item"):
            coreml_score = coreml_score_raw.item()
        elif isinstance(coreml_score_raw, list):
            coreml_score = coreml_score_raw[0]
        else:
            coreml_score = float(coreml_score_raw)
    else:
        # Fallback if output name is different (unlikely given conversion code)
        coreml_score = list(coreml_output.values())[0]

    # Compare results
    difference = abs(pytorch_score - coreml_score)
    print(f"PyTorch output: {pytorch_score:.6f}")
    print(f"Core ML output: {coreml_score:.6f}")
    print(f"Difference: {difference:.6f}")

    if difference < 1e-3:
        print("Validation passed! Outputs are very similar.")
    elif difference < 1e-2:
        print("Validation warning: Small difference detected.")
    else:
        print("Validation failed: Large difference detected.")

    return difference


def detect_model_type_from_path(path: str) -> Optional[str]:
    """Auto-detect the model type from the file path.

    Checks if the path contains 'merged' or 'siamese' as a directory component.

    Args:
        path: File path to analyze.

    Returns:
        'merged', 'siamese', or None if neither is detected.
    """
    normalized = os.path.normpath(path).lower()
    parts = normalized.split(os.sep)

    if MODEL_TYPE_MERGED in parts:
        return MODEL_TYPE_MERGED
    if MODEL_TYPE_SIAMESE in parts:
        return MODEL_TYPE_SIAMESE

    return None


def find_model_files(directory: str) -> List[str]:
    """Recursively find all model.pth files under a directory.

    Args:
        directory: Root directory to search.

    Returns:
        List of absolute paths to model.pth files.
    """
    model_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "model.pth":
                model_path = os.path.join(root, file)
                model_files.append(model_path)
    return model_files


def convert_single_model(
    model_path: str,
    output_path: Optional[str],
    model_name: Optional[str],
    validate: bool,
    use_image_type: bool,
    compute_precision: str,
    model_type: str = MODEL_TYPE_SIAMESE,
) -> bool:
    """Convert a single PyTorch model to MLPackage.

    Handles loading, conversion, and optional validation for one model file.
    If output_path or model_name are not provided, they are inferred from
    the model directory and its metrics.json.

    Args:
        model_path: Path to the .pth model file.
        output_path: Output path for the .mlpackage, or None to auto-generate.
        model_name: Backbone model name, or None to read from metrics.json.
        validate: Whether to validate the converted model against PyTorch.
        use_image_type: Whether to use ImageType inputs for iOS optimization.
        compute_precision: 'float32' or 'float16'.
        model_type: Either 'siamese' or 'merged'.

    Returns:
        True if conversion succeeded, False otherwise.
    """
    try:
        # Set default output path if not provided
        if output_path is None:
            model_dir = os.path.dirname(model_path) or "."
            filename_suffix = "_optimized" if use_image_type else ""
            output_path = os.path.join(model_dir, f"model{filename_suffix}.mlpackage")

        # Determine model name: use provided argument or read from metrics.json
        if model_name is None:
            model_dir = os.path.dirname(model_path) or "."
            model_name = get_model_name_from_metrics(model_dir)

            if model_name is None:
                model_name = "resnet18"
                print(f"Using default model name: {model_name}")

        # Load PyTorch model
        pytorch_model = load_pytorch_model(model_path, model_name, model_type)

        # Convert to MLPackage
        convert_to_mlpackage(
            pytorch_model, output_path, use_image_type, compute_precision, model_type
        )

        # Validate if requested
        if validate:
            validate_mlpackage(output_path, pytorch_model, model_type)

        print(f"\n✓ Successfully converted {model_path} to {output_path}")
        return True

    except Exception as e:
        print(f"\n✗ Conversion failed for {model_path}: {str(e)}")
        return False


def main():
    """Entry point for the MLPackage conversion CLI.

    Parses command-line arguments and runs either a single model conversion
    or a recursive batch conversion of all model.pth files under a directory.
    Supports auto-detection of model type from path and optional iOS
    optimization (ImageType inputs + float16 precision).
    """
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model (Siamese Network or Merged Classifier) to MLPackage"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/model.pth",
        help="Path to PyTorch model (.pth file) or directory containing model.pth files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for MLPackage (defaults to model.mlpackage in same directory as input). Only used for single file conversion.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Backbone model name (resnet18, efficientnet_b0, mobilenetv3_large_100, etc.). If not specified, will try to read from metrics.json",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=[MODEL_TYPE_SIAMESE, MODEL_TYPE_MERGED],
        help="Type of model to convert: 'siamese' (dual-input) or 'merged' (single-input). If not specified, auto-detected from path (e.g. results/siamese/, results/merged/). Defaults to 'siamese' if detection fails.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate converted model by comparing with PyTorch output (forces CPU execution)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize for iOS (uses ImageType for inputs and float16 precision)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        default=False,
        help="Recursively convert all model.pth files under the specified directory",
    )

    args = parser.parse_args()

    try:
        # Check if recursive mode is enabled
        if args.recursive:
            # Check if model_path is a directory
            if not os.path.isdir(args.model_path):
                print(
                    f"Error: --recursive requires a directory, but {args.model_path} is not a directory"
                )
                return

            # Find all model.pth files
            print(f"Searching for model.pth files under {args.model_path}...")
            model_files = find_model_files(args.model_path)

            if not model_files:
                print(f"No model.pth files found under {args.model_path}")
                return

            print(f"Found {len(model_files)} model.pth file(s):")
            for model_file in model_files:
                print(f"  - {model_file}")

            print("\nStarting batch conversion...")
            print("=" * 80)

            successful = 0
            failed = 0

            for i, model_file in enumerate(model_files, 1):
                print(f"\n[{i}/{len(model_files)}] Converting {model_file}")
                print("-" * 80)

                # Determine model type: use CLI arg, auto-detect from path, or default
                effective_model_type = args.model_type
                if effective_model_type is None:
                    effective_model_type = detect_model_type_from_path(model_file)
                    if effective_model_type:
                        print(
                            f"Auto-detected model type from path: {effective_model_type}"
                        )
                    else:
                        effective_model_type = MODEL_TYPE_SIAMESE
                        print(
                            f"Could not detect model type from path, defaulting to: {effective_model_type}"
                        )

                # Convert each model (output_path is None, so it will be auto-generated)
                if convert_single_model(
                    model_file,
                    None,
                    args.model_name,
                    args.validate,
                    args.optimize,
                    "float16" if args.optimize else "float32",
                    effective_model_type,
                ):
                    successful += 1
                else:
                    failed += 1

            print("\n" + "=" * 80)
            print("Batch conversion completed:")
            print(f"  ✓ Successful: {successful}")
            print(f"  ✗ Failed: {failed}")
            print(f"  Total: {len(model_files)}")

        else:
            # Single file conversion mode
            if os.path.isdir(args.model_path):
                print(
                    f"Error: {args.model_path} is a directory. Use --recursive to convert all model.pth files in it."
                )
                return

            if not os.path.exists(args.model_path):
                print(f"Error: Model file not found: {args.model_path}")
                return

            # Determine model type: use CLI arg, auto-detect from path, or default
            effective_model_type = args.model_type
            if effective_model_type is None:
                effective_model_type = detect_model_type_from_path(args.model_path)
                if effective_model_type:
                    print(f"Auto-detected model type from path: {effective_model_type}")
                else:
                    effective_model_type = MODEL_TYPE_SIAMESE
                    print(
                        f"Could not detect model type from path, defaulting to: {effective_model_type}"
                    )

            # Convert single model
            convert_single_model(
                args.model_path,
                args.output_path,
                args.model_name,
                args.validate,
                args.optimize,
                "float16" if args.optimize else "float32",
                effective_model_type,
            )

    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
