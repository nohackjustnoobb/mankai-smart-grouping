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


class SiameseNetworkWrapper(nn.Module):
    """
    Wrapper class to handle the dual-input structure for Core ML conversion.
    Core ML works better with models that have a single forward method.
    """

    def __init__(self, siamese_model: SiameseNetwork):
        super().__init__()
        self.siamese_model = siamese_model

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Forward pass with two image inputs."""
        return self.siamese_model(input1, input2)


def get_model_name_from_metrics(model_dir: str) -> Optional[str]:
    """
    Read the model name from metrics.json in the same directory as the model.

    Args:
        model_dir: Directory containing the model and metrics.json

    Returns:
        Model name from metrics.json, or None if not found
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


def load_pytorch_model(model_path: str, model_name: str = "resnet18") -> SiameseNetwork:
    """
    Load the PyTorch model from checkpoint.

    The SiameseNetwork class automatically detects the feature dimensions
    using its built-in auto-detection mechanism.

    Args:
        model_path: Path to the .pth file
        model_name: Name of the backbone model (e.g., 'resnet18', 'efficientnet_b0')

    Returns:
        Loaded SiameseNetwork model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}")
    print(f"Using backbone: {model_name}")

    # SiameseNetwork automatically detects feature dimensions
    model = SiameseNetwork(model_name=model_name)

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully!")
    return model


def convert_to_mlpackage(
    pytorch_model: SiameseNetwork,
    output_path: str,
    use_image_type: bool = False,
    compute_precision: Optional[str] = "float32",
) -> None:
    """
    Convert PyTorch model to MLPackage format.

    Args:
        pytorch_model: The PyTorch SiameseNetwork model
        output_path: Path where MLPackage should be saved
        use_image_type: Whether to use ImageType for inputs (expecting PIL images in app)
        compute_precision: Compute precision (float32 or float16)
    """
    print("Starting conversion to MLPackage...")

    # Wrap the model
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

        input_descriptions = [
            ct.ImageType(name="image1", shape=example_input1.shape, scale=scale, bias=bias),
            ct.ImageType(name="image2", shape=example_input2.shape, scale=scale, bias=bias),
        ]
        print("Using ImageType for inputs (expecting PIL images in app)")
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
    mlmodel.short_description = "Siamese Network for image similarity"
    mlmodel.version = "1.0.0"

    # Add input/output descriptions
    mlmodel.input_description["image1"] = "First image patch (left or reference)"
    mlmodel.input_description["image2"] = "Second image patch (right or comparison)"
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


def validate_mlpackage(mlpackage_path: str, pytorch_model: SiameseNetwork) -> None:
    """
    Validate the converted MLPackage by comparing outputs with PyTorch model.

    Args:
        mlpackage_path: Path to the MLPackage
        pytorch_model: Original PyTorch model for comparison
    """
    print("\nValidating converted model...")

    # Load the MLPackage
    mlmodel = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY)

    # Check if inputs are ImageType
    spec = mlmodel.get_spec()
    input_type_spec = spec.description.input[0].type.WhichOneof('Type')
    is_image_input = (input_type_spec == 'imageType')

    pytorch_model.eval()

    if is_image_input:
        print("Model expects image inputs. Generating random images...")
        height = int(spec.description.input[0].type.imageType.height)
        width = int(spec.description.input[0].type.imageType.width)

        # Generate random uint8 data
        img1_np = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img2_np = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Create PIL images for Core ML
        pil_img1 = Image.fromarray(img1_np)
        pil_img2 = Image.fromarray(img2_np)

        coreml_input = {"image1": pil_img1, "image2": pil_img2}

        # Create normalized tensors for PyTorch
        # Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).float()
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).float()

        # Convert to float tensor (0-1 range)
        input1_tensor = torch.from_numpy(img1_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        input2_tensor = torch.from_numpy(img2_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        # Normalize
        test_input1 = (input1_tensor - mean) / std
        test_input2 = (input2_tensor - mean) / std

    else:
        print("Model expects tensor inputs. Generating random tensors...")
        # Determine shape
        test_input1 = torch.randn(1, 3, 224, 224)
        test_input2 = torch.randn(1, 3, 224, 224)
        
        coreml_input = {"image1": test_input1.numpy(), "image2": test_input2.numpy()}

    # Get PyTorch prediction
    with torch.no_grad():
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


def find_model_files(directory: str) -> List[str]:
    """
    Recursively find all model.pth files under a directory.

    Args:
        directory: Root directory to search

    Returns:
        List of paths to model.pth files
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
) -> bool:
    """
    Convert a single model file to MLPackage.

    Args:
        model_path: Path to the .pth file
        output_path: Output path for MLPackage (optional)
        model_name: Backbone model name (optional)
        validate: Whether to validate the conversion

    Returns:
        True if conversion succeeded, False otherwise
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
        pytorch_model = load_pytorch_model(model_path, model_name)

        # Convert to MLPackage
        convert_to_mlpackage(
            pytorch_model, output_path, use_image_type, compute_precision
        )

        # Validate if requested
        if validate:
            validate_mlpackage(output_path, pytorch_model)

        print(f"\n✓ Successfully converted {model_path} to {output_path}")
        return True

    except Exception as e:
        print(f"\n✗ Conversion failed for {model_path}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch Siamese Network to MLPackage"
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

                # Convert each model (output_path is None, so it will be auto-generated)
                if convert_single_model(
                    model_file,
                    None,
                    args.model_name,
                    args.validate,
                    args.optimize,
                    "float16" if args.optimize else "float32",
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

            # Convert single model
            convert_single_model(
                args.model_path,
                args.output_path,
                args.model_name,
                args.validate,
                args.optimize,
                "float16" if args.optimize else "float32",
            )

    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
