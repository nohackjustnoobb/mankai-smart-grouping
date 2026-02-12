import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm


def get_image_files(directory: str) -> List[Path]:
    """Get all image files from the directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    image_files = []
    for ext in image_extensions:
        image_files.extend(directory_path.glob(f"*{ext}"))
        image_files.extend(directory_path.glob(f"*{ext.upper()}"))

    return sorted(image_files)


def extract_patch(
    image: Image.Image,
    side: str,
    part: str,
    patch_width: int = 224,
    patch_height: int = 224,
) -> Image.Image:
    """
    Extract a patch from the image.

    Args:
        image: PIL Image
        side: 'left' or 'right' (which half of the image)
        part: 'left' or 'right' (which edge of that half)
        patch_width: Width of the output patch (default 224)
        patch_height: Height of the output patch (default 224)

    Returns:
        PIL Image of size (patch_width, patch_height)
    """
    width, height = image.size
    mid_x = width // 2

    # First, split the image into left and right halves
    if side == "left":
        half = image.crop((0, 0, mid_x, height))
    else:  # right
        half = image.crop((mid_x, 0, width, height))

    half_width, half_height = half.size

    # Extract the patch from the specified edge
    if part == "left":
        # Take from the left edge
        x_start = 0
        x_end = min(patch_width, half_width)
    else:  # right
        # Take from the right edge
        x_start = max(0, half_width - patch_width)
        x_end = half_width

    # Crop the patch (full height, specified width range)
    patch = half.crop((x_start, 0, x_end, half_height))

    # Scale height to patch_height, keep width as is (crop)
    # If patch width is less than patch_width, we'll resize to fit
    if patch.width < patch_width:
        # If the image is too narrow, resize to fit
        patch = patch.resize((patch_width, patch_height), Image.Resampling.LANCZOS)
    else:
        # Scale height, crop width is already done
        patch = patch.resize((patch_width, patch_height), Image.Resampling.LANCZOS)

    return patch


def process_image(
    image_path: Path, output_dir: Path, image_index: int, patch_size: int = 224
) -> Dict[str, str]:
    """
    Process a single image and extract patches.

    Args:
        image_path: Path to the input image
        output_dir: Directory to save the patches
        image_index: Index for naming the output files
        patch_size: Size of the output patches (default 224)

    Returns:
        Dictionary with patch filenames: {'lr': filename, 'rl': filename, 'll': filename, 'rr': filename}
        Returns empty dict if processing fails
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return {}

    # Check if image is wide enough
    width, height = image.size
    if width < patch_size * 2:
        print(
            f"Skipping {image_path}: image too narrow (width={width}, need at least {patch_size * 2})"
        )
        return {}

    # Extract all four patches
    # Left half, left edge
    patch_ll = extract_patch(image, "left", "left", patch_size, patch_size)
    # Left half, right edge
    patch_lr = extract_patch(image, "left", "right", patch_size, patch_size)
    # Right half, left edge
    patch_rl = extract_patch(image, "right", "left", patch_size, patch_size)
    # Right half, right edge
    patch_rr = extract_patch(image, "right", "right", patch_size, patch_size)

    # Save patches
    base_name = f"{image_index:06d}"
    ll_filename = f"{base_name}_ll.jpg"
    lr_filename = f"{base_name}_lr.jpg"
    rl_filename = f"{base_name}_rl.jpg"
    rr_filename = f"{base_name}_rr.jpg"

    patch_ll.save(output_dir / ll_filename, quality=95)
    patch_lr.save(output_dir / lr_filename, quality=95)
    patch_rl.save(output_dir / rl_filename, quality=95)
    patch_rr.save(output_dir / rr_filename, quality=95)

    return {
        "ll": ll_filename,
        "lr": lr_filename,
        "rl": rl_filename,
        "rr": rr_filename,
    }


def generate_dataset(
    input_dir: str,
    output_dir: str,
    csv_path: str,
    patch_size: int = 224,
    max_images: int = None,
):
    """
    Generate dataset from images directory.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save the patches
        csv_path: Path to save the CSV file with pairs
        patch_size: Size of the output patches (default 224)
        max_images: Maximum number of images to process (None for all)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = get_image_files(input_dir)

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    if max_images:
        image_files = image_files[:max_images]

    print(f"Found {len(image_files)} images to process")

    # Process all images and collect patches
    all_patches = []
    for idx, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        patches = process_image(image_path, output_path, idx, patch_size)
        if patches:  # Only add if processing was successful
            all_patches.append(patches)

    if len(all_patches) < 2:
        print(
            f"\nError: Need at least 2 images to create false pairs. Got {len(all_patches)}"
        )
        return

    print(f"\nSuccessfully processed {len(all_patches)} images")
    print("Creating pairs...")

    # Create TRUE pairs: right edge of left half + left edge of right half (adjacent)
    true_pairs = []
    for patches in all_patches:
        true_pairs.append((patches["lr"], patches["rl"], 1))

    # Create FALSE pairs: pair patches from different images
    false_pairs = []
    random.seed(42)  # For reproducibility

    # For each image, create a false pair with a random different image
    for i, patches_i in enumerate(all_patches):
        # Choose a random different image
        j = random.choice([x for x in range(len(all_patches)) if x != i])
        patches_j = all_patches[j]

        # Pair right edge of image A with left edge of image B
        false_pairs.append((patches_i["rr"], patches_j["ll"], 0))

    # Combine all pairs
    all_pairs = true_pairs + false_pairs

    # Write CSV file
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["left_image", "right_image", "label"])
        for left, right, label in all_pairs:
            writer.writerow([left, right, label])

    print("\nDataset generation complete!")
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  True pairs: {len(true_pairs)}")
    print(f"  False pairs: {len(false_pairs)}")
    print(f"  Patches saved to: {output_path}")
    print(f"  CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset by splitting images and creating matching/non-matching pairs"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./raw_images",
        help="Directory containing input images (default: ./raw_images)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./images",
        help="Directory to save the patches (default: ./images)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="./pairs.csv",
        help="Path to save the CSV file with pairs (default: ./pairs.csv)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=224,
        help="Size of the output patches (default: 224)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all)",
    )

    args = parser.parse_args()

    generate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        csv_path=args.csv_path,
        patch_size=args.patch_size,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
