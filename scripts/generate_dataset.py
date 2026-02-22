import argparse
import csv
import random
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    ll_filename = f"{base_name}_ll.webp"
    lr_filename = f"{base_name}_lr.webp"
    rl_filename = f"{base_name}_rl.webp"
    rr_filename = f"{base_name}_rr.webp"

    patch_ll.save(output_dir / ll_filename, "WEBP")
    patch_lr.save(output_dir / lr_filename, "WEBP")
    patch_rl.save(output_dir / rl_filename, "WEBP")
    patch_rr.save(output_dir / rr_filename, "WEBP")

    return {
        "ll": ll_filename,
        "lr": lr_filename,
        "rl": rl_filename,
        "rr": rr_filename,
    }


def process_image_batch(
    image_paths_with_indices: List[tuple[Path, int]],
    output_dir: Path,
    patch_size: int,
    worker_id: int,
) -> List[Dict[str, str]]:
    """
    Process a batch of images.

    Args:
        image_paths_with_indices: List of tuples (image_path, image_index)
        output_dir: Directory to save the patches
        patch_size: Size of the output patches
        worker_id: ID of the worker thread for progress bar positioning

    Returns:
        List of patch dictionaries for successfully processed images
    """
    results = []

    # Create a progress bar for this worker
    with tqdm(
        total=len(image_paths_with_indices),
        desc=f"Worker {worker_id}",
        position=worker_id,
        leave=True,
        unit="img",
    ) as pbar:
        for image_path, image_index in image_paths_with_indices:
            patches = process_image(image_path, output_dir, image_index, patch_size)
            if patches:
                results.append(patches)
            pbar.update(1)

    return results


def generate_dataset(
    input_dir: str,
    output_dir: str,
    csv_path: str,
    patch_size: int = 224,
    max_images: int = None,
    num_workers: int = 4,
):
    """
    Generate dataset from images directory.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save the patches
        csv_path: Path to save the CSV file with pairs
        patch_size: Size of the output patches (default 224)
        max_images: Maximum number of images to process (None for all)
        num_workers: Number of parallel workers for processing (default 4)
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
    print(f"Using {num_workers} workers for parallel processing")

    # Split images into chunks for each worker
    chunk_size = (len(image_files) + num_workers - 1) // num_workers
    image_chunks = []
    for i in range(0, len(image_files), chunk_size):
        chunk = [
            (image_files[j], j) for j in range(i, min(i + chunk_size, len(image_files)))
        ]
        image_chunks.append(chunk)

    # Process all images and collect patches using multithreading
    all_patches = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit one task per worker with unique worker ID
        futures = [
            executor.submit(process_image_batch, chunk, output_path, patch_size, i)
            for i, chunk in enumerate(image_chunks)
        ]

        # Collect results as they complete
        for future in as_completed(futures):
            batch_results = future.result()
            all_patches.extend(batch_results)

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
        j = random.randint(0, len(all_patches) - 2)
        if j >= i:
            j += 1
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


def _merge_pair(
    args: tuple,
) -> tuple[int, str | None, str | None]:
    """
    Merge a single pair of images into one composite image.

    Returns:
        (index, merged_filename, label) on success, or (index, None, error_msg) on failure.
    """
    i, row, images_path, output_path = args
    left_path = images_path / row["left_image"]
    right_path = images_path / row["right_image"]

    if not left_path.exists():
        return (i, None, f"Missing left image: {left_path}")
    if not right_path.exists():
        return (i, None, f"Missing right image: {right_path}")

    try:
        left_img = (
            Image.open(left_path)
            .convert("RGB")
            .resize((224, 224), Image.Resampling.LANCZOS)
        )
        right_img = (
            Image.open(right_path)
            .convert("RGB")
            .resize((224, 224), Image.Resampling.LANCZOS)
        )
    except Exception as e:
        return (i, None, f"Error loading pair {i}: {e}")

    # Compose a 448x224 image and scale to 224x224
    composite = Image.new("RGB", (448, 224))
    composite.paste(left_img, (0, 0))
    composite.paste(right_img, (224, 0))
    merged = composite.resize((224, 224), Image.Resampling.LANCZOS)

    merged_filename = f"{i:06d}_merged.webp"
    merged.save(output_path / merged_filename, "WEBP")
    return (i, merged_filename, row["label"])


def merge_pairs(
    images_dir: str,
    csv_path: str,
    merged_output_dir: str,
    num_workers: int,
    merged_csv_path: str,
):
    images_path = Path(images_dir)
    csv_path = Path(csv_path)
    output_path = Path(merged_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Merging {len(rows)} pairs from {csv_path} ...")
    print(f"Using {num_workers} workers for parallel processing")

    task_args = [(i, row, images_path, output_path) for i, row in enumerate(rows)]

    results: list[tuple[int, str | None, str | None]] = [None] * len(rows)
    skipped = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_merge_pair, arg): arg[0] for arg in task_args}
        with tqdm(total=len(rows), desc="Merging pairs", unit="pair") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results[result[0]] = result
                pbar.update(1)

    # Collect results in original order, printing errors as we go
    merged_rows = []
    for i, merged_filename, label_or_error in results:
        if merged_filename is None:
            print(label_or_error)
            skipped += 1
        else:
            merged_rows.append([merged_filename, label_or_error])

    # Write a companion CSV for the merged images
    merged_csv_path = Path(merged_csv_path)
    merged_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        writer.writerows(merged_rows)

    print("\nMerge complete!")
    print(f"  Merged images saved to: {output_path}")
    print(f"  Merged CSV saved to: {merged_csv_path}")
    print(f"  Pairs merged: {len(merged_rows)}")
    if skipped:
        print(f"  Pairs skipped: {skipped}")


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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of parallel workers for processing (default: number of CPU cores)",
    )
    parser.add_argument(
        "--merge-pairs",
        action="store_true",
        default=False,
        help="After dataset generation, merge each pair from the CSV into a single 224x224 image",
    )
    parser.add_argument(
        "--merged-output-dir",
        type=str,
        default="./merged_images",
        help="Directory to save the merged pair images (default: ./merged_images)",
    )
    parser.add_argument(
        "--merge-csv-path",
        type=str,
        default="./merged_pairs.csv",
        help="Path for the output CSV of merged pairs (default: <merged-output-dir>/merged_pairs.csv)",
    )

    args = parser.parse_args()

    if args.merge_pairs:
        merge_pairs(
            images_dir=args.output_dir,
            csv_path=args.csv_path,
            merged_output_dir=args.merged_output_dir,
            num_workers=args.num_workers,
            merged_csv_path=args.merge_csv_path,
        )
    else:
        generate_dataset(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            csv_path=args.csv_path,
            patch_size=args.patch_size,
            max_images=args.max_images,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
