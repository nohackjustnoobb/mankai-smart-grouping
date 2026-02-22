> **Note:** This project is mostly vibe-coded. I have no idea what I'm doing, but it works (kind of).

# Mankai Smart Grouping

**Mankai Smart Grouping** is an intelligent tool designed to automatically group split image patches using deep learning. It uses binary classifiers to determine if two image patches belong side-by-side, enabling seamless reconstruction of spread pages.

Two model architectures are supported:

- **Siamese Network** — Takes two separate 224×224 image patches as input and compares their feature embeddings.
- **Merged Classifier** — Takes a single 224×224 merged image (two patches placed side-by-side) as input, simplifying the inference pipeline to a single forward pass.

## Installation

### Prerequisites

- [uv](https://github.com/astral-sh/uv)

### Steps

1.  Clone the repository:

    ```bash
    git clone https://github.com/nohackjustnoobb/mankai-smart-grouping.git
    cd mankai-smart-grouping
    ```

2.  Install dependencies:
    ```bash
    uv sync
    ```

## Dataset Structure

The dataset uses a CSV-based approach for organizing image pairs. The `generate_dataset.py` script can produce datasets for both model types.

### Siamese Network Dataset

#### Directory Layout

```
project/
├── images/
│   ├── pair_001_left.jpg
│   ├── pair_001_right.jpg
│   ├── pair_002_left.jpg
│   ├── pair_002_right.jpg
│   ├── pair_003_left.jpg
│   ├── pair_003_right.jpg
│   └── ...
└── pairs.csv
```

#### CSV Format

Create a `pairs.csv` file with the following structure:

```csv
left_image,right_image,label
pair_001_left.jpg,pair_001_right.jpg,1
pair_002_left.jpg,pair_002_right.jpg,0
pair_003_left.jpg,pair_003_right.jpg,1
```

**Columns:**

- `left_image`: Filename of the left patch (relative to the images directory)
- `right_image`: Filename of the right patch (relative to the images directory)
- `label`:
  - `1` = Adjacent (patches from the same source image, side-by-side)
  - `0` = Non-adjacent (patches from different images or not adjacent)

### Merged Classifier Dataset

#### Directory Layout

```
project/
├── merged_images/
│   ├── 000000_merged.webp
│   ├── 000001_merged.webp
│   ├── 000002_merged.webp
│   └── ...
└── merged_pairs.csv
```

#### CSV Format

Create a `merged_pairs.csv` file with the following structure:

```csv
image,label
000000_merged.webp,1
000001_merged.webp,0
000002_merged.webp,1
```

**Columns:**

- `image`: Filename of the merged image (two patches placed side-by-side, resized to 224×224)
- `label`:
  - `1` = Adjacent
  - `0` = Non-adjacent

### Image Requirements

- **Resolution**: All images must be pre-processed to **224×224 pixels**
- **Format**: JPG, PNG, or WebP
- **Color**: RGB color images

## Usage

### Training

Ensure your dataset is structured according to the [Dataset Structure](#dataset-structure) section above.

#### Siamese Network

```bash
uv run src/siamese_network.py --images ./images --csv ./pairs.csv --epochs 10 --model resnet18
```

**Arguments:**

- `--images`: Path to the directory containing images (default: `./images`).
- `--csv`: Path to the CSV file with image pairs (default: `./pairs.csv`).
- `--model`: Model backbone to use (e.g., `resnet18`, `efficientnet_b0`, `mobilenetv3_large_100`).
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size (default: `32`).
- `--learning_rate`: Learning rate (default: `0.001`).
- `--patience`: Patience for early stopping (default: `10`).
- `--dev`: Run in development mode with a smaller dataset for quick testing.

#### Merged Classifier

```bash
uv run src/merged_classifier.py --images ./merged_images --csv ./merged_pairs.csv --epochs 10 --model resnet18
```

**Arguments:**

- `--images`: Path to the directory containing merged images (default: `./merged_images`).
- `--csv`: Path to the CSV file with merged image labels (default: `./merged_pairs.csv`).
- `--results`: Path to results directory (default: `./results/merged`).
- `--model`: Model backbone to use (e.g., `resnet18`, `efficientnet_b0`, `mobilenetv3_large_100`).
- `--epochs`: Number of training epochs.
- `--batch_size`: Batch size (default: `32`).
- `--learning_rate`: Learning rate (default: `0.001`).
- `--patience`: Patience for early stopping (default: `10`).
- `--dev`: Run in development mode with a smaller dataset for quick testing.

### Converting to Core ML

For deployment on iOS, convert your trained PyTorch model to the Core ML format (`.mlpackage`) using the conversion script. The script supports both model types.

#### Siamese Network

```bash
uv run scripts/convert_to_mlpackage.py --model_path results/model.pth --model_type siamese --optimize --validate
```

#### Merged Classifier

```bash
uv run scripts/convert_to_mlpackage.py --model_path results/merged/model.pth --model_type merged --optimize --validate
```

**Key Arguments:**

- `--model_path`: Path to the trained PyTorch model (`.pth`).
- `--model_type`: Model type — `siamese` (default) or `merged`.
- `--optimize`: Optimizes the model for iOS (uses `ImageType` inputs and `float16` precision).
- `--validate`: Validates the converted model by comparing outputs with the original PyTorch model.

## Final Models

The final production models are located in the `models` directory, organized by type.

### Siamese Network

Located in `models/siamese/`. A **MobileNetV3 Large 100** model chosen for its superior balance of speed and accuracy.

- **PyTorch Model:** [`models/siamese/model.pth`](models/siamese/model.pth)
- **Core ML Model:** [`models/siamese/model.mlpackage`](models/siamese/model.mlpackage) / [`models/siamese/model_optimized.mlpackage`](models/siamese/model_optimized.mlpackage)
- **Architecture:** `mobilenetv3_large_100`
- **Test Accuracy:** 99.51%
- **Precision:** 0.9945
- **Recall:** 0.9955
- **F1 Score:** 0.9950

**Training Settings:**

- **Dataset:** ~520,000 pairs (~1,040,000 images)
- **Epochs:** 12 (Early Stopping, Max 50)
- **Batch Size:** 16
- **Learning Rate:** 0.0001
- **Patience:** 10

### Merged Classifier

> **Status:** Training in progress

## Benchmark Results

### Siamese Network

#### Training Environment

- **Epochs:** 10
- **GPU:** RTX 3080 10GB
- **Dataset:** 10,000 pairs (20,000 images)

#### Top Results Summary (Training)

| Model                      | Learning Rate | Batch Size | Accuracy | Loss   | Precision | Recall | F1 Score | Duration (s) |
| -------------------------- | ------------- | ---------- | -------- | ------ | --------- | ------ | -------- | ------------ |
| fastvit_t8.apple_dist_in1k | 0.0001        | 32         | 97.40%   | 0.1129 | 0.9959    | 0.9526 | 0.9737   | 485.37       |
| mobilenetv3_large_100      | 0.0001        | 16         | 97.20%   | 0.0968 | 0.9723    | 0.9723 | 0.9723   | 398.40       |
| efficientnet_b0            | 0.0001        | 16         | 96.80%   | 0.1810 | 0.9817    | 0.9545 | 0.9679   | 593.13       |
| efficientnet_b0            | 0.001         | 16         | 96.70%   | 0.0901 | 0.9797    | 0.9545 | 0.9670   | 597.14       |
| efficientnet_b0            | 0.0001        | 32         | 96.40%   | 0.2142 | 0.9719    | 0.9565 | 0.9641   | 303.55       |

#### Inference Environment

- **Device:** iPhone 15
- **OS:** iOS 26.2.1
- **Dataset:** 1,000 pairs (2,000 images)

#### Top Results Summary (Inference)

| Model                 | Type      | Batch Size | Learning Rate | Accuracy | Avg Time (ms) | Inf/Sec |
| :-------------------- | :-------- | :--------- | :------------ | :------- | :------------ | :------ |
| mobilenetv3_large_100 | Optimized | 16         | 0.0001        | 86.00%   | 5.35          | 186.92  |
| mobilenetv3_large_100 | Optimized | 64         | 0.001         | 86.90%   | 5.35          | 186.89  |
| mobilenetv3_large_100 | Optimized | 16         | 0.001         | 86.70%   | 5.35          | 186.77  |
| mobilenetv3_large_100 | Optimized | 16         | 1e-05         | 80.00%   | 5.37          | 186.11  |
| mobilenetv3_large_100 | Optimized | 64         | 1e-05         | 76.40%   | 5.38          | 185.74  |

For the full benchmark results, please refer to [benchmark_results.json](benchmark_results.json).

### Merged Classifier

> **Status:** Training in progress
