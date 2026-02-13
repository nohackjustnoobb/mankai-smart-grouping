> **Note:** This project is mostly vibe-coded. I have no idea what I'm doing, but it works (kind of).

# Mankai Smart Grouping

**Mankai Smart Grouping** is an intelligent tool designed to automatically group split image patches using deep learning. It leverages Siamese Networks to determine if two image patches belong side-by-side, enabling seamless reconstruction of spread pages.

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

The dataset uses a CSV-based approach for organizing image pairs. Here's how to structure your data:

### Directory Layout

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

### CSV Format

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

### Image Requirements

- **Resolution**: All images must be pre-processed to **224×224 pixels**
- **Format**: JPG, PNG, or WebP
- **Color**: RGB color images

## Usage

### Training

Ensure your dataset is structured according to the [Dataset Structure](#dataset-structure) section above. Then train the adjacency detection model:

```bash
uv run src/siamese_network.py --images ./images --csv ./pairs.csv --epochs 10 --model resnet18
```

**Common Arguments:**

- `--images`: Path to the directory containing images (default: `./images`).
- `--csv`: Path to the CSV file with image pairs (default: `./pairs.csv`).
- `--model`: Model backbone to use (e.g., `resnet18`, `efficientnet_b0`, `mobilenetv3_large_100`).
- `--epochs`: Number of training epochs.
- `--dev`: Run in development mode with a smaller dataset for quick testing.

### converting to Core ML

For deployment on iOS, convert your trained PyTorch model to the Core ML format (`.mlpackage`) using the conversion script.

```bash
uv run scripts/convert_to_mlpackage.py --model_path results/model.pth --optimize --validate
```

**Key Arguments:**

- `--model_path`: Path to the trained PyTorch model (`.pth`).
- `--optimize`: Optimizes the model for iOS (uses `ImageType` inputs and `float16` precision).
- `--validate`: Validates the converted model by comparing outputs with the original PyTorch model.

## Final Model

The final production model is located in the `models` directory. It is a **MobileNetV3 Large 100** model chosen for its superior balance of speed and accuracy.

- **PyTorch Model:** [`models/model.pth`](models/model.pth)
- **Core ML Model:** [`models/model.mlpackage`](models/model.mlpackage) / [`models/model_optimized.mlpackage`](models/model_optimized.mlpackage)
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

## Benchmark Results

### Training Environment

- **Epochs:** 10
- **GPU:** RTX 3080 10GB
- **Dataset:** 10,000 pairs (20,000 images)

### Top Results Summary (Training)

| Model                      | Learning Rate | Batch Size | Accuracy | Loss   | Precision | Recall | F1 Score | Duration (s) |
| -------------------------- | ------------- | ---------- | -------- | ------ | --------- | ------ | -------- | ------------ |
| fastvit_t8.apple_dist_in1k | 0.0001        | 32         | 97.40%   | 0.1129 | 0.9959    | 0.9526 | 0.9737   | 485.37       |
| mobilenetv3_large_100      | 0.0001        | 16         | 97.20%   | 0.0968 | 0.9723    | 0.9723 | 0.9723   | 398.40       |
| efficientnet_b0            | 0.0001        | 16         | 96.80%   | 0.1810 | 0.9817    | 0.9545 | 0.9679   | 593.13       |
| efficientnet_b0            | 0.001         | 16         | 96.70%   | 0.0901 | 0.9797    | 0.9545 | 0.9670   | 597.14       |
| efficientnet_b0            | 0.0001        | 32         | 96.40%   | 0.2142 | 0.9719    | 0.9565 | 0.9641   | 303.55       |

### Inference Environment

- **Device:** iPhone 15
- **OS:** iOS 26.2.1
- **Dataset:** 1,000 pairs (2,000 images)

### Top Results Summary (Inference)

| Model                 | Type      | Batch Size | Learning Rate | Accuracy | Avg Time (ms) | Inf/Sec |
| :-------------------- | :-------- | :--------- | :------------ | :------- | :------------ | :------ |
| mobilenetv3_large_100 | Optimized | 16         | 0.0001        | 86.00%   | 5.35          | 186.92  |
| mobilenetv3_large_100 | Optimized | 64         | 0.001         | 86.90%   | 5.35          | 186.89  |
| mobilenetv3_large_100 | Optimized | 16         | 0.001         | 86.70%   | 5.35          | 186.77  |
| mobilenetv3_large_100 | Optimized | 16         | 1e-05         | 80.00%   | 5.37          | 186.11  |
| mobilenetv3_large_100 | Optimized | 64         | 1e-05         | 76.40%   | 5.38          | 185.74  |

For the full benchmark results, please refer to [benchmark_results.json](benchmark_results.json).
