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

## Usage

### Training

To train the adjacency detection model, use the `src/siamese_network.py` script. You can configure the model architecture, batch size, learning rate, and more.

```bash
uv run src/siamese_network.py --images ./images --epochs 10 --model resnet18
```

**Common Arguments:**

- `--images`: Path to the directory containing images (default: `./images`).
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

The final production model is located in the `results` directory. It is a **MobileNetV3 Large 100** model chosen for its superior balance of speed and accuracy.

- **PyTorch Model:** [`models/model.pth`](models/model.pth)
- **Core ML Model:** [`models/model.mlpackage`](models/model.mlpackage) / [`models/model_optimized.mlpackage`](models/model_optimized.mlpackage)
- **Architecture:** `mobilenetv3_large_100`
- **Test Accuracy:** 99.25%
- **Precision:** 0.9914
- **Recall:** 0.9936
- **F1 Score:** 0.9925

**Training Settings:**

- **Dataset:** ~260,000 images
- **Epochs:** 16 (Early Stopping, Max 50)
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Patience:** 10

## Benchmark Results

### Training Environment

- **Epochs:** 10
- **GPU:** RTX 3080 10GB
- **Dataset:** 10,000 images

### Top Results Summary (Training)

| Model                      | Learning Rate | Batch Size | Status  | Accuracy | Loss   | Precision | Recall | F1 Score | Duration (s) |
| -------------------------- | ------------- | ---------- | ------- | -------- | ------ | --------- | ------ | -------- | ------------ |
| mobilenetv3_large_100      | 0.001         | 64         | Success | 98.00%   | 0.0759 | 0.9920    | 0.9688 | 0.9802   | 725.23       |
| efficientnet_b0            | 0.001         | 32         | Success | 97.70%   | 0.0714 | 0.9660    | 0.9877 | 0.9767   | 717.09       |
| resnet18                   | 0.001         | 16         | Success | 97.60%   | 0.0745 | 0.9776    | 0.9736 | 0.9756   | 1019.94      |
| fastvit_t8.apple_dist_in1k | 0.0001        | 16         | Success | 97.10%   | 0.0702 | 0.9773    | 0.9634 | 0.9703   | 1667.97      |
| efficientnet_b0            | 0.001         | 16         | Success | 96.80%   | 0.0945 | 0.9524    | 0.9886 | 0.9701   | 1711.09      |

### Runtime Benchmark (Inference)

**Environment:**

- **Device:** iPhone 15
- **OS:** iOS 26.2.1
- **Dataset:** 1,000 images

**Top Results:**

| Model                      | Type      | Batch Size | Learning Rate | Accuracy | Avg Time (ms) | Inf/Sec |
| :------------------------- | :-------- | :--------- | :------------ | :------- | :------------ | :------ |
| mobilenetv3_large_100      | Optimized | 32         | 0.001         | 97.50%   | 5.37          | 186.18  |
| mobilenetv3_large_100      | Standard  | 32         | 0.001         | 97.30%   | 8.63          | 115.90  |
| efficientnet_b0            | Optimized | 32         | 0.001         | 97.00%   | 6.82          | 146.72  |
| fastvit_t8.apple_dist_in1k | Standard  | 16         | 0.0001        | 96.90%   | 14.17         | 70.59   |
| mobilenetv3_large_100      | Standard  | 16         | 0.001         | 96.80%   | 8.79          | 113.78  |

For the full benchmark results, please refer to [benchmark_results.json](benchmark_results.json).
