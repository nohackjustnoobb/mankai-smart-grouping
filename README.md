> **Note:** This project is 100% vibe-coded. I have no idea what I'm doing, but it works (kind of).

# Mankai Smart Grouping

Mankai Smart Grouping uses deep learning to identify and group image fragments that belong together. It is built for [Mankai](https://github.com/nohackjustnoobb/mankai).

## Models

- [Final models](release/models)
- [Training report](release/training_report/report.md)

## Training

### 1. Add source images

Place your images in `raw/`.

### 2. Generate the dataset

```bash
uv run python scripts/generate_dataset.py
```

### 3. Train

```bash
uv run python -m src.train
```

### 4. Evaluate

```bash
uv run python -m src.evaluate
```

### 5. Convert to Core ML

```bash
uv run python scripts/convert_coreml.py
```

The trained TorchScript and Core ML models are saved below `output/deploy`.
