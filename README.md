> **Note:** This project is 100% vibe-coded. I have no idea what I'm doing, but it works (kind of).

# Mankai Smart Grouping

Mankai Smart Grouping uses deep learning to identify and group image fragments that belong together. It is built for [Mankai](https://github.com/nohackjustnoobb/mankai).

## Models

- [Final models](release/models)
- [Training report](release/report.md)

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

### 5. Generate a report

```bash
uv run python scripts/generate_report.py
```

### 6. Convert to Core ML

```bash
uv run python scripts/convert_coreml.py
```

The trained TorchScript and Core ML models are saved to `output/deploy/`, and the report is saved to `output/report.md`.
