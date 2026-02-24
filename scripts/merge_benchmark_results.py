"""Merge training and inference benchmark results into a unified report.

This script reads the training benchmark results and inference benchmark results
from their respective JSON files, merges them by run ID into a single JSON
structure, and prints summary tables for the top training and inference results.

It automatically discovers classifier types (e.g., siamese, merged) by scanning
subdirectories under results/.

Usage:
    python scripts/merge_benchmark_results.py

Expected input files (per classifier type):
    - results/<type>/benchmark_runs/benchmark_results.json   (training metrics)
    - results/<type>/benchmark_runs/inference_benchmark_results.json  (inference timings)

Output (per classifier type):
    - <type>_benchmark_results.json  (merged results with training + inference data)
"""

import json
import os
import re

# Base results directory
RESULTS_BASE_DIR = "results"


def load_json(path):
    """Load and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON data, or an empty list if the file does not exist.
    """
    if not os.path.exists(path):
        print(f"Warning: File not found: {path}")
        return []
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    """Save data to a JSON file with pretty formatting.

    Args:
        data: Python object to serialize.
        path: Output file path.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved merged results to {path}")


def discover_classifier_types():
    """Discover classifier types by scanning subdirectories under results/.

    Returns:
        List of classifier type names (e.g., ['merged', 'siamese']).
    """
    if not os.path.exists(RESULTS_BASE_DIR):
        print(f"Warning: Results directory not found: {RESULTS_BASE_DIR}")
        return []

    types = []
    for entry in sorted(os.listdir(RESULTS_BASE_DIR)):
        subdir = os.path.join(RESULTS_BASE_DIR, entry, "benchmark_runs")
        if os.path.isdir(subdir):
            types.append(entry)

    return types


def merge_results(classifier_type):
    """Merge training and inference results into a unified list.

    Reads training results and inference results from their respective JSON
    files, correlates them by run ID, and produces a list of merged entries
    each containing training metrics and inference benchmarks (both optimized
    and standard variants).

    Args:
        classifier_type: The classifier type name (e.g., 'siamese', 'merged').

    Returns:
        Sorted list of merged result dictionaries, each with keys:
        'modelId', 'settings', 'training', and 'inference'.
    """
    base = os.path.join(RESULTS_BASE_DIR, classifier_type, "benchmark_runs")
    training_path = os.path.join(base, "benchmark_results.json")
    inference_path = os.path.join(base, "inference_benchmark_results.json")

    training_data = load_json(training_path)
    inference_data = load_json(inference_path)

    merged_map = {}

    # 1. Process Training Data
    for item in training_data:
        run_id = item.get("run_id")
        if not run_id:
            continue

        raw_metrics = item.get("metrics", {})
        # Use parameters from metrics if available, otherwise fallback to top-level parameters
        settings_params = raw_metrics.get("parameters", item.get("parameters", {}))

        # Create a clean copy of metrics for the output, removing parameters since they are now in settings
        clean_metrics = raw_metrics.copy()
        if "parameters" in clean_metrics:
            del clean_metrics["parameters"]

        merged_map[run_id] = {
            "modelId": run_id,
            "settings": settings_params,
            "training": {
                "status": item.get("status"),
                "duration": item.get("duration"),
                "metrics": clean_metrics,
            },
            "inference": {"optimized": None, "standard": None},
        }

    # 2. Process Inference Data
    for item in inference_data:
        model_name = item.get("modelName", "")
        # Extract run_id from modelName (e.g., "run_1_20260212_130702_128197228668608/model")
        # Trying to capture run_{index}_{date}_{time}_{thread_id}
        run_id_match = re.search(r"(run_\d+_\d+_\d+_\d+)", model_name)

        run_id = run_id_match.group(1) if run_id_match else None

        if not run_id:
            continue

        if run_id not in merged_map:
            merged_map[run_id] = {
                "modelId": run_id,
                "settings": item.get("modelSettings", {}),
                "training": None,
                "inference": {"optimized": None, "standard": None},
            }

        # Determine type
        model_type_raw = item.get("modelType", "")
        inference_entry = item.copy()
        keys_to_remove = ["modelSettings", "modelName", "modelPath", "modelType"]
        for key in keys_to_remove:
            if key in inference_entry:
                del inference_entry[key]

        if "optimized" in model_type_raw:
            merged_map[run_id]["inference"]["optimized"] = inference_entry
        else:
            merged_map[run_id]["inference"]["standard"] = inference_entry

    merged_list = list(merged_map.values())

    merged_list.sort(key=lambda x: x["modelId"])

    return merged_list


def generate_inference_table(merged_data):
    """Generate a Markdown table of the top 5 inference benchmark results.

    Results are sorted by inferences per second (descending).

    Args:
        merged_data: List of merged result dictionaries.

    Returns:
        Markdown-formatted table string.
    """
    # Header
    table = "| Model | Type | Batch Size | Learning Rate | Accuracy | Avg Time (ms) | Inf/Sec |\n"
    table += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"

    flat_rows = []

    for entry in merged_data:
        settings = entry.get("settings", {})
        model_name = settings.get("model", "Unknown")
        bs = settings.get("batchSize", settings.get("batch_size", "-"))
        lr = settings.get("learningRate", settings.get("learning_rate", "-"))

        inf = entry.get("inference", {})
        if not inf:
            continue

        for type_key in ["optimized", "standard"]:
            data = inf.get(type_key)
            if data:
                row = {
                    "model": model_name,
                    "type": type_key.capitalize(),
                    "bs": bs,
                    "lr": lr,
                    "acc": data.get("accuracy", 0),
                    "time": data.get("avgTimeMs", 0),
                    "inf_sec": data.get("inferencePerSecond", 0),
                }
                flat_rows.append(row)

    # Sort by inf_sec descending
    flat_rows.sort(key=lambda x: x["inf_sec"], reverse=True)

    for r in flat_rows[:5]:
        acc_str = f"{r['acc']:.2f}%"
        time_str = f"{r['time']:.2f}"
        inf_str = f"{r['inf_sec']:.2f}"

        # Format LR if it's a number
        lr_val = r["lr"]

        table += f"| {r['model']} | {r['type']} | {r['bs']} | {lr_val} | {acc_str} | {time_str} | {inf_str} |\n"

    return table


def generate_training_table(merged_data):
    """Generate a Markdown table of the top 5 training results by accuracy.

    Only includes runs with a 'success' status.

    Args:
        merged_data: List of merged result dictionaries.

    Returns:
        Markdown-formatted table string.
    """
    # Filter only those with training metrics
    valid_runs = [
        x
        for x in merged_data
        if x.get("training") and x["training"].get("status") == "success"
    ]

    # Sort by Test Accuracy
    valid_runs.sort(
        key=lambda x: x["training"]["metrics"].get("test_acc", 0), reverse=True
    )

    header = "| Model | Learning Rate | Batch Size | Accuracy | Loss | Precision | Recall | F1 Score | Duration (s) |\n"
    header += "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"

    table = header
    for item in valid_runs[:5]:  # Top 5
        settings = item["settings"]
        training = item["training"]
        metrics = training.get("metrics", {})

        model = settings.get("model", "Unknown")
        lr = settings.get("learning_rate", settings.get("learningRate", "-"))
        bs = settings.get("batch_size", settings.get("batchSize", "-"))

        acc = f"{metrics.get('test_acc', 0):.2f}%"
        loss = f"{metrics.get('test_loss', 0):.4f}"
        prec = f"{metrics.get('precision', 0):.4f}"
        rec = f"{metrics.get('recall', 0):.4f}"
        f1 = f"{metrics.get('f1_score', 0):.4f}"
        dur = f"{training.get('duration', 0):.2f}"

        table += f"| {model} | {lr} | {bs} | {acc} | {loss} | {prec} | {rec} | {f1} | {dur} |\n"

    return table


def main():
    """Entry point: discover classifiers, merge results, save to JSON, and print summary tables."""
    classifier_types = discover_classifier_types()

    if not classifier_types:
        print("No classifier types found under results/.")
        return

    for classifier_type in classifier_types:
        print(f"\n{'=' * 60}")
        print(f"Processing classifier: {classifier_type}")
        print(f"{'=' * 60}")

        merged_data = merge_results(classifier_type)
        output_path = f"{classifier_type}_benchmark_results.json"
        save_json(merged_data, output_path)

        print(f"\n\n### Top Results Summary (Training) - {classifier_type}")
        train_table = generate_training_table(merged_data)
        print(train_table)

        print(f"\n### Runtime Benchmark (Inference) - {classifier_type}")
        inf_table = generate_inference_table(merged_data)
        print(inf_table)


if __name__ == "__main__":
    main()
