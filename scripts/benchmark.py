import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime
from itertools import product

# Global lock for printing
print_lock = threading.Lock()

# Configuration
SCRIPTS = {
    "siamese": "src/siamese_network.py",
    "merged": "src/merged_classifier.py",
}
RESULTS_BASE_DIR = "results/{script}/benchmark_runs"
SUMMARY_FILE = "benchmark_results.json"

# Parameters to sweep
PARAMETERS = {
    "learning_rate": [0.001, 0.0001, 0.00001],
    "batch_size": [16, 32, 64],
    "model": [
        "resnet18",
        "fastvit_t8.apple_dist_in1k",
        "efficientnet_b0",
        "mobilenetv3_large_100",
    ],
}


def run_experiment(args):
    """Runs a single experiment with the given parameters."""
    i = args["index"]
    total_combinations = args["total"]
    params = args["params"]
    script = args["script"]
    results_base_dir = args["results_base_dir"]
    run_id = args.get("run_id")

    print(f"\n--- Starting Experiment {i + 1}/{total_combinations} ---")
    print(f"Parameters: {params}")

    if run_id is None:
        # Create a unique directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include thread/process ID to avoid collision if started at exact same second
        import threading

        thread_id = threading.get_ident()
        run_id = f"run_{i + 1}_{timestamp}_{thread_id}"
    run_dir = os.path.join(results_base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Construct command
    cmd = [
        "uv",
        "run",
        script,
        "--results",
        run_dir,
        "--dev",
        "--epochs",
        "10",
        "--batch_size",
        str(params["batch_size"]),
        "--learning_rate",
        str(params["learning_rate"]),
        "--model",
        str(params["model"]),
        "--workers",
        "0",
    ]

    # Run the script
    start_time = time.time()

    # Use Popen to stream output
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            env={**os.environ, "PYTHONUNBUFFERED": "1"},  # Force unbuffered output
        )

        prefix = f"[Exp {i + 1}]"

        # Read output line by line
        for line in process.stdout:
            with print_lock:
                print(f"{prefix} {line}", end="")

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

        run_success = True

    except subprocess.CalledProcessError as e:
        with print_lock:
            print(f"{prefix} Error running experiment {run_id}: {e}")
        run_success = False
    end_time = time.time()

    # Collect results
    result_entry = {
        "run_id": run_id,
        "parameters": params,
        "status": "success" if run_success else "failed",
        "duration": end_time - start_time,
    }

    if run_success:
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
                result_entry["metrics"] = metrics
        else:
            result_entry["status"] = "metrics_missing"
            print(f"Warning: metrics.json not found for {run_id}.")

    print(f"Finished Experiment {i + 1}/{total_combinations}")
    return result_entry


def run_benchmark():
    parser = argparse.ArgumentParser(description="Run benchmark experiments.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry the last failed experiment from benchmark_results.json",
    )
    parser.add_argument(
        "--script",
        type=str,
        choices=list(SCRIPTS.keys()),
        default="siamese",
        help=f"Training script to use (default: siamese). Choices: {', '.join(SCRIPTS.keys())}",
    )
    args = parser.parse_args()

    script_path = SCRIPTS[args.script]
    results_base_dir = RESULTS_BASE_DIR.format(script=args.script)
    summary_file = os.path.join(results_base_dir, SUMMARY_FILE)
    print(f"Using training script: {script_path}")
    print(f"Results directory: {results_base_dir}")

    if args.retry_failed:
        print("Retrying failed experiments...")
        if not os.path.exists(summary_file):
            print(f"Error: {summary_file} not found. Cannot retry.")
            return

        with open(summary_file, "r") as f:
            try:
                previous_results = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Could not decode {summary_file}.")
                return

        # Find all failed experiments
        failed_indices = [
            i for i, res in enumerate(previous_results) if res["status"] != "success"
        ]

        if not failed_indices:
            print("No failed experiments found in summary.")
            return

        print(f"Found {len(failed_indices)} failed experiments to retry.")

        for i, idx in enumerate(failed_indices):
            failed_entry = previous_results[idx]
            print(
                f"\n--- Retrying Failed Experiment {i + 1}/{len(failed_indices)} (Run ID: {failed_entry['run_id']}) ---"
            )
            print(f"Parameters: {failed_entry['parameters']}")

            params = failed_entry["parameters"]
            # We'll treat this as a single experiment run
            new_result = run_experiment(
                {
                    "index": 0,
                    "total": 1,
                    "params": params,
                    "run_id": failed_entry["run_id"],
                    "script": script_path,
                    "results_base_dir": results_base_dir,
                }
            )

            # Update the entry
            previous_results[idx] = new_result

            # Save immediately
            with open(summary_file, "w") as f:
                json.dump(previous_results, f, indent=4)

        print(f"Updated {summary_file} with new results.")
        all_results = previous_results

    else:
        # Standard benchmark run
        # Clear previous results directory if it exists
        if os.path.exists(results_base_dir):
            print(f"Removing existing results directory: {results_base_dir}")
            shutil.rmtree(results_base_dir)

        # create results directory
        os.makedirs(results_base_dir, exist_ok=True)

        # Generate all combinations of parameters
        keys = PARAMETERS.keys()
        values = PARAMETERS.values()
        combinations = [dict(zip(keys, v)) for v in product(*values)]

        print(f"Found {len(combinations)} combinations to test.")
        print(f"Running with {args.workers} workers.")

        all_results = []

        # Prepare arguments for each task
        experiment_args = [
            {
                "index": i,
                "total": len(combinations),
                "params": params,
                "script": script_path,
                "results_base_dir": results_base_dir,
            }
            for i, params in enumerate(combinations)
        ]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.workers
        ) as executor:
            future_to_experiment = {
                executor.submit(run_experiment, arg): arg for arg in experiment_args
            }
            for future in concurrent.futures.as_completed(future_to_experiment):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    print(f"Experiment generated an exception: {e}")

                # Save summary
                with open(summary_file, "w") as f:
                    json.dump(all_results, f, indent=4)

        print(f"\nBenchmark complete. Summary saved to {summary_file}")

    # Print a quick table
    print("\nSummary Table:")
    print(
        f"{'Run ID':<30} | {'Model':<25} | {'LR':<10} | {'Batch':<8} | {'Acc':<8} | {'Loss':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}"
    )
    print("-" * 130)
    # Sort results by run_id (or index if we preserved it) for consistent display
    all_results.sort(key=lambda x: x["run_id"])

    for res in all_results:
        if res["status"] == "success" and "metrics" in res:
            metrics = res["metrics"]
            acc = metrics.get("test_acc", 0)
            loss = metrics.get("test_loss", 0)
            prec = metrics.get("precision", 0)
            rec = metrics.get("recall", 0)
            f1 = metrics.get("f1_score", 0)

            lr = res["parameters"]["learning_rate"]
            batch = res["parameters"]["batch_size"]
            mdl = res["parameters"]["model"]
            print(
                f"{res['run_id']:<30} | {mdl:<25} | {lr:<10} | {batch:<8} | {acc:.2f}%   | {loss:.4f}   | {prec:.4f}   | {rec:.4f}   | {f1:.4f}"
            )
        else:
            print(f"{res['run_id']:<30} | {res['status']}")


if __name__ == "__main__":
    run_benchmark()
