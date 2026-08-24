"""Generate a Markdown report from structured training and evaluation output."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _find_training_log(output_dir: Path, run_id: str | None) -> Path:
    if run_id:
        path = output_dir / "analysis" / run_id / "training_log.json"
        if not path.is_file():
            raise FileNotFoundError(f"training log not found for run {run_id}: {path}")
        return path

    # Match the report to the run associated with the deployed model.
    metadata_path = output_dir / "deploy" / "metadata.json"
    if metadata_path.is_file():
        metadata = _load_object(metadata_path)
        deployed_run_id = metadata.get("analysis_run_id")
        if isinstance(deployed_run_id, str) and deployed_run_id:
            deployed_log = (
                output_dir / "analysis" / deployed_run_id / "training_log.json"
            )
            if deployed_log.is_file():
                return deployed_log

    candidates = list((output_dir / "analysis").glob("*/training_log.json"))
    if not candidates:
        raise FileNotFoundError(
            f"no training logs found below {output_dir / 'analysis'}"
        )
    return max(candidates, key=lambda path: path.parent.name)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_markdown_cell(value) for value in row) + " |" for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format_number(value: Any, *, digits: int = 4) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—" if value is None else str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{digits}f}"


def _format_scientific(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—" if value is None else str(value)
    return f"{value:.2e}"


def _format_duration(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    seconds = max(0, round(float(value)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _best_epoch(epochs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    eligible = [
        epoch
        for epoch in epochs
        if isinstance(epoch.get("monitor_score"), (int, float))
        and not isinstance(epoch.get("monitor_score"), bool)
    ]
    if not eligible:
        return {}
    return max(eligible, key=lambda epoch: float(epoch["monitor_score"]))


def _validate_evaluation_matches_run(
    run: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> None:
    for key in ("manifest_sha256", "dataset_sha256"):
        run_value = run.get(key)
        evaluation_value = evaluation.get(key)
        if run_value and evaluation_value and run_value != evaluation_value:
            raise ValueError(
                f"evaluation {key} does not match the selected training run"
            )


def _metric_rows(
    run: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> list[tuple[str, Mapping[str, Any]]]:
    rows: list[tuple[str, Mapping[str, Any]]] = []
    calibration = run.get("calibration_metrics")
    if isinstance(calibration, Mapping):
        rows.append(("Calibration", calibration))

    if evaluation:
        split = evaluation.get("split", "evaluation")
        rows.append((str(split).replace("_", " ").title(), evaluation))
    else:
        test = run.get("test_metrics")
        if isinstance(test, Mapping):
            rows.append(("Test", test))
    return rows


def build_report(training: Mapping[str, Any], evaluation: Mapping[str, Any]) -> str:
    run = _mapping(training.get("run"))
    raw_epochs = training.get("epochs")
    if not isinstance(raw_epochs, list):
        raise TypeError("training_log.json requires an epochs list")
    if any(not isinstance(epoch, Mapping) for epoch in raw_epochs):
        raise TypeError("every training epoch must be a JSON object")
    epochs: list[Mapping[str, Any]] = list(raw_epochs)
    config = _mapping(run.get("config"))
    best = _best_epoch(epochs)
    best_epoch = run.get("best_epoch", best.get("epoch"))
    best_score = run.get("best_validation_deployment_pr_auc", best.get("monitor_score"))
    elapsed_seconds = sum(
        float(epoch["elapsed_seconds"])
        for epoch in epochs
        if isinstance(epoch.get("elapsed_seconds"), (int, float))
        and not isinstance(epoch.get("elapsed_seconds"), bool)
    )
    completed = bool(run.get("completed_at"))

    lines = [
        "# Training and evaluation report",
        "",
        "## Run summary",
        "",
        _table(
            ("Field", "Value"),
            (
                ("Run", run.get("run_id")),
                ("Status", "Completed" if completed else "In progress"),
                ("Started", run.get("started_at")),
                ("Completed", run.get("completed_at")),
                ("Device", run.get("device")),
                ("Model", config.get("model_name")),
                (
                    "Epochs",
                    f"{len(epochs)} / {config.get('epochs', '—')}",
                ),
                ("Best epoch", best_epoch),
                (
                    "Best validation deployment PR-AUC",
                    _format_number(best_score),
                ),
                ("Training time", _format_duration(elapsed_seconds)),
            ),
        ),
        "",
        "## Training configuration",
        "",
        _table(
            ("Setting", "Value"),
            (
                ("Batch size", _format_number(config.get("batch_size"))),
                ("Learning rate", _format_scientific(config.get("learning_rate"))),
                ("Weight decay", _format_scientific(config.get("weight_decay"))),
                ("Embedding dimension", config.get("embedding_dim")),
                ("Classifier hidden dimension", config.get("classifier_hidden_dim")),
                ("Samples per epoch", _format_number(config.get("samples_per_epoch"))),
                (
                    "Training positive fraction",
                    _format_number(config.get("sample_positive_fraction")),
                ),
                (
                    "Deployment positive fraction",
                    _format_number(config.get("deployment_positive_fraction")),
                ),
                ("Random seed", config.get("seed")),
            ),
        ),
        "",
        "## Dataset",
        "",
    ]

    dataset_rows = []
    for split in ("training", "validation", "calibration", "test"):
        counts = _mapping(run.get(f"{split}_counts"))
        if counts:
            dataset_rows.append(
                (
                    split.title(),
                    _format_number(counts.get("total")),
                    _format_number(counts.get("positive")),
                    _format_number(counts.get("negative")),
                )
            )
    if dataset_rows:
        lines.append(_table(("Split", "Samples", "Positive", "Negative"), dataset_rows))
    else:
        lines.append("Dataset counts were not recorded.")

    lines.extend(["", "## Training history", ""])
    if epochs:
        history_rows = []
        for epoch in epochs:
            history_rows.append(
                (
                    epoch.get("epoch"),
                    str(epoch.get("phase", "—")).replace("_", " "),
                    _format_number(epoch.get("training_loss"), digits=5),
                    _format_scientific(epoch.get("learning_rate")),
                    _format_number(epoch.get("roc_auc")),
                    _format_number(epoch.get("sample_average_precision")),
                    _format_number(epoch.get("deployment_pr_auc")),
                    _format_duration(epoch.get("elapsed_seconds")),
                    "✓" if epoch.get("is_best") else "",
                )
            )
        lines.append(
            _table(
                (
                    "Epoch",
                    "Phase",
                    "Loss",
                    "Learning rate",
                    "Validation ROC-AUC",
                    "Validation AP",
                    "Validation PR-AUC@deploy",
                    "Time",
                    "New best",
                ),
                history_rows,
            )
        )
    else:
        lines.append("No epochs have been recorded yet.")

    lines.extend(["", "## Evaluation", ""])
    metric_rows = _metric_rows(run, evaluation)
    if metric_rows:
        lines.append(
            _table(
                (
                    "Split",
                    "Samples",
                    "ROC-AUC",
                    "AP",
                    "PR-AUC@deploy",
                    "Precision@deploy",
                    "Recall",
                    "F1@deploy",
                    "FPR",
                    "Threshold",
                ),
                [
                    (
                        split,
                        _format_number(metrics.get("samples")),
                        _format_number(metrics.get("roc_auc")),
                        _format_number(metrics.get("sample_average_precision")),
                        _format_number(metrics.get("deployment_pr_auc")),
                        _format_number(metrics.get("deployment_precision")),
                        _format_number(metrics.get("recall")),
                        _format_number(metrics.get("deployment_f1")),
                        _format_number(metrics.get("false_positive_rate"), digits=6),
                        _format_number(metrics.get("threshold"), digits=6),
                    )
                    for split, metrics in metric_rows
                ],
            )
        )
        lines.extend(["", "### Confusion matrix counts", ""])
        lines.append(
            _table(
                (
                    "Split",
                    "True positive",
                    "False positive",
                    "True negative",
                    "False negative",
                ),
                [
                    (
                        split,
                        _format_number(metrics.get("true_positive")),
                        _format_number(metrics.get("false_positive")),
                        _format_number(metrics.get("true_negative")),
                        _format_number(metrics.get("false_negative")),
                    )
                    for split, metrics in metric_rows
                ],
            )
        )
        threshold_source = evaluation.get(
            "threshold_source", run.get("threshold_source")
        )
        if threshold_source:
            lines.extend(
                [
                    "",
                    f"Threshold source: **{_markdown_cell(threshold_source)}**.",
                ]
            )
    else:
        lines.append("Evaluation metrics are not available yet.")

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("output"),
        help="training output directory (default: output)",
    )
    parser.add_argument(
        "--run-id",
        help="analysis run to report (default: deployed run, then latest run)",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        help="evaluation JSON (default: OUTPUT_DIR/evaluation.json)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Markdown report path (default: OUTPUT_DIR/report.md)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    training_path = _find_training_log(output_dir, args.run_id)
    training = _load_object(training_path)

    evaluation_path = (
        args.evaluation.expanduser().resolve()
        if args.evaluation
        else output_dir / "evaluation.json"
    )
    evaluation = _load_object(evaluation_path) if evaluation_path.is_file() else {}
    run = _mapping(training.get("run"))
    if evaluation:
        _validate_evaluation_matches_run(run, evaluation)

    report_path = (
        args.report.expanduser().resolve() if args.report else output_dir / "report.md"
    )
    if report_path.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError("report path must end in .md or .markdown")
    report = build_report(training, evaluation)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_name(f".{report_path.name}.tmp")
    temporary_path.write_text(report, encoding="utf-8")
    temporary_path.replace(report_path)

    print(f"[info] training log: {training_path}")
    if evaluation:
        print(f"[info] evaluation: {evaluation_path}")
    else:
        print(f"[warning] evaluation not found: {evaluation_path}")
    print(f"[info] Markdown report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
