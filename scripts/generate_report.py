"""Plot every experiment training run found recursively below an output directory."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


PLOT_COLORS = (
    "#2563eb",
    "#ea580c",
    "#16a34a",
    "#9333ea",
    "#dc2626",
    "#0891b2",
    "#ca8a04",
    "#475569",
)
BEST_COLOR = "#2563eb"
MEDIAN_COLOR = "#334155"
RANGE_COLOR = "#bfdbfe"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#334155",
        "axes.titlecolor": "#0f172a",
        "axes.titleweight": "bold",
        "font.size": 10,
        "text.color": "#0f172a",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
    }
)


@dataclass
class TrainingRun:
    """A training log and its optional matching evaluation."""

    path: Path
    experiment_dir: Path
    run_id: str
    model: str
    experiment: str
    config: Mapping[str, Any]
    metadata: Mapping[str, Any]
    epochs: list[Mapping[str, Any]]
    evaluation: Mapping[str, Any]
    label: str = ""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _is_top_level_final_run(training_path: Path, output_dir: Path) -> bool:
    """Return whether a log belongs to the final run stored at output/ itself."""

    relative_path = training_path.relative_to(output_dir)
    return len(relative_path.parts) == 1 or relative_path.parts[0] == "analysis"


def _experiment_dir(training_path: Path, output_dir: Path) -> Path:
    """Resolve the directory owning analysis/, without assuming model depth."""

    for parent in training_path.parents:
        if parent == output_dir:
            break
        if parent.name == "analysis":
            return parent.parent
    return training_path.parent


def _experiment_name(experiment_dir: Path, output_dir: Path, model: str) -> str:
    parts = list(experiment_dir.relative_to(output_dir).parts)
    if parts and parts[0] == model:
        parts = parts[1:]
    return "/".join(parts) or experiment_dir.name or "default"


def _matching_evaluation(
    experiment_dir: Path,
    run_id: str,
) -> Mapping[str, Any]:
    """Load an evaluation only when it can be associated with this run."""

    evaluation_path = experiment_dir / "evaluation.json"
    if not evaluation_path.is_file():
        return {}

    metadata_path = experiment_dir / "deploy" / "metadata.json"
    if metadata_path.is_file():
        deployed_run_id = _load_object(metadata_path).get("analysis_run_id")
        if isinstance(deployed_run_id, str) and deployed_run_id != run_id:
            return {}
    else:
        sibling_logs = list((experiment_dir / "analysis").glob("*/training_log.json"))
        if len(sibling_logs) > 1:
            return {}

    return _load_object(evaluation_path)


def discover_runs(output_dir: Path) -> tuple[list[TrainingRun], int]:
    """Read every experiment JSON log, excluding the root final-model run."""

    all_paths = sorted(output_dir.rglob("training_log.json"))
    training_paths = [
        path for path in all_paths if not _is_top_level_final_run(path, output_dir)
    ]
    ignored_count = len(all_paths) - len(training_paths)
    if not training_paths:
        raise FileNotFoundError(
            f"no experiment training_log.json files found recursively below {output_dir} "
            "after excluding the top-level final-model run"
        )

    runs: list[TrainingRun] = []
    for path in training_paths:
        payload = _load_object(path)
        metadata = _mapping(payload.get("run"))
        raw_epochs = payload.get("epochs")
        if not isinstance(raw_epochs, list):
            raise TypeError(f"{path} requires an epochs list")
        if any(not isinstance(epoch, Mapping) for epoch in raw_epochs):
            raise TypeError(f"every epoch in {path} must be a JSON object")

        config = _mapping(metadata.get("config"))
        model = str(config.get("model_name") or metadata.get("model_name") or "unknown")
        run_id = str(metadata.get("run_id") or path.parent.name)
        experiment_dir = _experiment_dir(path, output_dir)
        runs.append(
            TrainingRun(
                path=path,
                experiment_dir=experiment_dir,
                run_id=run_id,
                model=model,
                experiment=_experiment_name(experiment_dir, output_dir, model),
                config=config,
                metadata=metadata,
                epochs=list(raw_epochs),
                evaluation=_matching_evaluation(experiment_dir, run_id),
            )
        )

    label_counts = Counter((run.model, run.experiment) for run in runs)
    for run in runs:
        run.label = run.experiment
        if label_counts[(run.model, run.experiment)] > 1:
            run.label = f"{run.experiment} · {run.run_id}"
    return runs, ignored_count


def _metric_series(
    run: TrainingRun,
    metric: str,
) -> tuple[list[float], list[float]]:
    points: list[tuple[float, float]] = []
    for epoch in run.epochs:
        epoch_number = _number(epoch.get("epoch"))
        value = _number(epoch.get(metric))
        if epoch_number is not None and value is not None:
            points.append((epoch_number, value))
    points.sort()
    return [point[0] for point in points], [point[1] for point in points]


def _best_metric(run: TrainingRun, metric: str) -> float | None:
    values = [
        value
        for epoch in run.epochs
        if (value := _number(epoch.get(metric))) is not None
    ]
    return max(values) if values else None


def _best_epoch(run: TrainingRun) -> int | None:
    candidates: list[tuple[float, int]] = []
    for epoch in run.epochs:
        score = _number(epoch.get("monitor_score"))
        epoch_number = _number(epoch.get("epoch"))
        if score is not None and epoch_number is not None:
            candidates.append((score, int(epoch_number)))
    return max(candidates)[1] if candidates else None


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._").lower()
    return slug or "unknown-model"


def _model_colors(models: Sequence[str]) -> dict[str, Any]:
    return {
        model: PLOT_COLORS[index % len(PLOT_COLORS)]
        for index, model in enumerate(sorted(set(models)))
    }


def _clean_axis(axis: Any) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot_run_bars(
    runs: Sequence[TrainingRun],
    values: Sequence[float | None],
    title: str,
    output_path: Path,
    *,
    dpi: int,
) -> Path:
    """Plot one run-level metric as a compact ranked dot plot."""

    ordered = sorted(
        (
            (run, value)
            for run, value in zip(runs, values, strict=True)
            if value is not None
        ),
        key=lambda item: (
            item[1],
            item[0].model,
            item[0].label,
        ),
    )
    ordered_runs = [run for run, _ in ordered]
    ordered_values = [float(value) for _, value in ordered]
    colors = _model_colors([run.model for run in ordered_runs])
    one_model = len(colors) == 1
    labels = [
        run.label if one_model else f"{run.model} · {run.label}" for run in ordered_runs
    ]
    figure_height = max(4.8, 0.4 * len(ordered_runs) + 1.5)
    figure, axis = plt.subplots(figsize=(12, figure_height))
    positions = list(range(len(ordered_runs)))
    bar_colors = [colors[run.model] for run in ordered_runs]

    minimum = min(ordered_values)
    maximum = max(ordered_values)
    spread = maximum - minimum
    padding = max(0.015, spread * 0.12)
    lower_bound = max(0.0, minimum - padding)
    upper_bound = min(1.0, maximum + padding)
    axis.hlines(
        positions,
        lower_bound,
        ordered_values,
        colors=bar_colors,
        linewidth=2.0,
        alpha=0.18,
    )
    axis.scatter(
        ordered_values,
        positions,
        c=bar_colors,
        s=58,
        edgecolors="white",
        linewidths=1.2,
        zorder=3,
    )
    axis.set_xlim(lower_bound, upper_bound)
    axis.set_xlabel("Score")
    axis.set_title(title, loc="left", pad=16, fontsize=14)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.grid(axis="x", alpha=0.25)
    axis.grid(axis="y", visible=False)
    axis.set_axisbelow(True)
    _clean_axis(axis)
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0, pad=8)
    axis.text(
        1.0,
        1.03,
        f"{len(ordered_runs)} evaluated runs",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#64748b",
        fontsize=9,
    )
    annotation_gap = max(0.002, (upper_bound - lower_bound) * 0.012)
    for position, value in zip(positions, ordered_values, strict=True):
        has_right_room = value + annotation_gap * 5 < upper_bound
        axis.text(
            value + annotation_gap if has_right_room else value - annotation_gap,
            position,
            f"{value:.4f}",
            va="center",
            ha="left" if has_right_room else "right",
            fontsize=8,
            fontweight="bold",
        )

    if not one_model:
        model_legend = [
            Patch(facecolor=colors[model], label=model) for model in sorted(colors)
        ]
        axis.legend(
            handles=model_legend,
            loc="lower right",
            frameon=False,
            ncol=min(3, len(model_legend)),
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_comparisons(
    runs: Sequence[TrainingRun],
    report_dir: Path,
    *,
    dpi: int,
) -> dict[str, Path]:
    """Create separate validation and test comparison charts."""

    validation_values = [_best_metric(run, "deployment_pr_auc") for run in runs]
    paths = {
        "Best validation deployment PR-AUC": _plot_run_bars(
            runs,
            validation_values,
            "Best validation deployment PR-AUC by experiment",
            report_dir / "validation_comparison.png",
            dpi=dpi,
        )
    }
    test_values = [_number(run.evaluation.get("deployment_pr_auc")) for run in runs]
    if any(value is not None for value in test_values):
        paths["Test deployment PR-AUC"] = _plot_run_bars(
            runs,
            test_values,
            "Test deployment PR-AUC by experiment",
            report_dir / "test_comparison.png",
            dpi=dpi,
        )
    return paths


def plot_model_history(
    model: str,
    runs: Sequence[TrainingRun],
    report_dir: Path,
    *,
    dpi: int,
) -> dict[str, Path]:
    """Summarize all runs without drawing every run as a separate line."""

    metrics = (
        ("training_loss", "Training loss", "Loss", "training_loss"),
        (
            "deployment_pr_auc",
            "Validation deployment PR-AUC",
            "Score",
            "validation_deployment_pr_auc",
        ),
        ("roc_auc", "Validation ROC-AUC", "Score", "validation_roc_auc"),
        (
            "sample_average_precision",
            "Validation average precision",
            "Score",
            "validation_average_precision",
        ),
    )
    ranked_runs = sorted(
        runs,
        key=lambda run: (
            _best_metric(run, "deployment_pr_auc") is not None,
            _best_metric(run, "deployment_pr_auc") or -math.inf,
        ),
        reverse=True,
    )
    best_run = ranked_runs[0]
    paths: dict[str, Path] = {}
    for metric, title, y_label, filename_suffix in metrics:
        values_by_epoch: dict[float, list[float]] = {}
        for run in runs:
            epoch_numbers, values = _metric_series(run, metric)
            for epoch_number, value in zip(epoch_numbers, values, strict=True):
                values_by_epoch.setdefault(epoch_number, []).append(value)

        aggregate_epochs = sorted(values_by_epoch)
        minima = [min(values_by_epoch[epoch]) for epoch in aggregate_epochs]
        medians = [median(values_by_epoch[epoch]) for epoch in aggregate_epochs]
        maxima = [max(values_by_epoch[epoch]) for epoch in aggregate_epochs]
        best_epochs, best_values = _metric_series(best_run, metric)

        figure, axis = plt.subplots(figsize=(12, 6.2))
        legend_handles: list[Any] = []
        if aggregate_epochs:
            axis.fill_between(
                aggregate_epochs,
                minima,
                maxima,
                color=RANGE_COLOR,
                alpha=0.6,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                aggregate_epochs,
                medians,
                color=MEDIAN_COLOR,
                linewidth=2.0,
                linestyle="--",
                zorder=2,
            )
            legend_handles.extend(
                [
                    Patch(
                        facecolor=RANGE_COLOR,
                        alpha=0.6,
                        label="All runs · min–max",
                    ),
                    Line2D(
                        [0],
                        [0],
                        color=MEDIAN_COLOR,
                        linewidth=2.0,
                        linestyle="--",
                        label="Median",
                    ),
                ]
            )
        if best_values:
            axis.plot(
                best_epochs,
                best_values,
                color=BEST_COLOR,
                linewidth=2.7,
                zorder=3,
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=BEST_COLOR,
                    linewidth=2.7,
                    label=f"Best · {best_run.label}",
                )
            )
            if metric != "training_loss":
                best_index = max(range(len(best_values)), key=best_values.__getitem__)
                axis.scatter(
                    [best_epochs[best_index]],
                    [best_values[best_index]],
                    color=BEST_COLOR,
                    edgecolor="white",
                    linewidth=1.2,
                    s=54,
                    zorder=4,
                )

        axis.set_title(f"{model} · {title}", loc="left", pad=16, fontsize=14)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(y_label)
        axis.grid(axis="y", alpha=0.7)
        axis.grid(axis="x", visible=False)
        axis.set_axisbelow(True)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        _clean_axis(axis)
        axis.text(
            1.0,
            1.03,
            f"{len(runs)} runs summarized",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color="#64748b",
            fontsize=9,
        )
        if metric == "training_loss":
            if minima and all(value > 0.0 for value in minima):
                axis.set_yscale("log")
        else:
            all_values = [*minima, *maxima, *best_values]
            if all_values:
                lower = min(all_values)
                upper = max(all_values)
                padding = max(0.01, (upper - lower) * 0.08)
                axis.set_ylim(max(0.0, lower - padding), min(1.0, upper + padding))

        if legend_handles:
            axis.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=len(legend_handles),
                fontsize=9,
                frameon=False,
            )
        else:
            axis.text(
                0.5,
                0.5,
                f"No {title.lower()} data recorded",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))

        output_path = report_dir / f"{_slugify(model)}_{filename_suffix}.png"
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        paths[title] = output_path
    return paths


def _format_markdown_number(value: float | None, *, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_markdown_report(
    runs: Sequence[TrainingRun],
    comparison_paths: Mapping[str, Path],
    model_history_paths: Mapping[str, Mapping[str, Path]],
    report_dir: Path,
) -> Path:
    """Create a portable Markdown report with charts and run data tables."""

    model_count = len(model_history_paths)
    model_label = "model" if model_count == 1 else "models"
    lines = [
        "# Training report",
        "",
        f"This report includes {len(runs)} experiment runs across "
        f"{model_count} {model_label}.",
        "",
        "## Experiment comparisons",
        "",
    ]
    for title, comparison_path in comparison_paths.items():
        lines.extend(
            [
                f"### {title}",
                "",
                f"![{title}]({comparison_path.relative_to(report_dir).as_posix()})",
                "",
            ]
        )
    lines.extend(["## Model details", ""])
    for model, history_paths in sorted(model_history_paths.items()):
        model_runs = sorted(
            (run for run in runs if run.model == model),
            key=lambda run: _best_metric(run, "deployment_pr_auc") or -math.inf,
            reverse=True,
        )
        lines.extend(
            [
                f"### {model}",
                "",
                "| Experiment | Status | Epochs | Best epoch | Best validation PR-AUC | Test PR-AUC | Learning rate | Dropout | Weight decay |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for run in model_runs:
            configured_epochs = run.config.get("epochs")
            epoch_count = str(len(run.epochs))
            if configured_epochs is not None:
                epoch_count = f"{epoch_count} / {configured_epochs}"
            status = "Completed" if run.metadata.get("completed_at") else "In progress"
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(run.experiment).replace("|", "\\|"),
                        status,
                        epoch_count,
                        str(_best_epoch(run) or "—"),
                        _format_markdown_number(_best_metric(run, "deployment_pr_auc")),
                        _format_markdown_number(
                            _number(run.evaluation.get("deployment_pr_auc"))
                        ),
                        _format_markdown_number(
                            _number(run.config.get("learning_rate")), digits=6
                        ),
                        _format_markdown_number(
                            _number(run.config.get("dropout")), digits=3
                        ),
                        _format_markdown_number(
                            _number(run.config.get("weight_decay")), digits=6
                        ),
                    )
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "#### Learning curves",
                "",
                "The solid blue line is the experiment with the highest validation "
                "deployment PR-AUC, the dashed line is the median, and the shaded "
                "band is the min–max range across all runs.",
                "",
            ]
        )
        for title, history_path in history_paths.items():
            lines.extend(
                [
                    f"##### {title}",
                    "",
                    f"![{model} {title}]({history_path.relative_to(report_dir).as_posix()})",
                    "",
                ]
            )
    report_path = report_dir / "report.md"
    temporary_path = report_path.with_name(f".{report_path.name}.tmp")
    temporary_path.write_text("\n".join(lines), encoding="utf-8")
    temporary_path.replace(report_path)
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("output"),
        help="directory searched recursively for training logs (default: output)",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="generated chart directory (default: OUTPUT_DIR/training_report)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="chart resolution in dots per inch (default: 160)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_dir():
        raise NotADirectoryError(f"output directory not found: {output_dir}")
    report_dir = (
        args.report_dir.expanduser().resolve()
        if args.report_dir
        else output_dir / "training_report"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    runs, ignored_count = discover_runs(output_dir)
    legacy_paths = [
        report_dir / "model_comparison.png",
        report_dir / "training_summary.csv",
        *(
            report_dir / f"{_slugify(model)}_training_history.png"
            for model in {run.model for run in runs}
        ),
    ]
    for legacy_path in legacy_paths:
        if legacy_path.is_file():
            legacy_path.unlink()

    comparison_paths = plot_comparisons(runs, report_dir, dpi=args.dpi)
    model_history_paths: dict[str, dict[str, Path]] = {}
    for model in sorted({run.model for run in runs}):
        model_runs = [run for run in runs if run.model == model]
        model_history_paths[model] = plot_model_history(
            model, model_runs, report_dir, dpi=args.dpi
        )
    markdown_path = write_markdown_report(
        runs,
        comparison_paths,
        model_history_paths,
        report_dir,
    )
    generated = [
        *comparison_paths.values(),
        *(
            path
            for history_paths in model_history_paths.values()
            for path in history_paths.values()
        ),
        markdown_path,
    ]

    model_count = len({run.model for run in runs})
    print(f"[info] discovered {len(runs)} runs across {model_count} models")
    if ignored_count:
        print(f"[info] ignored {ignored_count} top-level final-model run(s)")
    for path in generated:
        print(f"[info] generated: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
