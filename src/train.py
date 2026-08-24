"""Train and export the MobileNetV3 Siamese image-pair classifier."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PairDataset,
    build_transform,
    class_counts,
    dataset_sha256,
    file_sha256,
    make_balanced_sampler,
    read_manifest,
    split_records_for_experiment,
)
from .deploy import export_checkpoint_bundle
from .evaluate import (
    compute_metrics,
    compute_ranking_metrics,
    predict,
    print_metrics,
    print_ranking_metrics,
    resolve_device,
    select_threshold,
)
from .model import (
    SiameseNetwork,
    build_model,
    build_model_from_checkpoint,
)

TRAINING_LOG_FIELDS = (
    "epoch",
    "completed_at",
    "phase",
    "backbone_frozen",
    "training_loss",
    "learning_rate",
    "elapsed_seconds",
    "is_best",
    "epochs_without_improvement",
    "monitor_metric",
    "monitor_score",
    "roc_auc",
    "sample_average_precision",
    "deployment_pr_auc",
    "samples",
    "positives",
    "negatives",
)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(path)


def _write_training_history(
    run_dir: Path,
    run_metadata: dict[str, Any],
    history: list[dict[str, Any]],
) -> tuple[Path, Path]:
    csv_path = run_dir / "training_log.csv"
    json_path = run_dir / "training_log.json"

    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=TRAINING_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    temporary_csv.replace(csv_path)

    _atomic_write_json(
        {"run": run_metadata, "epochs": history},
        json_path,
    )
    return csv_path, json_path


def _capture_random_state(
    sampler: torch.utils.data.WeightedRandomSampler,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if sampler.generator is not None:
        state["sampler"] = sampler.generator.get_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_random_state(
    state: dict[str, Any], sampler: torch.utils.data.WeightedRandomSampler
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if sampler.generator is not None and "sampler" in state:
        sampler.generator.set_state(state["sampler"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _clone_model_state(model: SiameseNetwork) -> dict[str, Tensor]:
    """Keep an immutable CPU copy of the best weights inside resume checkpoints."""

    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_focal_loss(logits: Tensor, targets: Tensor, gamma: float) -> Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if gamma > 0.0:
        probability = torch.sigmoid(logits)
        probability_of_target = probability * targets + (1.0 - probability) * (
            1.0 - targets
        )
        loss = loss * (1.0 - probability_of_target).pow(gamma)
    return loss.mean()


def train_one_epoch(
    model: SiameseNetwork,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor]],
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    focal_gamma: float,
    max_gradient_norm: float,
    backbone_frozen: bool,
) -> float:
    model.train()
    if backbone_frozen:
        # Keep frozen BatchNorm statistics unchanged.
        model.encoder.backbone.eval()

    total_loss = 0.0
    total_samples = 0
    use_amp = device.type == "cuda"
    for image_1, image_2, labels in loader:
        image_1 = image_1.to(device, non_blocking=True)
        image_2 = image_2.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(image_1, image_2)
            loss = binary_focal_loss(logits, labels, focal_gamma)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
    return total_loss / total_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("data/dataset.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model-name", default="mobilenetv3_large_100")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--classifier-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--max-gradient-norm", type=float, default=5.0)
    parser.add_argument(
        "--sample-positive-fraction",
        type=float,
        default=0.5,
        help="positive fraction sampled during training (default: 0.5)",
    )
    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        help=(
            "balanced samples drawn per epoch (default: enough to draw each "
            "positive once in expectation)"
        ),
    )
    parser.add_argument(
        "--deployment-positive-fraction",
        type=float,
        default=0.01,
        help="expected real-world positive rate (default: 0.01)",
    )
    parser.add_argument(
        "--minimum-precision",
        type=float,
        help="choose maximum recall subject to this deployment precision",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        metavar="N",
        help="archive a checkpoint every N epochs. 0 disables archives",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume from latest_training_checkpoint.pt or an epoch checkpoint",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size < 2:
        raise ValueError("batch_size must be at least 2 for BatchNorm")
    if args.embedding_dim <= 0 or args.classifier_hidden_dim < 2:
        raise ValueError("model dimensions must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if args.freeze_backbone_epochs < 0:
        raise ValueError("freeze_backbone_epochs cannot be negative")
    if args.focal_gamma < 0.0:
        raise ValueError("focal_gamma cannot be negative")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("weight_decay cannot be negative")
    if args.max_gradient_norm <= 0.0:
        raise ValueError("max_gradient_norm must be positive")
    if args.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("checkpoint_every cannot be negative")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.samples_per_epoch is not None and args.samples_per_epoch < 2:
        raise ValueError("samples_per_epoch must be at least 2")
    if args.minimum_precision is not None and not 0.0 < args.minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be in (0, 1]")
    for name in ("sample_positive_fraction", "deployment_positive_fraction"):
        if not 0.0 < getattr(args, name) < 1.0:
            raise ValueError(f"{name} must be strictly between 0 and 1")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest.expanduser().resolve()
    manifest_digest = file_sha256(manifest_path)
    all_records = read_manifest(manifest_path)
    if file_sha256(manifest_path) != manifest_digest:
        raise ValueError("manifest changed while it was being parsed")
    dataset_digest = dataset_sha256(
        manifest_path,
        all_records,
        expected_manifest_sha256=manifest_digest,
    )
    splits = split_records_for_experiment(all_records)
    training_records = splits.training
    validation_records = splits.validation
    calibration_records = splits.calibration
    test_records = splits.test
    training_counts = class_counts(training_records)
    validation_counts = class_counts(validation_records)
    calibration_counts = class_counts(calibration_records)
    test_counts = class_counts(test_records)
    print(f"[info] device: {device}")
    print(
        f"[info] training: {training_counts} | validation: {validation_counts} | "
        f"calibration: {calibration_counts} | test: {test_counts}"
    )
    for split_name, counts in (
        ("calibration", calibration_counts),
        ("test", test_counts),
    ):
        if counts["negative"] < 10_000:
            print(
                f"[warning] fewer than 10,000 {split_name} negatives gives coarse "
                "FPR estimates for a "
                f"{args.deployment_positive_fraction:.2%} positive deployment rate"
            )

    training_dataset = PairDataset(training_records, build_transform(training=True))
    validation_dataset = PairDataset(
        validation_records, build_transform(training=False)
    )
    calibration_dataset = PairDataset(
        calibration_records, build_transform(training=False)
    )
    test_dataset = PairDataset(test_records, build_transform(training=False))
    samples_per_epoch = args.samples_per_epoch or math.ceil(
        training_counts["positive"] / args.sample_positive_fraction
    )
    sampler = make_balanced_sampler(
        training_dataset.labels,
        positive_fraction=args.sample_positive_fraction,
        seed=args.seed,
        num_samples=samples_per_epoch,
    )
    training_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        # Drop singleton batches because projection BatchNorm requires two samples.
        drop_last=samples_per_epoch % args.batch_size == 1,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    model = build_model(
        model_name=args.model_name,
        embedding_dim=args.embedding_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
        # Resume restores all model weights without downloading pretrained weights.
        pretrained=not args.no_pretrained and args.resume is None,
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    config = {
        "manifest_sha256": manifest_digest,
        "dataset_sha256": dataset_digest,
        "model_name": args.model_name,
        "embedding_dim": args.embedding_dim,
        "classifier_hidden_dim": args.classifier_hidden_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "focal_gamma": args.focal_gamma,
        "max_gradient_norm": args.max_gradient_norm,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "minimum_precision": args.minimum_precision,
        "epochs": args.epochs,
        "samples_per_epoch": samples_per_epoch,
        "sample_positive_fraction": args.sample_positive_fraction,
        "deployment_positive_fraction": args.deployment_positive_fraction,
        "seed": args.seed,
        "image_size": 224,
        "image_mean": list(IMAGENET_MEAN),
        "image_std": list(IMAGENET_STD),
    }
    checkpoint_path = output_dir / "best_checkpoint.pt"
    latest_checkpoint_path = output_dir / "latest_training_checkpoint.pt"

    if args.resume is None:
        run_started_at = datetime.now(UTC)
        run_id = run_started_at.strftime("run_%Y%m%dT%H%M%S_%fZ")
        history: list[dict[str, Any]] = []
        best_score = -1.0
        best_model_state: dict[str, Tensor] | None = None
        best_validation_metrics: dict[str, float | int] | None = None
        epochs_without_improvement = 0
        start_epoch = 1
        latest_training_state: dict[str, Any] | None = None
        run_metadata: dict[str, Any] = {
            "run_id": run_id,
            "started_at": run_started_at.isoformat(),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_digest,
            "dataset_sha256": dataset_digest,
            "output_dir": str(output_dir),
            "device": str(device),
            "torch_version": str(torch.__version__),
            "config": config,
            "training_counts": training_counts,
            "validation_counts": validation_counts,
            "calibration_counts": calibration_counts,
            "test_counts": test_counts,
        }
    else:
        resume_path = args.resume.expanduser().resolve()
        loaded_resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        if not isinstance(loaded_resume_checkpoint, dict):
            raise TypeError("resume checkpoint must contain a mapping")
        resume_checkpoint: dict[str, Any] = loaded_resume_checkpoint
        if resume_checkpoint.get("stage") != "training":
            raise ValueError(
                "only an unfinished stage='training' checkpoint can be resumed"
            )
        finalized_fields = {
            "recommended_threshold",
            "threshold_source",
            "threshold_reason",
            "calibration_metrics",
            "test_metrics",
        }
        present_finalized_fields = finalized_fields.intersection(resume_checkpoint)
        if present_finalized_fields:
            raise ValueError(
                "training checkpoint contains finalized fields: "
                f"{', '.join(sorted(present_finalized_fields))}"
            )
        required_training_keys = {
            "optimizer_state",
            "scheduler_state",
            "scaler_state",
            "random_state",
            "best_score",
            "best_model_state",
            "best_validation_metrics",
            "epochs_without_improvement",
            "run_id",
            "run_metadata",
            "history",
            "epoch",
            "model_state",
            "config",
        }
        missing_keys = required_training_keys - resume_checkpoint.keys()
        if missing_keys:
            raise ValueError(
                "checkpoint is not resumable. Missing keys: "
                f"{', '.join(sorted(missing_keys))}"
            )
        saved_config = resume_checkpoint["config"]
        if not isinstance(saved_config, dict):
            raise TypeError("resume checkpoint config must be a mapping")
        compatible_keys = (
            "manifest_sha256",
            "dataset_sha256",
            "model_name",
            "embedding_dim",
            "classifier_hidden_dim",
            "dropout",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "focal_gamma",
            "max_gradient_norm",
            "freeze_backbone_epochs",
            "early_stopping_patience",
            "minimum_precision",
            "epochs",
            "samples_per_epoch",
            "sample_positive_fraction",
            "deployment_positive_fraction",
            "seed",
        )
        mismatched = [
            key for key in compatible_keys if saved_config.get(key) != config.get(key)
        ]
        if mismatched:
            raise ValueError(
                "resume arguments differ from the checkpoint for: "
                f"{', '.join(mismatched)}"
            )

        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        scaler.load_state_dict(resume_checkpoint["scaler_state"])
        _restore_random_state(resume_checkpoint["random_state"], sampler)

        run_id = str(resume_checkpoint["run_id"])
        history = list(resume_checkpoint.get("history", []))
        best_score = float(resume_checkpoint["best_score"])
        best_model_state = dict(resume_checkpoint["best_model_state"])
        best_validation_metrics = dict(resume_checkpoint["best_validation_metrics"])
        epochs_without_improvement = int(
            resume_checkpoint["epochs_without_improvement"]
        )
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        run_metadata = dict(resume_checkpoint["run_metadata"])
        run_metadata["output_dir"] = str(output_dir)
        run_metadata.setdefault("resumed_at", []).append(datetime.now(UTC).isoformat())
        latest_training_state = resume_checkpoint
        print(f"[info] resuming {run_id} from epoch {start_epoch}: {resume_path}")

    if start_epoch > args.epochs + 1:
        raise ValueError(
            f"checkpoint epoch {start_epoch - 1} exceeds configured "
            f"epochs {args.epochs}"
        )
    finalize_without_training = (
        start_epoch == args.epochs + 1
        or epochs_without_improvement >= args.early_stopping_patience
    )
    if finalize_without_training:
        print("[info] training is complete. Resuming calibration, test, and export")

    run_dir = output_dir / "analysis" / run_id
    epoch_checkpoint_dir = run_dir / "checkpoints"
    epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(run_metadata, run_dir / "run_config.json")
    csv_log_path, json_log_path = _write_training_history(
        run_dir, run_metadata, history
    )
    print(f"[info] analysis run: {run_dir}")

    training_stop = start_epoch if finalize_without_training else args.epochs + 1
    for epoch in range(start_epoch, training_stop):
        started = time.monotonic()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        backbone_frozen = epoch <= args.freeze_backbone_epochs
        phase = "backbone_frozen" if backbone_frozen else "fine_tuning"
        model.encoder.freeze_backbone(backbone_frozen)
        training_loss = train_one_epoch(
            model,
            training_loader,
            optimizer,
            scaler,
            device,
            focal_gamma=args.focal_gamma,
            max_gradient_norm=args.max_gradient_norm,
            backbone_frozen=backbone_frozen,
        )
        scheduler.step()

        labels, probabilities = predict(
            model,
            validation_loader,
            device,
            training_positive_fraction=args.sample_positive_fraction,
            deployment_positive_fraction=args.deployment_positive_fraction,
        )
        metrics = compute_ranking_metrics(
            labels,
            probabilities,
            deployment_positive_fraction=args.deployment_positive_fraction,
        )
        elapsed = time.monotonic() - started
        print(
            f"[info] epoch={epoch:03d}/{args.epochs} | loss={training_loss:.5f} | "
            f"{elapsed:.1f}s | {'backbone frozen' if backbone_frozen else 'fine-tuning'}"
        )
        print_ranking_metrics(metrics)

        score = float(metrics["deployment_pr_auc"])
        is_best = score > best_score
        if is_best:
            best_score = score
            best_model_state = _clone_model_state(model)
            best_validation_metrics = dict(metrics)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "completed_at": datetime.now(UTC).isoformat(),
            "phase": phase,
            "backbone_frozen": backbone_frozen,
            "training_loss": training_loss,
            "learning_rate": learning_rate,
            "elapsed_seconds": elapsed,
            "is_best": is_best,
            "epochs_without_improvement": epochs_without_improvement,
            "monitor_metric": "validation_deployment_pr_auc",
            "monitor_score": score,
        }
        epoch_record.update(
            {field: metrics[field] for field in TRAINING_LOG_FIELDS if field in metrics}
        )
        history.append(epoch_record)

        run_metadata.update(
            {
                "last_epoch": epoch,
                "best_validation_deployment_pr_auc": best_score,
                "best_epoch": max(
                    (item for item in history if item["is_best"]),
                    key=lambda item: float(item["monitor_score"]),
                )["epoch"],
            }
        )
        _atomic_write_json(run_metadata, run_dir / "run_config.json")
        _write_training_history(run_dir, run_metadata, history)

        common_checkpoint: dict[str, Any] = {
            "stage": "training",
            "run_id": run_id,
            "run_metadata": run_metadata,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "config": config,
            "metrics": metrics,
            "metrics_split": "validation",
            "monitor_metric": "validation_deployment_pr_auc",
            "training_loss": training_loss,
            "learning_rate": learning_rate,
            "elapsed_seconds": elapsed,
            "training_counts": training_counts,
            "validation_counts": validation_counts,
            "calibration_counts": calibration_counts,
            "test_counts": test_counts,
            "best_score": best_score,
            "best_model_state": best_model_state,
            "best_validation_metrics": best_validation_metrics,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }
        resumable_checkpoint = {
            **common_checkpoint,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "random_state": _capture_random_state(sampler),
        }
        latest_training_state = resumable_checkpoint
        _atomic_torch_save(resumable_checkpoint, latest_checkpoint_path)

        if args.checkpoint_every and epoch % args.checkpoint_every == 0:
            epoch_checkpoint_path = epoch_checkpoint_dir / f"epoch_{epoch:03d}.pt"
            _atomic_torch_save(resumable_checkpoint, epoch_checkpoint_path)
            print(f"[info] saved epoch checkpoint: {epoch_checkpoint_path}")

        if is_best:
            best_checkpoint = {
                **common_checkpoint,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "random_state": _capture_random_state(sampler),
            }
            _atomic_torch_save(best_checkpoint, checkpoint_path)
            print(f"[info] saved new best checkpoint: {checkpoint_path}")

        if epochs_without_improvement >= args.early_stopping_patience:
            print("[info] early stopping: validation deployment PR-AUC did not improve")
            break

    if best_model_state is None or best_validation_metrics is None:
        raise RuntimeError("training completed without a best validation model")
    if latest_training_state is None:
        raise RuntimeError("training completed without a resumable checkpoint")
    best_epoch = int(run_metadata["best_epoch"])
    best_epoch_record = next(
        record for record in history if int(record["epoch"]) == best_epoch
    )
    best_checkpoint: dict[str, Any] = {
        "stage": "finalized",
        "run_id": run_id,
        "run_metadata": run_metadata,
        "epoch": best_epoch,
        "model_state": best_model_state,
        "config": config,
        "metrics": best_validation_metrics,
        "metrics_split": "validation",
        "monitor_metric": "validation_deployment_pr_auc",
        "training_loss": best_epoch_record["training_loss"],
        "learning_rate": best_epoch_record["learning_rate"],
        "elapsed_seconds": best_epoch_record["elapsed_seconds"],
        "training_counts": training_counts,
        "validation_counts": validation_counts,
        "calibration_counts": calibration_counts,
        "test_counts": test_counts,
        "best_score": best_score,
        "history": history,
    }
    best_model = build_model_from_checkpoint(best_checkpoint).to(device)
    calibration_labels, calibration_probabilities = predict(
        best_model,
        calibration_loader,
        device,
        training_positive_fraction=args.sample_positive_fraction,
        deployment_positive_fraction=args.deployment_positive_fraction,
    )
    recommended_threshold, threshold_reason = select_threshold(
        calibration_labels,
        calibration_probabilities,
        deployment_positive_fraction=args.deployment_positive_fraction,
        minimum_precision=args.minimum_precision,
    )
    calibration_metrics = compute_metrics(
        calibration_labels,
        calibration_probabilities,
        threshold=recommended_threshold,
        deployment_positive_fraction=args.deployment_positive_fraction,
    )
    test_labels, test_probabilities = predict(
        best_model,
        test_loader,
        device,
        training_positive_fraction=args.sample_positive_fraction,
        deployment_positive_fraction=args.deployment_positive_fraction,
    )
    test_metrics = compute_metrics(
        test_labels,
        test_probabilities,
        threshold=recommended_threshold,
        deployment_positive_fraction=args.deployment_positive_fraction,
    )
    final_dataset_digest = dataset_sha256(
        manifest_path,
        all_records,
        expected_manifest_sha256=manifest_digest,
    )
    if final_dataset_digest != dataset_digest:
        raise ValueError(
            "dataset image content changed during training or evaluation. "
            "regenerate or restore the dataset before finalization"
        )
    print("[info] calibration operating point")
    print_metrics(calibration_metrics)
    print("[info] independent test result")
    print_metrics(test_metrics)

    best_checkpoint.update(
        {
            "recommended_threshold": recommended_threshold,
            "threshold_reason": threshold_reason,
            "threshold_source": "calibration",
            "calibration_metrics": calibration_metrics,
            "test_metrics": test_metrics,
        }
    )
    run_metadata.update(
        {
            "completed_at": datetime.now(UTC).isoformat(),
            "recommended_threshold": recommended_threshold,
            "threshold_source": "calibration",
            "calibration_metrics": calibration_metrics,
            "test_metrics": test_metrics,
        }
    )
    best_checkpoint["run_metadata"] = run_metadata
    _atomic_torch_save(best_checkpoint, checkpoint_path)
    _atomic_write_json(run_metadata, run_dir / "run_config.json")
    _write_training_history(run_dir, run_metadata, history)

    evaluation_report: dict[str, Any] = {
        **test_metrics,
        "split": "test",
        "threshold_source": "calibration",
        "manifest_sha256": manifest_digest,
        "dataset_sha256": dataset_digest,
    }
    _atomic_write_json(evaluation_report, output_dir / "evaluation.json")
    encoder_path, classifier_path, metadata_path = export_checkpoint_bundle(
        best_checkpoint,
        output_dir / "deploy",
        extra_metadata={
            "training_log_csv": str(csv_log_path),
            "training_log_json": str(json_log_path),
        },
    )
    latest_training_state.update(
        {
            "stage": "closed",
            "closed_at": run_metadata["completed_at"],
            "run_metadata": run_metadata,
        }
    )
    _atomic_torch_save(latest_training_state, latest_checkpoint_path)
    print(f"[info] exported encoder: {encoder_path}")
    print(f"[info] exported classifier: {classifier_path}")
    print(f"[info] exported metadata: {metadata_path}")
    print(f"[info] closed training checkpoint: {latest_checkpoint_path}")
    print(f"[info] training log CSV: {csv_log_path}")
    print(f"[info] training log JSON: {json_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
