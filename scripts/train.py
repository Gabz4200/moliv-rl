from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from moliv_rl.models import MODEL_REGISTRY, get_model
from moliv_rl.train.trainer import ClassificationTrainer
from moliv_rl.utils.logger import get_logger
from moliv_rl.utils.reproducibility import set_seeds
from scripts.common import (
    add_data_args,
    add_model_args,
    build_datasets,
    build_dataloaders,
    build_local_datasets,
    build_streaming_datasets,
    create_config_parser,
    load_yaml_config,
)
from scripts.utils import load_config, resolve_device


def _serialize_args(
    args: argparse.Namespace | dict[str, Any],
) -> dict[str, Any]:
    r"""_serialize_args(args) -> dict

    Convert argparse namespace values into primitive types safe for
    PyTorch checkpoint serialization.

    Args:
        args (Namespace or dict): Raw CLI parsed arguments or dictionary.

    Returns:
        dict: Sanitized dictionary with :class:`~pathlib.Path` objects cast to strings.
    """
    items = vars(args).items() if isinstance(args, argparse.Namespace) else args.items()
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in items
    }


@dataclass
class ResultPaths:
    """Resolved output paths for a training run."""

    metrics_file: Path
    plots_dir: Path


def _build_result_paths(
    results_dir: Path,
    metrics_file: Path | None,
    plots_dir: Path | None,
) -> ResultPaths:
    r"""_build_result_paths(results_dir, metrics_file, plots_dir) -> ResultPaths

    Build default metrics/plots paths inside results_dir using the current
    timestamp when explicit paths are not provided.
    """
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H-%M")
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = (
        metrics_file
        if metrics_file is not None
        else results_dir / f"metrics_{timestamp}.txt"
    )
    plots_path = (
        plots_dir
        if plots_dir is not None
        else results_dir / f"plots_{timestamp}"
    )
    plots_path.mkdir(parents=True, exist_ok=True)

    return ResultPaths(metrics_file=metrics_path, plots_dir=plots_path)


def _add_optimizer_args(parser: argparse.ArgumentParser, cfg: dict[str, Any]) -> None:
    optimizer_cfg = cfg.get("optimizer", {})
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "adam", "sgd"),
        default=optimizer_cfg.get("name", "adamw"),
        help="Optimizer algorithm.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=optimizer_cfg.get("learning_rate", 1e-3),
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=optimizer_cfg.get("weight_decay", 1e-4),
        help="Optimizer weight decay.",
    )


def _add_loss_args(parser: argparse.ArgumentParser, cfg: dict[str, Any]) -> None:
    loss_cfg = cfg.get("loss", {})
    parser.add_argument(
        "--loss",
        choices=("cross_entropy", "focal_loss", "focal"),
        default=loss_cfg.get("name", "cross_entropy"),
        help="Loss function criterion.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=loss_cfg.get("label_smoothing", 0.0),
        help="Label smoothing factor.",
    )


def _add_scheduler_args(parser: argparse.ArgumentParser, cfg: dict[str, Any]) -> None:
    scheduler_cfg = cfg.get("scheduler", {})
    parser.add_argument(
        "--scheduler",
        choices=("cosine_annealing", "cosine", "none"),
        default=scheduler_cfg.get("name", "cosine_annealing"),
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--scheduler-interval",
        choices=("epoch", "step"),
        default=scheduler_cfg.get("interval", "epoch"),
        help="How often to step the learning-rate scheduler.",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=scheduler_cfg.get("eta_min", 0.0),
        help="Minimum learning rate for cosine annealing.",
    )


def _add_training_args(parser: argparse.ArgumentParser, cfg: dict[str, Any]) -> None:
    training_cfg = cfg.get("training", {})
    runtime_cfg = cfg.get("runtime", {})
    project_cfg = cfg.get("project", {})
    paths_cfg = cfg.get("paths", {})

    parser.add_argument(
        "--epochs",
        type=int,
        default=training_cfg.get("epochs", 10),
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=training_cfg.get("gradient_accumulation_steps", 1),
        help="Number of batches to accumulate before optimizer.step().",
    )
    parser.add_argument(
        "--precision-average",
        choices=("micro", "macro", "weighted"),
        default=training_cfg.get("precision_average", "macro"),
        help="Averaging strategy for validation precision.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=runtime_cfg.get("device", "auto"),
        help="Device, for example cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=project_cfg.get("seed", 42069),
        help="Random seed.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=training_cfg.get("use_amp", False),
        help="Use CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=runtime_cfg.get("save_best", True),
        help="Save best checkpoint on validation accuracy improvement.",
    )
    parser.add_argument(
        "--save-last",
        action=argparse.BooleanOptionalAction,
        default=runtime_cfg.get("save_last", True),
        help="Save checkpoint after every epoch.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(paths_cfg.get("results_dir", "results")),
        help="Root directory for metrics and plots outputs.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=paths_cfg.get("metrics_file"),
        help="Explicit metrics txt path. Defaults to a timestamped file",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=paths_cfg.get("plots_dir"),
        help="Explicit plots directory. Defaults to a timestamped directory",
    )


def parse_args() -> argparse.Namespace:
    r"""parse_args() -> argparse.Namespace

    Parse CLI options merged with default YAML configuration settings.

    Returns:
        argparse.Namespace: Populated argument namespace.
    """
    config_parser = create_config_parser()
    known_args, _ = config_parser.parse_known_args()
    cfg = load_yaml_config(known_args)

    parser = argparse.ArgumentParser(
        description="Train a moliv_rl PyTorch classifier.",
        parents=[config_parser],
    )

    add_data_args(parser, cfg)
    add_model_args(parser, cfg)
    _add_optimizer_args(parser, cfg)
    _add_loss_args(parser, cfg)
    _add_scheduler_args(parser, cfg)
    _add_training_args(parser, cfg)

    return parser.parse_args()






def build_trainer(
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
    steps_per_epoch: int,
) -> ClassificationTrainer:
    r"""build_trainer(args, device, logger, steps_per_epoch) -> ClassificationTrainer

    Instantiate model architecture, loss criterion, optimizer, scheduler, and trainer.

    Args:
        args (argparse.Namespace): Experiment hyperparameter configuration.
        device (torch.device): Compute device for model placement.
        logger (logging.Logger): Logger for tracking training progress.
        steps_per_epoch (int): Total batch steps per epoch used to size
            step-wise schedulers.

    Returns:
        ClassificationTrainer: Initialized trainer instance.
    """
    model_kwargs: dict[str, Any] = {
        "block_dims": args.block_dims,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "patch_size": args.patch_size,
        "dropout": args.dropout,
    }
    if args.model_name == "classification_model":
        model_kwargs["num_classes"] = args.num_classes

    model = get_model(
        model_name=args.model_name,
        optimize=args.optimize_model,
        **model_kwargs,
    )

    if args.loss in ("focal_loss", "focal"):
        criterion: nn.Module = FocalLoss(
            label_smoothing=args.label_smoothing,
        )
    elif args.loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing,
        )
    else:
        raise ValueError(f"Unsupported loss function: {args.loss}")

    if args.optimizer == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    if args.scheduler in ("cosine_annealing", "cosine"):
        optimizer_steps_per_epoch = (
            steps_per_epoch + args.grad_accum_steps - 1
        ) // args.grad_accum_steps
        scheduler_t_max = (
            args.epochs
            if args.scheduler_interval == "epoch"
            else args.epochs * optimizer_steps_per_epoch
        )

        scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_t_max,
                eta_min=args.eta_min,
            )
        )
    elif args.scheduler in ("none", None):
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    return ClassificationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        scheduler_interval=args.scheduler_interval,
        logger=logger,
        use_amp=args.use_amp,
    )


def _save_metrics(
    metrics_file: Path,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    best_val_acc: float,
    best_epoch: int,
) -> None:
    r"""_save_metrics(metrics_file, args, history, best_val_acc, best_epoch)

    Persist training run metadata and per-epoch metrics to a text file.
    """
    lines: list[str] = []
    lines.append("Training run metrics")
    lines.append("=" * 80)
    lines.append(f"timestamp: {datetime.now(tz=UTC).isoformat()}")
    lines.append(f"config: {args.config}")
    lines.append(f"dataset_type: {args.dataset_type}")
    lines.append(f"dataset_id: {args.dataset_id}")
    lines.append(f"train_split: {args.train_split}")
    lines.append(f"val_split: {args.val_split}")
    lines.append(f"image_size: {args.image_size}")
    lines.append(f"batch_size: {args.batch_size}")
    lines.append(f"device: {args.device}")
    lines.append(f"epochs: {args.epochs}")
    lines.append(f"gradient_accumulation_steps: {args.grad_accum_steps}")
    lines.append(f"use_amp: {args.use_amp}")
    lines.append(f"best_val_acc: {best_val_acc:.6f}")
    lines.append(f"best_epoch: {best_epoch}")
    lines.append("")
    lines.append("Per-epoch metrics")
    lines.append("-" * 80)
    lines.append(
        f"{'epoch':>5} | {'train_loss':>10} | {'val_loss':>10} | {'val_acc':>10} | {'val_precision':>13} | {'lr':>12}"
    )
    lines.append("-" * 80)
    for entry in history:
        lines.append(
            f"{entry['epoch']:>5} | {entry['train_loss']:>10.6f} | {entry['val_loss']:>10.6f} | {entry['val_acc']:>10.6f} | {entry['val_precision']:>13.6f} | {entry['lr']:>12.6g}"
        )
    lines.append("")
    lines.append("Raw history (JSON)")
    lines.append("-" * 80)
    lines.append(json.dumps(history, indent=2))

    metrics_file.write_text("\n".join(lines), encoding="utf-8")


def _save_plots(
    plots_dir: Path,
    history: list[dict[str, Any]],
) -> None:
    r"""_save_plots(plots_dir, history)

    Generate line plots for training/validation loss, accuracy, and precision.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    val_acc = [entry["val_acc"] for entry in history]
    val_precision = [entry["val_precision"] for entry in history]

    plots = {
        "loss": (train_loss, val_loss, "Loss", "train_loss", "val_loss"),
        "accuracy": (None, val_acc, "Accuracy", None, "val_acc"),
        "precision": (None, val_precision, "Precision", None, "val_precision"),
    }

    for name, (train_values, val_values, title, train_label, val_label) in plots.items():
        plt.figure(figsize=(8, 5))
        if train_values is not None:
            plt.plot(epochs, train_values, label=train_label, marker="o")
        plt.plot(epochs, val_values, label=val_label, marker="o")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plots_dir / f"{name}.png", dpi=150)
        plt.close()

    # Combined metrics plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_acc, label="val_acc", marker="o")
    plt.plot(epochs, val_precision, label="val_precision", marker="o")
    plt.title("Validation Accuracy and Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "combined_metrics.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()

    set_seeds(args.seed)

    device = resolve_device(args.device)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.checkpoint_dir / "logs"
    logger = get_logger(
        "train",
        log_dir=log_dir,
    )

    result_paths = _build_result_paths(
        results_dir=args.results_dir,
        metrics_file=args.metrics_file,
        plots_dir=args.plots_dir,
    )

    logger.info("Starting training")
    logger.info("Device: %s", device)
    logger.info("Arguments: %s", _serialize_args(vars(args)))
    logger.info("Metrics file: %s", result_paths.metrics_file)
    logger.info("Plots directory: %s", result_paths.plots_dir)

    train_dataset, val_dataset = build_datasets(args)

    if isinstance(train_dataset, ImageFolder):
        logger.info(
            "Classes: %s",
            train_dataset.classes,
        )
        logger.info(
            "Training samples: %d",
            len(train_dataset),
        )
        if val_dataset is not None:
            logger.info(
                "Validation samples: %d",
                len(val_dataset),
            )
    else:
        logger.info(
            "Dataset: %s",
            args.dataset_id,
        )
        if hasattr(train_dataset, "label2id") and train_dataset.label2id:
            logger.info(
                "Classes: %s",
                len(train_dataset.label2id),
            )

    train_loader, val_loader = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=args,
        device=device,
    )

    trainer = build_trainer(
        args=args,
        device=device,
        logger=logger,
        steps_per_epoch=len(train_loader),
    )

    best_val_acc = float("-inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = trainer.train_epoch(
            dataloader=train_loader,
            epoch=epoch,
            grad_accum_steps=args.grad_accum_steps,
        )

        val_metrics = trainer.evaluate(
            dataloader=val_loader,
            precision_average=cast(
                PrecisionAverage,
                args.precision_average,
            )
            if val_loader is not None
            else "macro",
        )

        learning_rate = (
            trainer.scheduler.get_last_lr()[0]
            if trainer.scheduler is not None
            else float(trainer.optimizer.param_groups[0]["lr"])
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "val_acc": val_metrics["val_acc"],
                "val_precision": val_metrics["val_precision"],
                "lr": learning_rate,
            }
        )

        logger.info(
            "Epoch %03d | train_loss=%.6f | val_loss=%.6f | "
            "val_acc=%.4f | val_precision=%.4f",
            epoch,
            train_loss,
            val_metrics["val_loss"],
            val_metrics["val_acc"],
            val_metrics["val_precision"],
        )

        checkpoint_metadata: dict[str, Any] = {
            "args": _serialize_args(vars(args)),
            **val_metrics,
        }

        if isinstance(train_dataset, ImageFolder):
            checkpoint_metadata["classes"] = train_dataset.classes
            checkpoint_metadata["class_to_idx"] = train_dataset.class_to_idx
        elif hasattr(train_dataset, "label2id"):
            checkpoint_metadata["classes"] = list(train_dataset.label2id.keys())
            checkpoint_metadata["class_to_idx"] = train_dataset.label2id

        if args.save_last:
            trainer.save_checkpoint(
                args.checkpoint_dir / "model_last.pth",
                epoch=epoch,
                extra=checkpoint_metadata,
            )

        if val_metrics["val_acc"] >= best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            best_epoch = epoch

            if args.save_best:
                trainer.save_checkpoint(
                    args.checkpoint_dir / "model_best.pth",
                    epoch=epoch,
                    extra={
                        **checkpoint_metadata,
                        "best_val_acc": best_val_acc,
                    },
                )

                logger.info(
                    "New best checkpoint: val_acc=%.4f",
                    best_val_acc,
                )

    _save_metrics(
        metrics_file=result_paths.metrics_file,
        args=args,
        history=history,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
    )
    logger.info("Metrics saved to %s", result_paths.metrics_file)

    _save_plots(plots_dir=result_paths.plots_dir, history=history)
    logger.info("Plots saved to %s", result_paths.plots_dir)

    logger.info(
        "Training complete. Best validation accuracy: %.4f at epoch %d",
        best_val_acc,
        best_epoch,
    )


if __name__ == "__main__":
    main()
