from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from torchvision.datasets import ImageFolder

from moliv_rl.data.dataset import MultiViewDataset, lejepa_collate
from moliv_rl.data.transforms import (
    MultiViewTransform,
    get_train_transforms,
)
from moliv_rl.losses import LeJepaLoss, SIGReg, WeakSIGReg
from moliv_rl.losses.focal_loss import FocalLoss
from moliv_rl.metrics import PrecisionAverage
from moliv_rl.models import MODEL_REGISTRY, get_model  # noqa: F401
from moliv_rl.train.factory import build_optimizer, build_scheduler
from moliv_rl.train.trainer import ClassificationTrainer, LeJepaTrainer
from moliv_rl.utils.logger import get_logger
from moliv_rl.utils.reproducibility import set_seeds
from scripts.common import (
    add_data_args,
    add_model_args,
    build_dataloaders,
    build_datasets,
    build_model,
    create_config_parser,
)
from scripts.utils import (
    load_config,  # noqa: F401
    resolve_device,
)


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


def _add_training_mode_args(parser: argparse.ArgumentParser, cfg: dict[str, Any]) -> None:
    training_cfg = cfg.get("training", {})
    parser.add_argument(
        "--training-mode",
        choices=("classification", "lejepa"),
        default=training_cfg.get("mode", "classification"),
        help="Training paradigm: standard classification or LeJEPA self-supervised learning.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=training_cfg.get("num_views", 2),
        help="Number of augmented views per sample for LeJEPA training.",
    )
    parser.add_argument(
        "--use-probe",
        type=bool,
        default=training_cfg.get("use_probe", True),
        help="Whether to attach a linear probe during LeJEPA evaluation.",
    )
    parser.add_argument(
        "--projector-hidden-dim",
        type=int,
        default=training_cfg.get("projector_hidden_dim", 2048),
        help="Hidden dimension for the LeJEPA projector MLP.",
    )
    parser.add_argument(
        "--projector-out-dim",
        type=int,
        default=training_cfg.get("projector_out_dim", 128),
        help="Output dimension for the LeJEPA projector MLP.",
    )
    parser.add_argument(
        "--sigreg-lamb",
        type=float,
        default=training_cfg.get("sigreg_lamb", 0.02),
        help="SIGReg regularization weight in the LeJEPA loss.",
    )
    parser.add_argument(
        "--sigreg-sketch-dim",
        type=int,
        default=training_cfg.get("sigreg_sketch_dim", 64),
        help="Sketch dimension for SIGReg.",
    )
    parser.add_argument(
        "--sigreg-integration-points",
        type=int,
        default=training_cfg.get("sigreg_integration_points", 17),
        help="Number of integration points for SIGReg.",
    )
    parser.add_argument(
        "--sigreg-integration-t-max",
        type=float,
        default=training_cfg.get("sigreg_integration_t_max", 3.0),
        help="Maximum integration time for SIGReg.",
    )


def _add_dataset_args(parser: argparse.ArgumentParser, cfg: dict[str, Any]) -> None:
    data_cfg = cfg.get("data", {})
    parser.add_argument(
        "--dataset-type",
        choices=("imagefolder", "streaming"),
        default=data_cfg.get("dataset_type", "imagefolder"),
        help="Dataset backend for training and validation data.",
    )
    parser.add_argument(
        "--dataset-id",
        default=data_cfg.get("dataset_id", "moliv-rl/gameqa-140k"),
        help="Hugging Face dataset id for streaming mode.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(data_cfg.get("data_dir", "data")),
        help="Root directory containing train/ and val/ folders for ImageFolder mode.",
    )
    parser.add_argument(
        "--train-split",
        default=data_cfg.get("train_split", "train"),
        help="Dataset split to use for training.",
    )
    parser.add_argument(
        "--val-split",
        default=data_cfg.get("val_split", "val"),
        help="Dataset split to use for validation.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=data_cfg.get("max_train_samples"),
        help="Optional cap on training samples for streaming datasets.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=data_cfg.get("max_val_samples"),
        help="Optional cap on validation samples for streaming datasets.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=data_cfg.get("image_size", 256),
        help="Target image size for transforms.",
    )
    parser.add_argument(
        "--pin-memory",
        type=bool,
        default=data_cfg.get("pin_memory"),
        help="Enable pinned memory for DataLoader. Defaults based on device when unset.",
    )
    parser.add_argument(
        "--persistent-workers",
        type=bool,
        default=data_cfg.get("persistent_workers"),
        help="Enable persistent DataLoader workers. Defaults based on num_workers when unset.",
    )


def parse_args() -> argparse.Namespace:
    r"""parse_args() -> Namespace

    Build the training CLI parser, merge defaults from the active config file,
    and return parsed arguments.
    """
    parser = create_config_parser()
    cfg: dict[str, Any] = {}

    add_data_args(parser, cfg)
    add_model_args(parser, cfg)
    _add_training_mode_args(parser, cfg)
    _add_dataset_args(parser, cfg)

    training_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(training_cfg.get("config", "configs/cpu_test.yaml")),
        help="Path to the YAML config for this run.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=training_cfg.get("epochs", 3),
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=training_cfg.get("batch_size", 1),
        help="Batch size for training and validation loaders.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=training_cfg.get("grad_accum_steps", 1),
        help="Number of gradient accumulation steps before optimizer update.",
    )
    parser.add_argument(
        "--optimizer",
        default=training_cfg.get("optimizer", "adam"),
        help="Optimizer name recognized by build_optimizer.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=training_cfg.get("lr", 1e-3),
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=training_cfg.get("weight_decay", 0.0),
        help="Weight decay applied to the optimizer.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=training_cfg.get("momentum", 0.9),
        help="Momentum factor for SGD-style optimizers.",
    )
    parser.add_argument(
        "--scheduler",
        default=training_cfg.get("scheduler", "none"),
        help="Learning rate scheduler name.",
    )
    parser.add_argument(
        "--scheduler-interval",
        choices=("epoch", "step"),
        default=training_cfg.get("scheduler_interval", "epoch"),
        help="Scheduler stepping frequency.",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=training_cfg.get("eta_min", 0.0),
        help="Minimum learning rate for cosine annealing.",
    )
    parser.add_argument(
        "--loss",
        default=training_cfg.get("loss", "cross_entropy"),
        help="Loss function name.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=training_cfg.get("label_smoothing", 0.0),
        help="Label smoothing factor for cross entropy / focal loss.",
    )
    parser.add_argument(
        "--precision-average",
        default=training_cfg.get("precision_average", "macro"),
        help="Precision averaging strategy.",
    )
    parser.add_argument(
        "--sigreg-type",
        default=training_cfg.get("sigreg_type", "strong"),
        help="SIGReg implementation to use.",
    )
    parser.add_argument(
        "--use-amp",
        type=bool,
        default=training_cfg.get("use_amp", False),
        help="Enable automatic mixed precision on CUDA.",
    )
    parser.add_argument(
        "--device",
        default=training_cfg.get("device", "auto"),
        help="Compute device. Use 'cpu' or 'cuda' or 'auto'.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=data_cfg.get("num_workers", 0),
        help="DataLoader worker processes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=training_cfg.get("seed", 42),
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(training_cfg.get("checkpoint_dir", "checkpoints")),
        help="Directory for saved checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=training_cfg.get("checkpoint"),
        help="Optional checkpoint path to resume training or evaluation.",
    )
    parser.add_argument(
        "--save-last",
        type=bool,
        default=training_cfg.get("save_last", True),
        help="Save a last checkpoint every epoch.",
    )
    parser.add_argument(
        "--save-best",
        type=bool,
        default=training_cfg.get("save_best", True),
        help="Save a best checkpoint when validation accuracy improves.",
    )
    parser.add_argument(
        "--safe-load",
        type=bool,
        default=training_cfg.get("safe_load", True),
        help="Use weights_only loading when restoring checkpoints.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(training_cfg.get("results_dir", "results")),
        help="Directory for metrics and plots.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path(training_cfg.get("metrics_file")) if training_cfg.get("metrics_file") else None,
        help="Explicit metrics file path.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path(training_cfg.get("plots_dir")) if training_cfg.get("plots_dir") else None,
        help="Explicit plots directory path.",
    )

    return parser.parse_args()


def _build_sigreg_loss(args: argparse.Namespace) -> nn.Module:
    r"""_build_sigreg_loss(args) -> nn.Module

    Select the configured SIGReg implementation.
    """
    sigreg_type = getattr(args, "sigreg_type", "strong")
    if sigreg_type == "weak":
        return WeakSIGReg(
            sketch_dim=getattr(args, "sigreg_sketch_dim", 64),
        )
    return SIGReg(
        sketch_dim=getattr(args, "sigreg_sketch_dim", 64),
        num_integration_points=getattr(args, "sigreg_integration_points", 17),
        integration_t_max=getattr(args, "sigreg_integration_t_max", 3.0),
    )


def _build_lejepa_projector(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
) -> nn.Module:
    r"""Build a simple 2-layer MLP projector for LeJEPA."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


def build_trainer(
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
    steps_per_epoch: int,
) -> Any:
    r"""build_trainer(args, device, logger, steps_per_epoch) -> Trainer

    Instantiate model architecture, loss criterion, optimizer, scheduler, and
    trainer. Dispatches between standard classification and LeJEPA
    self-supervised training based on ``args.training_mode``.

    Args:
        args (argparse.Namespace): Experiment hyperparameter configuration.
        device (torch.device): Compute device for model placement.
        logger (logging.Logger): Logger for tracking training progress.
        steps_per_epoch (int): Total batch steps per epoch used to size
            step-wise schedulers.

    Returns:
        ClassificationTrainer | LeJepaTrainer: Initialized trainer instance.
    """
    training_mode: Literal["classification", "lejepa"] = getattr(
        args, "training_mode", "classification"
    )

    if training_mode == "lejepa":
        if args.model_name in {"classification_model", "mobilenetv3_pretrained"}:
            raise ValueError(
                "LeJEPA training requires a backbone model without the classification head. "
                "Use --model-name my_model."
            )

        model = build_model(args, device)

        projector = _build_lejepa_projector(
            in_dim=args.out_channels,
            hidden_dim=getattr(args, "projector_hidden_dim", 2048),
            out_dim=getattr(args, "projector_out_dim", 128),
        )

        sigreg_fn = SIGReg(
            sketch_dim=getattr(args, "sigreg_sketch_dim", 64),
            num_integration_points=getattr(args, "sigreg_integration_points", 17),
            integration_t_max=getattr(args, "sigreg_integration_t_max", 3.0),
        )
        lejepa_criterion = LeJepaLoss(
            sigreg_loss_fn=sigreg_fn,
            lamb=getattr(args, "sigreg_lamb", 0.02),
            normalize_projections=True,
        )

        probe_criterion = None
        if getattr(args, "use_probe", True):
            probe_criterion = nn.Linear(
                getattr(args, "projector_out_dim", 128),
                getattr(args, "num_classes", 30),
            )

        parameters = list(model.parameters()) + list(projector.parameters())
        if probe_criterion is not None:
            parameters += list(probe_criterion.parameters())

        optimizer = build_optimizer(
            parameters,
            name=args.optimizer,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            momentum=getattr(args, "momentum", 0.9),
        )

        scheduler = None
        if args.scheduler in ("cosine_annealing", "cosine"):
            optimizer_steps_per_epoch = (
                steps_per_epoch + args.grad_accum_steps - 1
            ) // args.grad_accum_steps
            scheduler = build_scheduler(
                optimizer,
                name=args.scheduler,
                scheduler_interval=args.scheduler_interval,
                epochs=args.epochs,
                steps_per_epoch=optimizer_steps_per_epoch,
                eta_min=args.eta_min,
            )

        return LeJepaTrainer(
            model=model,
            projector=projector,
            optimizer=optimizer,
            lejepa_criterion=lejepa_criterion,
            probe_criterion=probe_criterion,
            device=device,
            scheduler=scheduler,
            scheduler_interval=args.scheduler_interval,
            logger=logger,
            use_amp=args.use_amp,
        )

    # Classification mode
    model = build_model(args, device)

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

    sigreg_loss_fn = _build_sigreg_loss(args)

    optimizer = build_optimizer(
        model.parameters(),
        name=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        momentum=getattr(args, "momentum", 0.9),
    )

    scheduler = None
    if args.scheduler in ("cosine_annealing", "cosine"):
        optimizer_steps_per_epoch = (
            steps_per_epoch + args.grad_accum_steps - 1
        ) // args.grad_accum_steps
        scheduler = build_scheduler(
            optimizer,
            name=args.scheduler,
            scheduler_interval=args.scheduler_interval,
            epochs=args.epochs,
            steps_per_epoch=optimizer_steps_per_epoch,
            eta_min=args.eta_min,
        )

    return ClassificationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        scheduler_interval=args.scheduler_interval,
        logger=logger,
        use_amp=args.use_amp,
        sigreg_loss_fn=sigreg_loss_fn,
        sigreg_weight=getattr(args, "sigreg_weight", 0.0),
    )


def _plot_history(history: list[dict[str, Any]], plots_dir: Path) -> None:
    r"""_plot_history(history, plots_dir)

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


def _log_class_distribution(
    train_dataset: Any,
    val_dataset: Any,
    logger: logging.Logger,
) -> None:
    r"""_log_class_distribution(train_dataset, val_dataset, logger)

    Log class distribution for datasets that expose it.
    """
    if hasattr(train_dataset, "class_distribution"):
        train_dist = train_dataset.class_distribution()
        if train_dist:
            logger.info("Train class distribution: %s", dict(sorted(train_dist.items())))

    if val_dataset is not None and hasattr(val_dataset, "class_distribution"):
        val_dist = val_dataset.class_distribution()
        if val_dist:
            logger.info("Val class distribution: %s", dict(sorted(val_dist.items())))


def _build_lejepa_dataloaders(
    train_dataset: Any,
    val_dataset: Any | None,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader | None, Any]:
    r"""_build_lejepa_dataloaders(train_dataset, val_dataset, args, device) -> Tuple[DataLoader, DataLoader | None, Any]

    Wrap datasets in multi-view transforms and build deterministic DataLoaders
    for LeJEPA self-supervised training.

    Args:
        train_dataset (Any): Raw training dataset.
        val_dataset (Any, optional): Raw validation dataset.
        args (argparse.Namespace): Experiment hyperparameter configuration.
        device (torch.device): Compute device for memory pinning.

    Returns:
        tuple: ``(train_loader, val_loader, train_mv_dataset)`` with ``val_loader`` possibly ``None``.
    """
    mv_transform = MultiViewTransform(
        transform=get_train_transforms(image_size=args.image_size),
        num_views=getattr(args, "num_views", 2),
    )
    train_mv_dataset = MultiViewDataset(
        dataset=train_dataset,
        transform=mv_transform,
        num_views=getattr(args, "num_views", 2),
    )

    val_mv_dataset = None
    if val_dataset is not None:
        val_mv_dataset = MultiViewDataset(
            dataset=val_dataset,
            transform=mv_transform,
            num_views=getattr(args, "num_views", 2),
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    pin_memory = args.pin_memory if args.pin_memory is not None else (device.type == "cuda")
    use_persistent = args.persistent_workers if args.persistent_workers is not None else (args.num_workers > 0)

    train_loader = DataLoader(
        train_mv_dataset,
        batch_size=args.batch_size,
        shuffle=not isinstance(train_mv_dataset, IterableDataset),
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        generator=generator,
        collate_fn=lejepa_collate,
    )

    val_loader = None
    if val_mv_dataset is not None:
        val_loader = DataLoader(
            val_mv_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=use_persistent,
            generator=generator,
            collate_fn=lejepa_collate,
        )

    return train_loader, val_loader, train_mv_dataset


def _build_checkpoint_metadata(
    args: argparse.Namespace,
    train_dataset: ImageFolder | Any,
    val_metrics: dict[str, float],
) -> dict[str, Any]:
    r"""_build_checkpoint_metadata(args, train_dataset, val_metrics) -> dict

    Construct checkpoint metadata payload with serialized args and dataset info.

    Args:
        args (argparse.Namespace): Experiment hyperparameter configuration.
        train_dataset (ImageFolder or Any): Training dataset used for class metadata.
        val_metrics (dict): Current validation metrics to embed in checkpoint.

    Returns:
        dict: Checkpoint metadata dictionary.
    """
    metadata: dict[str, Any] = {
        "args": _serialize_args(vars(args)),
        **val_metrics,
    }

    if isinstance(train_dataset, ImageFolder):
        metadata["classes"] = train_dataset.classes
        metadata["class_to_idx"] = train_dataset.class_to_idx
    elif hasattr(train_dataset, "label2id"):
        metadata["classes"] = list(train_dataset.label2id.keys())
        metadata["class_to_idx"] = train_dataset.label2id

    return metadata


def _train_and_validate(
    trainer: ClassificationTrainer | LeJepaTrainer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    args: argparse.Namespace,
    training_mode: Literal["classification", "lejepa"],
    logger: logging.Logger,
    train_dataset: ImageFolder | Any,
) -> tuple[list[dict[str, Any]], float, int]:
    r"""_train_and_validate(trainer, train_loader, val_loader, args, training_mode, logger, train_dataset) -> Tuple[list[dict], float, int]

    Execute the training loop, logging, and checkpointing for all epochs.

    Args:
        trainer (ClassificationTrainer | LeJepaTrainer): Initialized trainer instance.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader, optional): Validation data loader.
        args (argparse.Namespace): Experiment hyperparameter configuration.
        training_mode (str): Active training paradigm.
        logger (logging.Logger): Logger for training progress.
        train_dataset (ImageFolder or Any): Training dataset used for checkpoint metadata.

    Returns:
        tuple: ``(history, best_val_acc, best_epoch)`` from the training run.
    """
    best_val_acc = float("-inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(
            dataloader=train_loader,
            epoch=epoch,
            grad_accum_steps=args.grad_accum_steps,
        )

        if training_mode == "lejepa":
            assert isinstance(trainer, LeJepaTrainer)
            train_metrics = cast(dict[str, float], train_metrics)
            train_loss = float(train_metrics.get("train_loss", 0.0))
            val_metrics = (
                trainer.evaluate_probe(
                    dataloader=val_loader,
                    precision_average=cast(
                        PrecisionAverage,
                        args.precision_average,
                    )
                    if val_loader is not None
                    else "macro",
                )
                if val_loader is not None
                else {"val_loss": 0.0, "val_acc": 0.0, "val_precision": 0.0}
            )
        else:
            assert isinstance(trainer, ClassificationTrainer)
            train_metrics = cast(float, train_metrics)
            train_loss = float(train_metrics)
            val_metrics = (
                trainer.evaluate(
                    dataloader=val_loader,
                    precision_average=cast(
                        PrecisionAverage,
                        args.precision_average,
                    )
                    if val_loader is not None
                    else "macro",
                )
                if val_loader is not None
                else {"val_loss": 0.0, "val_acc": 0.0, "val_precision": 0.0}
            )

        if trainer.scheduler is not None:
            learning_rate = trainer.scheduler.get_last_lr()[0]
        else:
            assert trainer.optimizer is not None
            learning_rate = float(trainer.optimizer.param_groups[0]["lr"])

        history_entry: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics.get("val_loss", 0.0),
            "val_acc": val_metrics.get("val_acc", 0.0),
            "val_precision": val_metrics.get("val_precision", 0.0),
            "lr": learning_rate,
        }

        if training_mode == "lejepa":
            assert isinstance(train_metrics, dict)
            history_entry.update(
                {
                    "invariance_loss": train_metrics.get("invariance_loss", 0.0),
                    "sigreg_loss": train_metrics.get("sigreg_loss", 0.0),
                    "probe_loss": train_metrics.get("probe_loss", 0.0),
                }
            )

        history.append(history_entry)

        log_msg = (
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"val_loss={val_metrics.get('val_loss', 0.0):.6f} | "
            f"val_acc={val_metrics.get('val_acc', 0.0):.4f} | "
            f"val_precision={val_metrics.get('val_precision', 0.0):.4f}"
        )
        if training_mode == "lejepa":
            assert isinstance(train_metrics, dict)
            log_msg += (
                f" | inv={train_metrics.get('invariance_loss', 0.0):.6f} | "
                f"sigreg={train_metrics.get('sigreg_loss', 0.0):.6f} | "
                f"probe={train_metrics.get('probe_loss', 0.0):.6f}"
            )
        logger.info(log_msg)

        checkpoint_metadata = _build_checkpoint_metadata(
            args, train_dataset, val_metrics
        )

        if args.save_last:
            trainer.save_checkpoint(
                args.checkpoint_dir / "model_last.pth",
                epoch=epoch,
                extra=checkpoint_metadata,
            )

        if val_metrics.get("val_acc", 0.0) >= best_val_acc:
            best_val_acc = val_metrics.get("val_acc", 0.0)
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

    return history, best_val_acc, best_epoch


def _finalize_training(
    result_paths: ResultPaths,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    best_val_acc: float,
    best_epoch: int,
    logger: logging.Logger,
) -> None:
    r"""_finalize_training(result_paths, args, history, best_val_acc, best_epoch, logger)

    Persist metrics, plots, and final training summary logs.

    Args:
        result_paths (ResultPaths): Resolved output paths for the training run.
        args (argparse.Namespace): Experiment hyperparameter configuration.
        history (list[dict]): Per-epoch training history.
        best_val_acc (float): Best validation accuracy observed.
        best_epoch (int): Epoch at which best validation accuracy was achieved.
        logger (logging.Logger): Logger for final summary output.
    """
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

    if hasattr(train_dataset, "class_distribution"):
        train_dist = train_dataset.class_distribution()
        if train_dist:
            logger.info("Train class distribution: %s", dict(sorted(train_dist.items())))

    if val_dataset is not None and hasattr(val_dataset, "class_distribution"):
        val_dist = val_dataset.class_distribution()
        if val_dist:
            logger.info("Val class distribution: %s", dict(sorted(val_dist.items())))

    training_mode: Literal["classification", "lejepa"] = getattr(
        args, "training_mode", "classification"
    )

    if training_mode == "lejepa":
        train_loader, val_loader, train_dataset = _build_lejepa_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=args,
            device=device,
        )
    else:
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

    history, best_val_acc, best_epoch = _train_and_validate(
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args,
        training_mode=training_mode,
        logger=logger,
        train_dataset=train_dataset,
    )

    _finalize_training(
        result_paths=result_paths,
        args=args,
        history=history,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        logger=logger,
    )


if __name__ == "__main__":
    main()