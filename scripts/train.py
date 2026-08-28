from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from moliv_rl.data import (
    DataLoaderConfig,
    get_dataloaders,
    get_train_transforms,
    get_val_transforms,
)
from moliv_rl.losses import FocalLoss
from moliv_rl.metrics import PrecisionAverage
from moliv_rl.models import MODEL_REGISTRY, get_model
from moliv_rl.train.trainer import ClassificationTrainer
from moliv_rl.utils.logger import get_logger
from moliv_rl.utils.reproducibility import seed_worker, set_seeds
from scripts.utils import load_config, resolve_device


def _serialize_args(
    args: argparse.Namespace | dict[str, Any],
) -> dict[str, Any]:
    r"""_serialize_args(args) -> dict

    Convert argparse namespace values into primitive types safe for PyTorch checkpoint serialization.

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


def _create_config_parser() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    known_args, _ = config_parser.parse_known_args()
    return config_parser, known_args


def _load_config(known_args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if known_args.config and known_args.config.is_file():
        cfg = load_config(known_args.config)
    elif known_args.config and known_args.config != Path("configs/default.yaml"):
        raise FileNotFoundError(f"Config file not found: {known_args.config}")
    return cfg


def _add_data_args(parser: argparse.ArgumentParser, cfg: dict[str, Any]) -> None:
    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(paths_cfg.get("data_dir", "data")),
        help="Directory containing train and val subdirectories.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(paths_cfg.get("checkpoint_dir", "checkpoints")),
        help="Directory used for checkpoints and logs.",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default=data_cfg.get("train_split", "train"),
        help="Subdirectory name for training data.",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default=data_cfg.get("validation_split", "val"),
        help="Subdirectory name for validation data.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=data_cfg.get("image_size", 64),
        help="Input image height and width.",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        default=data_cfg.get("resize_scale", 1.15),
        help="Resize scale used before cropping.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=data_cfg.get("batch_size", 32),
        help="Training and validation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=data_cfg.get("num_workers", 2),
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=data_cfg.get("pin_memory", None),
        help="Pin memory in DataLoader.",
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=data_cfg.get("persistent_workers", None),
        help="Use persistent workers in DataLoader.",
    )


def _add_model_args(parser: argparse.ArgumentParser, cfg: dict[str, Any]) -> None:
    model_cfg = cfg.get("model", {})
    parser.add_argument(
        "--model-name",
        type=str,
        choices=tuple(MODEL_REGISTRY.keys()),
        default=model_cfg.get("name", "classification_model"),
        help="Model architecture name.",
    )
    parser.add_argument(
        "--block-dims",
        type=int,
        nargs="+",
        default=model_cfg.get("block_dims", [32, 64, 128]),
        help="Feature dimensions across model stages.",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=model_cfg.get("in_channels", 3),
        help="Number of input image channels.",
    )
    parser.add_argument(
        "--out-channels",
        type=int,
        default=model_cfg.get("out_channels", 512),
        help="Output feature dimension of the backbone.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=model_cfg.get("patch_size", 8),
        help="Patch/stride size for the initial stem convolution.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=model_cfg.get("dropout", 0.2),
        help="Dropout rate in model blocks.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=model_cfg.get("num_classes", 10),
        help="Number of output classes.",
    )
    parser.add_argument(
        "--optimize-model",
        action=argparse.BooleanOptionalAction,
        default=model_cfg.get("optimize", False),
        help="Compile model using torch.compile.",
    )


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


def parse_args() -> argparse.Namespace:
    r"""parse_args() -> argparse.Namespace

    Parse CLI options merged with default YAML configuration settings.

    Returns:
        argparse.Namespace: Populated argument namespace.
    """
    config_parser, known_args = _create_config_parser()
    cfg = _load_config(known_args)

    parser = argparse.ArgumentParser(
        description="Train a moliv_rl PyTorch classifier.",
        parents=[config_parser],
    )

    _add_data_args(parser, cfg)
    _add_model_args(parser, cfg)
    _add_optimizer_args(parser, cfg)
    _add_loss_args(parser, cfg)
    _add_scheduler_args(parser, cfg)
    _add_training_args(parser, cfg)

    return parser.parse_args()


def build_datasets(
    args: argparse.Namespace,
) -> tuple[ImageFolder, ImageFolder]:
    r"""build_datasets(args) -> Tuple[ImageFolder, ImageFolder]

    Construct train and validation :class:`~torchvision.datasets.ImageFolder` datasets from CLI arguments.

    Args:
        args (argparse.Namespace): Parsed experiment configuration.

    Returns:
        tuple: A pair ``(train_dataset, val_dataset)`` of instantiated datasets.
    """
    train_dir = args.data_dir / args.train_split
    val_dir = args.data_dir / args.val_split

    train_dataset = ImageFolder(
        root=train_dir,
        transform=get_train_transforms(
            image_size=args.image_size,
            resize_scale=args.resize_scale,
        ),
    )

    val_dataset = ImageFolder(
        root=val_dir,
        transform=get_val_transforms(
            image_size=args.image_size,
            resize_scale=args.resize_scale,
        ),
    )

    return train_dataset, val_dataset


def build_dataloaders(
    train_dataset: ImageFolder,
    val_dataset: ImageFolder,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    r"""build_dataloaders(train_dataset, val_dataset, args, device) -> Tuple[DataLoader, DataLoader]

    Construct hardware-optimized, seeded training and validation DataLoaders.

    Args:
        train_dataset (ImageFolder): Training dataset instance.
        val_dataset (ImageFolder): Validation dataset instance.
        args (argparse.Namespace): Parsed CLI arguments containing batch size, workers, and seed.
        device (torch.device): Execution device used to infer default memory pinning.

    Returns:
        tuple: A pair ``(train_loader, val_loader)`` of instantiated DataLoaders.
    """
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    pin_memory = (
        args.pin_memory if args.pin_memory is not None else (device.type == "cuda")
    )
    persistent_workers = (
        args.persistent_workers
        if args.persistent_workers is not None
        else (args.num_workers > 0)
    )

    config = DataLoaderConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=generator,
        worker_init_fn=seed_worker,
    )

    return get_dataloaders(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )


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
        steps_per_epoch (int): Total batch steps per epoch used to size step-wise schedulers.

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


def main() -> None:
    args = parse_args()

    set_seeds(args.seed)

    device = resolve_device(args.device)

    log_dir = args.checkpoint_dir / "logs"
    logger = get_logger(
        "train",
        log_dir=log_dir,
    )

    logger.info("Starting training")
    logger.info("Device: %s", device)
    logger.info("Arguments: %s", _serialize_args(vars(args)))

    train_dataset, val_dataset = build_datasets(args)

    logger.info(
        "Classes: %s",
        train_dataset.classes,
    )
    logger.info(
        "Training samples: %d",
        len(train_dataset),
    )
    logger.info(
        "Validation samples: %d",
        len(val_dataset),
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
            ),
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

        checkpoint_metadata = {
            "args": _serialize_args(vars(args)),
            "classes": train_dataset.classes,
            "class_to_idx": train_dataset.class_to_idx,
            **val_metrics,
        }

        if args.save_last:
            trainer.save_checkpoint(
                args.checkpoint_dir / "model_last.pth",
                epoch=epoch,
                extra=checkpoint_metadata,
            )

        if val_metrics["val_acc"] >= best_val_acc:
            best_val_acc = val_metrics["val_acc"]

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

    logger.info(
        "Training complete. Best validation accuracy: %.4f",
        best_val_acc,
    )


if __name__ == "__main__":
    main()
