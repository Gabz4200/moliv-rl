from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, cast

import torch
from moliv_rl.metrics.metrics import PrecisionAverage
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from moliv_rl.data.transforms import (
    get_train_transforms,
    get_val_transforms,
)
from moliv_rl.models.my_model import MyModel
from moliv_rl.train.trainer import ClassificationTrainer
from moliv_rl.utils.logger import get_logger
from moliv_rl.utils.reproducibility import seed_worker, set_seeds

# I am thinking about moving all that to Pytorch Lightning so it handles the training, but I am not sure yet, maybe later.


def _serialize_args(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Convert argparse values into checkpoint-safe metadata."""
    serialized: dict[str, Any] = {}

    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value

    return serialized


def resolve_device(device: str | None) -> torch.device:
    """Resolve the requested device or select a reasonable default."""
    if device is not None:
        resolved = torch.device(device)

        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        if resolved.type == "mps":
            if not hasattr(torch.backends, "mps"):
                raise RuntimeError("MPS is not available in this PyTorch build")

            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is not available")

        return resolved

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a moliv_rl PyTorch classifier.")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing train and val subdirectories.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory used for checkpoints and logs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training and validation batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of output classes.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Input image height and width.",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        default=1.15,
        help="Resize scale used before cropping.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of batches to accumulate before optimizer.step().",
    )
    parser.add_argument(
        "--precision-average",
        choices=("micro", "macro", "weighted"),
        default="macro",
        help="Averaging strategy for validation precision.",
    )
    parser.add_argument(
        "--scheduler-interval",
        choices=("epoch", "step"),
        default="epoch",
        help="How often to step the learning-rate scheduler.",
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=0.0,
        help="Minimum learning rate for cosine annealing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device, for example cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use CUDA automatic mixed precision.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}")

    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")

    if args.lr <= 0:
        raise ValueError(f"lr must be positive, got {args.lr}")

    if args.weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {args.weight_decay}")

    if args.num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {args.num_classes}")

    if args.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {args.image_size}")

    if args.resize_scale < 1.0:
        raise ValueError(f"resize_scale must be >= 1.0, got {args.resize_scale}")

    if args.num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {args.num_workers}")

    if args.grad_accum_steps <= 0:
        raise ValueError(
            f"grad_accum_steps must be positive, got {args.grad_accum_steps}"
        )

    if args.eta_min < 0:
        raise ValueError(f"eta_min must be non-negative, got {args.eta_min}")

    if args.use_amp and args.device == "cpu":
        raise ValueError("--use-amp requires a CUDA device")


def build_datasets(
    args: argparse.Namespace,
) -> tuple[ImageFolder, ImageFolder]:
    """Build train and validation ImageFolder datasets."""
    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training split directory not found: {train_dir}")

    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation split directory not found: {val_dir}")

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

    if len(train_dataset) == 0:
        raise ValueError(f"No training images found in {train_dir}")

    if len(val_dataset) == 0:
        raise ValueError(f"No validation images found in {val_dir}")

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError(
            "Training and validation class mappings differ: "
            f"train={train_dataset.class_to_idx}, "
            f"val={val_dataset.class_to_idx}"
        )

    if len(train_dataset.classes) != args.num_classes:
        raise ValueError(
            "Dataset class count does not match --num-classes: "
            f"dataset={len(train_dataset.classes)}, "
            f"argument={args.num_classes}"
        )

    return train_dataset, val_dataset


def build_dataloaders(
    train_dataset: ImageFolder,
    val_dataset: ImageFolder,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Build deterministic, seeded train and validation loaders."""
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0

    common_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "worker_init_fn": seed_worker,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_kwargs,
    )

    return train_loader, val_loader


def build_trainer(
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
    steps_per_epoch: int,
) -> ClassificationTrainer:
    """Build the model, optimizer, scheduler, and trainer."""
    model = MyModel(
        num_classes=args.num_classes,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.scheduler_interval == "epoch":
        scheduler_t_max = args.epochs
    else:
        optimizer_steps_per_epoch = (
            steps_per_epoch + args.grad_accum_steps - 1
        ) // args.grad_accum_steps
        scheduler_t_max = args.epochs * optimizer_steps_per_epoch

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_t_max,
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
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

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

        trainer.save_checkpoint(
            args.checkpoint_dir / "model_last.pth",
            epoch=epoch,
            extra=checkpoint_metadata,
        )

        if val_metrics["val_acc"] >= best_val_acc:
            best_val_acc = val_metrics["val_acc"]

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
