from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import cast

import torch
from moliv_rl.metrics.metrics import PrecisionAverage
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from moliv_rl.data.transforms import get_val_transforms
from moliv_rl.models.my_model import MyModel
from moliv_rl.train.trainer import ClassificationTrainer
from moliv_rl.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a moliv_rl classification checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing split subdirectories.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of output classes in MyModel.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Input image size expected by the model.",
    )
    parser.add_argument(
        "--precision-average",
        choices=("micro", "macro", "weighted"),
        default="macro",
        help="Averaging strategy for validation precision.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device, for example cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use CUDA automatic mixed precision during evaluation.",
    )
    parser.add_argument(
        "--safe-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use PyTorch's restricted checkpoint loader. Disable only for "
            "trusted checkpoints requiring unrestricted deserialization."
        ),
    )
    return parser.parse_args()


def resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")

    if args.num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {args.num_workers}")

    if args.num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {args.num_classes}")

    if args.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {args.image_size}")

    if not args.split:
        raise ValueError("split must not be empty")


def create_dataset(args: argparse.Namespace) -> ImageFolder:
    split_dir = args.data_dir / args.split

    if not split_dir.is_dir():
        raise FileNotFoundError(f"Evaluation split directory not found: {split_dir}")

    dataset = ImageFolder(
        root=split_dir,
        transform=get_val_transforms(
            image_size=args.image_size,
        ),
    )

    if len(dataset) == 0:
        raise ValueError(f"No images found in evaluation split: {split_dir}")

    if len(dataset.classes) != args.num_classes:
        raise ValueError(
            "Dataset class count does not match --num-classes: "
            f"dataset={len(dataset.classes)}, "
            f"argument={args.num_classes}"
        )

    return dataset


def create_dataloader(
    dataset: ImageFolder,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    pin_memory = device.type == "cuda"

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )


def create_trainer(
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> ClassificationTrainer:
    model = MyModel(num_classes=args.num_classes)

    # The optimizer is required by ClassificationTrainer even though it is
    # not used to calculate evaluation metrics.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.0,
    )

    return ClassificationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        logger=logger,
        use_amp=args.use_amp,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    logger = get_logger("evaluate")
    device = resolve_device(args.device)

    logger.info("Using device: %s", device)
    logger.info("Loading checkpoint: %s", args.checkpoint)

    trainer = create_trainer(
        args=args,
        device=device,
        logger=logger,
    )

    checkpoint = trainer.load_checkpoint(
        args.checkpoint,
        safe_load=args.safe_load,
    )

    logger.info(
        "Loaded checkpoint from %s at epoch %s",
        args.checkpoint,
        checkpoint.get("epoch", "unknown"),
    )

    dataset = create_dataset(args)
    dataloader = create_dataloader(
        dataset=dataset,
        args=args,
        device=device,
    )

    logger.info(
        "Evaluating %d samples across %d classes from split '%s'",
        len(dataset),
        len(dataset.classes),
        args.split,
    )

    metrics = trainer.evaluate(
        dataloader,
        precision_average=cast(
            PrecisionAverage,
            args.precision_average,
        ),
    )

    logger.info(
        "Evaluation results: val_loss=%.6f val_acc=%.4f val_precision=%.4f",
        metrics["val_loss"],
        metrics["val_acc"],
        metrics["val_precision"],
    )


if __name__ == "__main__":
    main()
