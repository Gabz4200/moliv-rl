from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
from collections.abc import Sized
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from moliv_rl.metrics import PrecisionAverage
from moliv_rl.models import MODEL_REGISTRY  # noqa: F401
from moliv_rl.train.trainer import ClassificationTrainer
from moliv_rl.utils.logger import get_logger
from scripts.common import (
    add_data_args,
    add_model_args,
    build_datasets,
    build_model,
    create_config_parser,
    load_yaml_config,
)
from scripts.utils import resolve_device


def parse_args() -> argparse.Namespace:
    r"""parse_args() -> argparse.Namespace

    Parse evaluation CLI arguments merged with default YAML configuration values.

    Returns:
        argparse.Namespace: Populated argument namespace.
    """
    config_parser = create_config_parser()
    known_args, _ = config_parser.parse_known_args()
    cfg = load_yaml_config(known_args)

    parser = argparse.ArgumentParser(
        description="Evaluate a moliv_rl classification checkpoint.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint file.",
    )
    add_data_args(parser, cfg)
    add_model_args(parser, cfg)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--precision-average",
        choices=("micro", "macro", "weighted"),
        default=cfg.get("training", {}).get("precision_average", "macro"),
        help="Averaging strategy for validation precision.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg.get("runtime", {}).get("device", "auto"),
        help="Compute device, for example cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=cfg.get("training", {}).get("use_amp", False),
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


def create_dataset(args: argparse.Namespace) -> Dataset:
    r"""create_dataset(args) -> Dataset

    Create a dataset for the specified evaluation split. Supports both local
    ImageFolder datasets and streaming Hugging Face datasets.

    Args:
        args: Parsed CLI arguments containing data configuration.

    Returns:
        Dataset: Instantiated dataset for evaluation.
    """
    train_dataset, val_dataset = build_datasets(args)

    if args.split == "train":
        return train_dataset

    if val_dataset is not None:
        return val_dataset

    if hasattr(train_dataset, "label2id") and train_dataset.label2id:
        return train_dataset

    raise ValueError(
        f"Unable to resolve evaluation split '{args.split}'. "
        "Expected 'train' or a validation split."
    )


def create_dataloader(
    dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    r"""create_dataloader(dataset, args, device) -> DataLoader

    Create an evaluation DataLoader.

    Args:
        dataset: Evaluation dataset instance.
        args: Parsed CLI arguments containing batch size and worker counts.
        device: Execution device used to infer memory pinning.

    Returns:
        DataLoader: Configured evaluation DataLoader.
    """
    pin_memory = (
        args.pin_memory if args.pin_memory is not None else (device.type == "cuda")
    )
    persistent_workers = (
        args.persistent_workers
        if args.persistent_workers is not None
        else (args.num_workers > 0)
    )
    use_persistent = persistent_workers if args.num_workers > 0 else False

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
    )


def create_trainer(
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> ClassificationTrainer:
    r"""create_trainer(args, device, logger) -> ClassificationTrainer

    Build model architecture and evaluation trainer without optimizer state.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        device (torch.device): Compute device for model placement.
        logger (logging.Logger): Logger instance for evaluation metrics.

    Returns:
        ClassificationTrainer: Initialized evaluation trainer.
    """
    model = build_model(args, device)

    return ClassificationTrainer(
        model=model,
        optimizer=None,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        logger=logger,
        use_amp=args.use_amp,
    )


def main() -> None:
    args = parse_args()

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

    if isinstance(dataset, ImageFolder):
        num_classes = len(dataset.classes)
    elif hasattr(dataset, "label2id") and dataset.label2id:
        num_classes = len(dataset.label2id)
    else:
        num_classes = getattr(args, "num_classes", "?")

    logger.info(
        "Evaluating %d samples across %d classes from split '%s'",
        len(cast(Sized, dataset)),
        num_classes,
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
