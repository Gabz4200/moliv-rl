from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from moliv_rl.data import get_val_transforms
from moliv_rl.metrics import PrecisionAverage
from moliv_rl.models import MODEL_REGISTRY, get_model
from moliv_rl.train.trainer import ClassificationTrainer
from moliv_rl.utils.logger import get_logger
from scripts.utils import load_config, resolve_device


def parse_args() -> argparse.Namespace:
    r"""parse_args() -> argparse.Namespace

    Parse evaluation CLI arguments merged with default YAML configuration values.

    Returns:
        argparse.Namespace: Populated argument namespace.
    """
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    known_args, _ = config_parser.parse_known_args()

    cfg: dict[str, Any] = {}
    if known_args.config and known_args.config.is_file():
        cfg = load_config(known_args.config)
    elif known_args.config and known_args.config != Path("configs/default.yaml"):
        raise FileNotFoundError(f"Config file not found: {known_args.config}")

    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    runtime_cfg = cfg.get("runtime", {})

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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(paths_cfg.get("data_dir", "data")),
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
        default=data_cfg.get("batch_size", 32),
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=data_cfg.get("num_workers", 0),
        help="Number of DataLoader worker processes.",
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
        help="Number of output classes in the model.",
    )
    parser.add_argument(
        "--optimize-model",
        action=argparse.BooleanOptionalAction,
        default=model_cfg.get("optimize", False),
        help="Compile model using torch.compile.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=data_cfg.get("image_size", 64),
        help="Input image size expected by the model.",
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
        help="Compute device, for example cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=training_cfg.get("use_amp", False),
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


def create_dataset(args: argparse.Namespace) -> ImageFolder:
    r"""create_dataset(args) -> ImageFolder

    Create an :class:`~torchvision.datasets.ImageFolder` dataset for the specified evaluation split.

    Args:
        args (argparse.Namespace): Parsed CLI arguments containing data directory and split name.

    Returns:
        ImageFolder: Instantiated dataset for evaluation.
    """
    split_dir = args.data_dir / args.split
    return ImageFolder(
        root=split_dir,
        transform=get_val_transforms(
            image_size=args.image_size,
        ),
    )


def create_dataloader(
    dataset: ImageFolder,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    r"""create_dataloader(dataset, args, device) -> DataLoader

    Create an evaluation :class:`~torch.utils.data.DataLoader`.

    Args:
        dataset (ImageFolder): Evaluation dataset instance.
        args (argparse.Namespace): Parsed CLI arguments containing batch size and worker counts.
        device (torch.device): Execution device used to infer memory pinning.

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
