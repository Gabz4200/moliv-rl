from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from moliv_rl.data.transforms import get_val_transforms
from moliv_rl.metrics import PrecisionAverage
from moliv_rl.models import get_model
from moliv_rl.train.trainer import ClassificationTrainer
from moliv_rl.utils.logger import get_logger


def load_config(config_path: Path | str | None) -> dict[str, Any]:
    """Load configuration from a YAML file if present."""
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def parse_args() -> argparse.Namespace:
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
        choices=("classification_model", "my_model"),
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


def resolve_device(device: str | None) -> torch.device:
    if device is not None and device != "auto":
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


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")

    if args.num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {args.num_workers}")

    if args.num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {args.num_classes}")

    if args.in_channels <= 0:
        raise ValueError(f"in_channels must be positive, got {args.in_channels}")

    if args.out_channels <= 0:
        raise ValueError(f"out_channels must be positive, got {args.out_channels}")

    if args.patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {args.patch_size}")

    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError(f"dropout must be in [0.0, 1.0), got {args.dropout}")

    if len(args.block_dims) < 2:
        raise ValueError(
            f"block_dims must contain at least 2 dimensions, got {args.block_dims}"
        )

    if any(dim <= 0 for dim in args.block_dims):
        raise ValueError(f"All block_dims must be positive, got {args.block_dims}")

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
