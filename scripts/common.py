from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from moliv_rl.data import (
    DataLoaderConfig,
    get_dataloaders,
    get_default_datasets,
    get_streaming_gameqa_datasets,
)
from moliv_rl.models import MODEL_REGISTRY
from moliv_rl.utils.reproducibility import seed_worker
from scripts.utils import load_config


def create_config_parser() -> argparse.ArgumentParser:
    """Return a parent parser that handles --config loading."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML configuration file.",
    )
    return config_parser


def load_yaml_config(known_args: argparse.Namespace) -> dict[str, Any]:
    """Load the YAML config referenced by known_args.config."""
    cfg: dict[str, Any] = {}
    if known_args.config and known_args.config.is_file():
        cfg = load_config(known_args.config)
    elif known_args.config and known_args.config != Path("configs/default.yaml"):
        raise FileNotFoundError(f"Config file not found: {known_args.config}")
    return cfg


def add_data_args(parser: argparse.ArgumentParser, cfg: dict[str, Any]) -> None:
    """Register shared data-related CLI arguments on parser."""
    paths_cfg = cfg.get("paths", {})
    data_cfg = cfg.get("data", {})

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(paths_cfg.get("data_dir", "data")),
        help="Directory containing split subdirectories.",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["local", "streaming"],
        default=data_cfg.get("dataset_type", "local"),
        help="Dataset source type.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=data_cfg.get("dataset_id", "OpenMOSS-Team/GameQA-140K"),
        help="Hugging Face dataset ID for streaming mode.",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default=data_cfg.get("train_split", "train"),
        help="Training split name.",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default=data_cfg.get("validation_split", "validation"),
        help="Validation split name.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=data_cfg.get("image_size", 64),
        help="Input image size expected by the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=data_cfg.get("batch_size", 32),
        help="Batch size.",
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
        "--max-train-samples",
        type=int,
        default=data_cfg.get("max_train_samples", None),
        help="Optional limit on training samples (streaming mode).",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=data_cfg.get("max_val_samples", None),
        help="Optional limit on validation samples (streaming mode).",
    )


def add_model_args(parser: argparse.ArgumentParser, cfg: dict[str, Any]) -> None:
    """Register shared model-related CLI arguments on parser."""
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
        default=model_cfg.get("num_classes", 30),
        help="Number of output classes.",
    )
    parser.add_argument(
        "--optimize-model",
        action=argparse.BooleanOptionalAction,
        default=model_cfg.get("optimize", False),
        help="Compile model using torch.compile.",
    )


def build_local_datasets(args: argparse.Namespace) -> tuple[Any, Any]:
    """Build local ImageFolder datasets from args."""
    from torchvision.datasets import ImageFolder

    from moliv_rl.data.transforms import get_train_transforms, get_val_transforms

    train_dir = args.data_dir / args.train_split
    val_dir = args.data_dir / args.val_split

    train_dataset = ImageFolder(
        root=train_dir,
        transform=get_train_transforms(image_size=args.image_size),
    )
    val_dataset = ImageFolder(
        root=val_dir,
        transform=get_val_transforms(image_size=args.image_size),
    )
    return train_dataset, val_dataset


def build_streaming_datasets(args: argparse.Namespace) -> tuple[Any, Any | None]:
    """Build streaming Hugging Face datasets from args."""
    return get_streaming_gameqa_datasets(
        image_size=args.image_size,
        train_split=args.train_split,
        val_split=args.val_split,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )


def build_datasets(args: argparse.Namespace) -> tuple[Any, Any]:
    """Dispatch dataset construction based on args.dataset_type."""
    if getattr(args, "dataset_type", "local") == "streaming":
        return build_streaming_datasets(args)
    return build_local_datasets(args)


def build_dataloaders(
    train_dataset: Any,
    val_dataset: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Build seeded, hardware-optimized dataloaders from args."""
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
