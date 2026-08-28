from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from .transforms import get_train_transforms, get_val_transforms


@dataclass
class DataLoaderConfig:
    """Configuration container for :func:`get_dataloaders`.

    Grouping DataLoader options here keeps the public factory signature stable
    as new options are added.
    """

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool | None = None
    generator: torch.Generator | None = None
    worker_init_fn: Any = None
    data_dir: Path | str = "data"
    image_size: int = 64


def get_default_datasets(
    data_dir: Path | str = "data",
    image_size: int = 64,
) -> tuple[ImageFolder, ImageFolder]:
    r"""get_default_datasets(data_dir='data', image_size=64) -> Tuple[ImageFolder, ImageFolder]

    Create train and validation :class:`~torchvision.datasets.ImageFolder` datasets on demand.

    Args:
        data_dir (Path or str, optional): Root directory path containing ``'train'`` and ``'val'`` subdirectories. Default: ``'data'``
        image_size (int, optional): Spatial image resolution for dataset transform pipelines. Default: ``64``

    Returns:
        tuple: A pair ``(train_dataset, val_dataset)`` of instantiated :class:`~torchvision.datasets.ImageFolder` objects.
    """
    root = Path(data_dir)
    train_ds = ImageFolder(
        root=str(root / "train"),
        transform=get_train_transforms(image_size=image_size),
    )
    val_ds = ImageFolder(
        root=str(root / "val"),
        transform=get_val_transforms(image_size=image_size),
    )
    return train_ds, val_ds


def get_dataloaders(
    config: DataLoaderConfig,
    train_dataset: Dataset | None = None,
    val_dataset: Dataset | None = None,
) -> tuple[DataLoader, DataLoader]:
    r"""get_dataloaders(config, train_dataset=None, val_dataset=None) -> Tuple[DataLoader, DataLoader]

    Create hardware-optimized :class:`~torch.utils.data.DataLoader` instances for training and validation.

    Args:
        config (DataLoaderConfig): Container for DataLoader hyperparameters and defaults.
        train_dataset (Dataset, optional): Custom training dataset. If ``None``, loads from :attr:`~DataLoaderConfig.data_dir`. Default: ``None``
        val_dataset (Dataset, optional): Custom validation dataset. If ``None``, loads from :attr:`~DataLoaderConfig.data_dir`. Default: ``None``

    Returns:
        tuple: A pair ``(train_loader, val_loader)`` of configured :class:`~torch.utils.data.DataLoader` objects.
    """
    train_ds = train_dataset
    val_ds = val_dataset

    if train_ds is None or val_ds is None:
        default_train_ds, default_val_ds = get_default_datasets(
            data_dir=config.data_dir,
            image_size=config.image_size,
        )
        if train_ds is None:
            train_ds = default_train_ds
        if val_ds is None:
            val_ds = default_val_ds

    use_persistent = (
        (config.persistent_workers if config.persistent_workers is not None else (config.num_workers > 0))
        if config.num_workers > 0
        else False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=use_persistent,
        generator=config.generator,
        worker_init_fn=config.worker_init_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=use_persistent,
        worker_init_fn=config.worker_init_fn,
    )

    return train_loader, val_loader


__all__ = [
    "get_dataloaders",
    "get_default_datasets",
]
