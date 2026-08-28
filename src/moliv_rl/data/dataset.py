from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

from .transforms import get_train_transforms, get_val_transforms


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
    batch_size: int = 32,
    num_workers: int = 4,
    train_dataset: Dataset | None = None,
    val_dataset: Dataset | None = None,
    pin_memory: bool = True,
    persistent_workers: bool | None = None,
    generator: torch.Generator | None = None,
    worker_init_fn: Any = None,
    data_dir: Path | str = "data",
    image_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    r"""get_dataloaders(batch_size=32, num_workers=4, train_dataset=None, val_dataset=None, pin_memory=True, persistent_workers=None, generator=None, worker_init_fn=None, data_dir='data', image_size=64) -> Tuple[DataLoader, DataLoader]

    Create hardware-optimized :class:`~torch.utils.data.DataLoader` instances for training and validation.

    Args:
        batch_size (int, optional): Mini-batch size per iteration. Default: ``32``
        num_workers (int, optional): Number of asynchronous worker subprocesses. Default: ``4``
        train_dataset (Dataset, optional): Custom training dataset. If ``None``, loads from :attr:`data_dir`. Default: ``None``
        val_dataset (Dataset, optional): Custom validation dataset. If ``None``, loads from :attr:`data_dir`. Default: ``None``
        pin_memory (bool, optional): Pin host memory to expedite host-to-device CUDA copies. Default: ``True``
        persistent_workers (bool, optional): Maintain active worker processes between epochs. Default: ``True`` when ``num_workers > 0``
        generator (Generator, optional): PyTorch random number generator for batch shuffling determinism. Default: ``None``
        worker_init_fn (callable, optional): Worker seeding initialization function. Default: ``None``
        data_dir (Path or str, optional): Root dataset directory used if datasets are omitted. Default: ``'data'``
        image_size (int, optional): Image resolution applied to default dataset transforms. Default: ``64``

    Returns:
        tuple: A pair ``(train_loader, val_loader)`` of configured :class:`~torch.utils.data.DataLoader` objects.
    """
    train_ds = train_dataset
    val_ds = val_dataset

    if train_ds is None or val_ds is None:
        default_train_ds, default_val_ds = get_default_datasets(
            data_dir=data_dir,
            image_size=image_size,
        )
        if train_ds is None:
            train_ds = default_train_ds
        if val_ds is None:
            val_ds = default_val_ds

    use_persistent = (
        (persistent_workers if persistent_workers is not None else (num_workers > 0))
        if num_workers > 0
        else False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, val_loader


__all__ = [
    "get_dataloaders",
    "get_default_datasets",
]
