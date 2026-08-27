from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder

from moliv_rl.data import get_dataloaders, get_default_datasets


@pytest.fixture
def dummy_data_dir(tmp_path: Path) -> Path:
    """Create a temporary dataset with train and val splits."""
    for split in ["train", "val"]:
        for cls_name in ["class_0", "class_1"]:
            cls_dir = tmp_path / split / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(2):
                img_arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                img = Image.fromarray(img_arr)
                img.save(cls_dir / f"img_{idx}.png")
    return tmp_path


class TestDatasetAndDataLoaders:
    """Behavioral tests for dataset and DataLoader creation."""

    def test_get_default_datasets(self, dummy_data_dir: Path) -> None:
        train_ds, val_ds = get_default_datasets(data_dir=dummy_data_dir, image_size=32)
        assert isinstance(train_ds, ImageFolder)
        assert isinstance(val_ds, ImageFolder)
        assert len(train_ds) == 4
        assert len(val_ds) == 4

    def test_get_dataloaders_default_directory(self, dummy_data_dir: Path) -> None:
        train_loader, val_loader = get_dataloaders(
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            data_dir=dummy_data_dir,
            image_size=32,
        )
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        batch = next(iter(train_loader))
        assert batch[0].shape == (2, 3, 32, 32)

    def test_get_dataloaders_custom_dataset(self) -> None:
        x_train = torch.randn(10, 3, 64, 64)
        y_train = torch.randint(0, 5, (10,))
        x_val = torch.randn(4, 3, 64, 64)
        y_val = torch.randint(0, 5, (4,))

        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)

        train_loader, val_loader = get_dataloaders(
            batch_size=2,
            num_workers=0,
            train_dataset=train_ds,
            val_dataset=val_ds,
            pin_memory=False,
        )

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert train_loader.batch_size == 2
        assert val_loader.batch_size == 2

        # Check train batch
        batch_train = next(iter(train_loader))
        assert batch_train[0].shape == (2, 3, 64, 64)
        assert batch_train[1].shape == (2,)

        # Check val batch
        batch_val = next(iter(val_loader))
        assert batch_val[0].shape == (2, 3, 64, 64)
        assert batch_val[1].shape == (2,)

    def test_dataloader_shuffle_contracts(self) -> None:
        x_train = torch.randn(8, 1)
        y_train = torch.arange(8)
        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_train, y_train)

        train_loader, val_loader = get_dataloaders(
            batch_size=8,
            num_workers=0,
            train_dataset=train_ds,
            val_dataset=val_ds,
            pin_memory=False,
        )

        assert train_loader.sampler is not None
        # Val loader uses SequentialSampler (no shuffle)
        assert isinstance(
            val_loader.sampler,
            torch.utils.data.SequentialSampler,
        )

    def test_persistent_workers_behavior_with_zero_workers(self) -> None:
        x_train = torch.randn(4, 1)
        y_train = torch.arange(4)
        train_ds = TensorDataset(x_train, y_train)

        train_loader, val_loader = get_dataloaders(
            batch_size=2,
            num_workers=0,
            train_dataset=train_ds,
            val_dataset=train_ds,
            persistent_workers=True,
            pin_memory=False,
        )

        # persistent_workers must be False if num_workers == 0
        assert not train_loader.persistent_workers
        assert not val_loader.persistent_workers
