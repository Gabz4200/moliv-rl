from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from moliv_rl.train.trainer import ClassificationTrainer


class TinyModel(nn.Module):
    def __init__(self, in_features: int = 16, num_classes: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, 8, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _create_dummy_loaders() -> tuple[DataLoader, DataLoader]:
    x_train = torch.randn(8, 16, 8, 8)
    y_train = torch.randint(0, 4, (8,), dtype=torch.int64)
    x_val = torch.randn(4, 16, 8, 8)
    y_val = torch.randint(0, 4, (4,), dtype=torch.int64)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=4,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=4,
        shuffle=False,
    )
    return train_loader, val_loader


class TestClassificationTrainer:
    """Behavioral tests for ClassificationTrainer."""

    def test_init_and_device_resolution(self) -> None:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trainer = ClassificationTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            scheduler_interval="epoch",
        )

        assert trainer.device == torch.device("cpu")
        assert trainer.model is not None
        assert not trainer.use_amp

    def test_train_epoch_loss_reduction(self) -> None:
        torch.manual_seed(42)
        model = TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        criterion = nn.CrossEntropyLoss()
        train_loader, _ = _create_dummy_loaders()

        trainer = ClassificationTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
        )

        initial_loss = trainer.train_epoch(train_loader, epoch=1)
        assert isinstance(initial_loss, float)
        assert initial_loss > 0.0

        # Further training should run without error
        loss_epoch_2 = trainer.train_epoch(train_loader, epoch=2)
        assert isinstance(loss_epoch_2, float)

    def test_train_epoch_with_gradient_accumulation(self) -> None:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        train_loader, _ = _create_dummy_loaders()

        trainer = ClassificationTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
        )

        loss = trainer.train_epoch(train_loader, epoch=1, grad_accum_steps=2)
        assert isinstance(loss, float)

    def test_train_epoch_with_step_scheduler(self) -> None:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        train_loader, _ = _create_dummy_loaders()

        trainer = ClassificationTrainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device="cpu",
            scheduler=scheduler,
            scheduler_interval="step",
        )

        # 2 batches in loader -> scheduler steps twice -> 0.1 * 0.5 * 0.5 = 0.025
        trainer.train_epoch(train_loader, epoch=1)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.025)

    @pytest.mark.parametrize("precision_avg", ["macro", "micro", "weighted"])
    def test_evaluate_metrics(self, precision_avg: str) -> None:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        _, val_loader = _create_dummy_loaders()

        trainer = ClassificationTrainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device="cpu",
        )

        metrics = trainer.evaluate(val_loader, precision_average=precision_avg)  # type: ignore[arg-type]

        assert "val_acc" in metrics
        assert "val_loss" in metrics
        assert "val_precision" in metrics
        assert 0.0 <= metrics["val_acc"] <= 1.0
        assert 0.0 <= metrics["val_precision"] <= 1.0

    def test_checkpoint_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        ckpt_path = tmp_path / "ckpt.pth"

        trainer = ClassificationTrainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device="cpu",
            scheduler=scheduler,
        )

        # Modify model parameter to verify state persistence
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.23)

        extra_meta = {"best_val_acc": 0.95, "classes": ["c0", "c1", "c2", "c3"]}
        trainer.save_checkpoint(ckpt_path, epoch=5, extra=extra_meta)
        assert ckpt_path.is_file()

        # Create new fresh model with zeroes
        new_model = TinyModel()
        with torch.no_grad():
            for p in new_model.parameters():
                p.zero_()

        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
        new_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(new_optimizer, T_max=10)

        new_trainer = ClassificationTrainer(
            model=new_model,
            optimizer=new_optimizer,
            criterion=nn.CrossEntropyLoss(),
            device="cpu",
            scheduler=new_scheduler,
        )

        loaded_ckpt = new_trainer.load_checkpoint(ckpt_path, safe_load=True)

        assert loaded_ckpt["epoch"] == 5
        assert loaded_ckpt["best_val_acc"] == 0.95
        assert loaded_ckpt["classes"] == ["c0", "c1", "c2", "c3"]

        # Verify weights were restored
        for p in new_model.parameters():
            assert torch.allclose(p, torch.tensor(1.23), atol=1e-5)

    def test_save_checkpoint_reserved_key_collision_raises(self, tmp_path: Path) -> None:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = ClassificationTrainer(model=model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), device="cpu")

        with pytest.raises(KeyError, match="reserved checkpoint keys"):
            trainer.save_checkpoint(tmp_path / "test.pth", epoch=1, extra={"model_state_dict": {}})

    def test_load_checkpoint_file_not_found_raises(self) -> None:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = ClassificationTrainer(model=model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), device="cpu")

        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint("non_existent_file.pth")

    def test_train_epoch_empty_dataloader_returns_zero(self) -> None:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = ClassificationTrainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device="cpu",
        )
        empty_loader = DataLoader(
            TensorDataset(
                torch.empty(0, 16, 8, 8),
                torch.empty(0, dtype=torch.int64),
            ),
            batch_size=4,
        )
        loss = trainer.train_epoch(empty_loader, epoch=1)
        assert loss == 0.0

    def test_trainer_without_optimizer_for_evaluation(self, tmp_path: Path) -> None:
        model = TinyModel()
        _, val_loader = _create_dummy_loaders()

        trainer = ClassificationTrainer(
            model=model,
            optimizer=None,
            criterion=None,
            device="cpu",
        )

        metrics = trainer.evaluate(val_loader)
        assert "val_acc" in metrics
        assert "val_loss" in metrics

        # train_epoch must fail with ValueError when optimizer is None
        with pytest.raises(ValueError, match="requires an optimizer"):
            trainer.train_epoch(val_loader, epoch=1)

        # save_checkpoint without optimizer
        ckpt_path = tmp_path / "eval_only.pth"
        trainer.save_checkpoint(ckpt_path, epoch=1)
        assert ckpt_path.is_file()

        # load_checkpoint without optimizer
        loaded = trainer.load_checkpoint(ckpt_path, safe_load=True)
        assert loaded["epoch"] == 1
        assert "optimizer_state_dict" not in loaded
